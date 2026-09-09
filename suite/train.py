"""Four explicit target-server training stages; no evaluation labels are loaded here."""
import contextlib,math,random,time
import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoModelForCausalLM
from . import common as C
from .engine import Engine
from .models import SlotCompressor,InterlatReceiver,InterlatCompressor,gradient_setup,frozen
from .losses import js,reverse_kl,dynamic_weight,latent_margin,uncertainty_kl

def build_module(family,stage,cfg,dtype=torch.float32,device='cuda:0'):
 path=C.model_path(family)
 if stage.startswith('latcom'):
  base=AutoModel.from_pretrained(path,torch_dtype=dtype,attn_implementation='sdpa',local_files_only=True);obj=SlotCompressor(base,cfg['latcom_slots'])
 elif stage=='interlat_receiver':
  base=AutoModelForCausalLM.from_pretrained(path,torch_dtype=dtype,attn_implementation='sdpa',local_files_only=True);obj=InterlatReceiver(base)
 else:
  base=AutoModel.from_pretrained(path,torch_dtype=dtype,attn_implementation='sdpa',local_files_only=True);obj=InterlatCompressor(base,cfg['interlat_slots_per_sender'])
 return obj.to(device=device,dtype=dtype)

def load_export(family,stage,cfg,device='cuda:0'):
 directory=C.stage_dir(family,stage);meta=C.read(directory/'complete.json')
 if not meta:raise RuntimeError('Required trained stage not complete: '+stage)
 if meta['run_identity']!=C.identity(cfg):raise ValueError('Trained-stage configuration/source identity differs')
 if C.sha(directory/'weights.pt')!=meta['weights_sha256']:raise ValueError('Trained export hash mismatch')
 obj=build_module(family,stage,cfg,torch.bfloat16,device)
 state=torch.load(directory/'weights.pt',map_location='cpu',weights_only=True);obj.load_state_dict(state,strict=True);del state
 return frozen(obj)

def train_stage(family,stage,cfg):
 C.install_signals();directory=C.stage_dir(family,stage);run_identity=C.identity(cfg);cache=C.STORE/'runs'/family/'training_cache';cachemeta=C.read(cache/'complete.json')
 if not cachemeta:raise RuntimeError('Training cache collection is incomplete')
 key=C.canon({'run':run_identity,'family':family,'stage':stage,'cache':cachemeta['identity']});C.seal(directory,key,{'stage':stage,'family':family,'run_identity':run_identity})
 if C.read(directory/'complete.json'):return
 records=[r for r in C.rows(cache/'records.jsonl') if r['status']=='retained' and (stage!='latcom_stage1' or not r['aux'])]
 if len(records)<2:raise ValueError('Need at least two distinct matched/mismatched examples')
 for r in records:
  if C.sha(cache/r['file'])!=r['sha256']:raise ValueError('Training cache corrupt')
 tr=cfg['training'];steps=tr[stage+'_steps'];batch=tr['latcom_global_batch'] if stage.startswith('latcom') else tr[stage+'_global_batch'];device='cuda:0'
 random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
 student=build_module(family,stage,cfg);gradient_setup(student)
 if stage=='latcom_stage2' and not (directory/'latest.pt').exists():
  previous=C.stage_dir(family,'latcom_stage1');meta=C.read(previous/'complete.json')
  if not meta or meta['run_identity']!=run_identity or C.sha(previous/'weights.pt')!=meta['weights_sha256']:raise ValueError('Invalid stage1 export')
  student.load_state_dict(torch.load(previous/'weights.pt',map_location='cpu',weights_only=True))
 engine=Engine(C.model_path(family),cfg)
 # Frozen receivers must retain input gradients for continuous-message learning.
 engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 engine.model.train()  # dropout is zero in pinned Qwen3 configs; activates checkpointing only.
 receiver=None
 if stage=='interlat_compression':
  # Free the generic frozen receiver before loading the trained one.
  del engine.model;torch.cuda.empty_cache();receiver=load_export(family,'interlat_receiver',cfg);engine.model=receiver.model
  receiver.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});receiver.model.train()
 optimizer=torch.optim.AdamW(student.parameters(),lr=tr['learning_rates'][stage],betas=tuple(tr['betas']),eps=1e-8,weight_decay=tr['weight_decay'],foreach=False)
 start_step=0;latest=directory/'latest.pt'
 if latest.exists():
  checkpoint=torch.load(latest,map_location='cpu',weights_only=True)
  if checkpoint['identity']!=key:raise ValueError('Training resume identity differs')
  student.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);start_step=checkpoint['step'];random.setstate(checkpoint['python_rng']);torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state(checkpoint['cuda_rng']);del checkpoint
 metrics_path=directory/'metrics.jsonl'
 old_metrics=C.rows(metrics_path,repair=True)
 if any(r['step']>start_step for r in old_metrics):
  C.write(directory/'superseded_metrics.json',[r for r in old_metrics if r['step']>start_step]);metrics_path.write_text(''.join(__import__('json').dumps(r)+'\n' for r in old_metrics if r['step']<=start_step))
 def asset(index):
  obj=torch.load(cache/records[index]['file'],map_location='cpu',weights_only=True)
  return {k:([t.to(device) for t in v] if k=='z_sources' else v.to(device) if torch.is_tensor(v) else v) for k,v in obj.items()}
 def save(step):
  temp=directory/'latest.pt.tmp';torch.save({'identity':key,'step':step,'model':student.state_dict(),'optimizer':optimizer.state_dict(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},temp);temp.replace(latest)
 def ce(logits,y):return F.cross_entropy(logits,y)
 def public_prefix(a,model=None):return (model or engine.model).get_input_embeddings()(a['qids'])
 def actor_prefix(actor,a,h,plan_ids=None,fraction=0):
  q=public_prefix(a,actor.model);carrier=actor.carrier(h)
  if fraction>0:
   cut=min(int(len(h)*fraction),len(plan_ids));plan=actor.model.get_input_embeddings()(plan_ids[:cut]);carrier=torch.cat([actor.bop,plan,actor.adapter(h[cut:]),actor.eop]) if cut<len(h) else torch.cat([actor.bop,plan,actor.eop])
  # Insert before the assistant marker, after the user question. Find the last im_end ID.
  end_id=engine.tokenizer.convert_tokens_to_ids('<|im_end|>');where=(a['qids']==end_id).nonzero().flatten();b=int(where[-1]) if len(where) else len(q)
  return torch.cat([q[:b],carrier,q[b:]])
 def loss_for(a,b,progress):
  y=a['target_ids'];q=public_prefix(a)
  with engine.autocast():
   if stage.startswith('latcom'):
    src=[a['z_single']] if stage=='latcom_stage1' else a['z_sources']
    wrong=[b['z_single']] if stage=='latcom_stage1' else [b['z_sources'][0] if role=='gold' and j==0 else z for j,(role,z) in enumerate(zip(a['source_roles'],src))]
    m=student(a['qids'],src);negative=student(a['qids'],wrong)
    logits=engine.prediction_logits(torch.cat([m,q]),y);neglogits=engine.prediction_logits(torch.cat([negative,q]),y);task=ce(logits,y);contrast=latent_margin(task,ce(neglogits,y),tr['latcom_contrast_margin'],tr['latcom_contrast_temperature']);weight=dynamic_weight(task,contrast,.5);total=task+weight*contrast
    values={'task':task,'contrast':contrast,'contrast_weight':weight}
    if stage=='latcom_stage1':
     with torch.no_grad():full=engine.prediction_logits(torch.cat([a['z_single'],q]),y)
     alignment=js(logits,full)+tr['latcom_anchor_weight']*(m.float().mean(0)-a['z_single'].float().mean(0)).square().sum();wa=dynamic_weight(task,alignment,.2);total=total+wa*alignment;values.update(alignment=alignment,alignment_weight=wa)
   elif stage=='interlat_receiver':
    fraction=max(0.,1-progress/.8);pos=actor_prefix(student,a,a['full_hidden'],a['plan_ids'],fraction);neg=actor_prefix(student,a,b['full_hidden'],b['plan_ids'],fraction)
    logits=engine.prediction_logits(pos,y,student.model);negative=engine.prediction_logits(neg,y,student.model)
    with torch.no_grad():
     _,text_prefix,_,_=engine.render(engine.qa_messages(a['question'],a['plan_text']),False);teacher=engine.prediction_logits(text_prefix,y)
    task=ce(logits,y);sep=-js(logits,negative);alpha=tr['interlat_kl_fraction'];align=alpha*reverse_kl(logits,teacher)+(1-alpha)*(1-F.cosine_similarity(logits,teacher,dim=-1)).mean();total=task+tr['interlat_separation_weight']*sep+tr['interlat_alignment_weight']*align;values={'task':task,'separation':sep,'alignment':align}
   else:
    hidden=student(a['sender_ids']);pos=actor_prefix(receiver,a,hidden);logits=engine.prediction_logits(pos,y)
    with torch.no_grad():
     full=engine.prediction_logits(actor_prefix(receiver,a,a['full_hidden']),y);empty=engine.prediction_logits(q,y);target=receiver.adapter(a['full_hidden']);target=F.adaptive_avg_pool1d(target.T[None],len(hidden))[0].T
    task=ce(logits,y);pref=uncertainty_kl(logits,full,empty);geom=1-F.cosine_similarity(receiver.adapter(hidden).float().mean(0),target.float().mean(0),dim=0);total=task+tr['interlat_compression_pref_weight']*pref+tr['interlat_compression_geometry_weight']*geom;values={'task':task,'preference':pref,'geometry':geom}
  return total,{**{k:float(v.detach()) for k,v in values.items()},'total':float(total.detach())}
 for step in range(start_step+1,steps+1):
  if C.STOP:save(step-1);C.write(directory/'status.json',{'status':'paused','step':step-1,'total_steps':steps,'time':C.now()});raise SystemExit(75)
  warm=max(1,int(steps*tr['warmup_ratio']));factor=step/warm if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm)))
  for group in optimizer.param_groups:group['lr']=tr['learning_rates'][stage]*factor
  optimizer.zero_grad(set_to_none=True);start=time.perf_counter();sums={};indices=[random.randrange(len(records)) for _ in range(batch)]
  for i in indices:
   j=random.randrange(len(records)-1);j+=j>=i;a=asset(i);b=asset(j)
   ctx=torch.autograd.graph.save_on_cpu(pin_memory=True) if tr['cpu_saved_activations'] else contextlib.nullcontext()
   with ctx:loss,values=loss_for(a,b,step/steps);scaled=loss/batch
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss; no evaluation export produced')
   scaled.backward()
   for k,v in values.items():sums[k]=sums.get(k,0)+v/batch
   del a,b,loss,scaled
  grad=torch.nn.utils.clip_grad_norm_(student.parameters(),tr['clip_grad_norm'])
  if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
  optimizer.step();torch.cuda.synchronize();entry={'step':step,'total_steps':steps,'global_batch':batch,'learning_rate':optimizer.param_groups[0]['lr'],'grad_norm':float(grad),'seconds':time.perf_counter()-start,'time':C.now(),**sums};C.append(directory/'metrics.jsonl',entry);C.write(directory/'status.json',{'status':'training',**entry})
  if step%tr['save_every']==0 or step==steps:save(step)
 # Publish only after all requested updates. Save BF16 CPU inference weights, one tensor at a time.
 optimizer.zero_grad(set_to_none=True);del optimizer;torch.cuda.empty_cache();student.to(device='cpu',dtype=torch.bfloat16)
 temporary=directory/'weights.pt.tmp';torch.save(student.state_dict(),temporary);temporary.replace(directory/'weights.pt')
 C.write(directory/'complete.json',{'run_identity':run_identity,'identity':key,'stage':stage,'family':family,'steps':steps,'weights_sha256':C.sha(directory/'weights.pt'),'provenance':'paper_derived_port','time':C.now()});C.write(directory/'status.json',{'status':'completed','step':steps,'total_steps':steps,'time':C.now()})
 if not tr['retain_final_optimizer']:latest.unlink(missing_ok=True)

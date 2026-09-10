#!/usr/bin/env python3
"""Opt-in, isolated A100 engineering checks. Never produces benchmark scores."""
import argparse,copy,gc,json,os,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--store',required=True);p.add_argument('--phase',choices=['real','tiny'],required=True);a=p.parse_args()
store=Path(a.store).resolve();store.relative_to(ROOT);store.mkdir(parents=True,exist_ok=True)
os.environ['LATEN_STORE']=str(store);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,transformers
from suite import common as C
from suite.engine import Engine
from suite.models import InterlatReceiver,InterlatCompressor,SlotCompressor,frozen
from suite.txt_reports import write_parts,flatten
cfg=copy.deepcopy(C.config());results={'purpose':'engineering_canary_not_benchmark','phase':a.phase,'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),'started_beijing':C.now(),'checks':{}}
assert 'A100' in results['gpu'];torch.set_num_threads(4);torch.manual_seed(42)

def emit():
 out=store/'STATS_TXT';out.mkdir(exist_ok=True)
 results['updated_beijing']=C.now()
 write_parts(out,[('02_canary','A100 engineering checks; no benchmark performance claims',''.join(flatten(results)))])
 # Use a canary-specific introduction so these files cannot be mistaken for a formal suite report.
 (out/'00_README.txt').write_text('A100 工程测试统计，非正式 Benchmark 跑分。\n读取 01_INDEX.txt，再读取 02_canary 各分片。\n真实 4B 与随机小模型阶段分别记录；没有推理输出。\n',encoding='utf-8')

def check(name,fn):
 torch.cuda.reset_peak_memory_stats();start=time.perf_counter();print('START',name,flush=True)
 try:
  value=fn();torch.cuda.synchronize();results['checks'][name]={'status':'passed','seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),**(value or {})};emit();print('PASS',name,results['checks'][name],flush=True)
 except Exception as e:
  results['checks'][name]={'status':'failed','error_type':type(e).__name__,'seconds':time.perf_counter()-start};emit();raise

def cleanup():gc.collect();torch.cuda.empty_cache()

model=Path(a.model).resolve();models=store/'models';models.mkdir(exist_ok=True)
if a.phase=='real':
 (models/'4b').symlink_to(model,target_is_directory=True) if not (models/'4b').exists() else None
 results['model_source']='existing_local_Qwen3_4B';results['model_config_sha256']=C.sha(model/'config.json')
 results['deviations']={'synthetic_collection_candidates':4,'formal_minimum_main_retained':32,'general_output_cap_for_smoke':48,'semantic_output_cap':2048,'learned_inference_modules':'untrained_module_interface_checks_only','real_training_check':'adapter_only_one_update_not_full_model_training'}
 holder={}
 def init():
  engine=Engine(model,cfg);engine.realign_init();holder['engine']=engine
  return {'realignment_finite':bool(torch.isfinite(engine.W).all()),'realignment_shape':list(engine.W.shape)}
 check('real_4b_load_and_realignment',init);engine=holder['engine']
 def collect_check():
  from suite import collect as K
  candidates=[dict(id='synthetic_private/'+str(i),question='According to the private registry, what is the assigned city for parcel '+key+'? Return only Answer: city.',answer=city,gold_docs=['The private registry assigns parcel '+key+' to '+city+'.'],other_docs=['The warehouse opens at noon.'],source='synthetic_engineering_private_fact',aux=False) for i,(key,city) in enumerate([('QX-718','Paris'),('RM-294','Tokyo')])]
  candidates += [dict(id='synthetic_aux/'+str(i),question=q,answer=ans,gold_docs=[e],other_docs=['A square has four sides.'],source='synthetic_engineering_aux',aux=True) for i,(q,ans,e) in enumerate([('What is 2 plus 3?','5','Adding 2 and 3 gives 5.'),('What is 7 minus 3?','4','Subtracting 3 from 7 gives 4.')])]
  file=store/'data/training_candidates.jsonl'
  if not file.exists():
   for row in candidates:C.append(file,row)
  C.write(store/'data/manifest.json',{'identity':C.canon(candidates),'files':{file.name:C.sha(file)},'synthetic_engineering_only':True})
  with patch.object(K,'Engine',return_value=engine):K.collect('4b',cfg)
  records=C.rows(C.cache_dir('4b')/'records.jsonl');meta=C.read(C.cache_dir('4b')/'complete.json')
  assert len(records)==4
  before=C.sha(C.cache_dir('4b')/'records.jsonl')
  with patch.object(K,'Engine',side_effect=AssertionError('Completed collect must not reload')):K.collect('4b',cfg)
  assert before==C.sha(C.cache_dir('4b')/'records.jsonl')
  assert all('raw_text' not in (probe or {}) for row in records for probe in row['probes'].values())
  return {'scanned':len(records),'main_retained_by_method':meta['main_retained_by_method'],'formal_method_ready':meta['method_ready'],'assets':sum(r['status']=='retained' for r in records),'resume_zero_model_calls':True}
 check('real_4b_collect_and_resume',collect_check);cleanup()
 def inference(method):
  # Reuse frozen pretrained backbone to check interfaces without loading untrained duplicate 4B copies.
  modules={}
  if method=='latcom':modules={'compressor':frozen(SlotCompressor(engine.model.model,cfg['latcom_slots']).to('cuda',dtype=torch.bfloat16))}
  if method=='interlat':modules={'receiver':frozen(InterlatReceiver(engine.model).to('cuda',dtype=torch.bfloat16)),'compressor':frozen(InterlatCompressor(engine.model.model,cfg['interlat_slots_per_sender']).to('cuda',dtype=torch.bfloat16))}
  with torch.inference_mode():row=engine.pipeline({'example_id':'synthetic_math/1','question':'There are 2 apples. Add 3 more apples. How many apples are there?','dataset':'gsm8k','cap':48},method,modules)
  assert row['output']['output_tokens']>0
  from suite.scoring import simple_score
  score=simple_score({'dataset':'gsm8k','gold':'5'},row['output']['text'])
  C.append(store/'interface_checks'/method/'results.jsonl',{**row,'example_id':'synthetic_math/1','dataset':'gsm8k','correct':score['correct'],'score':score})
  return {'generated_tokens':row['output']['output_tokens'],'stop_reason':row['output']['stop_reason'],'sender_count':len(row['traces']),'efficiency':row['efficiency'],'untrained_learned_interface':method in ('latcom','interlat')}
 for method in cfg['methods']:
  check('real_4b_interface_'+method,lambda m=method:inference(m));cleanup()
 def semantic(method):
  from suite.semantic import load_frozen
  from suite.semantic_engine import pipeline,action_diagnostic
  variants,cases,programs,_=load_frozen();v=next(v for v in variants if v.split=='dev' and v.graph_level=='G4' and v.information_level=='I3' and v.target_fact_id is None)
  with torch.inference_mode():
   row=pipeline(engine,cases[v.case_id],v,programs[v.policy_program_id],method,cfg,{})
   if row['status']=='success':label,scores,tokens=action_diagnostic(engine,cases[v.case_id],programs[v.policy_program_id],row['predicted_information_vector']);row['model_action_exact']=label==v.gold.candidate_label
  C.append(store/'semantic_interface_checks'/method/'results.jsonl',row)
  return {'graph':'G4','information_level':'I3','case_count':1,'execution_status':row['status'],'reasoning_tokens':row['reasoning_token_count'],'reasoning_cap_hit':row['reasoning_cap_hit'],'efficiency':row['efficiency']}
 for method in ('latentmas','latentmas_h2o','latentmas_hidden'):
  check('real_4b_semantic_'+method,lambda m=method:semantic(m));cleanup()
 def adapter_update():
  actor=InterlatReceiver(engine.model).to('cuda');frozen(actor.model);actor.adapter.float();actor.adapter.train()
  params=[p for n,p in actor.named_parameters() if not n.startswith('model.')];opt=torch.optim.AdamW(params,lr=1e-4,foreach=False)
  q=engine.ids('What is the private value?');y=engine.ids('Answer: 5')
  hidden=torch.randn(4,engine.model.config.hidden_size,device='cuda',dtype=torch.bfloat16)
  before=actor.adapter.output_scale.detach().clone()
  with engine.autocast():
   prefix=torch.cat([actor.model.get_input_embeddings()(q),actor.carrier(hidden)]);logits=engine.prediction_logits(prefix,y,actor.model);loss=torch.nn.functional.cross_entropy(logits,y)
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad) and grad>0;opt.step()
  delta=float((actor.adapter.output_scale-before).abs());assert delta>0
  return {'loss':float(loss),'grad_norm':float(grad),'parameter_delta':delta,'base_gradients_absent':all(p.grad is None for p in actor.model.parameters())}
 check('real_4b_receiver_adapter_backward_update',adapter_update)
else:
 from transformers import Qwen3Config,Qwen3ForCausalLM,AutoTokenizer
 from suite import train as T
 tiny=models/'4b';tiny.mkdir(exist_ok=True)
 tc=Qwen3Config(vocab_size=151936,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=40960,attention_dropout=0.,eos_token_id=151645,tie_word_embeddings=True)
 if not (tiny/'config.json').exists():
  base=Qwen3ForCausalLM(tc).to(torch.bfloat16);base.save_pretrained(tiny,max_shard_size='5MB');AutoTokenizer.from_pretrained(model,local_files_only=True).save_pretrained(tiny);del base
 cfg['sender_latent_steps']=2;cfg['latcom_slots']=4;cfg['interlat_slots_per_sender']=3
 tr=cfg['training'];tr['minimum_retained']=2;tr['latcom_global_batch']=tr['interlat_receiver_global_batch']=tr['interlat_compression_global_batch']=1;tr['save_every']=1;tr['retain_final_optimizer']=True
 stages=['latcom_stage1','latcom_stage2','interlat_receiver','interlat_compression']
 for stage in stages:tr[stage+'_steps']=2
 C.write(store/'canary_config.json',cfg);results['deviations']={'random_tiny_Qwen3_hidden_size':32,'layers':2,'optimizer_steps_per_stage':2,'global_batch':1,'synthetic_assets':2,'formal_training_unchanged':True}
 cache=C.cache_dir('4b');cache.mkdir(parents=True,exist_ok=True)
 if not (cache/'complete.json').exists():
  rows=[]
  for i in range(2):
   item={'id':str(i),'question':'Fixture question','answer':str(i),'qids':torch.tensor([151644,872,198,40,151645]),'target_ids':torch.tensor([32,33,34+i]),'z_single':torch.randn(2,32,dtype=torch.bfloat16),'z_sources':[torch.randn(2,32,dtype=torch.bfloat16),torch.randn(2,32,dtype=torch.bfloat16)],'source_roles':['gold','irrelevant'],'full_hidden':torch.randn(3,32,dtype=torch.bfloat16),'plan_ids':torch.tensor([32,33,34]),'sender_ids':torch.tensor([151644,872,198,40,151645]),'plan_text':'Fixture plan','source':'synthetic_engineering','aux':False,'eligible_for':['latcom','interlat']}
   file=cache/(str(i)+'.pt');torch.save(item,file);row={'id':str(i),'status':'retained','file':file.name,'sha256':C.sha(file),'aux':False,'eligible_for':['latcom','interlat']};C.append(cache/'records.jsonl',row);rows.append(row)
  C.write(cache/'complete.json',{'identity':C.canon(rows),'synthetic_engineering_only':True})
 for stage in stages:
  def run_stage(stage=stage):
   resumed=False
   if stage=='latcom_stage1' and not (C.stage_dir('4b',stage)/'latest.pt').exists():
    append=C.append
    def stop_after_first(path,row):
     append(path,row)
     if Path(path).name=='metrics.jsonl' and row['step']==1:C.STOP=True
    try:
     with patch.object(C,'append',side_effect=stop_after_first):T.train_stage('4b',stage,cfg)
    except SystemExit as e:
     assert e.code==75
    else:raise AssertionError('Expected graceful pause after first step')
    assert C.read(C.stage_dir('4b',stage)/'status.json')['step']==1
    C.STOP=False;resumed=True
   T.train_stage('4b',stage,cfg);root=C.stage_dir('4b',stage);metrics=C.rows(root/'metrics.jsonl');assert len(metrics)==2 and all(m['grad_norm']>0 for m in metrics)
   exported=T.load_export('4b',stage,cfg);assert all(torch.isfinite(p).all() for p in exported.parameters());del exported
   digest=C.sha(root/'weights.pt')
   with patch.object(T,'build_module',side_effect=AssertionError('Completed stage must not reload')):T.train_stage('4b',stage,cfg)
   assert digest==C.sha(root/'weights.pt')
   return {'optimizer_steps':len(metrics),'paused_after_step_one_and_resumed':resumed,'metrics':[{k:v for k,v in r.items() if isinstance(v,(int,float))} for r in metrics],'export_load':True,'completed_resume_zero_model_calls':True}
  check('tiny_cuda_full_train_'+stage,run_stage);cleanup()
results['completed_beijing']=C.now();results['all_checks_passed']=True;emit();print('ALL CHECKS PASSED',a.phase,flush=True)

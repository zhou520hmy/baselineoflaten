"""Resumable LATEN evaluation. No benchmark examples enter training."""
import time
import torch
from . import common as C
from .semantic import load_frozen,summarize
from .semantic_engine import pipeline,action_diagnostic
from .engine import Engine
from .train import load_export

def evaluate_semantic(family,method,cfg,smoke=False):
 C.install_signals();sem=cfg['semantic_benchmark'];variants,cases,programs,data=load_frozen()
 if (sem['variants'],sem['pairs'],sem['cases'])!=(648,486,162):raise ValueError('Frozen LATEN denominators must not be changed')
 if sem.get('do_sample') is not False or sem['reasoning_max_new_tokens']!=2048:raise ValueError('Preserve the authorized greedy 2048-token semantic protocol')
 if smoke:
  variants=[next(v for v in variants if v.split=='dev' and v.graph_level==g and v.information_level==i and v.target_fact_id is None) for g,i in zip(('G4','G5','G6'),('I3','I6','I9'))]
 directory=C.STORE/'runs'/family/('semantic_smoke' if smoke else 'semantic')/method
 key=C.canon({'run':C.identity(cfg),'family':family,'method':method,'data':data,'smoke':smoke});C.seal(directory,key,{'family':family,'method':method,'smoke':smoke,'test_scope':'previously_exposed_test_diagnostic','test_inference_authorized_by_current_user_request':True})
 allowed={v.variant_id for v in variants};saved=C.rows(directory/'results.jsonl',repair=True);raw=C.rows(directory/'generations.jsonl',repair=True);done={r['variant_id']:r for r in saved};generated={r['variant_id']:r for r in raw}
 if len(done)!=len(saved) or len(generated)!=len(raw) or not (set(done)|set(generated))<=allowed or any(r['run_identity']!=key for r in saved+raw):raise ValueError('Semantic resume identity or matrix differs')
 def status(phase,**extra):C.write(directory/'status.json',{'status':phase,'completed':len(done),'expected':len(variants),'time':C.now(),**extra})
 if len(done)==len(variants):status('completed');C.write(directory/'summary.json',summarize(saved,variants));return
 status('loading_model');engine=Engine(C.model_path(family),cfg);engine.realign_init();modules={};exports={}
 for name,stage in ([('compressor','latcom_stage2')] if method=='latcom' else [('receiver','interlat_receiver'),('compressor','interlat_compression')] if method=='interlat' else []):
  modules[name]=load_export(family,stage,cfg);exports[name]=C.read(C.stage_dir(family,stage)/'complete.json')
 C.write(directory/'checkpoint_provenance.json',exports)
 with torch.inference_mode():engine.forward_cache(engine.model.get_input_embeddings()(torch.tensor([1,2,3],device=engine.device)))
 for v in variants:
  if v.variant_id in done:continue
  if C.STOP:status('paused');raise SystemExit(75)
  status('running',current=v.variant_id)
  try:
   if v.variant_id not in generated:
    with torch.inference_mode():row=pipeline(engine,cases[v.case_id],v,programs[v.policy_program_id],method,cfg,modules)
    row.update(run_identity=key,run_id=method+'::'+v.variant_id,arm_id=method,family=family,engineering_smoke=smoke,time=C.now());C.append(directory/'generations.jsonl',row);generated[v.variant_id]=row
   row=dict(generated[v.variant_id]);engine.sync();t=time.perf_counter();action_time=0.
   if row['status']=='success':
    with torch.inference_mode():selected,scores,prompt_tokens=action_diagnostic(engine,cases[v.case_id],programs[v.policy_program_id],row['predicted_information_vector'])
    engine.sync();action_time=time.perf_counter()-t;row.update(selected_action_label=selected,action_candidate_scores=scores,model_action_exact=selected==v.gold.candidate_label,model_action_diagnostic_prompt_tokens=prompt_tokens)
   row['efficiency']={**row['efficiency'],'model_action_diagnostic_seconds':action_time,'pipeline_with_action_diagnostic_seconds':row['efficiency']['task_wall_seconds']+action_time}
   C.append(directory/'results.jsonl',row);done[v.variant_id]=row
  except Exception as e:
   C.append(directory/'infrastructure_errors.jsonl',{'variant_id':v.variant_id,'type':type(e).__name__,'error':str(e),'time':C.now(),'scored':False});status('blocked',current=v.variant_id,error=str(e));raise
  status('running',last=v.variant_id)
  if len(done)%10==0 or smoke:C.write(directory/'summary_progress.json',summarize(list(done.values()),variants))
 C.write(directory/'summary.json',summarize(list(done.values()),variants));status('completed')

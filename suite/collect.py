"""Versioned train-only caches with explicit LatCom and Interlat readiness."""
import re,time
import torch
from . import common as C
from .engine import Engine
from .collection_policy import normalized,final_answer,answer_matches,clean_controls,probe,eligible_methods,pool_counts

def collect(family,cfg):
 C.install_signals();directory=C.cache_dir(family);data=C.read(C.STORE/'data/manifest.json');tr=cfg['training']
 if not data:raise RuntimeError('Prepared data manifest missing; run download first')
 for name,digest in data['files'].items():
  if C.sha(C.STORE/'data'/name)!=digest:raise ValueError('Prepared data was modified: '+name)
 candidates=C.rows(C.STORE/'data/training_candidates.jsonl');candidate_ids={r['id'] for r in candidates}
 if len(candidate_ids)!=len(candidates):raise ValueError('Duplicate candidate IDs')
 key=C.canon({'run':C.identity(cfg),'family':family,'data':data['identity'],'collector':'method_gates_v2'});C.seal(directory,key,{'family':family,'training_only':True,'collector':'method_gates_v2'})
 records=C.rows(directory/'records.jsonl',repair=True);done={r['id'] for r in records}
 if len(done)!=len(records) or not done<=candidate_ids:raise ValueError('Invalid collection resume IDs')
 for row in records:
  if row['status']=='retained' and C.sha(directory/row['file'])!=row['sha256']:raise ValueError('Retained asset hash mismatch')
 counts=pool_counts(records)
 def status(phase,**extra):
  reasons={}
  for r in records:reasons[r.get('reason','retained')]=reasons.get(r.get('reason','retained'),0)+1
  gate_rejections={m:{} for m in ('latcom','interlat')}
  for r in records:
   for method,reason in r.get('rejection_reasons',{}).items():gate_rejections[method][reason]=gate_rejections[method].get(reason,0)+1
  C.write(directory/'status.json',{'status':phase,'gate_rejections':gate_rejections,'main_retained_by_method':counts,'candidate_count':len(candidates),'scanned':len(records),'retained_assets':sum(r['status']=='retained' for r in records),'reasons':reasons,'time':C.now(),**extra})
 def finish():
  ready={m:n>=tr['minimum_retained'] for m,n in counts.items()}
  C.write(directory/'complete.json',{'identity':key,'main_retained_by_method':counts,'method_ready':ready,'scanned':len(records),'retained':sum(r['status']=='retained' for r in records),'time':C.now(),'files':{r['file']:r['sha256'] for r in records if r['status']=='retained'}})
  status('completed' if all(ready.values()) else 'completed_with_blocked_methods',method_ready=ready)
 if C.read(directory/'complete.json'):finish();return
 status('loading_model');engine=Engine(C.model_path(family),cfg);engine.realign_init()
 eos=engine.model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
 for row in candidates:
  if row['id'] in done:continue
  if C.STOP:status('paused');raise SystemExit(75)
  if not row['aux'] and all(n>=tr['retained_limit'] for n in counts.values()):continue
  engine.seed(row['id'],'training_cache_v2');start=time.perf_counter();status('collecting',current=row['id'])
  def bounded(doc):return engine.tokenizer.decode(engine.tokenizer.encode(doc,add_special_tokens=False)[:tr['max_evidence_tokens']],skip_special_tokens=True)
  gold_docs=[bounded(s) for s in row['gold_docs']];other=[bounded(s) for s in row['other_docs'][:2]]
  if not other:other=[bounded(next(x['gold_docs'][0] for x in candidates if x['id']!=row['id']))]
  if len(gold_docs)>4:gold_docs=gold_docs[:3]+['\n'.join(gold_docs[3:])]
  gold_text='\n'.join(gold_docs)
  qids,qemb,_,_=engine.render(engine.qa_messages(row['question']),False)
  direct=probe(engine.decode(qemb,tr['filter_max_new_tokens'],temperature=0),row['answer']) if not row['aux'] else {'correct':False,'not_run':'auxiliary'}
  # Text-plan eligibility is independent of the frozen Receiver's raw-Z readability.
  h,plan_ids,plan,sender_ids=engine.full_plan_hidden(engine.qa_messages(row['question'],gold_text),tr['plan_max_new_tokens'])
  plan_probe=probe({'text':plan,'stop_reason':'eos' if len(plan_ids) and int(plan_ids[-1]) in eos else 'max_new_tokens','output_tokens':len(plan_ids)},row['answer'])
  single=None;single_probe=None
  if row['aux'] or not direct['correct']:
   _,emb,_,_=engine.render(engine.qa_messages(row['question'],gold_text),False);single=engine.rollout(emb,cfg['sender_latent_steps'])['z']
   # Exactly the same full-Z prefix used by stage-one JS alignment, not a different multi-source carrier.
   single_probe=probe(engine.decode(torch.cat([single,qemb]),tr['filter_max_new_tokens'],temperature=0),row['answer'])
  eligible=eligible_methods(direct,single_probe,plan_probe,row['aux'])
  if not row['aux']:eligible=[m for m in eligible if counts[m]<tr['retained_limit']]
  info={'id':row['id'],'aux':row['aux'],'source':row['source'],'status':'filtered','eligible_for':eligible,'probes':{'question_only':direct,'single_gold_latent':single_probe,'evidence_text_plan':plan_probe}}
  info['rejection_reasons']={}
  if not row['aux']:
   if 'latcom' not in eligible:info['rejection_reasons']['latcom']='single_model_solvable' if direct['correct'] else 'full_latent_unanswerable' if not single_probe or not single_probe['correct'] else 'pool_full'
   if 'interlat' not in eligible:info['rejection_reasons']['interlat']='text_plan_unanswerable' if not plan_probe['correct'] else 'pool_full'
  if eligible:
   mixed=gold_docs[0][:len(gold_docs[0])//2]+'\n'+other[0][:len(other[0])//2]
   sources=(gold_docs+other[:1]+[mixed])[:6];roles=(['gold']*len(gold_docs)+['irrelevant','mixed'])[:len(sources)]
   zs=[];cache=None;absolute=0
   for doc in sources:
    _,emb,_,_=engine.render(engine.qa_messages(row['question'],doc),False);r=engine.rollout(emb,cfg['sender_latent_steps'],cache,absolute);cache=r['cache'];absolute=r['absolute_next'];zs.append(r['z'].cpu())
   del cache
   if single is None:
    _,emb,_,_=engine.render(engine.qa_messages(row['question'],gold_text),False);single=engine.rollout(emb,cfg['sender_latent_steps'])['z']
   messages=engine.qa_messages(row['question'],gold_text+'\nVerified answer: '+str(row['answer'])+'\nExplain briefly how the evidence supports that answer. Do not invent evidence.')
   _,emb,_,_=engine.render(messages,False);rationale=engine.decode(emb,tr['rationale_max_new_tokens'],temperature=0)['text']
   rationale=re.split(r'(?i)\banswer\s*:',clean_controls(rationale))[0].strip()
   suffixids=engine.ids('\nAnswer: '+str(row['answer']))
   if len(suffixids)>=tr['max_target_tokens']:raise ValueError('Training answer itself exceeds target budget')
   target=torch.cat([engine.ids('Reasoning: '+rationale)[:tr['max_target_tokens']-len(suffixids)],suffixids])
   asset={'id':row['id'],'question':row['question'],'answer':row['answer'],'target_ids':target.cpu(),'qids':qids.cpu(),'z_single':single.cpu(),'z_sources':zs,'source_roles':roles,'full_hidden':h.cpu(),'plan_ids':plan_ids.cpu(),'sender_ids':sender_ids.cpu(),'plan_text':clean_controls(plan),'source':row['source'],'aux':row['aux'],'eligible_for':eligible}
   filename=C.canon(row['id'])[:24]+'.pt';path=directory/filename;temp=path.with_suffix('.tmp');torch.save(asset,temp);temp.replace(path)
   info.update(status='retained',file=filename,sha256=C.sha(path),rationale_conditioned_on_gold=True,rationale_semantic_validation='not_independently_judged',plan_answer_verified=plan_probe['correct'])
  else:info['reason']='no_method_eligible_or_pool_full'
  info['seconds']=time.perf_counter()-start;C.append(directory/'records.jsonl',info);records.append(info);done.add(row['id']);counts=pool_counts(records);status('collecting',last=row['id'])
  print(f'collect {family}: {len(records)}/{len(candidates)} candidates; main pools {counts}',flush=True)
 finish()

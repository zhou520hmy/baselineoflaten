"""Build evidence-filtered training caches on the target GPU, separate from evaluation."""
import re,time
import torch
from . import common as C
from .engine import Engine
from .scoring import boxes,numeric

def normalized(s):return ' '.join(re.sub(r'[^\w\s]',' ',str(s).lower()).split())
def final_answer(text):
 if not text.strip():return ''
 b=boxes(text)
 if b:return b[-1]
 split=re.split(r'(?i)\b(?:final\s+)?answer\s*:',text.split('</think>')[-1]);return (split[-1].strip().splitlines() or [''])[0] if len(split)>1 else text.strip().splitlines()[-1]
def answer_matches(text,gold):
 predicted=final_answer(text)
 a,b=numeric(predicted),numeric(str(gold))
 return a==b if a is not None and b is not None else normalized(predicted)==normalized(gold)

def collect(family,cfg):
 C.install_signals();directory=C.STORE/'runs'/family/'training_cache';key=C.canon({'run':C.identity(cfg),'family':family,'data':C.read(C.STORE/'data/manifest.json')['identity']});C.seal(directory,key,{'family':family,'training_only':True})
 previous=C.rows(directory/'records.jsonl',repair=True);done={x['id'] for x in previous};main_kept=sum(r['status']=='retained' and not r['aux'] for r in previous);tr=cfg['training']
 if C.read(directory/'complete.json'):return
 engine=Engine(C.model_path(family),cfg);engine.realign_init();candidates=C.rows(C.STORE/'data/training_candidates.jsonl')
 for row in candidates:
  if row['id'] in done:continue
  if C.STOP:C.write(directory/'status.json',{'status':'paused','time':C.now()});raise SystemExit(75)
  if main_kept>=tr['retained_limit'] and not row['aux']:continue
  engine.seed(row['id'],'training_cache');start=time.perf_counter()
  def bounded(doc):return engine.tokenizer.decode(engine.tokenizer.encode(doc,add_special_tokens=False)[:tr['max_evidence_tokens']],skip_special_tokens=True)
  gold_docs=[bounded(s) for s in row['gold_docs']];other=[bounded(s) for s in row['other_docs'][:2]]
  if not other:other=[bounded(next(x['gold_docs'][0] for x in candidates if x['id']!=row['id']))]
  mixed=gold_docs[0][:len(gold_docs[0])//2]+'\n'+other[0][:len(other[0])//2]
  # Preserve every supporting source within the six-source training cap.
  if len(gold_docs)>4:gold_docs=gold_docs[:3]+['\n'.join(gold_docs[3:])]
  sources=gold_docs+other[:1]+[mixed];sources=sources[:6];roles=['gold']*len(gold_docs)+['irrelevant','mixed'];roles=roles[:len(sources)]
  info={'id':row['id'],'aux':row['aux'],'source':row['source'],'status':'filtered'}
  public=engine.qa_messages(row['question']);qids,qemb,_,_=engine.render(public,False)
  if not row['aux']:
   direct=engine.decode(qemb,tr['filter_max_new_tokens'],temperature=0)
   if answer_matches(direct['text'],row['answer']):
    info.update(reason='single_model_solvable',seconds=time.perf_counter()-start);C.append(directory/'records.jsonl',info);continue
  zs=[];cache=None;absolute=0
  for doc in sources:
   _,emb,_,_=engine.render(engine.qa_messages(row['question'],doc),False);r=engine.rollout(emb,cfg['sender_latent_steps'],cache,absolute);cache=r['cache'];absolute=r['absolute_next'];zs.append(r['z'].cpu())
  del cache
  full=engine.decode(torch.cat([torch.cat([z.to(engine.device) for z,role in zip(zs,roles) if role=='gold']),qemb]),tr['filter_max_new_tokens'],temperature=0)
  if not row['aux'] and not answer_matches(full['text'],row['answer']):
   info.update(reason='full_latent_unanswerable',seconds=time.perf_counter()-start);C.append(directory/'records.jsonl',info);continue
  gold_text='\n'.join(gold_docs);messages=engine.qa_messages(row['question'],gold_text+'\nVerified answer: '+str(row['answer'])+'\nExplain briefly how the evidence supports that answer. Do not invent evidence.')
  _,emb,_,_=engine.render(messages,False);rationale=engine.decode(emb,tr['rationale_max_new_tokens'],temperature=0)['text']
  rationale=re.split(r'(?i)\banswer\s*:',rationale)[0].replace('<|im_end|>','').strip()
  suffix='\nAnswer: '+str(row['answer']);suffixids=engine.ids(suffix)
  if len(suffixids)>=tr['max_target_tokens']:raise ValueError('Training answer itself exceeds target budget; increase max_target_tokens')
  prefixids=engine.ids('Reasoning: '+rationale)[:tr['max_target_tokens']-len(suffixids)];target=torch.cat([prefixids,suffixids])
  _,emb,_,_=engine.render(engine.qa_messages(row['question'],gold_text),False);single=engine.rollout(emb,cfg['sender_latent_steps'])['z']
  h,plan_ids,plan,sender_ids=engine.full_plan_hidden(engine.qa_messages(row['question'],gold_text),tr['plan_max_new_tokens'])
  asset={'id':row['id'],'question':row['question'],'answer':row['answer'],'target_ids':target.cpu(),'qids':qids.cpu(),'z_single':single.cpu(),'z_sources':zs,'source_roles':roles,'full_hidden':h.cpu(),'plan_ids':plan_ids.cpu(),'sender_ids':sender_ids.cpu(),'plan_text':plan,'source':row['source'],'aux':row['aux']}
  filename=C.canon(row['id'])[:24]+'.pt';path=directory/filename;temp=path.with_suffix('.tmp');torch.save(asset,temp);temp.replace(path)
  info.update(status='retained',file=filename,sha256=C.sha(path),rationale_conditioned_on_gold=True,rationale_semantic_validation='not_independently_judged',seconds=time.perf_counter()-start);C.append(directory/'records.jsonl',info)
  if not row['aux']:main_kept+=1
  C.write(directory/'status.json',{'status':'collecting','main_retained':main_kept,'candidate':row['id'],'time':C.now()})
 records=C.rows(directory/'records.jsonl');retained=[r for r in records if r['status']=='retained']
 if main_kept<tr['minimum_retained']:raise RuntimeError(f'Only {main_kept} evidence-filtered main examples; need {tr["minimum_retained"]}. No random learned-method fallback. Increase candidate_limit in a new store or inspect filtering.')
 C.write(directory/'complete.json',{'identity':key,'main_retained':main_kept,'retained':len(retained),'scanned':len(records),'time':C.now(),'files':{r['file']:r['sha256'] for r in retained}});C.write(directory/'status.json',{'status':'completed','main_retained':main_kept,'retained':len(retained),'time':C.now()})

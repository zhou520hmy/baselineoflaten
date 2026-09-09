"""Pinned data downloads and one shared evaluation/training manifest."""
from pathlib import Path
import json,random,re,shutil
from .common import ROOT,STORE,read,write,rows,append,sha,canon,seal
HF=read(ROOT/'evidence/hf_sources.json')

def download(repo,filename):
 from huggingface_hub import hf_hub_download
 return Path(hf_hub_download(repo,filename,repo_type=HF[repo]['kind'].rstrip('s'),revision=HF[repo]['sha'],local_dir=STORE/'raw'/repo.replace('/','--')))

def parquet(repo,filename):
 import pyarrow.parquet as pq
 return pq.read_table(download(repo,filename)).to_pylist()

def build_data(cfg):
 directory=STORE/'data';key=canon({'hf':HF,'datasets':cfg['datasets'],'training':cfg['training'],'seed':cfg['seed'],'medqa':sha(ROOT/'vendor/medqa_300.json')})
 existing=read(directory/'manifest.json')
 if existing:
  if existing['identity']!=key:raise ValueError('Prepared data config differs; use new LATEN_STORE')
  for name,digest in existing['files'].items():
   if sha(directory/name)!=digest:raise ValueError('Prepared data was modified: '+name)
  return
 directory.mkdir(parents=True,exist_ok=True)
 sources={};all_items=[]
 def label(x):return {'1':'A','2':'B','3':'C','4':'D','5':'E'}.get(str(x),str(x).strip().upper())
 def save(name,items):
  if len(items)!=cfg['datasets'][name]:raise ValueError(f'{name}: expected {cfg["datasets"][name]}, got {len(items)}')
  if len({r['example_id'] for r in items})!=len(items):raise ValueError('Duplicate dataset IDs')
  p=directory/(name+'.jsonl');p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items));all_items.extend(items);sources[p.name]=sha(p)
 for task in cfg['datasets']:
  out=[]
  if task=='gsm8k':
   raw=parquet('openai/gsm8k','main/test-00000-of-00001.parquet')
   for i,r in enumerate(raw):out.append(dict(example_id=f'gsm8k/{i}',question=r['question'].strip(),gold=r['answer'].split('####')[-1].strip()))
  elif task.startswith('arc_'):
   subset='ARC-Easy' if task=='arc_easy' else 'ARC-Challenge'
   raw=parquet('allenai/ai2_arc',f'{subset}/test-00000-of-00001.parquet')
   for i,r in enumerate(raw):out.append(dict(example_id=f'{task}/{r["id"]}',question=r['question'].strip()+'\n'+'\n'.join(f'{label(a)}: {b.strip()}' for a,b in zip(r['choices']['label'],r['choices']['text'])),gold=label(r['answerKey'])))
  elif task=='medqa':
   for i,r in enumerate(read(ROOT/'vendor/medqa_300.json')):
    matched=[j for j,t in enumerate(r['options']) if re.sub(r'^[A-D][.)]\s*','',t).strip()==str(r['answer']).strip()]
    if len(matched)!=1:raise ValueError('Ambiguous MedQA gold')
    out.append(dict(example_id=f'medqa/{i}',question=r['query'],gold='ABCD'[matched[0]]))
  elif task=='gpqa':
   for i,r in enumerate(parquet('fingertap/GPQA-Diamond','test/gpqa_diamond.parquet')):out.append(dict(example_id=f'gpqa/{i}',question=r['question'].strip(),gold=label(r['answer'])))
  else:
   repo='evalplus/'+task;filename=next(n for n in HF[repo]['files'] if n.endswith('.parquet'))
   for r in parquet(repo,filename):
    tid=str(r['task_id']);question='Provide a self-contained Python solution in a ```python code block.\n'+r['prompt']
    if task=='mbppplus':
     question+='\nPublic examples:\n'+'\n'.join(r['test_list'][:3]);tests='\n'.join(r.get('test_imports') or [])+'\n'+r['test'];reference=r['code'];entry=None
    else:tests=r['test']+'\ncheck('+r['entry_point']+')';reference=r['prompt']+r['canonical_solution'];entry=r['entry_point']
    out.append(dict(example_id=task+'/'+tid,question=question,gold=None,task_id=tid,test_code=tests,reference_code=reference,entry_point=entry))
  for r in out:r['dataset']=task
  save(task,out)
 # Training sources: only original training splits. Exact question and code-task-ID exclusion.
 eval_questions={re.sub(r'\s+',' ',x['question']).strip().lower() for x in all_items}
 eval_code_ids={str(x.get('task_id')).split('/')[-1] for x in all_items if x['dataset']=='mbppplus'}
 tr=cfg['training'];rng=random.Random(cfg['seed']);candidates=[]
 hot=[]
 for f in ('distractor/train-00000-of-00002.parquet','distractor/train-00001-of-00002.parquet'):hot+=parquet('hotpotqa/hotpot_qa',f)
 rng.shuffle(hot)
 target_hot=int(tr['candidate_limit']*tr['candidate_fraction_hotpot'])
 for r in hot:
  support=r['supporting_facts'];gold_titles=set(support['title']);context=r['context'];gold_docs=[];other=[]
  for title,sentences in zip(context['title'],context['sentences']):
   doc=title+': '+' '.join(sentences)
   (gold_docs if title in gold_titles else other).append(doc)
  if not gold_docs:continue
  candidates.append(dict(id='hotpot/'+r['id'],source='hotpot_train',question=r['question'],answer=r['answer'],gold_docs=gold_docs,other_docs=other,aux=False))
  if len(candidates)>=target_hot:break
 musique=rows(download('dgslibisey/MuSiQue','musique_ans_v1.0_train.jsonl'));rng.shuffle(musique)
 for r in musique:
  if not r.get('answerable',True):continue
  gold_docs=[p['title']+': '+p['paragraph_text'] for p in r['paragraphs'] if p['is_supporting']]
  other=[p['title']+': '+p['paragraph_text'] for p in r['paragraphs'] if not p['is_supporting']]
  if not gold_docs:continue
  candidates.append(dict(id='musique/'+r['id'],source='musique_train',question=r['question'],answer=r['answer'],gold_docs=gold_docs,other_docs=other,aux=False))
  if len(candidates)>=tr['candidate_limit']:break
 math=parquet('openai/gsm8k','main/train-00000-of-00001.parquet');rng.shuffle(math)
 for i,r in enumerate(math[:tr['aux_math']]):
  reasoning,answer=r['answer'].split('####');candidates.append(dict(id=f'gsm8k_train/{i}',source='gsm8k_train',question=r['question'],answer=answer.strip(),gold_docs=[reasoning.strip()],other_docs=[],aux=True))
 code=parquet('google-research-datasets/mbpp','full/train-00000-of-00001.parquet');rng.shuffle(code);count=0
 for r in code:
  if str(r['task_id']) in eval_code_ids:continue
  candidates.append(dict(id='mbpp_train/'+str(r['task_id']),source='mbpp_train',question=r['text'],answer=r['code'],gold_docs=[r['code']],other_docs=[],aux=True));count+=1
  if count>=tr['aux_code']:break
 filtered=[r for r in candidates if re.sub(r'\s+',' ',r['question']).strip().lower() not in eval_questions]
 if len(filtered)!=len(candidates):raise ValueError('Train/eval exact question overlap detected')
 if count<tr['aux_code']:raise ValueError('Insufficient disjoint auxiliary code training data')
 p=directory/'training_candidates.jsonl';p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in candidates));sources[p.name]=sha(p)
 write(directory/'manifest.json',dict(identity=key,files=sources,counts=cfg['datasets'],total=sum(cfg['datasets'].values()),candidate_count=len(candidates),hf_revisions={k:v['sha'] for k,v in HF.items()},test_data_not_used_for_training=True))

def evaluation_items(cfg,task):
 manifest=read(STORE/'data/manifest.json');p=STORE/'data'/(task+'.jsonl')
 if sha(p)!=manifest['files'][p.name]:raise ValueError('Evaluation data hash differs')
 data=rows(p)
 limit=cfg.get('evaluation_limit_per_dataset')
 if limit is not None:data=data[:int(limit)]
 return data

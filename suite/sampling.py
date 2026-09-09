"""Outcome-independent fixed subsets. No model scores are used for selection."""
import ast, json, math, random
from collections import Counter
from . import common as C


def enabled(cfg):
 return cfg.get('evaluation_sampling',{}).get('enabled',False)


def expected_general(cfg,task):
 n=cfg['datasets'][task]
 if enabled(cfg):
  if cfg['evaluation_sampling']['fraction']!=0.1 or cfg['evaluation_sampling']['strategy']!='complexity_proxy_terciles_v1':raise ValueError('This sampled protocol fixes fraction=0.1 and complexity proxy terciles')
  if cfg.get('evaluation_limit_per_dataset') is not None:raise ValueError('Do not combine fixed sampling with prefix limits')
  n=max(1,math.floor(n*cfg['evaluation_sampling']['fraction']+0.5))
 elif cfg.get('evaluation_limit_per_dataset') is not None:n=min(n,int(cfg['evaluation_limit_per_dataset']))
 return n


def output_kind(cfg,kind):
 if kind=='semantic':return 'semantic_parallel' if 'inference_parallel' in cfg else kind
 return kind+'_sample10' if enabled(cfg) else kind


def complexity(item):
 branches=0
 if item['dataset'] in ('mbppplus','humanevalplus'):
  try:
   tree=ast.parse(item['reference_code'])
   branches=sum(isinstance(n,(ast.If,ast.For,ast.While,ast.IfExp,ast.comprehension,ast.Try,ast.BoolOp)) for n in ast.walk(tree))
  except SyntaxError:branches=0
 return (branches,len(item['question']))


def select_general(data,cfg,task):
 n=expected_general(cfg,task)
 if not enabled(cfg):return data[:n],None
 if len(data)!=cfg['datasets'][task]:raise ValueError('Full source denominator differs before sampling')
 seed=cfg['evaluation_sampling']['seed'];ranked=sorted(data,key=lambda r:(complexity(r),C.canon(r['example_id'])))
 bins=[ranked[len(data)*i//3:len(data)*(i+1)//3] for i in range(3)]
 quotas=[n//3+int(i<n%3) for i in range(3)]
 # For tiny fixture datasets, redistribute quotas into available bins.
 for i in range(3):
  excess=max(0,quotas[i]-len(bins[i]));quotas[i]-=excess
  for j in range(3):
   move=min(excess,len(bins[j])-quotas[j]);quotas[j]+=move;excess-=move
 selected=[];stats=[]
 for i,(pool,k) in enumerate(zip(bins,quotas)):
  pool=sorted(pool,key=lambda r:r['example_id']);picked=random.Random(C.keyed_seed(seed,task,i)).sample(pool,k);selected+=picked
  stats.append({'stratum':i,'source_count':len(pool),'selected_count':k,'proxy_range':[list(complexity(ranked_item)) for ranked_item in (bins[i][0],bins[i][-1])] if pool else None})
 selected.sort(key=lambda r:C.canon([seed,task,r['example_id']]))
 meta={'dataset':task,'source_count':len(data),'selected_count':n,'seed':seed,'strategy':'complexity_proxy_terciles_v1','proxy_is_verified_difficulty':False,'strata':stats,'selected_ids':[r['example_id'] for r in selected]}
 return selected,meta


def save_selection(store,name,records,meta,source):
 if meta is None:return
 directory=store/'data/sampled10';payload=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in records)
 import hashlib
 meta={**meta,'source':source,'selected_records_sha256':hashlib.sha256(payload.encode()).hexdigest()}
 old=C.read(directory/(name+'.manifest.json'))
 if old is not None and old!=meta:raise ValueError('Saved sampling manifest differs; choose a new store')
 p=directory/(name+'.jsonl')
 if old is not None and (not p.exists() or C.sha(p)!=meta['selected_records_sha256']):raise ValueError('Saved sampled records were modified')
 if old is None:
  directory.mkdir(parents=True,exist_ok=True);p.write_text(payload);C.write(directory/(name+'.manifest.json'),meta)

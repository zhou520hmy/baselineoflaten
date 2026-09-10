"""Read-only completed-work checks. No inference or downloads during verification."""
from . import common as C
from .sampling import method_output_kind

def evaluation_complete(cfg,family,method,kind):
 root=C.STORE/'runs'/family/method_output_kind(cfg,kind,method)/method
 meta=C.read(root/'manifest.json')
 if not meta:return False
 count=cfg.get('inference_parallel',{}).get('workers',1)
 if kind=='evaluation':
  from .data import evaluation_items
  items=[row for task in cfg['datasets'] for row in evaluation_items(cfg,task)];ids=[r['example_id'] for r in items];field='example_id';action='evaluate'
  data=C.read(C.STORE/'data/manifest.json')['identity'];parts=[ids[i::count] for i in range(count)];row_identity='identity'
 else:
  from .semantic import load_frozen
  from .parallel import partition_cases
  variants,_,_,data=load_frozen();ids=[v.variant_id for v in variants];field='variant_id';action='semantic';parts=[[v.variant_id for v in partition_cases(variants,i,count)] for i in range(count)];row_identity='run_identity'
 expected=C.canon({'run':C.evaluation_identity(cfg,method),'family':family,'method':method,'action':action,'ids':ids,'shards':count})
 if meta['identity']!=expected:raise ValueError('Existing evaluation identity differs; preserve old data and use its original config: '+str(root))
 path=root/'results.jsonl'
 if not path.exists():return False
 try:rows=C.rows(path)
 except ValueError:return False  # Worker coordinator can recover an incomplete final record.
 seen=[r[field] for r in rows]
 if len(seen)!=len(set(seen)) or not set(seen)<=set(ids):raise ValueError('Invalid saved evaluation coverage')
 expected_rows={key:C.canon({'run':C.evaluation_identity(cfg,method),'family':family,'method':method,'data':data,'smoke':False,'shard':(i,count)}) for i,part in enumerate(parts) for key in part}
 if any(r.get(row_identity)!=expected_rows[r[field]] for r in rows):raise ValueError('Saved row identity differs')
 return len(rows)==len(ids)

def can_skip_smoke(cfg,family,method):
 if evaluation_complete(cfg,family,method,'evaluation'):return True
 # Reuse a verified completed smoke from either known storage layout.
 from .data import evaluation_items
 data=C.read(C.STORE/'data/manifest.json')
 if not data:return False
 subset=evaluation_items(cfg,'gsm8k')[:1];ids={r['example_id'] for r in subset}
 expected=C.canon({'run':C.evaluation_identity(cfg,method),'family':family,'method':method,'data':data['identity'],'smoke':True,'shard':(0,1)})
 for kind in ('smoke_collect_v2','smoke'):
  root=C.STORE/'runs'/family/kind/method;meta=C.read(root/'manifest.json')
  if meta and meta['identity']==expected:
   rows=C.rows(root/'results.jsonl')
   if len(rows)==len(ids) and {r['example_id'] for r in rows}==ids and all(r['identity']==expected for r in rows):return True
 return False

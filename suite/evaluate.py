import time
import torch
from . import common as C
from .data import evaluation_items
from .engine import Engine
from .train import load_export
from .scoring import code_score,simple_score

def evaluate(family,method,cfg,smoke=False):
 C.install_signals();directory=C.STORE/'runs'/family/('smoke' if smoke else 'evaluation')/method
 key=C.canon({'run':C.identity(cfg),'family':family,'method':method,'data':C.read(C.STORE/'data/manifest.json')['identity'],'smoke':smoke})
 C.seal(directory,key,{'family':family,'method':method,'provenance':'paper_derived_port' if method in ('latcom','interlat') else 'explicit_upstream_protocol_reimplementation','smoke':smoke})
 items=[]
 for task in (['gsm8k'] if smoke else cfg['datasets']):
  subset=evaluation_items(cfg,task)
  for r in (subset[:1] if smoke else subset):items.append({**r,'cap':64 if smoke else cfg['caps'][task]})
 done_rows=C.rows(directory/'results.jsonl',repair=True);done={r['example_id']:r for r in done_rows};generated=C.rows(directory/'generations.jsonl',repair=True);generation={r['example_id']:r for r in generated}
 if len(done)!=len(done_rows) or len(generation)!=len(generated):raise ValueError('Duplicate saved evaluation IDs')
 allowed={r['example_id'] for r in items}
 if not set(done)<=allowed or not set(generation)<=allowed:raise ValueError('Saved IDs are outside this fixed cell')
 if any(r['identity']!=key for r in done_rows+generated):raise ValueError('Saved rows have a different run identity')
 if len(done)==len(items):C.write(directory/'status.json',{'status':'completed','completed':len(done),'expected':len(items),'time':C.now()});return
 engine=Engine(C.model_path(family),cfg);engine.realign_init();modules={}
 if method=='latcom':modules['compressor']=load_export(family,'latcom_stage2',cfg)
 elif method=='interlat':modules={'receiver':load_export(family,'interlat_receiver',cfg),'compressor':load_export(family,'interlat_compression',cfg)}
 # Synthetic warmup only; not included in any task statistics.
 with torch.inference_mode():engine.forward_cache(engine.model.get_input_embeddings()(torch.tensor([1,2,3],device=engine.device)))
 for item in items:
  example=item['example_id']
  if example in done:continue
  if C.STOP:C.write(directory/'status.json',{'status':'paused','completed':len(done),'expected':len(items),'time':C.now()});raise SystemExit(75)
  C.write(directory/'status.json',{'status':'running','completed':len(done),'expected':len(items),'current':example,'time':C.now()})
  if example not in generation:
   try:
    with torch.inference_mode():result=engine.pipeline(item,method,modules)
   except Exception as e:
    C.append(directory/'infrastructure_errors.jsonl',{'example_id':example,'type':type(e).__name__,'error':str(e),'time':C.now(),'scored':False});raise
   result.update(example_id=example,dataset=item['dataset'],method=method,family=family,identity=key,time=C.now(),max_new_tokens=item['cap'],engineering_smoke=smoke)
   C.append(directory/'generations.jsonl',result);generation[example]=result
  result=generation[example];start=time.perf_counter()
  score=code_score(item,result['output']['text'],cfg) if item['dataset'] in ('mbppplus','humanevalplus') else simple_score(item,result['output']['text'])
  row={**result,'score':score,'correct':score['correct'],'scoring_seconds':time.perf_counter()-start};C.append(directory/'results.jsonl',row);done[example]=row
  C.write(directory/'status.json',{'status':'running','completed':len(done),'expected':len(items),'last':example,'correct':sum(r['correct'] for r in done.values()),'time':C.now()})
 C.write(directory/'status.json',{'status':'completed','completed':len(done),'expected':len(items),'time':C.now()})

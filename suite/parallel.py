"""Bounded same-GPU inference shards; the parent alone writes merged outputs."""
import contextlib, fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path
from . import common as C
from .sampling import method_output_kind,output_kind

OOM_EXIT=86

def partition(items,index,count):
 if count<1 or not 0<=index<count:raise ValueError('Invalid shard')
 return items[index::count]


def partition_cases(variants,index,count):
 case_ids=list(dict.fromkeys(v.case_id for v in variants))
 selected=set(partition(case_ids,index,count))
 return [v for v in variants if v.case_id in selected]


def worker_info(cfg):
 return {'configured_workers':cfg.get('inference_parallel',{}).get('workers',1),
         'runtime_concurrency_bound':int(os.environ.get('LATEN_RUNTIME_CONCURRENCY','1')),
         'shard_index':int(os.environ.get('LATEN_SHARD_INDEX','0')),
         'shard_count':int(os.environ.get('LATEN_SHARD_COUNT','1')),
         'retry_mode':os.environ.get('LATEN_RETRY_MODE','none'),
         'latency_comparable_as_isolated_run':False,
         'timing_interpretation':'contended_task_wall_time_not_isolated_latency'}


def configure_allocator(cfg):
 import torch
 concurrency=int(os.environ.get('LATEN_RUNTIME_CONCURRENCY','1'))
 fraction=cfg.get('inference_parallel',{}).get('total_allocator_fraction',0.9)/concurrency
 if not 0<fraction<=0.9:raise ValueError('Invalid allocator budget')
 if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(fraction,0)
 return fraction


@contextlib.contextmanager
def model_setup_lock():
 path=C.STORE/'inference_model_setup.lock';path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)


def complete_lines(path):
 if not path.exists():return []
 data=path.read_bytes();end=data.rfind(b'\n')+1
 return [json.loads(line) for line in data[:end].splitlines() if line]


def atomic_rows(path,rows):
 from .stat_records import prepare_record
 rows=[value for row in rows if (value:=prepare_record(path,row)) is not None]
 if path.name=='generations.jsonl' and not rows:return
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
 temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));os.replace(temp,path)


def merge(directory,count,allowed,id_field):
 order={example:i for i,example in enumerate(allowed)}
 for kind in ('results','generations'):
  merged=[]
  for index in range(count):merged+=complete_lines(directory/'shards'/str(index)/(kind+'.jsonl'))
  ids=[r[id_field] for r in merged]
  if len(ids)!=len(set(ids)) or not set(ids)<=set(allowed):raise ValueError('Shard merge has duplicate/out-of-sample IDs')
  merged.sort(key=lambda r:order[r[id_field]]);atomic_rows(directory/(kind+'.jsonl'),merged)
  if kind=='results':n=len(merged)
 return n


def run_pool(commands,max_active,logdir,env_base,on_poll=lambda:None):
 """Testable process launcher. Non-OOM errors also retain completed journals."""
 pending=list(commands);active={};codes={};logdir.mkdir(parents=True,exist_ok=True)
 try:
  while pending or active:
   if C.STOP:
    for process,_ in active.values():process.terminate()
    pending=[]
   while pending and len(active)<max_active and not C.STOP:
    index,command,extra=pending.pop(0);env={**env_base,**extra};log=(logdir/f'worker_{index}_{extra.get("LATEN_RETRY_MODE","none")}.log').open('a')
    active[index]=(subprocess.Popen(command,cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log)
   for index,(process,log) in list(active.items()):
    rc=process.poll()
    if rc is not None:codes[index]=rc;log.close();del active[index]
   on_poll()
   if active:time.sleep(1)
  if C.STOP:raise SystemExit(75)
  return codes
 finally:
  # Do not leave GPU children behind if merging or the parent fails.
  for process,log in active.values():
   process.terminate()
   try:process.wait(timeout=30)
   except subprocess.TimeoutExpired:process.kill();process.wait()
   log.close()


def parallel_evaluate(action,family,method,cfg,cfg_path):
 C.install_signals();count=int(cfg.get('inference_parallel',{}).get('workers',1))
 if count not in (1,2):raise ValueError('This B200 profile supports one or two inference workers')
 kind='evaluation' if action=='evaluate' else 'semantic'
 directory=C.STORE/'runs'/family/method_output_kind(cfg,kind,method)/method
 if action=='evaluate':
  from .data import evaluation_items
  items=[r for task in cfg['datasets'] for r in evaluation_items(cfg,task)];allowed=[r['example_id'] for r in items];id_field='example_id'
 else:
  from .semantic import selected_frozen
  variants,_,_,_=selected_frozen(cfg);allowed=[v.variant_id for v in variants];id_field='variant_id'
 key=C.canon({'run':C.evaluation_identity(cfg,method),'family':family,'method':method,'action':action,'ids':allowed,'shards':count})
 C.seal(directory,key,{'family':family,'method':method,'expected':len(allowed),'shards':count,'sampled':action=='evaluate' and bool(cfg.get('evaluation_sampling',{}).get('enabled'))})
 # Single writer even for manually invoked evaluation; do not race repairs/merges.
 with (directory/'coordinator.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  start=time.perf_counter()
  def progress():
   n=merge(directory,count,allowed,id_field)
   states={str(i):C.read(directory/'shards'/str(i)/'status.json',{}) for i in range(count)}
   C.write(directory/'status.json',{'status':'completed' if n==len(allowed) else 'running','completed':n,'expected':len(allowed),'workers':states,'time':C.now()})
  progress()
  if C.read(directory/'status.json')['completed']==len(allowed):return
  command=[sys.executable,'-m','suite.cli',action,'--config',str(cfg_path),'--family',family,'--method',method]
  def commands(indices,concurrency,retry):
   return [(i,command,{'LATEN_INFERENCE_WORKER':'1','LATEN_SHARD_INDEX':str(i),'LATEN_SHARD_COUNT':str(count),'LATEN_RUNTIME_CONCURRENCY':str(concurrency),'LATEN_RETRY_MODE':retry}) for i in indices]
  env=os.environ.copy();codes=run_pool(commands(range(count),count,'none'),count,directory/'logs',env,progress)
  other={i:rc for i,rc in codes.items() if rc not in (0,OOM_EXIT)}
  oom=[i for i,rc in codes.items() if rc==OOM_EXIT]
  C.append(directory/'scheduler_attempts.jsonl',{'exit_codes':codes,'concurrency':count,'time':C.now()})
  if not other and oom and cfg.get('inference_parallel',{}).get('oom_serial_retry',True):
   print('OOM: resuming affected shards serially with saved generations retained',flush=True)
   retry=run_pool(commands(oom,1,'serial_after_oom'),1,directory/'logs',env,progress)
   C.append(directory/'scheduler_attempts.jsonl',{'exit_codes':retry,'concurrency':1,'time':C.now()});codes.update(retry)
  progress();failed={i:rc for i,rc in codes.items() if rc}
  C.append(directory/'scheduler_wall.jsonl',{'stage_wall_seconds_including_setup_scoring_retries':time.perf_counter()-start,'exit_codes':codes,'time':C.now(),'isolated_latency_comparable':False})
  if failed:
   state=C.read(directory/'status.json');C.write(directory/'status.json',{**state,'status':'blocked','failed_shards':failed})
   raise RuntimeError(f'Inference shards failed: {failed}; see {directory}/logs; no examples dropped')
  if C.read(directory/'status.json')['completed']!=len(allowed):raise RuntimeError('Workers exited without complete coverage')

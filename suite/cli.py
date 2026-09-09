"""Single-GPU stages: serial training, bounded parallel evaluation."""
import argparse,contextlib,fcntl,json,os,shutil,subprocess,sys,time,traceback
from . import common as C

def host_checks(require_idle=True):
 if sys.version_info[:2] not in ((3,10),(3,11),(3,12)):raise RuntimeError('Use Python3.10–3.12 with venv support')
 for tool in ('nvidia-smi','git','docker'):
  if not shutil.which(tool):raise RuntimeError(f'Required host executable missing: {tool}. Install it before one-click execution; no sudo/system changes are made by this bundle.')
 C.run(['docker','info'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 gpu=int(os.environ.get('GPU_ID','0'))
 p=C.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True)
 values={int(row.split(',')[0]):[x.strip() for x in row.split(',')] for row in p.stdout.splitlines()};row=values[gpu]
 if 'B200' not in row[1] or int(row[2])<170*1024:raise RuntimeError('This full suite requires one B200 with at least 170 GiB; refusing a large run on this host')
 if require_idle:
  process=C.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
  if int(row[3])>=1024 or int(row[4])>0 or any(x.split(',')[0].strip()==row[6] for x in process.stdout.splitlines()):raise RuntimeError('Selected B200 is in use; no jobs will be terminated')
 free=shutil.disk_usage(C.STORE).free/2**30
 required_disk=150 if (C.STORE/'run_identity/manifest.json').exists() else 500
 if free<required_disk:raise RuntimeError(f'Need at least {required_disk} GiB free in LATEN_STORE; found {free:.1f}')
 meminfo=dict(line.split(':',1) for line in open('/proc/meminfo'));memory=int(meminfo['MemTotal'].split()[0])/2**20
 if memory<240:raise RuntimeError(f'Full 8B optimization/resume needs about 256 GB host RAM; found {memory:.1f} GiB')
 return {'gpu':gpu,'name':row[1],'memory_mib':int(row[2]),'driver':row[5],'free_disk_gib':free,'host_ram_gib':memory}

def doctor():
 import torch
 if torch.__version__.split('+')[0]!='2.7.1' or torch.version.cuda!='12.8':raise RuntimeError('Pinned torch2.7.1+cu128 is required for the B200 run')
 if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
 if torch.cuda.get_device_capability(0)!=(10,0):raise RuntimeError('Expected Blackwell sm_100')
 x=torch.randn(64,64,device='cuda',dtype=torch.bfloat16);y=x@x
 if not torch.isfinite(y).all():raise RuntimeError('B200 BF16 smoke failed')
 torch.cuda.synchronize();C.write(C.STORE/'environment/gpu_doctor.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'capability':list(torch.cuda.get_device_capability()),'time':C.now()})

def downloads(cfg):
 from huggingface_hub import snapshot_download
 from .data import build_data
 for family,spec in cfg['families'].items():
  snapshot_download(spec['repo'],revision=spec['revision'],local_dir=C.model_path(family),allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken','LICENSE*'])
  modelconfig=C.read(C.model_path(family)/'config.json')
  if modelconfig.get('model_type')!='qwen3':raise ValueError('Unexpected model family')
  index=C.read(C.model_path(family)/'model.safetensors.index.json')
  if not index or any(not (C.model_path(family)/name).is_file() for name in set(index['weight_map'].values())):raise ValueError('Model weight shards incomplete')
  hashes={str(p.relative_to(C.model_path(family))):C.sha(p) for p in C.model_path(family).glob('*') if p.is_file()}
  manifest=C.STORE/'environment'/('model_'+family+'.json');previous=C.read(manifest)
  if previous and previous['files']!=hashes:raise ValueError('Downloaded model bytes differ from the saved manifest')
  C.write(manifest,{'source':spec,'files':hashes,'time':C.now()})
 for name,url,commit in [('LatentMAS','https://github.com/Gen-Verse/LatentMAS.git','9a9e4d331eb11430bd9e64754c6b252b06d73031'),('Interlat','https://github.com/XiaoDu-flying/Interlat.git','66a89cb4d4097b2f86cbe48ed9851d6e8578f821')]:
  path=C.STORE/'upstream'/name
  if not path.exists():
   path.mkdir(parents=True);C.run(['git','init',str(path)]);C.run(['git','-C',str(path),'remote','add','origin',url]);C.run(['git','-C',str(path),'fetch','--depth','1','origin',commit]);C.run(['git','-C',str(path),'checkout','--detach',commit])
  head=C.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
  if head!=commit:raise ValueError('Upstream checkout identity differs')
 build_data(cfg)
 image_meta=C.read(C.STORE/'sandbox/image.json');docker_hash=C.canon({p.name:C.sha(p) for p in (C.ROOT/'sandbox').iterdir() if p.is_file()})
 if not image_meta or image_meta['build_hash']!=docker_hash:
  C.run(['docker','build','--pull','-t',cfg['code_tests']['docker_image'],str(C.ROOT/'sandbox')]);image=C.run(['docker','image','inspect','--format','{{.Id}}',cfg['code_tests']['docker_image']],capture_output=True,text=True).stdout.strip();C.write(C.STORE/'sandbox/image.json',{'image_id':image,'build_hash':docker_hash,'time':C.now()})
 from .scoring import code_score
 fixture={'test_code':'assert add(2, 3) == 5','dataset':'mbppplus'}
 if not code_score(fixture,'```python\ndef add(a,b): return a+b\n```',cfg)['correct']:raise RuntimeError('Code sandbox positive canary failed')
 if code_score(fixture,'```python\ndef add(a,b): return 0\n```',cfg)['correct']:raise RuntimeError('Code sandbox negative canary failed')
 # Score every selected canonical solution once before model evaluation. Never drop failing tasks.
 from .data import evaluation_items
 refdir=C.STORE/('sandbox/reference_checks_sample10' if cfg.get('evaluation_sampling',{}).get('enabled') else 'sandbox/reference_checks');refkey=C.canon({'data':C.read(C.STORE/'data/manifest.json')['identity'],'sampling':cfg.get('evaluation_sampling'),'image':C.read(C.STORE/'sandbox/image.json')['image_id'],'tests':cfg['code_tests']})
 C.seal(refdir,refkey,{'purpose':'validate_expanded_test_execution_not_model_evaluation'})
 checked={r['example_id'] for r in C.rows(refdir/'results.jsonl',repair=True) if r['correct']}
 for task in ('mbppplus','humanevalplus'):
  if task not in cfg['datasets']:continue
  for item in evaluation_items({**cfg,'evaluation_limit_per_dataset':None},task):
   if item['example_id'] in checked:continue
   score=code_score(item,'```python\n'+item['reference_code']+'\n```',cfg)
   C.append(refdir/'results.jsonl',{'example_id':item['example_id'],**score,'time':C.now()})
   if not score['correct']:raise RuntimeError('Canonical code/test gate failed: '+item['example_id']+'; inspect sandbox/reference_checks/results.jsonl; no denominator changed')
   checked.add(item['example_id'])
 C.write(refdir/'complete.json',{'verified':len(checked),'identity':refkey,'time':C.now()})
 C.write(C.STORE/'environment/downloads.json',{'models':cfg['families'],'data':C.read(C.STORE/'data/manifest.json')['identity'],'time':C.now()})

def child(action,cfg_path,*extra):
 command=[sys.executable,'-m','suite.cli',action,'--config',str(cfg_path),*extra];print('START',action,*extra,flush=True)
 return subprocess.run(command,cwd=C.ROOT).returncode

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['all','preflight-host','_doctor','download','collect','train','evaluate','semantic','semantic-all','eval-all','semantic-smoke','smoke','report','status']);p.add_argument('--config',default=str(C.ROOT/'configs/default.json'));p.add_argument('--family',choices=['4b','8b']);p.add_argument('--method',choices=['latentmas','latentmas_h2o','latentmas_hidden','latcom','interlat']);p.add_argument('--stage',choices=['latcom_stage1','latcom_stage2','interlat_receiver','interlat_compression']);a=p.parse_args();cfg=C.config(a.config);C.STORE.mkdir(parents=True,exist_ok=True)
 if a.action=='preflight-host':print(json.dumps(host_checks()));return
 if a.action=='_doctor':doctor();return
 if a.action in ('status','report'):
  from .report import report
  report(cfg);return
 if a.action=='download':downloads(cfg);return
 if a.action=='collect':
  from .collect import collect
  if not a.family:p.error('--family required')
  collect(a.family,cfg);return
 if a.action=='train':
  from .train import train_stage
  if not a.family or not a.stage:p.error('--family and --stage required')
  train_stage(a.family,a.stage,cfg);return
 if a.action in ('evaluate','semantic') and not os.environ.get('LATEN_INFERENCE_WORKER'):
  from .parallel import parallel_evaluate
  if not a.family or not a.method:p.error('--family and --method required')
  parallel_evaluate(a.action,a.family,a.method,cfg,__import__('pathlib').Path(a.config).resolve());return
 if a.action=='evaluate':
  from .evaluate import evaluate
  if not a.family or not a.method:p.error('--family and --method required')
  evaluate(a.family,a.method,cfg);return
 if a.action in ('semantic','semantic-smoke'):
  from .evaluate_semantic import evaluate_semantic
  if not a.family or not a.method:p.error('--family and --method required')
  evaluate_semantic(a.family,a.method,cfg,smoke=a.action=='semantic-smoke');return
 # Single coordinator owns a store lock; no simultaneous model stages.
 lock=C.STORE/'run.lock'
 with lock.open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);host=C.STORE/'environment/host.json';C.write(host,host_checks())
  cfg_path=__import__('pathlib').Path(a.config).resolve();key=C.identity(cfg);C.seal(C.STORE/('sampled_eval_run_identity' if a.action=='eval-all' else 'semantic_parallel_run_identity' if a.action=='semantic-all' else 'run_identity'),key,{'config':cfg})
  if child('_doctor',cfg_path)!=0:raise RuntimeError('B200 environment validation failed')
  if a.action!='semantic-all' and child('download',cfg_path)!=0:raise RuntimeError('Download/setup failed')
  from .report import report
  for family in ([a.family] if a.family else cfg['families']):
   if a.action in ('semantic-all','eval-all'):
    for method in cfg['methods']:
     if a.action=='eval-all' and child('evaluate',cfg_path,'--family',family,'--method',method):raise RuntimeError('General evaluation stopped; saved progress retained')
     rc=child('semantic',cfg_path,'--family',family,'--method',method);report(cfg,False)
     if rc:raise RuntimeError(f'{family}/{method} semantic evaluation stopped; saved progress retained')
    continue
   # Short real-model interface checks; separate directory, never counted as benchmark scores.
   for method in ('latentmas','latentmas_h2o','latentmas_hidden'):
    command=[sys.executable,'-c','from suite.common import config; from suite.evaluate import evaluate; import sys; evaluate(sys.argv[1],sys.argv[2],config(sys.argv[3]),smoke=True)',family,method,str(cfg_path)]
    if subprocess.run(command,cwd=C.ROOT).returncode:raise RuntimeError('Real-model interface smoke failed: '+method)
   if a.action=='smoke':continue
   for method in ('latentmas','latentmas_h2o','latentmas_hidden'):
    if method not in cfg['methods']:continue
    rc=child('evaluate',cfg_path,'--family',family,'--method',method);report(cfg,False)
    if rc:raise RuntimeError(f'{family}/{method} stopped with code {rc}; saved results retained')
    if cfg.get('semantic_benchmark',{}).get('enabled') and child('semantic',cfg_path,'--family',family,'--method',method):raise RuntimeError('Semantic evaluation stopped; saved progress retained')
   if any(m in cfg['methods'] for m in ('latcom','interlat')):
    rc=child('collect',cfg_path,'--family',family)
    if rc:raise RuntimeError('Training data collection stopped; saved progress retained')
   for method,stages in [('latcom',('latcom_stage1','latcom_stage2')),('interlat',('interlat_receiver','interlat_compression'))]:
    if method not in cfg['methods']:continue
    for stage in stages:
     rc=child('train',cfg_path,'--family',family,'--stage',stage);report(cfg,False)
     if rc:raise RuntimeError(f'{family}/{stage} stopped with code {rc}; checkpoint retained')
    rc=child('evaluate',cfg_path,'--family',family,'--method',method);report(cfg,False)
    if rc:raise RuntimeError(f'{family}/{method} stopped; saved generations/results retained')
    if cfg.get('semantic_benchmark',{}).get('enabled') and child('semantic',cfg_path,'--family',family,'--method',method):raise RuntimeError('Semantic evaluation stopped; saved progress retained')
  complete=report(cfg)
  if a.action=='semantic-all' and C.read(C.STORE/'reports/semantic_summary.json')['complete']:C.write(C.STORE/'SEMANTIC_COMPLETE.json',{'identity':key,'completed_cells':len(cfg['families'])*len(cfg['methods']),'time':C.now()})
  if a.action in ('all','eval-all') and complete:C.write(C.STORE/'COMPLETE.json',{'identity':key,'completed_cells':len(cfg['families'])*len(cfg['methods'])*(len(cfg['datasets'])+int(cfg.get('semantic_benchmark',{}).get('enabled',False))),'time':C.now()})
if __name__=='__main__':
 try:main()
 except Exception as e:
  oom=type(e).__name__=='OutOfMemoryError' or ('out of memory' in str(e).lower() and 'cuda' in str(e).lower())
  path=C.STORE/('worker_errors/'+str(os.getpid())+'.json' if os.environ.get('LATEN_INFERENCE_WORKER') else 'last_error.json')
  C.write(path,{'type':type(e).__name__,'error':str(e),'time':C.now(),'benchmark_failure_score':False,'cuda_oom':oom});traceback.print_exc();sys.exit(86 if oom else 1)

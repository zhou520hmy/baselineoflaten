from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,random,signal,subprocess
ROOT=Path(__file__).resolve().parents[1]
STORE=Path(os.environ.get('LATEN_STORE',ROOT/'storage')).resolve()
STOP=False

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def canon(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except FileNotFoundError:return default

def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)
def append(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def rows(path,repair=False):
 path=Path(path)
 if not path.exists():return []
 data=path.read_bytes();end=data.rfind(b'\n')+1
 if end!=len(data):
  if not repair:raise ValueError(f'Incomplete last JSONL record: {path}')
  backup=path.with_suffix(path.suffix+'.incomplete_tail');backup.write_bytes(data[end:])
  with path.open('r+b') as f:f.truncate(end)
 result=[json.loads(x) for x in data[:end].splitlines() if x]
 return result

def config(path=None):return read(path or ROOT/'configs/default.json')
def identity(config):
 sources={str(p.relative_to(ROOT)):sha(p) for folder in ('suite','vendor') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.jsonl')}
 return canon({'config':config,'sources':sources,'dependencies':sha(ROOT/'requirements.txt')})
def seal(directory,key,metadata):
 path=Path(directory)/'manifest.json';old=read(path)
 if old and old['identity']!=key:raise ValueError(f'Resume identity differs: {path}; choose a new run directory')
 if not old:write(path,{'identity':key,'created_at':now(),**metadata})
def keyed_seed(seed,*keys):return (seed+int(canon(keys)[:8],16))%(2**31)
def install_signals():
 def stop(*_):
  global STOP
  STOP=True
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
def run(command,**kwargs):return subprocess.run(command,check=True,**kwargs)
def model_path(family):return STORE/'models'/family

def stage_dir(family,stage):return STORE/'runs'/family/'training'/stage

def compatible_export_identities(cfg):
 current=identity(cfg);allowed={current};previous=read(ROOT/'evidence/previous_release.json')
 if cfg.get('semantic_benchmark',{}).get('reuse_completed_previous_release_exports') and previous and {k:v for k,v in cfg.items() if k!='semantic_benchmark'}==previous['config']:
  allowed.add(previous['identity'])
 return allowed

"""Reuse completed downloads and reference checks without network or GPU execution."""
import subprocess
from . import common as C


def model_ready(family,cfg):
 root=C.model_path(family);meta=C.read(C.STORE/'environment'/('model_'+family+'.json'))
 if not meta:return False
 if meta['source']!=cfg['families'][family]:raise ValueError('Saved model source/revision differs')
 for name,digest in meta['files'].items():
  path=root/name
  if not path.is_file():return False
  if C.sha(path)!=digest:raise ValueError('Downloaded model hash differs: '+name)
 index=C.read(root/'model.safetensors.index.json')
 return bool(index and all((root/name).is_file() for name in set(index['weight_map'].values())))


def references_ready(cfg):
 image=C.read(C.STORE/'sandbox/image.json');data=C.read(C.STORE/'data/manifest.json')
 if not image or not data:return False
 build=C.canon({p.name:C.sha(p) for p in (C.ROOT/'sandbox').iterdir() if p.is_file()})
 if image['build_hash']!=build:return False
 check=subprocess.run(['docker','image','inspect',image['image_id']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 if check.returncode:return False
 ref=C.STORE/('sandbox/reference_checks_sample10' if cfg.get('evaluation_sampling',{}).get('enabled') else 'sandbox/reference_checks')
 expected=C.canon({'data':data['identity'],'sampling':cfg.get('evaluation_sampling'),'image':image['image_id'],'tests':cfg['code_tests']})
 meta=C.read(ref/'complete.json');manifest=C.read(ref/'manifest.json')
 if not meta or not manifest or meta['identity']!=expected or manifest['identity']!=expected:return False
 from .data import evaluation_items
 needed={r['example_id'] for task in ('mbppplus','humanevalplus') if task in cfg['datasets'] for r in evaluation_items({**cfg,'evaluation_limit_per_dataset':None},task)}
 rows=C.rows(ref/'results.jsonl');passed={r['example_id'] for r in rows if r['correct']}
 return needed<=passed


def assets_ready(cfg,families):
 if not all(model_ready(f,cfg) for f in families):return False
 if not C.read(C.STORE/'data/manifest.json'):return False
 from .data import build_data
 build_data(cfg)  # Existing prepared data path only: hashes and original config, no downloads.
 for name,commit in [('LatentMAS','9a9e4d331eb11430bd9e64754c6b252b06d73031'),('Interlat','66a89cb4d4097b2f86cbe48ed9851d6e8578f821')]:
  path=C.STORE/'upstream'/name
  if not path.exists():return False
  head=C.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
  if head!=commit:raise ValueError('Upstream source revision differs')
 return references_ready(cfg)

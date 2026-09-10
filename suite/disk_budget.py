"""Check atomic full-optimizer checkpoint headroom before GPU allocation."""
import shutil
from . import common as C

def training_space(family,stage,cfg):
 index=C.read(C.model_path(family)/'model.safetensors.index.json')
 if not index or not index.get('metadata',{}).get('total_size'):raise RuntimeError('Model index lacks total_size for training disk admission')
 # BF16 base bytes -> FP32 parameter + two FP32 Adam moments: 6x.
 # Include 10% for learned adapter/slot weights and serialization overhead.
 one=int(index['metadata']['total_size']*6*1.10)
 old=C.stage_dir(family,stage)/'latest.pt'
 reserve=max(0,2*one-(old.stat().st_size if old.exists() else 0))+20*2**30
 return {'required_free_bytes':reserve,'estimated_checkpoint_bytes':one,'free_bytes':shutil.disk_usage(C.STORE).free}

def check_training_space(family,stage,cfg):
 budget=training_space(family,stage,cfg);C.write(C.stage_dir(family,stage)/'disk_budget.json',budget)
 if budget['free_bytes']<budget['required_free_bytes']:
  raise RuntimeError(f"Insufficient disk for {family}/{stage}: need {budget['required_free_bytes']/2**30:.1f} GiB free including old/new optimizer checkpoint replacement; have {budget['free_bytes']/2**30:.1f}. Free space or use a larger LATEN_STORE; no files were deleted.")
 return budget

"""Independent branch failures do not suppress the remaining experiment matrix."""
from . import common as C

def run_remaining(families,methods,semantic_enabled,call,report):
 failures=[]
 def attempt(action,family,method=None,stage=None):
  extra=['--family',family]
  if method:extra+=['--method',method]
  if stage:extra+=['--stage',stage]
  rc=call(action,*extra)
  if rc:
   event={'action':action,'family':family,'method':method,'stage':stage,'exit_code':rc,'time':C.now(),'scored':False};failures.append(event);C.append(C.STORE/'branch_failures.jsonl',event)
   print('BLOCKED BRANCH',event,flush=True)
  report()
  return rc==0
 for family in families:
  for method in ('latentmas','latentmas_h2o','latentmas_hidden'):
   if method not in methods:continue
   if not attempt('_smoke_one',family,method):continue
   attempt('evaluate',family,method)
   if semantic_enabled:attempt('semantic',family,method)
  learned=[m for m in ('latcom','interlat') if m in methods]
  if not learned:continue
  if not attempt('collect',family):continue
  for method in learned:
   stages=('latcom_stage1','latcom_stage2') if method=='latcom' else ('interlat_receiver','interlat_compression')
   success=True
   for stage in stages:
    if not attempt('train',family,stage=stage):success=False;break
   if not success:continue
   attempt('evaluate',family,method)
   if semantic_enabled:attempt('semantic',family,method)
 return failures

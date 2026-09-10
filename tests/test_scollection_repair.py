"""Collection policy and resume regressions use known fixtures, not model evidence."""
import copy,hashlib,json,re,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from suite import common as C
from suite.collection_policy import answer_matches,final_answer,probe,eligible_methods,pool_counts,stage_records
from suite.schedule import run_remaining
from suite.disk_budget import training_space,check_training_space

class CollectionRepair(unittest.TestCase):
 def test_original_control_token_false_negative_is_fixed(self):
  raw='Answer: Paris<|im_end|>'
  old=' '.join(re.sub(r'[^\w\s]',' ',raw.split(':')[-1].lower()).split())
  self.assertNotEqual(old,'paris');self.assertTrue(answer_matches(raw,'Paris'))
  for raw,gold in [('Answer: 42<|endoftext|>','42'),('Answer: The United States.\n<|im_end|>','United States'),('<think>maybe Rome</think>\nAnswer: Paris<|im_end|>','Paris'),('Final \\boxed{1/2}<|im_end|>','0.5')]:self.assertTrue(answer_matches(raw,gold))
 def test_no_answer_leakage_from_reasoning_or_unclosed_think(self):
  for raw in ('<think>Answer: Paris','<think>Paris</think>Answer: Rome<|im_end|>','Paris is one possible option.\nAnswer: Rome','We know Paris but cannot decide.'):
   self.assertFalse(answer_matches(raw,'Paris'))
  self.assertFalse(probe({'text':'Answer: Paris','stop_reason':'max_new_tokens','output_tokens':512},'Paris')['correct'])
 def test_separate_method_eligibility_and_aux_minimum(self):
  yes={'correct':True};no={'correct':False}
  self.assertEqual(eligible_methods(no,yes,yes),['latcom','interlat'])
  self.assertEqual(eligible_methods(no,no,yes),['interlat'])
  self.assertEqual(eligible_methods(yes,None,yes),['interlat'])
  self.assertEqual(eligible_methods(no,yes,no),['latcom'])
  records=[dict(status='retained',aux=True,eligible_for=['latcom','interlat']) for _ in range(64)]
  self.assertEqual(pool_counts(records),{'latcom':0,'interlat':0})
  with self.assertRaises(RuntimeError):stage_records(records,'interlat_receiver',32)
  records.append(dict(status='retained',aux=False,eligible_for=['interlat']))
  self.assertEqual(len(stage_records(records,'interlat_receiver',1)),65)
  with self.assertRaises(RuntimeError):stage_records(records,'latcom_stage1',1)
 def test_training_free_identity_reuses_only_unchanged_inference(self):
  cfg=C.config();prior=C.read(C.ROOT/'evidence/previous_parallel_release.json')
  self.assertEqual(C.evaluation_identity(cfg,'latentmas'),prior['identity'])
  self.assertNotEqual(C.evaluation_identity(cfg,'latcom'),prior['identity'])
  changed=copy.deepcopy(cfg);changed['caps']['gsm8k']+=1
  self.assertNotEqual(C.evaluation_identity(changed,'latentmas'),prior['identity'])
  original=C.sha
  with patch.object(C,'sha',side_effect=lambda p:'changed' if str(p).endswith('suite/engine.py') else original(p)):
   self.assertNotEqual(C.evaluation_identity(cfg,'latentmas'),prior['identity'])
 def test_old_new_cache_training_and_eval_paths_separate(self):
  from suite.sampling import method_output_kind
  self.assertEqual(C.cache_dir('4b').name,'training_cache_v2');self.assertEqual(C.stage_dir('8b','latcom_stage1').parent.name,'training_collect_v2')
  self.assertEqual(method_output_kind(C.config(),'evaluation','latentmas'),'evaluation_sample10');self.assertEqual(method_output_kind(C.config(),'evaluation','latcom'),'evaluation_sample10_collect_v2')
 def test_latcom_failure_continues_interlat_and_8b(self):
  calls=[]
  def call(action,*args):
   calls.append((action,args))
   return int(action=='train' and args[-1]=='latcom_stage1' and args[1]=='4b')
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)):
   failures=run_remaining(['4b','8b'],C.config()['methods'],True,call,lambda:None)
   self.assertEqual(len(failures),1)
   self.assertTrue(any(a=='evaluate' and '--family' in x and x[1]=='4b' and x[-1]=='interlat' for a,x in calls))
   self.assertTrue(any(a=='evaluate' and x[1]=='8b' and x[-1]=='latentmas' for a,x in calls))
   self.assertFalse(any(a=='evaluate' and x[1]=='4b' and x[-1]=='latcom' for a,x in calls))
 def test_collect_failure_does_not_skip_later_family(self):
  calls=[]
  def call(a,*x):calls.append((a,x));return int(a=='collect' and x[1]=='4b')
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)):
   failures=run_remaining(['4b','8b'],['latentmas','latcom','interlat'],True,call,lambda:None)
   self.assertEqual(len(failures),1);self.assertTrue(any(a=='train' and x[1]=='8b' for a,x in calls));self.assertFalse(any(a=='train' and x[1]=='4b' for a,x in calls))
 def test_checkpoint_disk_admission_counts_atomic_replacement(self):
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)),patch('suite.disk_budget.shutil.disk_usage',return_value=SimpleNamespace(free=90*2**30)):
   C.write(C.model_path('8b')/'model.safetensors.index.json',{'metadata':{'total_size':16_000_000_000}})
   result=training_space('8b','latcom_stage1',C.config());self.assertGreater(result['required_free_bytes'],190*2**30)
   with self.assertRaises(RuntimeError):check_training_space('8b','latcom_stage1',C.config())

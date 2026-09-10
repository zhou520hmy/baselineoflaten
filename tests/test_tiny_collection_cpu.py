"""End-to-end collection journal fixtures. Deterministic fake responses, real CPU tensors."""
import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from suite import common as C,collect as K

class FakeEngine:
 latent_correct=False
 def __init__(self,*args):self.device='cpu';self.model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=0));self.tokenizer=SimpleNamespace(encode=lambda s,**kw:[ord(c) for c in s],decode=lambda ids,**kw:''.join(chr(i) for i in ids))
 def realign_init(self):pass
 def seed(self,*args):pass
 def qa_messages(self,q,evidence=None):return [{'q':q,'evidence':evidence}]
 def render(self,messages,*args):return torch.tensor([1,2]),torch.zeros(2,3),0,1
 def ids(self,s):return torch.tensor([ord(c) for c in s])
 def decode(self,emb,cap,**kwargs):return {'text':'Answer: '+('Paris' if len(emb)>2 and self.latent_correct else 'Rome')+'<|im_end|>','stop_reason':'eos','output_tokens':3}
 def rollout(self,emb,steps,cache=None,absolute=None):return {'z':torch.ones(2,3),'cache':None,'absolute_next':4}
 def full_plan_hidden(self,messages,cap):return torch.ones(3,3),torch.tensor([5,6,0]),'Answer: Paris<|im_end|>',torch.tensor([1,2])

class CollectionCPU(unittest.TestCase):
 def setup_store(self,root):
  candidates=[dict(id=f'main/{i}',question=f'Q{i}',answer='Paris',gold_docs=['Paris facts'],other_docs=['Rome facts'],source='hotpot_train',aux=False) for i in range(2)]
  candidates+=[dict(id='aux/1',question='aux',answer='Paris',gold_docs=['Paris facts'],other_docs=[],source='gsm8k_train',aux=True)]
  p=root/'data/training_candidates.jsonl'
  for row in candidates:C.append(p,row)
  C.write(root/'data/manifest.json',{'identity':'fixture','files':{p.name:C.sha(p)}})
  old=root/'runs/4b/training_cache';old.mkdir(parents=True);(old/'failure.txt').write_text('original failure remains')
 def test_interlat_collects_when_frozen_latents_fail_and_resumes(self):
  cfg=copy.deepcopy(C.config());cfg['training']['minimum_retained']=1
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)),patch.object(C,'install_signals'),patch.object(K,'Engine',FakeEngine):
   self.setup_store(Path(d));K.collect('4b',cfg);root=C.cache_dir('4b');meta=C.read(root/'complete.json')
   self.assertEqual(meta['main_retained_by_method'],{'latcom':0,'interlat':2});self.assertEqual(meta['method_ready'],{'latcom':False,'interlat':True});self.assertEqual(C.read(root/'status.json')['gate_rejections']['latcom']['full_latent_unanswerable'],2)
   rows=C.rows(root/'records.jsonl');self.assertNotIn('raw_text',rows[0]['probes']['evidence_text_plan']);self.assertTrue(rows[0]['probes']['evidence_text_plan']['correct']);self.assertEqual(rows[0]['eligible_for'],['interlat'])
   saved=torch.load(root/rows[0]['file'],weights_only=True);self.assertEqual(saved['plan_text'],'Answer: Paris')
   self.assertEqual((Path(d)/'runs/4b/training_cache/failure.txt').read_text(),'original failure remains')
   with patch.object(K,'Engine',side_effect=AssertionError('No model reload for a completed scan')):K.collect('4b',cfg)
   (root/rows[0]['file']).write_bytes(b'broken')
   with self.assertRaises(ValueError):K.collect('4b',cfg)
 def test_correct_eos_answers_populate_both_pools(self):
  cfg=copy.deepcopy(C.config());cfg['training']['minimum_retained']=1
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)),patch.object(C,'install_signals'),patch.object(K,'Engine',FakeEngine),patch.object(FakeEngine,'latent_correct',True):
   self.setup_store(Path(d));K.collect('4b',cfg);meta=C.read(C.cache_dir('4b')/'complete.json');self.assertTrue(all(meta['method_ready'].values()));self.assertEqual(meta['main_retained_by_method'],{'latcom':2,'interlat':2})

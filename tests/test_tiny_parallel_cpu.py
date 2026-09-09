"""Worker evaluator integration with tiny random CPU models, never benchmark scores."""
import copy,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from test_tiny_cpu import Tokenizer
from suite import common as C,evaluate as E,evaluate_semantic as S
from suite.engine import Engine
from suite.parallel import merge
from suite.semantic import load_frozen
from suite.semantic_engine import pipeline as semantic_pipeline

class WorkerCPU(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);torch.manual_seed(31)
  model=Qwen3ForCausalLM(Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=48,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=32768,eos_token_id=0)).float().eval();model.generation_config.eos_token_id=0
  cls.cfg=C.config();cls.cfg.update(sender_latent_steps=2,temperature=0.)
  cls.cfg['caps']={k:2 for k in cls.cfg['caps']};cls.cfg['semantic_benchmark']['sender_latent_steps']=2
  cls.engine=Engine(None,cls.cfg,'cpu',model=model,tokenizer=Tokenizer());cls.engine.realign_init()
 def test_actual_general_worker_journals_rng_tokens_and_resume(self):
  cfg=copy.deepcopy(self.cfg);cfg['datasets']={'gsm8k':40};items=[dict(example_id=f'tiny/{i}',dataset='gsm8k',question='What is 1+1?',gold='2') for i in range(4)]
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)),patch.object(C,'install_signals'),patch.object(E,'Engine',return_value=self.engine),patch.object(E,'configure_allocator'),patch.object(E,'evaluation_items',return_value=items):
   C.write(Path(d)/'data/manifest.json',{'identity':'cpu_fixture'})
   for i in range(2):
    with patch.dict(os.environ,{'LATEN_INFERENCE_WORKER':'1','LATEN_SHARD_INDEX':str(i),'LATEN_SHARD_COUNT':'2','LATEN_RUNTIME_CONCURRENCY':'2'}):E.evaluate('4b','latentmas',cfg)
   root=Path(d)/'runs/4b/evaluation_sample10/latentmas';self.assertEqual(merge(root,2,[r['example_id'] for r in items],'example_id'),4)
   saved=C.rows(root/'results.jsonl')
   for item,row in zip(items,saved):
    with torch.inference_mode():direct=self.engine.pipeline({**item,'cap':2},'latentmas')
    self.assertEqual(row['output']['token_ids'],direct['output']['token_ids']);self.assertEqual(row['efficiency']['generated_text_tokens'],direct['efficiency']['generated_text_tokens']);self.assertEqual(row['execution']['runtime_concurrency_bound'],2)
   with patch.dict(os.environ,{'LATEN_INFERENCE_WORKER':'1','LATEN_SHARD_INDEX':'0','LATEN_SHARD_COUNT':'2','LATEN_RUNTIME_CONCURRENCY':'1','LATEN_RETRY_MODE':'serial_after_oom'}),patch.object(E,'Engine',side_effect=AssertionError('Do not load for complete shard')):E.evaluate('4b','latentmas',cfg)
 def test_actual_semantic_worker_whole_pairs_and_resume(self):
  variants,cases,programs,manifest=load_frozen();ids=list(dict.fromkeys(v.case_id for v in variants))[:2];variants=[v for v in variants if v.case_id in ids]
  def tiny_pipeline(engine,case,v,program,method,cfg,modules):
   cfg=copy.deepcopy(cfg);cfg['semantic_benchmark']['reasoning_max_new_tokens']=2
   return semantic_pipeline(engine,case,v,program,method,cfg,modules)
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)),patch.object(C,'install_signals'),patch.object(S,'Engine',return_value=self.engine),patch.object(S,'configure_allocator'),patch.object(S,'selected_frozen',return_value=(variants,cases,programs,manifest)),patch.object(S,'pipeline',side_effect=tiny_pipeline):
   for i in range(2):
    with patch.dict(os.environ,{'LATEN_INFERENCE_WORKER':'1','LATEN_SHARD_INDEX':str(i),'LATEN_SHARD_COUNT':'2','LATEN_RUNTIME_CONCURRENCY':'2'}):S.evaluate_semantic('4b','latentmas',self.cfg)
   root=Path(d)/'runs/4b/semantic_parallel/latentmas';self.assertEqual(merge(root,2,[v.variant_id for v in variants],'variant_id'),8)
   for i in range(2):
    summary=C.read(root/'shards'/str(i)/'summary.json');self.assertTrue(summary['complete']);self.assertEqual(summary['pairs']['pair_count'],3)
   with patch.dict(os.environ,{'LATEN_INFERENCE_WORKER':'1','LATEN_SHARD_INDEX':'1','LATEN_SHARD_COUNT':'2','LATEN_RUNTIME_CONCURRENCY':'1'}),patch.object(S,'Engine',side_effect=AssertionError('Do not reload for completed shard')):S.evaluate_semantic('4b','latentmas',self.cfg)

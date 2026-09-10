"""Deterministic subset and real CPU-process scheduler checks; no GPU scoring."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from suite import common as C,parallel as P
from suite.sampling import select_general,expected_general,output_kind,save_selection
from suite.semantic import load_frozen,selected_frozen,summarize,vector_score,vector_from_variant

class SamplingParallel(unittest.TestCase):
 def test_counts_full_private_and_balanced_general(self):
  cfg=C.config();counts={k:expected_general(cfg,k) for k in cfg['datasets']}
  self.assertEqual(list(counts.values()),[132,238,117,30,38,16,20]);self.assertEqual(10*(sum(counts.values())+648),12390)
  self.assertEqual(len(selected_frozen(cfg)[0]),648)
  for task,n in cfg['datasets'].items():
   items=[dict(example_id=f'{task}/{i}',dataset=task,question='x'*(i+1),reference_code='def f(): return 1') for i in range(n)]
   selected,meta=select_general(items,cfg,task);again,meta2=select_general(list(reversed(items)),cfg,task)
   self.assertEqual(selected,again);self.assertEqual(meta,meta2);self.assertEqual(len(selected),counts[task])
   quotas=[s['selected_count'] for s in meta['strata']];self.assertLessEqual(max(quotas)-min(quotas),1);self.assertGreater(min(quotas),0)
   self.assertEqual(len({r['example_id'] for r in selected}),counts[task])
 def test_sample_manifest_tamper_and_seed_guard(self):
  cfg=C.config();cfg['datasets']={'arc_easy':30};items=[dict(example_id=str(i),dataset='arc_easy',question='x'*i) for i in range(30)]
  subset,meta=select_general(items,cfg,'arc_easy')
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d:
   store=Path(d);save_selection(store,'arc_easy',subset,meta,{'sha':'fixture'});save_selection(store,'arc_easy',subset,meta,{'sha':'fixture'})
   with self.assertRaises(ValueError):save_selection(store,'arc_easy',subset,{**meta,'seed':43},{'sha':'fixture'})
   (store/'data/sampled10/arc_easy.jsonl').write_text('tampered')
   with self.assertRaises(ValueError):save_selection(store,'arc_easy',subset,meta,{'sha':'fixture'})
 def test_case_shards_keep_pairs_and_all_frozen_rows(self):
  variants,_,programs,_=load_frozen();parts=[P.partition_cases(variants,i,2) for i in range(2)]
  self.assertEqual([len(x) for x in parts],[324,324]);self.assertFalse({v.variant_id for v in parts[0]}&{v.variant_id for v in parts[1]})
  for part in parts:
   fixture=[vector_score(v,programs[v.policy_program_id],dict(vector=vector_from_variant(v),reasoning_stop_reason='thinking_close')) for v in part]
   result=summarize(fixture,part);self.assertTrue(result['complete']);self.assertEqual(result['pairs']['pair_count'],243)
 def test_merge_partial_tail_and_duplicate_rejection(self):
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d:
   root=Path(d);P.atomic_rows(root/'shards/0/results.jsonl',[{'example_id':'a'}]);p=root/'shards/1/results.jsonl';P.atomic_rows(p,[{'example_id':'b'}])
   with p.open('ab') as f:f.write(b'{"example_id":')
   self.assertEqual(P.merge(root,2,['b','a'],'example_id'),2);self.assertEqual([r['example_id'] for r in C.rows(root/'results.jsonl')],['b','a'])
   P.atomic_rows(p,[{'example_id':'a'}])
   with self.assertRaises(ValueError):P.merge(root,2,['b','a'],'example_id')
 def test_real_process_pool_concurrency_bound(self):
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d:
   root=Path(d);code="import pathlib,sys,time,json; start=time.time(); time.sleep(1.3); pathlib.Path(sys.argv[1]).write_text(json.dumps([start,time.time()]))"
   commands=[(i,[sys.executable,'-c',code,str(root/f'{i}.json')],{}) for i in range(3)]
   codes=P.run_pool(commands,2,root/'logs',os.environ.copy());self.assertEqual(codes,{0:0,1:0,2:0})
   spans=[C.read(root/f'{i}.json') for i in range(3)]
   self.assertLess(max(spans[0][0],spans[1][0]),min(spans[0][1],spans[1][1]));self.assertGreaterEqual(spans[2][0],min(spans[0][1],spans[1][1]))
 def test_oom_serial_retry_preserves_completed_shard_and_resume(self):
  cfg=C.config();cfg['datasets']={'gsm8k':20};items=[{'example_id':'a'},{'example_id':'b'}]
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)),patch.object(C,'install_signals'),patch('suite.data.evaluation_items',return_value=items):
   directory=Path(d)/'runs/4b/evaluation_sample10/latentmas';calls=[]
   def pool(commands,max_active,logs,env,on_poll):
    calls.append((max_active,[c[0] for c in commands]))
    if len(calls)==1:
     P.atomic_rows(directory/'shards/0/results.jsonl',[{'example_id':'a'}]);on_poll();return {0:0,1:86}
    self.assertEqual(commands[0][2]['LATEN_RETRY_MODE'],'serial_after_oom');P.atomic_rows(directory/'shards/1/results.jsonl',[{'example_id':'b'}]);on_poll();return {1:0}
   with patch.object(P,'run_pool',side_effect=pool):P.parallel_evaluate('evaluate','4b','latentmas',cfg,C.ROOT/'configs/default.json')
   self.assertEqual(calls,[(2,[0,1]),(1,[1])]);self.assertEqual(C.read(directory/'status.json')['status'],'completed')
   with patch.object(P,'run_pool',side_effect=AssertionError('completed tasks must not rerun')):P.parallel_evaluate('evaluate','4b','latentmas',cfg,C.ROOT/'configs/default.json')
 def test_other_error_not_retried_or_scored(self):
  cfg=C.config();cfg['datasets']={'gsm8k':20}
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)),patch.object(C,'install_signals'),patch('suite.data.evaluation_items',return_value=[{'example_id':'a'}]),patch.object(P,'run_pool',return_value={0:1,1:0}) as pool:
   with self.assertRaises(RuntimeError):P.parallel_evaluate('evaluate','4b','latentmas',cfg,C.ROOT/'configs/default.json')
   self.assertEqual(pool.call_count,1);root=Path(d)/'runs/4b/evaluation_sample10/latentmas';self.assertEqual(C.read(root/'status.json')['status'],'blocked');self.assertEqual(C.rows(root/'results.jsonl'),[])
 def test_legacy_export_training_and_output_isolation(self):
  cfg=C.config();prior=C.read(C.ROOT/'evidence/previous_full_release.json');self.assertIn(prior['identity'],C.compatible_export_identities(cfg));cfg['sender_latent_steps']+=1;self.assertNotIn(prior['identity'],C.compatible_export_identities(cfg))
  self.assertEqual(output_kind(cfg,'evaluation'),'evaluation_sample10');self.assertEqual(output_kind(cfg,'semantic'),'semantic_parallel')
 def test_empty_reports_use_new_denominators(self):
  from suite import report as R,report_semantic as S
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)),patch.object(R,'STORE',Path(d)):
   self.assertFalse(R.report(C.config(),False));general=R.general_statistics(C.config());sem=S.semantic_statistics(C.config());self.assertTrue((Path(d)/'STATS_TXT/00_README.txt').exists());self.assertFalse((Path(d)/'reports/summary.json').exists())
   self.assertEqual(sum(r['expected'] for r in general['cells']),5910);self.assertEqual(sum(r['expected'] for r in sem['cells']),6480);self.assertEqual(len(general['cells'])+len(sem['cells']),80)

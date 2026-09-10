"""Statistics-only export and zero-execution continuation fixtures; no GPU work."""
import hashlib,json,os,shutil,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from suite import common as C,report as R,report_semantic as S
from suite.stat_records import compact_general,compact_semantic,prepare_record
from suite.txt_reports import byte_chunks,write_parts,MAX_BYTES
from suite.schedule import run_remaining
from suite.resume import evaluation_complete

class StatsResume(unittest.TestCase):
 def test_utf8_byte_splitting_reconstructs_chinese_and_long_lines(self):
  source=('统计数据🙂'*20000)+'\n'+('step = 10\n'*20000)
  chunks=byte_chunks(source);self.assertEqual(''.join(chunks),source);self.assertTrue(all(len(x.encode())<=86000 for x in chunks))
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d:
   root=Path(d);entries=write_parts(root,[('06_training','Training statistics',source)])
   self.assertGreater(len(entries),2)
   for name,size,digest,_ in entries:
    raw=(root/name).read_bytes();self.assertEqual(size,len(raw));self.assertEqual(digest,hashlib.sha256(raw).hexdigest());self.assertLess(size,90000)
   self.assertTrue(all(p.suffix=='.txt' and p.stat().st_size<=MAX_BYTES for p in root.iterdir()))
 def test_stale_generated_parts_removed_and_index_complete(self):
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d:
   root=Path(d);write_parts(root,[('05_laten_4b_latentmas','Metrics','x'*190000)])
   entries=write_parts(root,[('05_laten_4b_latentmas','Metrics','new stats\n')]);self.assertEqual(len(entries),1)
   self.assertFalse((root/'05_laten_4b_latentmas.p002.txt').exists());index=(root/'01_INDEX.txt').read_text();self.assertIn(entries[0][0],index)
 def test_score_journals_drop_text_code_prompts_and_token_ids(self):
  row=dict(example_id='fixture/1',dataset='mbppplus',correct=False,output={'text':'SECRET_OUTPUT','token_ids':[98],'output_tokens':1,'stop_reason':'eos'},receiver_prompt='SECRET_PROMPT',traces=['SECRET_TRACE'],score={'correct':False,'metric':'pass@1','prediction':'SECRET_CODE','error':'SECRET_TRACEBACK','test_status':'failed'},efficiency={'generated_text_tokens':1})
  compact=compact_general(row);self.assertNotIn('SECRET',json.dumps(compact));self.assertEqual(compact['efficiency']['generated_text_tokens'],1)
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d:
   root=Path(d);C.append(root/'generations.jsonl',row);self.assertFalse((root/'generations.jsonl').exists())
   C.append(root/'results.jsonl',row);self.assertNotIn('SECRET',(root/'results.jsonl').read_text())
 def test_semantic_compaction_preserves_all_pair_and_task_metrics(self):
  from suite.semantic import load_frozen,vector_score,vector_from_variant,summarize
  variants,_,programs,_=load_frozen();rows=[vector_score(v,programs[v.policy_program_id],dict(vector=vector_from_variant(v),reasoning_stop_reason='thinking_close',reasoning_text='SECRET_THINK',reasoning_token_ids=[9])) for v in variants]
  compact=[compact_semantic(r) for r in rows];before=summarize(rows,variants);after=summarize(compact,variants)
  for key in ('metrics','by','pairs','pairs_by','rates','pair_rates','fact_error_given_task_success'):self.assertEqual(before[key],after[key])
  self.assertNotIn('SECRET_THINK',json.dumps(compact));self.assertEqual(after['pairs']['pair_count'],486)
 def test_old_raw_results_export_as_txt_only_without_reasoning(self):
  cfg=C.config()
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)),patch.object(R,'STORE',Path(d)):
   p=Path(d)/'runs/4b/evaluation_sample10/latentmas/results.jsonl';p.parent.mkdir(parents=True)
   keys=('generated_text_tokens','generated_latent_positions','prompt_tokens','model_prefill_positions','transmitted_positions','transmitted_bytes','receiver_message_positions','internal_cache_relay_positions','internal_cache_relay_bytes','compressor_prefill_positions')
   row=dict(example_id='fixture/1',dataset='gsm8k',correct=True,output={'text':'SECRET_REASONING'},efficiency={**dict.fromkeys(keys,7),'task_wall_seconds':2})
   original=json.dumps(row)+'\n';p.write_text(original);self.assertFalse(R.report(cfg,False));self.assertEqual(p.read_text(),original)
   files=list((Path(d)/'STATS_TXT').iterdir());self.assertGreater(len(files),10)
   for f in files:self.assertEqual(f.suffix,'.txt');self.assertLess(f.stat().st_size,90000);self.assertNotIn('SECRET_REASONING',f.read_text())
   self.assertFalse((Path(d)/'reports').exists())
 def test_workload_identity_keeps_collect_training_and_learned_eval(self):
  prior=C.read(C.ROOT/'evidence/previous_collect_release.json');cfg=C.config()
  self.assertEqual(C.identity(cfg),prior['identity']);self.assertEqual(C.evaluation_identity(cfg,'interlat'),prior['identity'])
  cfg['training']['latcom_stage1_steps']+=1;self.assertNotEqual(C.identity(cfg),prior['identity'])
 def test_ready_downloads_do_not_call_network_or_reference_tests(self):
  from suite.cli import downloads
  download=Mock(side_effect=AssertionError('Must not download verified assets'))
  with patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(snapshot_download=download)}),patch('suite.assets.assets_ready',return_value=True),patch('suite.data.build_data',side_effect=AssertionError('No rebuild')):
   downloads(C.config(),['4b'])
  download.assert_not_called()
 def test_scheduler_does_not_execute_completed_tests(self):
  calls=[]
  def skip(a,f,m,s):return a in ('_smoke_one','evaluate','semantic') and m=='latentmas'
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)):
   result=run_remaining(['4b'],['latentmas'],True,lambda a,*x:calls.append((a,x)) or 0,lambda:None,skip=skip)
   self.assertEqual(calls,[]);self.assertEqual(result,[])
 def test_completed_evaluation_reuse_requires_exact_ids_and_identities(self):
  cfg=C.config();cfg['datasets']={'gsm8k':20};items=[{'example_id':'a'},{'example_id':'b'}];ids=['a','b']
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d,patch.object(C,'STORE',Path(d)),patch('suite.data.evaluation_items',return_value=items):
   root=Path(d)/'runs/4b/evaluation_sample10/latentmas';data='fixture';C.write(Path(d)/'data/manifest.json',{'identity':data})
   run=C.evaluation_identity(cfg,'latentmas');key=C.canon({'run':run,'family':'4b','method':'latentmas','action':'evaluate','ids':ids,'shards':2});C.seal(root,key,{})
   for i,example in enumerate(ids):C.append(root/'results.jsonl',{'example_id':example,'identity':C.canon({'run':run,'family':'4b','method':'latentmas','data':data,'smoke':False,'shard':(i,2)})})
   self.assertTrue(evaluation_complete(cfg,'4b','latentmas','evaluation'))
   C.append(root/'results.jsonl',{'example_id':'a','identity':'wrong'})
   with self.assertRaises(ValueError):evaluation_complete(cfg,'4b','latentmas','evaluation')
 def test_resume_shell_reuses_environment_without_pip(self):
  with tempfile.TemporaryDirectory(dir=C.ROOT/'tests') as d:
   root=Path(d);shutil.copy(C.ROOT/'run.sh',root/'run.sh');fake=root/'.venv/bin/python';fake.parent.mkdir(parents=True)
   fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_CALL_LOG"\nif [[ "$*" == *pip* ]]; then exit 99; fi\nexit 0\n');fake.chmod(0o755);log=root/'calls.txt'
   result=subprocess.run(['bash','run.sh','resume','--family','8b'],cwd=root,env={**os.environ,'PYTHON_BIN':str(fake),'LATEN_STORE':str(root/'storage'),'FAKE_CALL_LOG':str(log)},capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr);calls=log.read_text();self.assertIn('_env_check',calls);self.assertIn('suite.cli resume --family 8b',calls);self.assertNotIn('pip',calls)

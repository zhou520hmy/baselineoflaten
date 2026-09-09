"""Full-data protocol fixtures; injected gold is never a model score."""
import ast,copy,json,unittest
from collections import Counter
from pathlib import Path
from suite import common as C
from suite.semantic import load_frozen,vector_score,summarize,vector_from_variant,build_action_prompt,build_deliberative_joint_vector_prompt
from vendor.laten.schema import record_to_dict,canonical_json
from vendor.laten.prompts import build_prompt_snapshot
from vendor.laten.summary import summarize as source_summary

class SemanticContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.variants,cls.cases,cls.programs,cls.manifest=load_frozen()
 def decoded(self,vector):return dict(vector=vector,reasoning_stop_reason='thinking_close',reasoning_cap_hit=False,reasoning_token_count=0)
 def fixture(self,v):return vector_score(v,self.programs[v.policy_program_id],self.decoded(vector_from_variant(v)))
 def test_648_486_and_exact_dataclass_roundtrip(self):
  self.assertEqual(len(self.variants),648);self.assertEqual(sum(v.target_fact_id is not None for v in self.variants),486)
  for split in ('dev','test'):
   original=C.rows(C.ROOT/'vendor/laten/data'/split/'variants.jsonl');self.assertEqual(canonical_json([record_to_dict(v) for v in self.variants if v.split==split]),canonical_json(original))
  self.assertEqual(Counter(v.graph_level for v in self.variants),Counter(G4=216,G5=216,G6=216))
 def test_all_gold_policy_vectors_and_original_metric_parity(self):
  rows=[self.fixture(v) for v in self.variants];summary=summarize(rows,self.variants);original=source_summary(rows,self.variants)
  for key in ('metrics','by','pairs','pairs_by'):self.assertEqual(summary[key],original[key])
  self.assertEqual(summary['metrics']['vector_exact'],648);self.assertEqual(summary['metrics']['functional_action_exact'],648);self.assertEqual(summary['pairs']['target_pair_exact'],486);self.assertEqual(summary['metrics']['bit_total'],3888)
 def test_failed_cf_and_partial_pair_denominators(self):
  base=self.variants[0];group=[v for v in self.variants if v.case_id==base.case_id];rows=[self.fixture(v) for v in group]
  failed=vector_score(group[1],self.programs[group[1].policy_program_id],{'vector':None,'reasoning_stop_reason':'max_new_tokens','reasoning_cap_hit':True,'reasoning_token_count':2048});rows[1]=failed
  report=summarize(rows,self.variants);self.assertEqual(report['pairs']['pair_count'],3);self.assertEqual(report['pairs']['target_pair_exact'],2);self.assertEqual(report['metrics']['reasoning_cap_count'],1);self.assertFalse(report['complete']);self.assertEqual(summarize(rows[:1],self.variants)['pairs']['pair_count'],0)
 def test_original_648_prompt_snapshots(self):
  snapshots={r['variant_id']:r for split in ('dev','test') for r in C.rows(C.ROOT/'vendor/laten/data'/split/'prompt_snapshots.jsonl')}
  for v in self.variants:
   actual=record_to_dict(build_prompt_snapshot(self.cases[v.case_id],v,self.programs[v.policy_program_id]));self.assertEqual(actual,snapshots[v.variant_id])
 def test_receiver_prompt_invariant_under_cf(self):
  for v in self.variants:
   c=self.cases[v.case_id];p=self.programs[v.policy_program_id];text=json.dumps(build_deliberative_joint_vector_prompt(c,p))
   self.assertNotIn(v.variant_id,text);self.assertNotIn('Gold',text)
   self.assertNotIn('Recovered information vector',text)
 def test_wrong_fact_can_preserve_action_conditional_metric(self):
  found=False
  for v in self.variants:
   gold=vector_from_variant(v)
   for i in range(len(gold)):
    wrong=gold[:i]+str(1-int(gold[i]))+gold[i+1:];row=vector_score(v,self.programs[v.policy_program_id],self.decoded(wrong))
    if row['functional_action_exact']:
     report=summarize([row],self.variants);self.assertEqual(report['fact_error_given_task_success']['rate'],1);found=True;break
   if found:break
  self.assertTrue(found)
 def test_legacy_export_identity_is_config_guarded(self):
  cfg=C.config();prior=C.read(C.ROOT/'evidence/previous_release.json');self.assertIn(prior['identity'],C.compatible_export_identities(cfg));cfg['training']['latcom_stage1_steps']+=1;self.assertNotIn(prior['identity'],C.compatible_export_identities(cfg))
 def test_matrix_is_80_cells_65550_attempts(self):
  cfg=C.config();self.assertEqual(2*5*(len(cfg['datasets'])+1),80);self.assertEqual(2*5*(sum(cfg['datasets'].values())+cfg['semantic_benchmark']['variants']),65550)
if __name__=='__main__':unittest.main()

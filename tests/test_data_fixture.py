"""Schema/denominator/disjointness test using tiny local fixtures, no downloads."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from suite import data
from suite.common import ROOT,config

class DataFixture(unittest.TestCase):
 def test_all_seven_schemas_and_training_split(self):
  cfg=copy.deepcopy(config());cfg['datasets']={k:1 for k in cfg['datasets']};cfg['datasets']['medqa']=300
  cfg['training'].update(candidate_limit=2,aux_math=1,aux_code=1)
  def parquet(repo,name):
   if repo=='openai/gsm8k':return [{'question':'train question' if 'train' in name else 'test question','answer':'Reasoning. #### 3'}]
   if repo=='allenai/ai2_arc':return [{'id':'a','question':'Choose','choices':{'label':['1','2'],'text':['yes','no']},'answerKey':'2'}]
   if repo=='fingertap/GPQA-Diamond':return [{'question':'Physics options A X B Y','answer':'A'}]
   if repo=='evalplus/mbppplus':return [{'task_id':11,'prompt':'code task','test_list':['assert f()==1'],'test':'assert f()==1','test_imports':[],'code':'def f(): return 1'}]
   if repo=='evalplus/humanevalplus':return [{'task_id':'HumanEval/1','prompt':'def f():\n','test':'def check(candidate): assert candidate()==1','entry_point':'f','canonical_solution':'    return 1'}]
   if repo=='hotpotqa/hotpot_qa':return [{'id':'h','question':'Who is H?','answer':'H','supporting_facts':{'title':['H']},'context':{'title':['H','I'],'sentences':[['gold'],['noise']]}}]
   if repo=='google-research-datasets/mbpp':return [{'task_id':11,'text':'excluded by ID','code':'pass'},{'task_id':12,'text':'different train code','code':'pass'}]
   raise AssertionError((repo,name))
  with tempfile.TemporaryDirectory(dir=ROOT/'tests') as t:
   store=Path(t);p=store/'musique.jsonl';p.write_text(json.dumps({'id':'m','question':'Who is M?','answer':'M','paragraphs':[{'title':'M','paragraph_text':'gold','is_supporting':True}]})+'\n')
   with patch.object(data,'STORE',store),patch.object(data,'parquet',parquet),patch.object(data,'download',return_value=p):
    data.build_data(cfg);manifest=json.loads((store/'data/manifest.json').read_text());self.assertEqual(manifest['total'],306);self.assertEqual(manifest['candidate_count'],4)
    self.assertEqual(data.evaluation_items(cfg,'arc_easy')[0]['gold'],'B')
    self.assertIn('check(f)',data.evaluation_items(cfg,'humanevalplus')[0]['test_code'])
    records=data.rows(store/'data/training_candidates.jsonl');self.assertEqual([r['id'] for r in records if r['source']=='mbpp_train'],['mbpp_train/12'])
    data.build_data(cfg)  # unchanged resume checks the saved hashes
    (store/'data/gsm8k.jsonl').write_text('{}\n')
    with self.assertRaises(ValueError):data.build_data(cfg)
if __name__=='__main__':unittest.main()

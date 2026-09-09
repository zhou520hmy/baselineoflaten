import ast,json,tempfile,unittest
from pathlib import Path
from suite.common import ROOT,rows
from suite.scoring import numeric,parse_answer,simple_score,extract_code
class PortableContract(unittest.TestCase):
 def test_sources_parse(self):
  for folder in ('suite','tests','sandbox'):
   for path in (ROOT/folder).glob('*.py'):ast.parse(path.read_text(),filename=str(path))
 def test_matrix_and_pins(self):
  c=json.loads((ROOT/'configs/default.json').read_text());self.assertEqual(len(c['families'])*len(c['methods'])*len(c['datasets']),70);self.assertEqual(sum(c['datasets'].values()),5907)
  for source in json.loads((ROOT/'evidence/hf_sources.json').read_text()).values():self.assertEqual(len(source['sha']),40)
 def test_numeric_choice(self):
  self.assertEqual(numeric('1,000'),numeric('1000.0'));self.assertEqual(numeric(r'\frac{1}{2}'),numeric('0.5'));self.assertEqual(parse_answer(r'Final \boxed{1,000}','gsm8k'),'1,000');self.assertEqual(parse_answer(r'42 items. \boxed{C}','arc_easy'),'C');self.assertIsNone(parse_answer('3 molecules.','medqa'));self.assertFalse(simple_score({'dataset':'gsm8k','gold':'0'},'No answer')['correct'])
 def test_code(self):
  self.assertEqual(extract_code('analysis</think>\n```python\ndef f(): return 1\n```'),'def f(): return 1');self.assertIsNone(extract_code('Reasoning without code'))
 def test_journal_recovery(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tests') as d:
   p=Path(d)/'rows.jsonl';p.write_bytes(b'{"id":1}\n{"id":')
   with self.assertRaises(ValueError):rows(p)
   self.assertEqual(rows(p,repair=True),[{'id':1}]);self.assertEqual(p.read_bytes(),b'{"id":1}\n')
 def test_portable(self):
  for p in (ROOT/'suite').glob('*.py'):self.assertNotIn('/home/',p.read_text())
 def test_gold_not_in_engine(self):
  source=(ROOT/'suite/engine.py').read_text();self.assertNotIn("item['gold']",source);self.assertNotIn("item['test_code']",source)
if __name__=='__main__':unittest.main()

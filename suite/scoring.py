"""Numeric/MC parsing and supplied expanded code tests; Docker exit codes are typed."""
import json,re,subprocess,tempfile,time,uuid
from fractions import Fraction
from .common import STORE,read

def boxes(text):
 result=[]
 for m in re.finditer(r'\\boxed\s*\{',text):
  depth=1;start=m.end();i=start
  while i<len(text) and depth:
   if text[i]=='{':depth+=1
   elif text[i]=='}':depth-=1
   i+=1
  if depth==0:result.append(text[start:i-1].strip())
 return result

def numeric(text):
 text=text.strip().replace(',','').replace('$','')
 text=re.sub(r'\\(?:dfrac|tfrac|frac)\{([+-]?[\d.]+)\}\{([+-]?[\d.]+)\}',r'\1/\2',text)
 if re.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?',text):
  try:
   parts=text.split('/');return Fraction(parts[0])/(Fraction(parts[1]) if len(parts)>1 else 1)
  except (ValueError,ZeroDivisionError):return None
 return None

def parse_answer(text,task):
 answer_text=text.split('</think>')[-1];b=boxes(answer_text)
 if task=='gsm8k':
  if b and numeric(b[-1]) is not None:return b[-1]
  numbers=re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?',answer_text)
  return numbers[-1] if numbers else None
 if b:
  x=re.sub(r'\\(?:text|mathrm)\{([^{}]+)\}',r'\1',b[-1]).strip().upper()
  return x if x in 'ABCDE' and len(x)==1 else None
 matches=re.findall(r'(?:final\s+answer|answer|option)\s*(?:is|:|=)?\s*\(?([A-E])\)?\b',answer_text,re.I)
 if matches:return matches[-1].upper()
 matches=re.findall(r'^\s*\(?([A-E])\)?[.\s]*$',answer_text,re.M)
 return matches[-1].upper() if matches else None

def extract_code(text):
 blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text.split('</think>')[-1],re.S|re.I)
 if blocks:return blocks[-1].strip()
 # Explicit extraction failure, rather than executing reasoning prose.
 return None

def simple_score(item,text):
 pred=parse_answer(text,item['dataset']);gold=item['gold']
 ok=pred is not None and (numeric(pred)==numeric(gold) and numeric(gold) is not None if item['dataset']=='gsm8k' else pred==gold.upper())
 return {'correct':bool(ok),'prediction':pred,'metric':'numeric_accuracy' if item['dataset']=='gsm8k' else 'multiple_choice_accuracy'}

def code_score(item,text,cfg):
 code=extract_code(text)
 if code is None:return {'correct':False,'prediction':None,'metric':'expanded_test_pass@1','test_status':'no_code_block'}
 image=read(STORE/'sandbox/image.json')['image_id'];name='laten-baseline-'+uuid.uuid4().hex[:18];timeout=cfg['code_tests']['timeout_seconds']
 command=['docker','run','--rm','-i','--name',name,'--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534','--pids-limit','64','--memory',cfg['code_tests']['memory'],'--cpus',str(cfg['code_tests']['cpus']),'--tmpfs','/tmp:rw,nosuid,nodev,size=128m',image]
 scratch=STORE/'tmp';scratch.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 try:
  with tempfile.TemporaryFile(dir=scratch) as err:
   p=subprocess.run(command,input=json.dumps({'program':code+'\n\n'+item['test_code'],'timeout':timeout}).encode(),stdout=subprocess.DEVNULL,stderr=err,timeout=timeout+30)
   err.seek(0);error=err.read(4096).decode(errors='replace')
  if p.returncode in (125,126,127,137):raise RuntimeError('Code sandbox infrastructure failure: '+error)
  return {'correct':p.returncode==0,'prediction':code,'metric':'expanded_test_pass@1','test_status':'passed' if p.returncode==0 else ('timeout' if p.returncode==124 else 'failed'),'error':error,'seconds':time.perf_counter()-start,'image_id':image}
 except subprocess.TimeoutExpired:
  subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
  raise RuntimeError('Docker failed to complete outside the inner code timeout')

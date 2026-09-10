"""Train-only QA parsing and explicit, separate eligibility for each learned method."""
import re
from .scoring import boxes,numeric

CONTROL=re.compile(r'<\|(?:im_start|im_end|endoftext|eot_id|end_of_text)\|>')

def clean_controls(text):return CONTROL.sub('',str(text)).strip()

def normalized(text):
 text=clean_controls(text).lower()
 text=re.sub(r'[^\w\s]',' ',text)
 return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())

def final_answer(text):
 text=clean_controls(text)
 if '<think>' in text and '</think>' not in text:return ''
 text=text.split('</think>')[-1].strip()
 b=boxes(text)
 if b:return clean_controls(b[-1])
 parts=re.split(r'(?i)\b(?:final\s+)?answer\s*:',text)
 if len(parts)>1:return (parts[-1].strip().splitlines() or [''])[0].strip()
 # Single short span is valid; no substring search through multi-line reasoning.
 return text if len(text.splitlines())==1 else ''

def answer_matches(text,gold):
 predicted=final_answer(text)
 if not predicted:return False
 a,b=numeric(predicted),numeric(clean_controls(gold))
 return a==b if a is not None and b is not None else bool(normalized(predicted)) and normalized(predicted)==normalized(gold)

def probe(output,gold):
 return {'raw_text':output['text'],'parsed_answer':final_answer(output['text']),'gold_answer':str(gold),'stop_reason':output['stop_reason'],'output_tokens':output['output_tokens'],'correct':output['stop_reason']=='eos' and answer_matches(output['text'],gold)}

def eligible_methods(direct,single,plan,aux=False):
 if aux:return ['latcom','interlat']
 return (['latcom'] if not direct['correct'] and single and single['correct'] else [])+(['interlat'] if plan['correct'] else [])

def pool_counts(records):
 return {m:sum(r['status']=='retained' and not r['aux'] and m in r.get('eligible_for',[]) for r in records) for m in ('latcom','interlat')}

def stage_records(records,stage,minimum):
 method='latcom' if stage.startswith('latcom') else 'interlat'
 selected=[r for r in records if r['status']=='retained' and method in r.get('eligible_for',[]) and (stage!='latcom_stage1' or not r['aux'])]
 count=sum(not r['aux'] for r in selected)
 if count<minimum:raise RuntimeError(f'{method}: only {count} method-eligible main examples; need {minimum}. Inspect training_cache_v2 diagnostics; auxiliary examples do not satisfy this gate.')
 return selected

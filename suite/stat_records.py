"""Minimal local resume statistics. No evaluation text, code, prompts or token IDs."""
from pathlib import Path

def numeric_tree(value):
 if value is None or isinstance(value,(int,float,bool)):return value
 if isinstance(value,dict):return {k:numeric_tree(v) for k,v in value.items() if v is None or isinstance(v,(int,float,bool,dict))}
 return None

def compact_general(row):
 keys=('example_id','dataset','method','family','identity','time','max_new_tokens','engineering_smoke','correct','scoring_seconds','execution','generation_stop_reason','output_token_count')
 out={k:row[k] for k in keys if k in row};out['efficiency']=numeric_tree(row.get('efficiency',{}))
 if 'score' in row:out['score']={k:row['score'][k] for k in ('correct','metric','test_status','seconds') if k in row['score']}
 if 'output' in row:out.update(generation_stop_reason=row['output'].get('stop_reason'),output_token_count=row['output'].get('output_tokens'))
 out['storage_policy']='statistics_only_v1'
 return out

def compact_semantic(row):
 keys=('variant_id','case_id','semantic_family_id','source_family_id','split','domain','information_level','reasoning_level','graph_level','target_fact_id','status','information_bit_total','information_bit_correct_count','information_vector_exact','functional_action_exact','model_action_exact','reasoning_cap_hit','reasoning_stop_reason','reasoning_token_count','per_agent_block_exact','predicted_information_vector','run_identity','run_id','arm_id','family','engineering_smoke','time','execution','model_action_diagnostic_prompt_tokens','evidence_label')
 out={k:row[k] for k in keys if k in row};out['efficiency']=numeric_tree(row.get('efficiency',{}));out['storage_policy']='statistics_only_v1'
 # Bit vectors are local sufficient statistics for exact BASE/CF aggregation, never exported as per-case answers.
 return out

def compact_collection(row):
 out=dict(row)
 out['probes']={name:({k:v for k,v in value.items() if k in ('correct','stop_reason','output_tokens','not_run')} if isinstance(value,dict) else value) for name,value in row.get('probes',{}).items()}
 return out

def prepare_record(path,row):
 path=Path(path)
 if path.name=='records.jsonl' and path.parent.name=='training_cache_v2':return compact_collection(row)
 if path.name not in ('results.jsonl','generations.jsonl'):return row
 if 'variant_id' in row:return compact_semantic(row)
 if 'example_id' in row and 'dataset' in row:
  # Raw text stays in memory until scored. An unscored interrupted generation may need recomputation.
  if path.name=='generations.jsonl':return None
  return compact_general(row)
 return row

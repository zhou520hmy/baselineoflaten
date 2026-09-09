"""Frozen project dataset and source-identical scoring, without torch dependencies."""
from pathlib import Path
from collections import Counter
from . import common as C
from vendor.laten import schema as S
from vendor.laten.prompts import build_role_prompt,canonical_fact_assertion,vector_from_variant,assignment_from_vector,build_action_prompt
from vendor.laten.deliberative_prompts import build_deliberative_joint_vector_prompt
from vendor.laten.public_context import build_role_public_context
from vendor.laten.policy import evaluate_policy,program_sha256
from vendor.laten.summary import summarize as original_summary
from vendor.laten.efficiency import summarize_efficiency

DATA=C.ROOT/'vendor/laten/data'
def _gold(x):return S.GoldRecord(**{**x,'active_action_names':tuple(x['active_action_names']),'trace':tuple(S.TraceStep(**t) for t in x['trace'])})
def _variant(x):return S.CaseVariant(**{**x,'facts':tuple(S.FactValue(**f) for f in x['facts']),'gold':_gold(x['gold'])})
def _case(x):return S.BenchmarkCase(**{**x,'roles':tuple(S.RoleSpec(**{**r,'owned_fact_ids':tuple(r['owned_fact_ids'])}) for r in x['roles']),'facts':tuple(S.FactValue(**f) for f in x['facts']),'base_gold':_gold(x['base_gold']),'counterfactuals':tuple(S.CounterfactualRecord(**r) for r in x['counterfactuals'])})
def load_frozen():
 manifest=C.read(DATA/'manifest.json')
 for name,meta in manifest['files'].items():
  if C.sha(DATA/name)!=meta['sha256']:raise ValueError('Frozen LATEN data hash changed: '+name)
 cases={c.case_id:c for split in ('dev','test') for c in map(_case,C.rows(DATA/split/'cases.jsonl'))}
 programs={p.policy_program_id:p for p in map(S.policy_program_from_dict,C.rows(DATA/'policy_programs/programs.jsonl'))}
 variants=[v for split in ('dev','test') for v in map(_variant,C.rows(DATA/split/'variants.jsonl'))]
 if len(cases)!=162 or len(variants)!=648 or len({v.variant_id for v in variants})!=648:raise ValueError('LATEN fixed matrix changed')
 if Counter(v.split for v in variants)!=Counter(dev=108,test=540):raise ValueError('LATEN split denominator changed')
 bycase={}
 for v in variants:
  c=cases[v.case_id];p=programs[v.policy_program_id];gold=vector_from_variant(v)
  if program_sha256(p)!=p.program_sha256 or p.program_sha256!=c.policy_program_sha256:raise ValueError('Policy source identity differs')
  if evaluate_policy(p,assignment_from_vector(p,gold)).action_vector!=v.gold.action_vector:raise ValueError('Frozen gold and deterministic policy differ')
  if len(c.roles)!=int(c.graph_level[1:]) or c.roles[-1].role_type!='judger':raise ValueError('Invalid role graph')
  if sorted(r.position for r in c.roles)!=list(range(1,len(c.roles)+1)):raise ValueError('Role chronology differs')
  receiver=build_deliberative_joint_vector_prompt(c,p)
  for role in c.roles[:-1]:
   own='\n'.join(m['content'] for m in build_role_prompt(c,v,role_id=role.role_id,channel='latent'))
   public='\n'.join(m['content'] for m in build_role_public_context(c,p,role))
   if c.public_policy in own or c.public_policy in public:raise ValueError('Policy leaked to producer')
   for fact in v.facts:
    assertion=canonical_fact_assertion(fact)
    if (assertion in own)!=(fact.fact_id in role.owned_fact_ids):raise ValueError('Private fact ownership changed')
    if assertion in public or any(assertion in m['content'] for m in receiver):raise ValueError('Gold leaked into public query')
  bycase.setdefault(v.case_id,[]).append(v)
 for group in bycase.values():
  base=[v for v in group if v.target_fact_id is None];cf=[v for v in group if v.target_fact_id is not None]
  if len(base)!=1 or len(cf)!=3:raise ValueError('Incomplete BASE/CF group')
  for v in cf:
   changed=[j for j,(a,b) in enumerate(zip(vector_from_variant(base[0]),vector_from_variant(v))) if a!=b]
   if len(changed)!=1 or v.facts[changed[0]].fact_id!=v.target_fact_id:raise ValueError('CF does not change exactly its named target')
 return variants,cases,programs,manifest

def vector_score(v,program,decoded):
 gold=vector_from_variant(v);pred=decoded.get('vector');ok=decoded['reasoning_stop_reason'] in ('thinking_close','eos_mapped_to_thinking_close')
 if ok and (not isinstance(pred,str) or len(pred)!=len(gold) or set(pred)-set('01')):raise ValueError('Invalid supposedly completed binary vector')
 row={k:getattr(v,k) for k in ('variant_id','case_id','semantic_family_id','source_family_id','split','domain','information_level','reasoning_level','graph_level','target_fact_id')}
 row.update(status='success' if ok else 'failed',gold_information_vector=gold,information_bit_total=len(gold),information_bit_correct_count=0,information_vector_exact=False,functional_action_exact=False,model_action_exact=False,expected_action_vector=v.gold.action_vector,expected_action_label=v.gold.candidate_label,evidence_label='previously_exposed_test_diagnostic' if v.split=='test' else 'development_diagnostic',**{k:value for k,value in decoded.items() if k!='vector'})
 if ok:
  recovered=assignment_from_vector(program,pred);functional=evaluate_policy(program,recovered);correct=[a==b for a,b in zip(pred,gold)]
  row.update(predicted_information_vector=pred,recovered_assignment=recovered,functional_action_vector=functional.action_vector,functional_action_exact=functional.action_vector==v.gold.action_vector,information_bit_correct_count=sum(correct),information_vector_exact=all(correct),per_agent_block_exact={f'agent_{owner}':all(correct[j] for j,f in enumerate(program.facts) if f.owner_agent_index==owner) for owner in (1,2,3)})
 return row

def summarize(rows,variants):
 allowed={v.variant_id for v in variants};ids=[r['variant_id'] for r in rows]
 if len(set(ids))!=len(ids) or not set(ids)<=allowed:raise ValueError('Invalid semantic result coverage')
 result=original_summary(rows,variants);m=result['metrics'];n=len(rows);pair_total=sum(v.target_fact_id is not None for v in variants)
 result.update(expected_pairs=pair_total,pending_pairs=pair_total-result['pairs']['pair_count'],complete=n==len(variants) and result['pairs']['pair_count']==pair_total)
 result['rates']={k:(m[k]/n if n else None) for k in ('vector_exact','functional_action_exact','model_action_exact','success_count','reasoning_cap_count')};result['rates']['bit_accuracy']=m['bit_correct']/m['bit_total'] if m['bit_total'] else None
 pairs=result['pairs'];result['pair_rates']={k:pairs[k]/pairs['pair_count'] if pairs['pair_count'] else None for k in ('target_pair_exact','non_target_pair_exact','both_vectors_exact')}
 successful_actions=[r for r in rows if r['status']=='success' and r.get('functional_action_exact')];errors=sum(not r['information_vector_exact'] for r in successful_actions)
 result['fact_error_given_task_success']={'fact_error_count':errors,'task_success_count':len(successful_actions),'rate':errors/len(successful_actions) if successful_actions else None}
 result['efficiency']=summarize_efficiency(rows,success_field='information_vector_exact',expected_case_count=len(variants))
 result['efficiency_by']={field:{value:summarize_efficiency([r for r in rows if r[field]==value],expected_case_count=sum(getattr(v,field)==value for v in variants)) for value in sorted({getattr(v,field) for v in variants})} for field in ('split','graph_level','information_level')}
 return result

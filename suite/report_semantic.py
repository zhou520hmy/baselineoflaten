"""Project metrics stay separate from the seven general-task accuracy macro."""
import csv,io
from . import common as C
from .semantic import load_frozen,summarize
from .sampling import output_kind

def report_semantic(cfg,print_status=True):
 if not cfg.get('semantic_benchmark',{}).get('enabled'):return True
 variants,_,_,_=load_frozen();cells=[];details={};text=['# LATEN Benchmark — frozen 648-variant comparison','',f'Updated (Beijing): {C.now()}','', 'Test split: previously_exposed_test_diagnostic. No new training on this Benchmark.','', '| Model | Method | Done | Vector | Bits | Functional action | Model action | Target CF | Non-target CF | Both CF vectors | Mean task s |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for family in cfg['families']:
  for method in cfg['methods']:
   p=C.STORE/'runs'/family/output_kind(cfg,'semantic')/method/'results.jsonl'
   if p.exists():
    data=p.read_bytes();end=data.rfind(b'\n')+1;rows=[__import__('json').loads(x) for x in data[:end].splitlines() if x]
   else:rows=[]
   report=summarize(rows,variants);details[family+'/'+method]=report;m=report['metrics'];pairs=report['pairs'];eff=report['efficiency'];n=len(rows)
   cell=dict(family=family,method=method,latency_comparable_as_isolated_run=False,completed=n,expected=648,status='completed' if report['complete'] else 'incomplete',vector_exact=m['vector_exact'],vector_accuracy=report['rates']['vector_exact'],bit_correct=m['bit_correct'],bit_total=m['bit_total'],bit_accuracy=report['rates']['bit_accuracy'],functional_action_exact=m['functional_action_exact'],functional_action_accuracy=report['rates']['functional_action_exact'],model_action_exact=m['model_action_exact'],model_action_accuracy=report['rates']['model_action_exact'],success_count=m['success_count'],reasoning_cap_count=m['reasoning_cap_count'],expected_pairs=486,**pairs,**{k+'_rate':v for k,v in report['pair_rates'].items()},conditional_fact_error_rate=report['fact_error_given_task_success']['rate'],accumulated_task_seconds_not_stage_wall=eff['task_wall_seconds']['total'],mean_task_seconds=eff['task_wall_seconds']['mean'],p50_task_seconds=eff['task_wall_seconds']['p50'],p95_task_seconds=eff['task_wall_seconds']['p95'])
   for key,value in eff['counters'].items():cell['total_'+key]=value['total'];cell['mean_'+key]=value['mean_per_case']
   cells.append(cell)
   def rate(k,d):return f'{k}/{d}' if d else 'Pending'
   mean=f'{cell["mean_task_seconds"]:.3f}' if n else '—'
   text.append('| '+' | '.join([family,method,f'{n}/648',rate(m['vector_exact'],n),rate(m['bit_correct'],m['bit_total']),rate(m['functional_action_exact'],n),rate(m['model_action_exact'],n),rate(pairs['target_pair_exact'],pairs['pair_count']),rate(pairs['non_target_pair_exact'],pairs['pair_count']),rate(pairs['both_vectors_exact'],pairs['pair_count']),mean])+' |')
 text+=['','Task seconds are contended wall-time diagnostics. Summed overlapping task durations are not stage elapsed time; throughput is not an isolated-method comparison. See scheduler_wall.jsonl for coordinator wall time.']
 out=C.STORE/'reports';out.mkdir(parents=True,exist_ok=True);complete=all(c['status']=='completed' for c in cells)
 C.write(out/'semantic_summary.json',{'time':C.now(),'complete':complete,'expected_cells':len(cells),'cells':cells,'details':details})
 stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells);(out/'semantic_summary.csv').write_text(stream.getvalue());(out/'semantic_summary.md').write_text('\n'.join(text)+'\n')
 if print_status:print('\n'.join(text))
 return complete

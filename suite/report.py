import csv,io,statistics
from .common import STORE,read,rows,write,now,stage_dir
from .sampling import method_output_kind,output_kind,expected_general

def general_statistics(cfg):
 cells=[];table=[]
 for family in cfg['families']:
  for method in cfg['methods']:
   directory=STORE/'runs'/family/method_output_kind(cfg,'evaluation',method)/method
   try:raw=rows(directory/'results.jsonl')
   except ValueError:
    # Reporting never repairs a file while a worker may be writing it.
    data=(directory/'results.jsonl').read_bytes();end=data.rfind(b'\n')+1;raw=[__import__('json').loads(r) for r in data[:end].splitlines() if r]
   if len({r['example_id'] for r in raw})!=len(raw):raise ValueError('Duplicate evaluation IDs in report')
   for dataset,count in cfg['datasets'].items():
    expected=expected_general(cfg,dataset)
    rr=[r for r in raw if r['dataset']==dataset];n=len(rr)
    if n>expected:raise ValueError('More rows than the locked denominator')
    correct=sum(r['correct'] for r in rr);timings=[r['efficiency']['task_wall_seconds'] for r in rr]
    cell={'family':family,'method':method,'dataset':dataset,'provenance':'paper_derived_port' if method in ('latcom','interlat') else 'protocol_reimplementation','completed':n,'expected':expected,'source_expected':count,'sampled':expected!=count,'latency_comparable_as_isolated_run':False,'status':'completed' if n==expected else 'incomplete','correct':correct,'accuracy':correct/n if n else None,'mean_task_seconds':statistics.mean(timings) if n else None,'median_task_seconds':statistics.median(timings) if n else None}
    for key in ('generated_text_tokens','generated_latent_positions','prompt_tokens','model_prefill_positions','transmitted_positions','transmitted_bytes','receiver_message_positions','internal_cache_relay_positions','internal_cache_relay_bytes','compressor_prefill_positions'):
     cell['mean_'+key]=statistics.mean([r['efficiency'][key] for r in rr]) if n else None
     cell['total_'+key]=sum(r['efficiency'][key] for r in rr)
    cell['accumulated_task_seconds_not_stage_wall']=sum(timings);cell['total_generated_text_tokens']=sum(r['efficiency']['generated_text_tokens'] for r in rr);cells.append(cell)
 text=['# B200 baseline comparison',f'Updated (Beijing): {now()}','', '**Actual portable-suite results; not copied paper scores.** Incomplete denominators are shown explicitly.','']
 for family in cfg['families']:
  text += [f'## Qwen3-{family.upper()}','','| Method | '+' | '.join(cfg['datasets'])+' | Macro accuracy |','|---|'+'---|'*(len(cfg['datasets'])+1)]
  for method in cfg['methods']:
   cc=[c for c in cells if c['family']==family and c['method']==method];v=[]
   for c in cc:v.append(f'{c["correct"]}/{c["completed"]} ({100*c["accuracy"]:.2f}%)'+(' [partial]' if c['status']!='completed' else '') if c['completed'] else 'Pending')
   macro=f'{100*statistics.mean(c["accuracy"] for c in cc):.2f}%' if all(c['status']=='completed' for c in cc) else 'Incomplete'
   text.append('| '+method+' | '+' | '.join(v)+' | '+macro+' |')
  text.append('')
 text+=['## Training stages','', '| Family | Stage | State | Step |','|---|---|---|---|']
 for family in cfg['families']:
  for stage in ('latcom_stage1','latcom_stage2','interlat_receiver','interlat_compression'):
   state=read(stage_dir(family,stage)/'status.json',{});text.append(f'| {family} | {stage} | {state.get("status","pending")} | {state.get("step","—")} |')
 text+=['','Efficiency CSV covers every sender and final receiver; setup, training, code judging are excluded from inference time. Generated text tokens, latent positions, prefill and bytes are distinct. Parallel timings are contention diagnostics, not comparable isolated latency. Accumulated overlapping task seconds are not stage elapsed time; see scheduler_wall.jsonl. Token counts retain whole-path definitions.']
 return {'time':now(),'cells':cells,'complete':all(c['status']=='completed' for c in cells),'expected_cells':len(cells),'display_text':'\n'.join(text)+'\n'}


def report(cfg,print_status=True):
 from .report_semantic import semantic_statistics
 from .txt_reports import export_statistics
 general=general_statistics(cfg);semantic=semantic_statistics(cfg)
 result=export_statistics(cfg,general,semantic)
 if print_status:
  print(f"{result['completed_cells']}/{result['expected_cells']} completed cells; read {result['directory']}/00_README.txt",flush=True)
 return result['complete']

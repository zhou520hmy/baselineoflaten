"""Only aggregate statistics are exported. All parts are bounded by UTF-8 BYTES."""
import fcntl,hashlib,json,math,os,re
from pathlib import Path
from . import common as C
from .stat_records import numeric_tree

MAX_BYTES=88_000
PAYLOAD_BYTES=86_000

def byte_chunks(text,limit=PAYLOAD_BYTES):
 # Line boundaries preferred; oversized lines split at a UTF-8 codepoint boundary.
 chunks=[];current='';size=0
 for line in text.splitlines(keepends=True):
  while len(line.encode('utf-8'))>limit:
   if current:chunks.append(current);current='';size=0
   raw=line.encode('utf-8');piece=raw[:limit].decode('utf-8',errors='ignore')
   if not piece:raise ValueError('Byte budget cannot hold one Unicode character')
   chunks.append(piece);line=line[len(piece):]
  n=len(line.encode('utf-8'))
  if size+n>limit:chunks.append(current);current='';size=0
  current+=line;size+=n
 if current or not chunks:chunks.append(current)
 return chunks

def flatten(value,prefix=''):
 if isinstance(value,dict):
  for key in sorted(value):yield from flatten(value[key],prefix+'.'+str(key) if prefix else str(key))
 elif isinstance(value,list):
  for i,item in enumerate(value):yield from flatten(item,f'{prefix}[{i}]')
 else:
  shown='NA' if value is None else str(value)
  yield f'{prefix} = {shown}\n'

def table(cells):
 if not cells:return 'No cells enabled.\n'
 keys=list(cells[0]);lines=['\t'.join(keys)+'\n']
 for row in cells:lines.append('\t'.join('NA' if row.get(k) is None else str(row[k]) for k in keys)+'\n')
 return ''.join(lines)

def general_totals(cells):
 result={}
 for cell in cells:result.setdefault(cell['family']+'/'+cell['method'],[]).append(cell)
 return {key:{'completed':sum(c['completed'] for c in group),'expected':sum(c['expected'] for c in group),'correct':sum(c['correct'] for c in group),'micro_accuracy_completed':sum(c['correct'] for c in group)/sum(c['completed'] for c in group) if sum(c['completed'] for c in group) else None,'macro_accuracy_complete_only':sum(c['accuracy'] for c in group)/len(group) if all(c['status']=='completed' for c in group) else None} for key,group in result.items()}

def scheduler_statistics(cfg):
 from .sampling import method_output_kind
 result={}
 for family in cfg['families']:
  for kind in ('evaluation','semantic'):
   for method in cfg['methods']:
    path=C.STORE/'runs'/family/method_output_kind(cfg,kind,method)/method/'scheduler_wall.jsonl'
    attempts=[]
    if path.exists():
     for line in path.read_text().splitlines():
      try:row=json.loads(line)
      except json.JSONDecodeError:continue
      attempts.append(numeric_tree(row))
    result[family+'/'+kind+'/'+method]={'recorded_attempt_count':len(attempts),'attempts':attempts}
 return result

def safe_training_state(cfg):
 result={}
 for family in cfg['families']:
  for stage in ('latcom_stage1','latcom_stage2','interlat_receiver','interlat_compression'):
   root=C.stage_dir(family,stage);state=C.read(root/'status.json',{});metrics=root/'metrics.jsonl';series=[]
   if metrics.exists():
    for line in metrics.read_text().splitlines():
     try:row=json.loads(line)
     except json.JSONDecodeError:continue
     # Per-step NUMBERS only, no examples, prompts, paths or free-form exception text.
     series.append(numeric_tree(row))
   result[family+'/'+stage]={'status':state.get('status','pending'),'current_step':state.get('step',0),'target_steps':cfg['training'][stage+'_steps'],'complete':bool(C.read(root/'complete.json')),'metrics':series}
 return result

def safe_collection_state(cfg):
 output={}
 for family in cfg['families']:
  root=C.cache_dir(family);state=C.read(root/'status.json',{});ready=C.read(root/'complete.json',{})
  output[family]={'status':state.get('status','pending'),'main_retained_by_method':state.get('main_retained_by_method',{}),'gate_rejections':state.get('gate_rejections',{}),'candidate_count':state.get('candidate_count'),'scanned':state.get('scanned',0),'retained_assets':state.get('retained_assets',0),'method_ready':ready.get('method_ready',{}),'minimum_main_required':cfg['training']['minimum_retained']}
 return output

def failure_counts():
 from collections import Counter
 path=C.STORE/'branch_failures.jsonl';counts=Counter()
 if path.exists():
  for line in path.read_text().splitlines():
   try:row=json.loads(line)
   except json.JSONDecodeError:continue
   # Controlled stage identifiers and integer exit codes; never export raw exception messages.
   action=row.get('action');family=row.get('family');method=row.get('method');stage=row.get('stage')
   if family not in ('4b','8b') or action not in ('collect','train','evaluate','semantic','_smoke_one'):continue
   if method not in (None,'latentmas','latentmas_h2o','latentmas_hidden','latcom','interlat'):continue
   if stage not in (None,'latcom_stage1','latcom_stage2','interlat_receiver','interlat_compression'):continue
   rc=row.get('exit_code')
   if isinstance(rc,int):counts[f'{family}/{action}/{method or stage or "all"}/exit_{rc}']+=1
 latest=C.read(C.STORE/'last_error.json',{})
 error_type=latest.get('type')
 if isinstance(error_type,str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,80}',error_type):counts['latest_recorded_error_type/'+error_type]+=1
 return dict(counts)

def write_parts(directory,sections):
 entries=[];wanted={'00_README.txt','01_INDEX.txt'}
 for stem,title,text in sections:
  parts=byte_chunks(text)
  for i,part in enumerate(parts,1):
   name=f'{stem}.p{i:03d}.txt';payload=f'TITLE: {title}\nPART: {i}/{len(parts)}\nENCODING: UTF-8\n--- CONTENT ---\n'+part
   raw=payload.encode('utf-8')
   if len(raw)>MAX_BYTES:raise ValueError('TXT part exceeds byte limit')
   temp=directory.parent/('stats_export_'+name+'.tmp');temp.write_bytes(raw);os.replace(temp,directory/name);wanted.add(name)
   entries.append((name,len(raw),hashlib.sha256(raw).hexdigest(),title))
 # Index is deliberately small because sections/part count are bounded by the suite size.
 index='READ ORDER | BYTES | SHA256 | DESCRIPTION\n'+'\n'.join(f'{n} | {b} | {h} | {t}' for n,b,h,t in entries)+'\n'
 if len(index.encode())>MAX_BYTES:raise ValueError('Too many report parts for a bounded index; reduce retained step statistics')
 intro='''统计结果交接目录 / STATISTICS ONLY

先读本文件，再按 01_INDEX.txt 顺序读取各 TXT 分片。
每个 TXT 都小于 90,000 字节，实际硬上限为 88,000 字节。
分片带 PART 编号；只需按编号阅读，无需还原 JSONL 或拷贝其它运行文件。
03 为通用任务汇总，04 为 LATEN 总指标，05 为 LATEN 分层指标，
06 为训练步数/损失，07 为 collect 保留数/门槛，08 为累计阻塞次数。
09 为已记录的调度阶段耗时；当前未结束阶段不会计入完整历时。
02 包含覆盖率、评测口径和文件目录范围。NA 表示尚无统计，不等于零分。

本目录仅有聚合统计，没有逐题问题、答案、代码、推理文本或 token IDs。
Token/latent/prefill/通信字节分别统计；并行时间不作独占 GPU 速度比较。
传回本目录内全部 TXT 即可。不要传 storage/runs、models、data 或旧日志。
其它目录是本机续跑资产；旧原始日志不会被自动删除，也不会混入本目录。
采集缓存中的训练目标是训练资产，不属于本目录的交付结果。

索引内有每个统计分片的 UTF-8 字节数和 SHA256，可用于检查传输完整性。
更新读取时请等本轮 status/export-txt 返回，避免在索引更新期间拷贝。
'''
 for name,payload in [('00_README.txt',intro),('01_INDEX.txt',index)]:
  if len(payload.encode())>MAX_BYTES:raise ValueError('TXT directory file too large')
  tmp=directory.parent/('stats_export_'+name+'.tmp');tmp.write_text(payload,encoding='utf-8');os.replace(tmp,directory/name)
 for path in directory.glob('*.txt'):
  if path.name not in wanted and re.fullmatch(r'\d\d_[a-z0-9_]+\.p\d+\.txt',path.name):path.unlink()
 return entries

def export_statistics(cfg,general,semantic):
 directory=C.STORE/'STATS_TXT';directory.mkdir(parents=True,exist_ok=True)
 cells=general['cells']+semantic['cells'];complete=general['complete'] and semantic['complete'];done=sum(c['status']=='completed' for c in cells)
 overview={'updated_beijing':C.now(),'completed_cells':done,'expected_cells':len(cells),'complete':complete,'general_tasks_per_method_model':sum(c['expected'] for c in general['cells'] if c['family']==next(iter(cfg['families'])) and c['method']==cfg['methods'][0]),'semantic_variants_per_method_model':648,'semantic_pairs_per_method_model':486,'general_sample_fraction':cfg.get('evaluation_sampling',{}).get('fraction',1),'reasoning_output_included':False,'execution_identity':C.identity(cfg),'models':cfg['families'],'seed':cfg['seed'],'output_token_caps':cfg['caps'],'metric_notes':'Accuracy is a fraction from 0 to 1; NA is pending. General macro weights datasets equally. LATEN metrics are separate. Accumulated contended task seconds are not stage elapsed time.'}
 sections=[('02_overview','Coverage and metric definitions',''.join(flatten(overview))),('03_general','General task counts, accuracy, whole-path token and resource statistics','AGGREGATES\n'+''.join(flatten(general_totals(general['cells'])))+'\nPER-DATASET CELLS\n'+table(general['cells'])),('04_laten','Full LATEN counts, rates, CF and token statistics',table(semantic['cells']))]
 for family in cfg['families']:
  for method in cfg['methods']:
   detail=semantic['details'].get(family+'/'+method)
   if detail is not None:sections.append((f'05_laten_{family}_{method}',f'LATEN breakdown {family}/{method}',''.join(flatten(detail))))
 sections.extend([('06_training','Training stage status and numeric step metrics',''.join(flatten(safe_training_state(cfg)))),('07_collect','Method-specific collection gates and counts',''.join(flatten(safe_collection_state(cfg)))),('08_failures','Cumulative branch failure counts; historical failures can remain after successful retry',''.join(flatten(failure_counts())) or 'No recorded branch failures.\n'),('09_scheduler','Recorded coordinator attempt durations including setup, scoring and retries',''.join(flatten(scheduler_statistics(cfg))))])
 # Lock lives outside deliverables; STATS_TXT contains TXT only.
 with (C.STORE/'stats_export.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);write_parts(directory,sections)
 return {'directory':str(directory),'complete':complete,'completed_cells':done,'expected_cells':len(cells)}

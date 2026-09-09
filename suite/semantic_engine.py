"""Five explicit communication operators on the frozen serial G4/G5/G6 graph."""
import time,hashlib
import torch
from .semantic import build_role_prompt,build_role_public_context,build_deliberative_joint_vector_prompt,build_action_prompt,vector_score
from .engine import cache_length,cache_bytes,headwise_prune
from vendor.laten.efficiency import native_decode_counts


def native_vector_decode(engine,prefix,length,cap,*,cache=None,absolute_start=0,model=None):
 """Original native greedy close/EOS/cap rules, with method-specific prefix or KV."""
 model=model or engine.model;tokenizer=engine.tokenizer
 close=tuple(tokenizer.encode('</think>',add_special_tokens=False));answer=tuple(tokenizer.encode('\n\n',add_special_tokens=False));separator=tuple(tokenizer.encode(',',add_special_tokens=False));digits=[tokenizer.encode(x,add_special_tokens=False) for x in ('0','1')]
 if not close or any(len(x)!=1 for x in digits):raise ValueError('Tokenizer does not support native binary readout')
 digits=[x[0] for x in digits];eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
 out=engine.forward_cache(prefix,cache,absolute_start,model=model);cache=out.past_key_values;next_position=absolute_start+len(prefix)
 def logits(out):
  with engine.autocast():return model.lm_head(out.last_hidden_state[0,-1]).float()
 current=logits(out);reasoning=[];emitted=[];stop='max_new_tokens';scores=[];bits=[]
 def consume(value):
  nonlocal cache,next_position,current
  token=torch.tensor([value],device=engine.device);out=engine.forward_cache(model.get_input_embeddings()(token),cache,next_position,model=model);cache=out.past_key_values;next_position+=1;current=logits(out);emitted.append(value)
 for _ in range(cap):
  chosen=int(current.argmax())
  if chosen in eos:stop='eos_mapped_to_thinking_close';break
  reasoning.append(chosen);consume(chosen)
  if tuple(reasoning[-len(close):])==close:stop='thinking_close';break
 if stop=='eos_mapped_to_thinking_close':
  for value in close:reasoning.append(value);consume(value)
 completed=stop!='max_new_tokens'
 if completed:
  for value in answer:consume(value)
  for i in range(length):
   logp=current.log_softmax(-1);restricted=[float(logp[x]) for x in digits];bit=int(restricted[1]>restricted[0]);bits.append(str(bit));scores.append(restricted);consume(digits[bit])
   if i+1<length:
    for value in separator:consume(value)
 counts=native_decode_counts(reasoning_token_count=len(reasoning),stop_reason=stop,close_token_count=len(close),answer_prefix_token_count=len(answer),separator_token_count=len(separator),vector_length=length,completed=completed)
 return dict(vector=''.join(bits) if completed else None,reasoning_text='<think>\n'+tokenizer.decode(reasoning,skip_special_tokens=False,clean_up_tokenization_spaces=False),reasoning_token_ids=reasoning,reasoning_token_count=len(reasoning),reasoning_stop_reason=stop,reasoning_cap_hit=not completed,reasoning_max_new_tokens=cap,bit_logprobs=scores,emitted_token_ids=emitted,native_token_counts=counts)


def render_semantic_receiver(engine,messages,model):
 # Exact benchmark renderer appends THINKING_OPEN to the native generation template.
 text=engine.tokenizer.apply_chat_template(list(messages),tokenize=False,add_generation_prompt=True,enable_thinking=True)+'<think>\n'
 ids=engine.ids(text)
 if len(ids)>engine.cfg['max_prompt_tokens']:raise ValueError('Semantic own prompt exceeds explicit limit')
 marker='<|im_start|>user\n';start=text.index(marker)+len(marker);boundary=len(engine.tokenizer.encode(text[:start],add_special_tokens=False));end=text.rfind('<|im_end|>');end_boundary=len(engine.tokenizer.encode(text[:end],add_special_tokens=False))
 if engine.tokenizer.encode(text[:start],add_special_tokens=False)!=ids[:boundary].tolist():raise ValueError('Semantic user header is not a token prefix')
 return ids,model.get_input_embeddings()(ids),boundary,end_boundary


def pipeline(engine,case,variant,program,method,cfg,modules=None):
 modules=modules or {};sem=cfg['semantic_benchmark'];depth=sem['sender_latent_steps'];engine.realign_init();engine.seed(variant.variant_id,'semantic');engine.sync()
 if engine.device.type=='cuda':torch.cuda.reset_peak_memory_stats(engine.device)
 start=time.perf_counter();cache=None;absolute=0;message=None;ledger=[];traces=[];cache_prompt=0;cache_latent=0
 totals={k:0 for k in ('prompt_tokens','received_message_positions','generated_latent_positions','transmitted_positions','transmitted_prompt_positions','transmitted_latent_positions','transmitted_bytes','model_prefill_positions','adapter_context_tokens','compressor_prefill_positions','local_boundary_positions')}
 stages={'producer_rollout':0.,'message_adaptation':0.,'receiver_decode':0.}
 def incoming(own,qids,boundary,end):
  nonlocal message
  if message is None:return own,0
  before=time.perf_counter();n=len(message);totals['received_message_positions']+=n
  if method=='latcom':
   with engine.autocast():m=modules['compressor'](qids,[message])
   c=len(qids)+len(message)+2+len(m);totals['compressor_prefill_positions']+=c;totals['model_prefill_positions']+=c;totals['adapter_context_tokens']+=len(qids)
   totals['transmitted_positions']+=len(m);totals['transmitted_latent_positions']+=len(m);totals['transmitted_bytes']+=m.numel()*m.element_size();totals['received_message_positions']+=len(m)
   prefix=torch.cat([m.to(own),own]);received=len(m)
  elif method=='interlat':
   with engine.autocast():m=modules['receiver'].carrier(message)
   # Channel payload is raw hidden states; adapter/boundary vectors are receiver-local.
   prefix=torch.cat([own[:end],m.to(own),own[end:]]);received=len(m);totals['local_boundary_positions']+=len(m)-len(message)
  else:prefix=torch.cat([own[:boundary],message.to(own),own[boundary:]]);received=n
  engine.sync();stages['message_adaptation']+=time.perf_counter()-before
  return prefix,received
 producers=case.roles[:-1]
 for i,role in enumerate(producers):
  messages=build_role_prompt(case,variant,role_id=role.role_id,channel='latent');ids,own,boundary,end=engine.render(messages,False)
  if method=='interlat':own=modules['compressor'].backbone.get_input_embeddings()(ids)
  own_raw=engine.model.get_input_embeddings()(ids).detach();qids=ids
  if method=='latcom' and message is not None:qids=engine.render(build_role_public_context(case,program,role),False)[0]
  if method in ('latentmas','latentmas_h2o'):
   received=cache_length(cache);totals['received_message_positions']+=received;prefix=own
  else:prefix,received=incoming(own,qids,boundary,end)
  engine.sync();t=time.perf_counter()
  if method=='interlat':
   with engine.autocast():outgoing=modules['compressor'].cached(ids,prefix=prefix)
   generated=len(outgoing);pcount=0;lcount=generated;metadata={}
  else:
   r=engine.rollout(prefix,depth,cache if method in ('latentmas','latentmas_h2o') else None,absolute if method in ('latentmas','latentmas_h2o') else 0,h2o=method=='latentmas_h2o');generated=len(r['z']);metadata={'absolute_next':r['absolute_next']}
   if method in ('latentmas','latentmas_h2o'):
    cache=r['cache'];absolute=r['absolute_next'];cache_prompt+=len(own);cache_latent+=generated
    if method=='latentmas_h2o':
     cache=headwise_prune(cache,r['scores'],r['old_length'],len(own),generated,cfg['h2o_prompt_budget_per_sender_per_head']);cache_prompt-=max(0,len(own)-cfg['h2o_prompt_budget_per_sender_per_head'])
    outgoing=None;pcount=cache_prompt;lcount=cache_latent
   elif method=='latentmas_hidden':ledger.append(own_raw);outgoing=torch.cat([*ledger,r['z']]);pcount=sum(len(x) for x in ledger);lcount=generated
   else:outgoing=r['z'];pcount=0;lcount=generated
  engine.sync();seconds=time.perf_counter()-t;stages['producer_rollout']+=seconds
  positions=cache_length(cache) if outgoing is None else len(outgoing);nbytes=cache_bytes(cache) if outgoing is None else outgoing.numel()*outgoing.element_size()
  if positions!=pcount+lcount:raise AssertionError('Outgoing prompt/latent partition differs')
  totals['prompt_tokens']+=len(ids);totals['model_prefill_positions']+=len(prefix);totals['generated_latent_positions']+=generated;totals['transmitted_positions']+=positions;totals['transmitted_prompt_positions']+=pcount;totals['transmitted_latent_positions']+=lcount;totals['transmitted_bytes']+=nbytes
  def digest(tensor):return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  provenance={'own_prompt_embedding_sha256':digest(own_raw),'current_latent_sha256':digest(outgoing if method=='interlat' else r['z'])}
  traces.append(dict(**provenance,role_id=role.role_id,role_type=role.role_type,receiver_role_id=case.roles[i+1].role_id,own_messages=messages,own_prompt_token_ids=ids.tolist(),incoming_message_positions=received,prefill_positions=len(prefix),generated_latent_positions=generated,outgoing_prompt_positions=pcount,outgoing_latent_positions=lcount,outgoing_message_positions=positions,outgoing_bytes=nbytes,rollout_seconds=seconds,**metadata));message=outgoing
 messages=build_deliberative_joint_vector_prompt(case,program);model=modules['receiver'].model if method=='interlat' else engine.model
 ids,own,boundary,end=render_semantic_receiver(engine,messages,model)
 if method in ('latentmas','latentmas_h2o'):prefix=own;received=cache_length(cache);totals['received_message_positions']+=received
 else:prefix,received=incoming(own,ids,boundary,end);cache=None;absolute=0
 totals['prompt_tokens']+=len(ids);totals['model_prefill_positions']+=len(prefix);engine.sync();t=time.perf_counter()
 decoded=native_vector_decode(engine,prefix,len(program.facts),sem['reasoning_max_new_tokens'],cache=cache,absolute_start=absolute,model=model);engine.sync();stages['receiver_decode']=time.perf_counter()-t
 row=vector_score(variant,program,decoded);engine.sync();wall=time.perf_counter()-start
 totals.update(decoded['native_token_counts']);totals.update(task_wall_seconds=wall,stage_seconds=stages,cuda_synchronized=engine.device.type=='cuda',warmup=False,receiver_message_positions=received,peak_allocated_memory_gib=torch.cuda.max_memory_allocated(engine.device)/2**30 if engine.device.type=='cuda' else None,timing_scope='instrumented_warm_whole_role_path_through_vector_and_policy_excludes_model_action_diagnostic')
 if totals['received_message_positions']!=totals['transmitted_positions']:raise AssertionError('Semantic communication edge accounting mismatch')
 row.update(efficiency=totals,role_inputs=traces,receiver_messages=messages,receiver_prompt_token_ids=ids.tolist(),method=method,carrier_format=method,communication_protocol=sem['protocol'])
 return row


def action_diagnostic(engine,case,program,vector):
 """Original frozen-base A-H conditional scorer, no communication or gold input."""
 ids,own,_,_=engine.render(build_action_prompt(case,program,vector),False);out=engine.forward_cache(own)
 with engine.autocast():logp=engine.model.lm_head(out.last_hidden_state[0,-1]).float().log_softmax(-1)
 scores=[]
 for letter in 'ABCDEFGH':
  token=engine.tokenizer.encode(letter,add_special_tokens=False)
  if len(token)!=1:raise ValueError('A-H diagnostic requires single-token candidates')
  value=float(logp[token[0]]);scores.append(dict(candidate=letter,sum_logprob=value,mean_logprob=value,token_count=1))
 selected=min(scores,key=lambda x:(-x['mean_logprob'],x['candidate']))['candidate']
 return selected,scores,len(ids)

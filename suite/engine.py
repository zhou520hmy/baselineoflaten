from types import SimpleNamespace
import contextlib,hashlib,time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from vendor.latentmas_prompts import build_agent_message_hierarchical_latent_mas
from .common import model_path,keyed_seed
from .models import frozen

def cache_length(cache):return 0 if cache is None else cache.get_seq_length()
def cache_bytes(cache):return sum(l.keys.numel()*l.keys.element_size()+l.values.numel()*l.values.element_size() for l in cache.layers if l.is_initialized)
def headwise_prune(cache,scores,old_length,prompt_length,latent_steps,budget):
 k=min(budget,prompt_length);full=old_length+prompt_length+latent_steps
 for layer,importance in zip(cache.layers,scores):
  if layer.keys.shape[-2]!=full:raise AssertionError('Cache length differs from relay provenance')
  heads=layer.keys.shape[1];chosen=importance.topk(k,dim=-1).indices.sort(dim=-1).values+old_length
  old=torch.arange(old_length,device=chosen.device)[None].expand(heads,-1);tail=torch.arange(old_length+prompt_length,full,device=chosen.device)[None].expand(heads,-1)
  indices=torch.cat([old,chosen,tail],dim=-1)[None,:,:,None].expand(layer.keys.shape[0],-1,-1,layer.keys.shape[-1])
  layer.keys=torch.gather(layer.keys,2,indices).contiguous();layer.values=torch.gather(layer.values,2,indices).contiguous()
 return cache

class Engine:
 def __init__(self,path,cfg,device='cuda:0',model=None,tokenizer=None):
  self.cfg=cfg;self.device=torch.device(device);self.dtype=torch.bfloat16 if self.device.type=='cuda' else torch.float32
  self.tokenizer=tokenizer or AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
  self.model=model or AutoModelForCausalLM.from_pretrained(path,torch_dtype=self.dtype,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).to(device)
  frozen(self.model);self.W=None;self.embedding_norm=None
 def sync(self):
  if self.device.type=='cuda':torch.cuda.synchronize(self.device)
 def autocast(self):return torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else contextlib.nullcontext()
 def seed(self,*keys):
  seed=keyed_seed(self.cfg['seed'],*keys);torch.manual_seed(seed)
  if self.device.type=='cuda':torch.cuda.manual_seed_all(seed)
  return seed
 @torch.no_grad()
 def realign_init(self):
  if self.W is not None:return
  ein=self.model.get_input_embeddings().weight;eout=self.model.get_output_embeddings().weight;d=ein.shape[1]
  gram=torch.zeros(d,d,device=self.device,dtype=torch.float32);cross=torch.zeros_like(gram);norm=torch.zeros((),device=self.device)
  with torch.autocast(device_type=self.device.type,enabled=False):
   for start in range(0,len(ein),4096):
    i=ein[start:start+4096].float();o=eout[start:start+4096].float();gram.add_(o.T@o);cross.add_(o.T@i);norm+=i.norm(dim=-1).sum()
   gram.diagonal().add_(1e-5);self.W=torch.linalg.solve(gram,cross);self.embedding_norm=norm/len(ein)
  if not torch.isfinite(self.W).all():raise ValueError('Nonfinite realignment matrix')
 def realign(self,h):
  z=h.float()@self.W;return (z*(self.embedding_norm/z.norm(dim=-1,keepdim=True).clamp_min(1e-6))).to(self.dtype)
 def ids(self,text):return torch.tensor(self.tokenizer.encode(text,add_special_tokens=False),device=self.device,dtype=torch.long)
 def render(self,messages,thinking=False,model=None):
  text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
  # Qwen3's thinking template may stop before <think>; never add a duplicate opener.
  if thinking and not text.rstrip().endswith('<think>'):text+='<think>\n'
  ids=self.ids(text)
  if len(ids)>self.cfg['max_prompt_tokens']:raise ValueError('Prompt exceeds explicit limit; no silent truncation')
  user='<|im_start|>user\n';start=text.index(user)+len(user)
  while self.tokenizer.encode(text[:start],add_special_tokens=False)!=ids[:len(self.tokenizer.encode(text[:start],add_special_tokens=False))].tolist():
   if start>=len(text) or not text[start].isspace():raise ValueError('No token-exact injection boundary')
   start+=1
  boundary=len(self.tokenizer.encode(text[:start],add_special_tokens=False))
  end=text.rfind('<|im_end|>');end_boundary=len(self.tokenizer.encode(text[:end],add_special_tokens=False))
  emb=(model or self.model).get_input_embeddings()(ids)
  return ids,emb,boundary,end_boundary
 def role_messages(self,item,role):return build_agent_message_hierarchical_latent_mas(role,item['question'],context='',method='latent_mas',args=SimpleNamespace(model_name='Qwen/Qwen3',task=item['dataset']))
 def qa_messages(self,question,evidence=None):
  prompt=question if evidence is None else question+'\n\nEvidence:\n'+evidence
  return [{'role':'system','content':'You are a helpful reasoning assistant. Use the available evidence and end with Answer: followed by the answer.'},{'role':'user','content':prompt}]
 def base(self,model=None):return (model or self.model).model
 def forward_cache(self,embeds,cache=None,absolute_start=None,model=None,attentions=False):
  model=model or self.model;physical=cache_length(cache);absolute=physical if absolute_start is None else absolute_start
  if cache is None:cache=DynamicCache(config=model.config)
  positions=torch.arange(physical,physical+len(embeds),device=self.device);rotary=torch.arange(absolute,absolute+len(embeds),device=self.device)
  old=model.config._attn_implementation;model.config._attn_implementation='eager' if attentions else 'sdpa'
  try:
   with self.autocast():return model.model(inputs_embeds=embeds[None],past_key_values=cache,position_ids=rotary[None],cache_position=positions,attention_mask=torch.ones(1,physical+len(embeds),device=self.device,dtype=torch.long),use_cache=True,output_attentions=attentions,return_dict=True)
  finally:model.config._attn_implementation=old
 @torch.no_grad()
 def rollout(self,embeds,steps,cache=None,absolute_start=None,h2o=False):
  old_length=cache_length(cache);absolute=old_length if absolute_start is None else absolute_start
  out=self.forward_cache(embeds,cache,absolute);cache=out.past_key_values;last=out.last_hidden_state[0,-1];zs=[];raw=[];scores=None
  for i in range(steps):
   raw.append(last.detach());z=self.realign(last);zs.append(z.detach());out=self.forward_cache(z[None],cache,absolute+len(embeds)+i,attentions=True);cache=out.past_key_values;last=out.last_hidden_state[0,-1]
   if h2o:
    if out.attentions is None:raise ValueError('H2O requires actual attention probabilities')
    if scores is None:scores=[torch.zeros(l.keys.shape[1],len(embeds),device=self.device) for l in cache.layers]
    for j,a in enumerate(out.attentions):
     heads=scores[j].shape[0];selected=a[0,:,:,old_length:old_length+len(embeds)].float();scores[j]+=selected.reshape(heads,-1,selected.shape[-1]).sum(1)
  return {'z':torch.stack(zs),'raw':torch.stack(raw),'cache':cache,'absolute_next':absolute+len(embeds)+steps,'scores':scores,'old_length':old_length}
 @torch.no_grad()
 def decode(self,embeds,cap,cache=None,absolute_start=None,model=None,temperature=None):
  model=model or self.model;absolute=cache_length(cache) if absolute_start is None else absolute_start;out=self.forward_cache(embeds,cache,absolute,model=model)
  h=out.last_hidden_state[0,-1];cache=out.past_key_values;tokens=[];temperature=self.cfg['temperature'] if temperature is None else temperature
  eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos);stop='max_new_tokens'
  for i in range(cap):
   with self.autocast():logits=model.lm_head(h).float()
   if temperature<=0:token=logits.argmax()
   else:
    values,indices=torch.sort(logits/temperature,descending=True);probs=values.softmax(-1);remove=probs.cumsum(-1)-probs>self.cfg['top_p'];values[remove]=-torch.inf
    k=self.cfg.get('top_k',0)
    if k:values[k:]=-torch.inf
    token=indices[torch.multinomial(values.softmax(-1),1)[0]]
   value=int(token);tokens.append(value)
   if value in eos:stop='eos';break
   if i+1<cap:
    e=model.get_input_embeddings()(token[None]);out=self.forward_cache(e,cache,absolute+len(embeds)+i,model=model);h=out.last_hidden_state[0,-1];cache=out.past_key_values
  return {'text':self.tokenizer.decode(tokens,skip_special_tokens=False),'token_ids':tokens,'output_tokens':len(tokens),'stop_reason':stop}
 def prediction_logits(self,prefix,target_ids,model=None):
  model=model or self.model
  y=model.get_input_embeddings()(target_ids[:-1]);x=torch.cat([prefix,y])
  with self.autocast():
   h=model.model(inputs_embeds=x[None],use_cache=False,return_dict=True).last_hidden_state[0,len(prefix)-1:]
   logits=model.lm_head(h)
  if len(logits)!=len(target_ids):raise AssertionError('Teacher/student next-token shift mismatch')
  return logits.float()
 @torch.no_grad()
 def full_plan_hidden(self,messages,cap):
  ids,emb,_,_=self.render(messages,False);out=self.decode(emb,cap,temperature=0);tokens=torch.tensor(out['token_ids'],device=self.device);allids=torch.cat([ids,tokens]);h=self.model.model(input_ids=allids[None],use_cache=False,return_dict=True).last_hidden_state[0,len(ids):]
  return h.detach(),tokens,out['text'],ids

 def pipeline(self,item,method,modules=None):
  modules=modules or {};self.seed(item['example_id'],'evaluation');self.realign_init();self.sync()
  if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
  start=time.perf_counter();cache=None;absolute=0;payloads=[];zs=[];traces=[]
  totals=dict(generated_text_tokens=0,generated_latent_positions=0,prompt_tokens=0,model_prefill_positions=0,transmitted_positions=0,transmitted_bytes=0,internal_cache_relay_positions=0,internal_cache_relay_bytes=0,compressor_prefill_positions=0,sender_seconds=0.,compression_seconds=0.,receiver_seconds=0.)
  for role in ('planner','critic','refiner'):
   messages=self.role_messages(item,role);ids,own,_,_=self.render(messages,self.cfg['sender_thinking']);t=time.perf_counter();self.seed(item['example_id'],role)
   if method=='interlat':
    sender=modules['compressor'];receiver=modules['receiver'];
    with self.autocast():hidden=sender.cached(ids);message=receiver.carrier(hidden)
    payloads.append(message);positions=len(message);nbytes=message.numel()*message.element_size();generated=len(hidden);received=0
   else:
    r=self.rollout(own,self.cfg['sender_latent_steps'],cache,absolute,h2o=method=='latentmas_h2o');cache=r['cache'];absolute=r['absolute_next'];zs.append(r['z']);generated=len(r['z']);received=r['old_length']
    if method=='latentmas_h2o':cache=headwise_prune(cache,r['scores'],r['old_length'],len(own),generated,self.cfg['h2o_prompt_budget_per_sender_per_head'])
    if method=='latentmas_hidden':message=torch.cat([own,r['z']]);payloads.append(message);positions=len(message);nbytes=message.numel()*message.element_size()
    elif method=='latcom':positions=generated;nbytes=r['z'].numel()*r['z'].element_size()
    else:positions=cache_length(cache);nbytes=cache_bytes(cache)
   relay_positions=cache_length(cache) if method in ('latcom','latentmas_hidden') and role!='refiner' else 0
   relay_bytes=cache_bytes(cache) if relay_positions else 0
   totals['internal_cache_relay_positions']+=relay_positions;totals['internal_cache_relay_bytes']+=relay_bytes
   totals['transmitted_positions']+=relay_positions;totals['transmitted_bytes']+=relay_bytes
   self.sync();seconds=time.perf_counter()-t;totals['sender_seconds']+=seconds;totals['generated_latent_positions']+=generated;totals['prompt_tokens']+=len(ids);totals['model_prefill_positions']+=len(ids);totals['transmitted_positions']+=positions;totals['transmitted_bytes']+=nbytes
   traces.append(dict(role=role,prompt=messages,prompt_ids=ids.tolist(),prompt_tokens=len(ids),received_cache_positions=received,latent_steps=generated,internal_cache_relay_positions=relay_positions,internal_cache_relay_bytes=relay_bytes,transmitted_positions=positions,transmitted_bytes=nbytes,seconds=seconds))
  messages=self.role_messages(item,'judger');receiver_model=modules['receiver'].model if method=='interlat' else self.model
  ids,own,boundary,end_boundary=self.render(messages,self.cfg['receiver_thinking'],receiver_model);t=time.perf_counter();self.seed(item['example_id'],'judger')
  if method=='latcom':
   cache=None
   with self.autocast():carrier=modules['compressor'](ids,zs)
   totals['compressor_prefill_positions']=len(ids)+sum(len(z)+2 for z in zs)+len(carrier)
   totals['model_prefill_positions']+=totals['compressor_prefill_positions']
   totals['transmitted_positions']+=len(carrier);totals['transmitted_bytes']+=carrier.numel()*carrier.element_size()
   prefix=torch.cat([carrier,own]);absolute=0
  elif method in ('latentmas_hidden','interlat'):
   cache=None;carrier=torch.cat(payloads);where=boundary if method=='latentmas_hidden' else end_boundary;prefix=torch.cat([own[:where],carrier,own[where:]]);absolute=0
  else:carrier=None;prefix=own
  self.sync();totals['compression_seconds']=time.perf_counter()-t;totals['prompt_tokens']+=len(ids);totals['model_prefill_positions']+=len(prefix)
  totals['receiver_message_positions']=len(carrier) if carrier is not None else cache_length(cache)
  totals['compressor_output_bytes']=carrier.numel()*carrier.element_size() if method=='latcom' else 0
  t=time.perf_counter();output=self.decode(prefix,item['cap'],cache,absolute,model=receiver_model);self.sync();totals['receiver_seconds']=time.perf_counter()-t;totals['generated_text_tokens']+=output['output_tokens'];totals['task_wall_seconds']=time.perf_counter()-start
  totals['peak_allocated_memory_gib']=torch.cuda.max_memory_allocated(self.device)/2**30 if self.device.type=='cuda' else None
  totals['communication_scope']='sum_over_logical_edges_including_internal_sender_KV_relays_and_compressor_output;logical_bytes_not_network_measurement'
  totals['timing_scope']='all_sender_rollouts_compression_receiver_prefill_and_decode_excludes_setup_scoring';totals['cuda_synchronized']=self.device.type=='cuda'
  return dict(output=output,traces=traces,receiver_prompt=messages,receiver_prompt_ids=ids.tolist(),efficiency=totals,carrier_format=method)

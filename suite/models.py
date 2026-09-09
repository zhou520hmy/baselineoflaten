"""Explicit method modules. Learned methods require trained exports, never random fallbacks."""
import torch
from torch import nn
from transformers import AutoModel,AutoModelForCausalLM

class SlotCompressor(nn.Module):
 def __init__(self,backbone,slots=64):
  super().__init__();self.backbone=backbone;d=backbone.config.hidden_size
  self.slots=nn.Parameter(torch.randn(slots,d)*.02);self.roles=nn.Parameter(torch.randn(6,d)*.02);self.separator=nn.Parameter(torch.randn(1,d)*.02)
 def forward(self,qids,trajectories):
  emb=self.backbone.get_input_embeddings();parts=[emb(qids)]
  for i,z in enumerate(trajectories):
   if i>=6:raise ValueError('At most six training senders')
   parts += [self.roles[i:i+1],z.to(self.slots.dtype),self.separator]
  parts.append(self.slots);x=torch.cat(parts)[None]
  return self.backbone(inputs_embeds=x,use_cache=False,return_dict=True).last_hidden_state[0,-len(self.slots):]

class ReceiverAdapter(nn.Module):
 def __init__(self,d):
  super().__init__();self.pre_ln=nn.LayerNorm(d,eps=1e-6);self.mha=nn.MultiheadAttention(d,8,dropout=.1,batch_first=True);self.post_ln=nn.LayerNorm(d,eps=1e-6)
  self.proj=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.LayerNorm(d),nn.Linear(d,d));self.scale=nn.Parameter(torch.tensor(.2));self.output_scale=nn.Parameter(torch.tensor(.1))
  nn.init.normal_(self.proj[0].weight,std=.02);nn.init.zeros_(self.proj[0].bias);nn.init.xavier_uniform_(self.proj[-1].weight,gain=.01);nn.init.zeros_(self.proj[-1].bias)
 def forward(self,h):
  dtype=self.pre_ln.weight.dtype;x=self.pre_ln(h.to(dtype))[None];a,_=self.mha(x,x,x,need_weights=False);x=self.post_ln(x+a)[0];r=x*self.scale
  return (r+self.proj(r))*self.output_scale

class InterlatReceiver(nn.Module):
 def __init__(self,model):
  super().__init__();self.model=model;d=model.config.hidden_size;self.adapter=ReceiverAdapter(d);self.bop=nn.Parameter(torch.randn(1,d)*.02);self.eop=nn.Parameter(torch.randn(1,d)*.02)
 def carrier(self,hidden):return torch.cat([self.bop,self.adapter(hidden),self.eop])

class InterlatCompressor(nn.Module):
 def __init__(self,backbone,k=21):
  super().__init__();self.backbone=backbone;d=backbone.config.hidden_size;self.k=k;self.bos=nn.Parameter(torch.randn(1,d)*.02);self.h2e=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d,bias=False))
 def forward(self,ids):
  # Recomputed prefixes preserve the autoregressive dependency while allowing activation checkpointing.
  x=self.backbone.get_input_embeddings()(ids);x=torch.cat([x,self.bos]);states=[]
  for _ in range(self.k):
   h=self.backbone(inputs_embeds=x[None],use_cache=False,return_dict=True).last_hidden_state[0,-1]
   states.append(h);x=torch.cat([x,self.h2e(h)[None]])
  return torch.stack(states)
 @torch.no_grad()
 def cached(self,ids):
  from transformers.cache_utils import DynamicCache
  cache=DynamicCache(config=self.backbone.config)
  out=self.backbone(input_ids=ids[None],past_key_values=cache,use_cache=True,return_dict=True);cache=out.past_key_values;x=self.bos[None];states=[]
  for _ in range(self.k):
   out=self.backbone(inputs_embeds=x,past_key_values=cache,use_cache=True,return_dict=True);h=out.last_hidden_state[0,-1];states.append(h);cache=out.past_key_values;x=self.h2e(h)[None,None]
  return torch.stack(states)

def gradient_setup(module):
 module.train()
 for p in module.parameters():p.requires_grad_(True)
 base=getattr(module,'backbone',getattr(module,'model',None))
 base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 base.config.use_cache=False

def frozen(module):
 module.eval()
 for p in module.parameters():p.requires_grad_(False)
 return module

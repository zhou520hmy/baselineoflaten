import copy,unittest
from types import SimpleNamespace
import torch
from torch import nn
from transformers import Qwen3Config,Qwen3ForCausalLM
from suite.engine import Engine
from suite.models import SlotCompressor,InterlatReceiver,InterlatCompressor,frozen
from suite.semantic import load_frozen
from suite.semantic_engine import pipeline,native_vector_decode
from vendor.laten.native_reference import NativeReference,CapturedGenerationError
from test_tiny_cpu import Tokenizer

class ScriptHead(nn.Module):
 def __init__(self,script,default):super().__init__();self.script=list(script);self.default=default;self.index=0
 def forward(self,h):
  value=self.script[self.index] if self.index<len(self.script) else self.default;self.index+=1;logits=torch.full((*h.shape[:-1],128),-20.,device=h.device);logits[...,value]=20.;return logits
class SemanticCPU(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);torch.manual_seed(12);config=Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=48,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=32768,eos_token_id=0)
  cls.base=Qwen3ForCausalLM(config).float().eval();cls.base.generation_config.eos_token_id=0
  cls.cfg={'seed':42,'temperature':0.,'top_p':.95,'top_k':0,'max_prompt_tokens':12000,'sender_latent_steps':2,'sender_thinking':False,'receiver_thinking':True,'h2o_prompt_budget_per_sender_per_head':3,'semantic_benchmark':{'sender_latent_steps':2,'reasoning_max_new_tokens':2,'protocol':'tiny_CPU_fixture_not_formal'}}
  cls.variants,cls.cases,cls.programs,_=load_frozen()
 def test_all_five_methods_all_three_graphs(self):
  e=Engine(None,self.cfg,'cpu',model=copy.deepcopy(self.base),tokenizer=Tokenizer());e.realign_init()
  modules={'latcom':{'compressor':frozen(SlotCompressor(copy.deepcopy(self.base.model),4))},'interlat':{'receiver':frozen(InterlatReceiver(copy.deepcopy(self.base))),'compressor':frozen(InterlatCompressor(copy.deepcopy(self.base.model),3))}}
  for method in ('latentmas','latentmas_h2o','latentmas_hidden','latcom','interlat'):
   for graph in ('G4','G5','G6'):
    v=next(v for v in self.variants if v.graph_level==graph and v.information_level=='I3');c=self.cases[v.case_id]
    with torch.inference_mode():row=pipeline(e,c,v,self.programs[v.policy_program_id],method,self.cfg,modules.get(method))
    ef=row['efficiency'];self.assertEqual(len(row['role_inputs']),int(graph[1])-1);self.assertEqual(ef['received_message_positions'],ef['transmitted_positions']);self.assertEqual(ef['transmitted_positions'],ef['transmitted_prompt_positions']+ef['transmitted_latent_positions']);self.assertEqual(ef['generated_latent_positions'],(int(graph[1])-1)*(3 if method=='interlat' else 2));self.assertFalse(ef['cuda_synchronized'])
    if method=='latentmas_hidden':self.assertTrue(all(r['outgoing_latent_positions']==2 for r in row['role_inputs']))
 def test_original_native_decoder_parity_close_eos_cap(self):
  tokenizer=Tokenizer();close=tuple(tokenizer.encode('</think>'));one=tokenizer.encode('1')[0]
  for script,cap in (([0],20),(list(close),20),([],2)):
   model=copy.deepcopy(self.base);model.lm_head=ScriptHead(script,one);e=Engine(None,self.cfg,'cpu',model=model,tokenizer=tokenizer);prefix=model.get_input_embeddings()(torch.tensor([1,2,3]))
   with torch.inference_mode():actual=native_vector_decode(e,prefix,3,cap)
   refmodel=copy.deepcopy(self.base);refmodel.lm_head=ScriptHead(script,one);ref=NativeReference();ref.model=refmodel;ref.tokenizer=tokenizer;ref.eos_ids={0};ref.close_ids=close;ref.answer_prefix_ids=tuple(tokenizer.encode('\n\n'));ref.separator_ids=tuple(tokenizer.encode(','));ref.digit_ids=tuple(tokenizer.encode(x)[0] for x in ('0','1'));ref._prompt_embeddings=lambda messages,carrier:prefix[None]
   try:expected=ref.generate(carrier=prefix,messages=[],reasoning_max_new_tokens=cap,vector_length=3)
   except CapturedGenerationError as error:
    self.assertTrue(actual['reasoning_cap_hit']);self.assertEqual(actual['reasoning_token_ids'],error.token_ids);self.assertEqual(actual['native_token_counts']['answer_tokens'],0)
   else:
    for key in ('vector','reasoning_token_count','reasoning_stop_reason'):self.assertEqual(actual[key],getattr(expected,key))
    self.assertEqual(actual['reasoning_token_ids'],list(expected.reasoning_token_ids));self.assertEqual(actual['native_token_counts']['answer_tokens'],3)
 def test_compressor_optional_prefix_preserves_original_recurrence(self):
  module=frozen(InterlatCompressor(copy.deepcopy(self.base.model),3));ids=torch.tensor([1,2,3])
  with torch.inference_mode():self.assertTrue(torch.allclose(module.cached(ids),module.cached(ids,prefix=module.backbone.get_input_embeddings()(ids)),atol=1e-5))
if __name__=='__main__':unittest.main()

import copy,unittest
from types import SimpleNamespace
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from suite.engine import Engine,headwise_prune,cache_length
from suite.models import SlotCompressor,InterlatCompressor,InterlatReceiver,frozen
from suite.losses import js,latent_margin
class Tokenizer:
 def encode(self,s,add_special_tokens=False):return [ord(c)%95+1 for c in s]
 def decode(self,ids,skip_special_tokens=False,**kwargs):return ' '.join(str(int(i)) for i in ids)
 def apply_chat_template(self,messages,**kw):return '<|im_start|>system\nhelp<|im_end|>\n<|im_start|>user\n'+messages[-1]['content']+'<|im_end|>\n<|im_start|>assistant\n'
class TinyCPU(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);torch.manual_seed(7);conf=Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=4096,attention_dropout=0.,eos_token_id=0)
  cls.base=Qwen3ForCausalLM(conf).float().eval();cls.base.generation_config.eos_token_id=0
  cfg={'seed':42,'temperature':0.,'top_p':.95,'top_k':0,'max_prompt_tokens':2000,'sender_latent_steps':2,'sender_thinking':False,'receiver_thinking':True,'h2o_prompt_budget_per_sender_per_head':3}
  cls.engine=Engine(None,cfg,'cpu',model=cls.base,tokenizer=Tokenizer());cls.engine.realign_init()
 def test_alignment_actual_inputs(self):
  e=self.engine;emb=e.model.get_input_embeddings()(torch.tensor([1,2,3]));r=e.rollout(emb,2);self.assertEqual(cache_length(r['cache']),5);self.assertTrue(torch.allclose(r['z'].norm(dim=-1),e.embedding_norm.expand(2),atol=1e-5));self.assertTrue(torch.allclose(r['z'][0],e.realign(r['raw'][0])))
 def test_headwise_selection(self):
  keys=torch.arange(8.).reshape(1,1,8,1).expand(1,2,8,1).clone();layer=SimpleNamespace(keys=keys,values=keys.clone());cache=SimpleNamespace(layers=[layer]);scores=[torch.tensor([[9.,0.,8.,1.],[0.,9.,1.,8.]])];headwise_prune(cache,scores,2,4,2,2);self.assertEqual(layer.keys[0,0,:,0].tolist(),[0,1,2,4,6,7]);self.assertEqual(layer.keys[0,1,:,0].tolist(),[0,1,3,5,6,7])
 def test_pruned_absolute_positions(self):
  e=self.engine;emb=e.model.get_input_embeddings()(torch.tensor([1,2,3,4]));r=e.rollout(emb,2,h2o=True);cache=headwise_prune(r['cache'],r['scores'],0,4,2,2);self.assertEqual(cache_length(cache),4);out=e.forward_cache(emb[:1],cache,absolute_start=6);self.assertEqual(cache_length(out.past_key_values),5);self.assertTrue(torch.isfinite(out.last_hidden_state).all())
 def test_frozen_receiver_slot_gradient(self):
  e=self.engine;backbone=copy.deepcopy(e.model.model)
  for p in backbone.parameters():p.requires_grad_(True)
  compressor=SlotCompressor(backbone,4);qids=torch.tensor([1,2,3]);m=compressor(qids,[torch.randn(5,32)]);prefix=torch.cat([m,e.model.get_input_embeddings()(qids)]);logits=e.prediction_logits(prefix,torch.tensor([4,5,6]));torch.nn.functional.cross_entropy(logits,torch.tensor([4,5,6])).backward();self.assertGreater(float(compressor.slots.grad.abs().sum()),0);self.assertTrue(all(p.grad is None for p in e.model.parameters()))
 def test_interlat_recurrence(self):
  module=InterlatCompressor(copy.deepcopy(self.base.model),3).eval();ids=torch.tensor([1,2,3]);a=module(ids);b=module.cached(ids);self.assertTrue(torch.allclose(a,b,atol=2e-5));a.square().mean().backward();self.assertGreater(float(module.bos.grad.abs().sum()),0)
 def test_loss_signs(self):
  a=torch.tensor([[8.,-8.]],requires_grad=True);self.assertAlmostEqual(float(js(a,a)),0,places=6);self.assertGreater(float(js(a,-a)),.69);self.assertLess(float(js(a,-a)),.694);self.assertLess(float(latent_margin(torch.tensor(.1),torch.tensor(2.))),float(latent_margin(torch.tensor(2.),torch.tensor(.1))))
 def test_three_pipelines(self):
  for method in ('latentmas','latentmas_h2o','latentmas_hidden'):
   with torch.inference_mode():r=self.engine.pipeline({'example_id':'tiny/1','question':'What is 1+1?','dataset':'gsm8k','cap':2},method)
   self.assertEqual(len(r['traces']),3);self.assertEqual(r['efficiency']['generated_latent_positions'],6);self.assertEqual(r['output']['output_tokens'],2)
 def test_learned_pipelines(self):
  modules={'latcom':{'compressor':frozen(SlotCompressor(copy.deepcopy(self.base.model),4))},'interlat':{'receiver':frozen(InterlatReceiver(copy.deepcopy(self.base))),'compressor':frozen(InterlatCompressor(copy.deepcopy(self.base.model),3))}}
  for method,module in modules.items():
   with torch.inference_mode():r=self.engine.pipeline({'example_id':'tiny/learned','question':'1+1?','dataset':'gsm8k','cap':2},method,module)
   self.assertEqual(r['output']['output_tokens'],2)
   self.assertEqual(r['efficiency']['receiver_message_positions'],4 if method=='latcom' else 15)
   if method=='latcom':
    self.assertGreater(r['efficiency']['internal_cache_relay_bytes'],0);self.assertGreater(r['efficiency']['compressor_prefill_positions'],4)
 def test_receiver_and_compression_gradients(self):
  e=self.engine;receiver=InterlatReceiver(copy.deepcopy(self.base));receiver.requires_grad_(True);qids=torch.tensor([1,2]);y=torch.tensor([3,4]);prefix=torch.cat([receiver.model.get_input_embeddings()(qids),receiver.carrier(torch.randn(5,32))]);e.prediction_logits(prefix,y,receiver.model).square().mean().backward();self.assertGreater(float(receiver.adapter.pre_ln.weight.grad.abs().sum()),0)
  receiver.zero_grad(set_to_none=True);frozen(receiver);compressor=InterlatCompressor(copy.deepcopy(self.base.model),3);compressor.requires_grad_(True);h=compressor(qids);prefix=torch.cat([receiver.model.get_input_embeddings()(qids),receiver.carrier(h)]);e.prediction_logits(prefix,y,receiver.model).square().mean().backward();self.assertGreater(float(compressor.bos.grad.abs().sum()),0);self.assertTrue(all(p.grad is None for p in receiver.parameters()))
if __name__=='__main__':unittest.main()

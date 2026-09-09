import torch
import torch.nn.functional as F

def js(a,b):
 la=a.log_softmax(-1);lb=b.log_softmax(-1);lm=torch.logaddexp(la,lb)-torch.log(torch.tensor(2.,device=a.device))
 return .5*((la.exp()*(la-lm)).sum(-1)+(lb.exp()*(lb-lm)).sum(-1)).mean()
def reverse_kl(student,teacher):
 ls=student.log_softmax(-1);lt=teacher.log_softmax(-1)
 return (ls.exp()*(ls-lt)).sum(-1).mean()
def dynamic_weight(task,aux,maximum):return (.1*task.detach()/aux.detach().abs().clamp_min(1e-8)).clamp(.01,maximum)
def latent_margin(pos,neg,margin=.57,temperature=1.):return F.softplus((pos-neg+margin)/temperature)
def uncertainty_kl(student,full,empty):
 lf=full.log_softmax(-1);le=empty.log_softmax(-1);ls=student.log_softmax(-1)
 hf=-(lf.exp()*lf).sum(-1);he=-(le.exp()*le).sum(-1);w=(he-hf).clamp_min(0).detach()
 if bool((w>0).any()):w=w.clamp_max(torch.quantile(w[w>0],.95))
 return (w*(lf.exp()*(lf-ls)).sum(-1)).sum()/w.sum().clamp_min(1e-8)

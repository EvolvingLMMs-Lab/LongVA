"""Test the correctness (up to certain tolerance of numerical error) of low-memory cross-ent

Yao Fu, University of Edinburgh
yao.fu@ed.ac.uk
"""

import sys 
sys.path.append("..")

import torch 
import torch.nn.functional as F
from low_mem_cross_ent import low_mem_cross_ent, cross_ent_normal

bsz = 1
seqlen = 50000
hidden = 4096
vocab = 15000
dtype = torch.bfloat16
rtol=1e-05 # relative tolerance when comparing the gradients from two implementations
atol=1e-07 # absolute tolerance when comparing the gradients from two implementations
           # in Pytorch its default is 1e-8 but our implementation cannot pass this threshold
           # 1e-7 seems to be the smallest rolerance we can pass

x = torch.normal(mean=0, std=0.01, size=(bsz, seqlen, hidden), 
    device="cuda", dtype=dtype, requires_grad=True)
weight = torch.normal(mean=0, std=0.01, size=(vocab, hidden), 
    device="cuda", dtype=dtype, requires_grad=True)
labels = torch.randint(low=0, high=vocab - 1, size=(bsz, seqlen), device="cuda")

loss_normal = cross_ent_normal(x, weight, labels)
print("loss normal: %.4f" % loss_normal.cpu().item())
loss_normal.backward()
x_grad = x.grad.clone()
weight_grad = weight.grad.clone()
# print(x.grad)
# print(weight.grad)


# TODO: this one almost reduce memory to half. Maybe further increase sp
x.grad = None
weight.grad = None
loss_low_mem = low_mem_cross_ent(x, weight, labels)
print("loss low mem: %.4f" % loss_low_mem.cpu().item())
loss_low_mem.backward()
# print(x.grad)
# print(weight.grad) 

## Test implementation by asserting close
assert(torch.allclose(x_grad, x.grad, rtol=rtol, atol=atol))
assert(torch.allclose(weight_grad, weight.grad, rtol=rtol, atol=atol))
print("PASS: gradients from normal computation and low memory computation are close.")


# #### Test gradient of logits
# x.grad = None
# weight.grad = None
# logits = torch.einsum("bsh, vh -> bsv", x, weight)
# loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))
# d_logits = torch.autograd.grad(loss, logits)
# p = F.softmax(torch.einsum("blh, vh -> blv", x, weight), dim=-1)
# p_ = p / (bsz * seqlen)

# #### test index add 
# x = torch.tensor([1, 2, 3, 4, 5, 6, 7])
# index = torch.tensor([1, 3, 4])
# source = torch.tensor([1, 1, 1])
# x.index_add_(dim=0, index=index, source=source)

# #### test index add 2 
# sp = 4
# micro_seqlen = seqlen // sp
# p = torch.normal(mean=0, std=0.01, size=(bsz, micro_seqlen, vocab), 
#     device="cuda", dtype=torch.bfloat16)
# labels_ = labels[:, :micro_seqlen].view(-1)
# index = torch.arange(bsz * micro_seqlen, device="cuda") * vocab
# index += labels_
# d_logits = -p.view(-1)
# source = torch.tensor([1] * bsz * micro_seqlen, dtype=torch.bfloat16, device="cuda")
# d_logits.index_add_(0, index, source)
# d_logits = d_logits.view(bsz, micro_seqlen, vocab)


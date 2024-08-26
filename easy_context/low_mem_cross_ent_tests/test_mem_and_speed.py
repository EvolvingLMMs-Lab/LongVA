"""Test the memory and speed and MFU of low memory cross entropy

Yao Fu, University of Edinburgh
yao.fu@ed.ac.uk

bf16, seqlen=50000, vocab=150000, without torch.compile
|          | normal | low_mem | -     |
| sp       | -      | 4       | 16    |
| peak mem | 43.4G  | 18.5G   | 8.1G  |
| forward  | 0.307  | 0.310   | 0.315 |
| backward | 0.631  | 0.896   | 0.914 | 
| MFU      | 0.57   | 0.45    | 0.44  |

NOTE: tried torch.compile and it takes significantly larger memory, so do not use
TODO: profile and check why backward is slower
"""
import sys 
sys.path.append("..")

import torch 
import numpy as np 
import torch.nn.functional as F
from low_mem_cross_ent import low_mem_cross_ent, cross_ent_normal

implementation = "low_mem" # "normal", "low_mem"
device_type = "A100"
bsz = 1
seqlen = 50000
hidden = 4096
vocab = 150000
sp=16
dtype = torch.bfloat16
# dtype = torch.float
G = 1024 ** 3
T = 1024 ** 4

x = torch.normal(mean=0, std=0.01, size=(bsz, seqlen, hidden), 
    device="cuda", dtype=dtype, requires_grad=True)
weight = torch.normal(mean=0, std=0.01, size=(vocab, hidden), 
    device="cuda", dtype=dtype, requires_grad=True)
labels = torch.randint(low=0, high=vocab - 1, size=(bsz, seqlen), device="cuda")

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

n_runs = 50
flop = 6 * bsz * seqlen * hidden * vocab
if(implementation == "normal"):
    forward_times, backward_times = [], []
    for _ in range(n_runs): 
        loss_normal, time_elapse = timed(lambda: cross_ent_normal(x, weight, labels))
        forward_times.append(time_elapse)
        _, time_elapse = timed(lambda: loss_normal.backward())
        backward_times.append(time_elapse)
    mem = torch.cuda.max_memory_allocated()
elif(implementation == "low_mem"):
    forward_times, backward_times = [], []
    for _ in range(n_runs): 
        loss_low_mem, time_elapse = timed(lambda: low_mem_cross_ent(x, weight, labels, sp))
        forward_times.append(time_elapse)
        _, time_elapse = timed(lambda: loss_low_mem.backward())
        backward_times.append(time_elapse)
    mem = torch.cuda.max_memory_allocated()
else: raise NameError("Implementation %s not recognized" % implementation)

forward_time = np.median(forward_times)
backward_time = np.median(backward_times)
flops = (flop / T) / (forward_time + backward_time)
if(device_type == "A100"):
    device_flop = 312
else: raise NameError("device %s not recognized" % device_type)

print("%s, peak memory %.1fG, forward time %.4f, backward time %.4f, flops %.2fT, util %.2f" % 
    (implementation, mem / G, forward_time, backward_time, flops, flops / device_flop))

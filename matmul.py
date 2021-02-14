import torch
from torch import tensor

def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)

    for i in range(ar):
        # Convince yourself this works
        c[i,:] = (a[i].unsqueeze(-1) * b).sum(dim=0)
    return c

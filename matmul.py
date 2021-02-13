import torch
from torch import tensor

def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)

    for i in range(ar):
        for j in range(bc):
            for k in range(ac): # could use br here
                c[i,j] += a[i,k] * b[k,j]
    return c

import torch
from torch import tensor

def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac==br
    #c = torch.zeros(ar, bc)

    #for i in range(ar):
    #    # Convince yourself this works
    #    c[i,:] = (a[i,:,None] * b).sum(dim=0)
    #return c

    return torch.einsum('ik,kj->ij', a, b)

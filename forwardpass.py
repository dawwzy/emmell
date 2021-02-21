import torch
from torch import tensor

def forwardpass():
    return

def lin(x, w, b): return x @ w + b

def normalize(x, m, s): return (x-m)/s

def test_near_zero(a, tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"

def init_params(m, nh):
    w1 = torch.randn(m,nh)/math.sqrt(m)
    b1 = torch.zeros(nh)
    w2 = torch.randn(nh,1)/math.sqrt(nh)
    b2 = torch.zeros(1)

    return w1, b1, w2, b2

def relu(x): return x.clamp_min(0.0)

x = torch.randn(200,100)
y = torch.randn(200)
m = 100
nh = 50

w1, b1, w2, b2 = init_params(100, 50)

l1 = lin(x, w1, b1)

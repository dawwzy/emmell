import torch
from torch import tensor
import math

def forwardpass():
    return

def lin(x, w, b): return x @ w + b

def normalize(x, m, s): return (x-m)/s

def test_near_zero(a, tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"

def init_params(m, nh):
    # kaiming init
    w1 = torch.randn(m,nh)*math.sqrt(2/m)
    b1 = torch.zeros(nh)
    w2 = torch.randn(nh,1)*math.sqrt(2/nh)
    b2 = torch.zeros(1)

    return w1, b1, w2, b2

def relu(x): return x.clamp(min=0.0)

def mse (output, targ): return (output.squeeze(-1)- targ).pow(2).mean()

x = torch.randn(200,100)
y = torch.randn(200)
m = 100
nh = 50

w1, b1, w2, b2 = init_params(100, 50)

def model(x):
    l1 = lin(x, w1, b1)
    l2 = relu(l1)
    l3 = lin(l1, w2, b2)
    return l3

print(model(x))

import torch
from torch import tensor

def forwardpass():
    return

def lin(x, w, b): return x @ w + b

def normalize(x, m, s): return (x-m)/s

def test_near_zero(a, tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"

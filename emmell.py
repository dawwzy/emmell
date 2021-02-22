import torch
from torch import tensor
import math

def forwardpass():
    return

def normalize(x, m, s): return (x-m)/s

def test_near_zero(a, tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"

def init_params(m, nh):
    # kaiming init
    w1 = torch.randn(m,nh)*math.sqrt(2/m)
    b1 = torch.zeros(nh)
    w2 = torch.randn(nh,1)*math.sqrt(2/nh)
    b2 = torch.zeros(1)

    return w1, b1, w2, b2

class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp(min = 0.0)
        return self.out

    def backward(self): self.inp.g = (self.inp>0).float() * self.out.g

class Lin():
    def __init__(self, w, b): self.w, self.b = w,b

    def __call__(self, inp):
        self.inp = inp
        self.out = inp@self.w + self.b
        return self.out

    def backward(self): 
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)

class Mse():
    def __call__(self, inp, targ):
        self.inp = inp
        self.targ = targ
        self.out = ( inp.squeeze() - targ ).pow(2).mean()
        return self.out

    def backward(self):
        x = 2.0*( self.inp.squeeze() - self.targ ).unsqueeze(-1)
        self.inp.g = x/self.targ.shape[0]

class Model():
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1, b1), Relu(), Lin(w2, b2)]
        self.loss = Mse()

    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x, targ)

    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()

x = torch.randn(200,100)
y = torch.randn(200)
m = 100
nh = 50

w1, b1, w2, b2 = init_params(100, 50)

model = Model(w1, b1, w2, b2)
loss = model(x,y)
model.backward()

print(loss)

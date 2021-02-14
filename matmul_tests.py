from matmul import *
import timeit

def test1():
    m1 = torch.randn(5, 28*28)
    m2 = torch.randn(784, 10)
    return matmul(m1, m2)

t1 = timeit.timeit("test1()", number = 10, globals=globals())

print("Test 1 timing: ", t1)

def test2():
    m1 = tensor([[1, 2],[3, 4]])
    m2 = tensor([[5, 6],[7, 8]])
    return matmul(m1, m2)

print("Test 2 ... ", end="")
assert torch.allclose(test2(), tensor([[19.0, 22.0],[43.0, 50.0]]), 
       rtol=1e-3, atol=1e-5), "Failed"
print("OK")

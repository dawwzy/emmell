from matmul import *
import timeit

def test1():
    m1 = torch.randn(5, 28*28)
    m2 = torch.randn(784, 10)
    return matmul(m1, m2)

t1 = timeit.timeit("test1()", number = 10, globals=globals())

print(t1)

def test2():
    m1 = tensor([[1, 2],[3, 4]])
    m2 = tensor([[5, 6],[7, 8]])
    return matmul(m1, m2)

assert torch.all(torch.eq(test2(), tensor([[19, 22],[43, 50]])))
print("Passed Test 2")


import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
print(x.size())

X = x.reshape(3, 4)
print(X)

print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))
print(torch.randn((3, 4)))
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
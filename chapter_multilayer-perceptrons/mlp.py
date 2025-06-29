
import torch
import d2l

import matplotlib.pyplot as plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

figure, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x.detach(), y.detach())
# y.backward(torch.ones_like(x), retain_graph=True)
y.backward(torch.ones_like(x), retain_graph=False)
ax2.plot(x.detach(), x.grad)
plt.show()

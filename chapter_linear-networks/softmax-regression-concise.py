
import torch
import d2l
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10


def train(net, train_iter, loss, trainer):
    epoch_loss = 0
    num_batch = 0
    net.train()

    for x, y in train_iter:
        trainer.zero_grad()
        l = loss(net(x), y)
        epoch_loss += l.sum()
        num_batch += y.numel()
        l.mean().backward()
        trainer.step()

    return epoch_loss / num_batch


def test(net, test_iter, loss):
    test_loss = 0
    num_batch = 0
    net.eval()

    for x, y in test_iter:
        l = loss(net(x), y)
        test_loss += l.sum()
        num_batch += y.numel()

    return test_loss / num_batch


for epoch in range(num_epochs):
    total_loss = train(net, train_iter, loss, trainer)
    print(f"Epoch: {epoch + 1}, Train Loss: {total_loss:.4f}")

print(f"Test Loss: {test(net, test_iter, loss):.4f}")
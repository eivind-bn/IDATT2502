import torch
import torch.nn.functional
import torchvision
import matplotlib.pyplot as plt

mnist_train = torchvision.datasets.MNIST('', train=True, download=True)
mnist_test = torchvision.datasets.MNIST('', train=False, download=True)

train_in = mnist_train.data.reshape(-1, 784).float()
test_in = mnist_test.data.reshape(-1, 784).float()

train_out = torch.zeros((mnist_train.targets.shape[0], 10))
train_out[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

test_out = torch.zeros((mnist_test.targets.shape[0], 10))
test_out[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1


class HandwritingClassificationModel:
    def __init__(self):
        self.W = torch.ones([784, 10], requires_grad=True)
        self.b = torch.ones([1, 10], requires_grad=True)

    def f(self, x):
        return torch.nn.functional.softmax(x @ self.W + self.b, dim=1)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = HandwritingClassificationModel()
optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)

for _ in range(250):
    model.loss(train_in, train_out).backward()
    optimizer.step()
    optimizer.zero_grad()

print(f'loss = {model.loss(train_in, train_out).item()}, '
      f'accuracy = {model.accuracy(test_in, test_out).item()}')

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(model.W[:, i].detach().numpy().reshape(28, 28))
    plt.title(f'img: {i}')
    plt.xticks([])
    plt.yticks([])

plt.show()

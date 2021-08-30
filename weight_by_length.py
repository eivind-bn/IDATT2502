import re
import matplotlib.pyplot as plt
from torch import tensor, optim
from torch.nn.functional import mse_loss

sample_lengths = []
sample_weights = []

with open('length_weight.csv') as file:
    formatted = re.findall("(\\d+\\.\\d+e[+-]\\d+),(\\d\\.\\d+e[+-]\\d+)", file.read())

    for(length, weight) in formatted:
        sample_lengths.append(float(length))
        sample_weights.append(float(weight))

sample_lengths = tensor(sample_lengths).reshape(-1, 1)
sample_weights = tensor(sample_weights).reshape(-1, 1)


class LinearRegressionModel:

    def __init__(self):
        self.W = tensor([[0.0]], requires_grad=True)
        self.b = tensor([[0.0]], requires_grad=True)

    def f(self, arg):
        return arg @ self.W + self.b

    def loss(self, arg, result):
        return mse_loss(self.f(arg), result)


model = LinearRegressionModel()
optimizer = optim.SGD([model.W, model.b], 0.0001)

for epoch in range(100_000):
    model.loss(sample_lengths, sample_weights).backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"W = {model.W}, b = {model.b}, loss = {model.loss(sample_lengths, sample_weights)}")

plt.xlabel('x = length')
plt.ylabel('y = weight')

length_interval = tensor([[sample_lengths.min()], [sample_lengths.max()]])

plt.plot(sample_lengths, sample_weights, 'o', label='$(x^{(i)},y^{(i)})$')
plt.plot(length_interval.flatten(), model.f(length_interval).detach().flatten(), label='$\\hat y = f(x) = xW+b$')

plt.legend()
plt.show()

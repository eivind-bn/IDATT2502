import re

import matplotlib.pyplot as plt
from torch import tensor, optim, exp
from torch.nn.functional import mse_loss

sample_days = []
sample_head_circumferences = []

with open('day_head_circumference.csv') as file:
    formatted = re.findall("(\\d+\\.\\d+e[+-]\\d+),(\\d\\.\\d+e[+-]\\d+)", file.read())

    for(day, head_circumference) in formatted:
        sample_days.append(float(day))
        sample_head_circumferences.append(float(head_circumference))

sample_days = tensor(sample_days).reshape(-1, 1)
sample_head_circumferences = tensor(sample_head_circumferences).reshape(-1, 1)


class NonLinearRegressionModel:

    def __init__(self):
        self.W = tensor([[0.0]], requires_grad=True)
        self.b = tensor([[0.0]], requires_grad=True)

    def f(self, arg):
        return 20 * 1 / (1 + exp(-(arg @ self.W + self.b))) + 31

    def loss(self, arg, result):
        return mse_loss(self.f(arg), result)


model = NonLinearRegressionModel()
optimizer = optim.SGD([model.W, model.b], 0.000001)

for epoch in range(100_000):
    model.loss(sample_days, sample_head_circumferences).backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"W = {model.W}, b = {model.b}, loss = {model.loss(sample_days, sample_head_circumferences)}")

plt.xlabel('x = length')
plt.ylabel('y = weight')

plt.scatter(sample_days, sample_head_circumferences)
plt.scatter(sample_days, model.f(sample_days).detach().flatten(), color='orange', label='$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \dfrac{1}{1+e^{-z}}$')

plt.legend()
plt.show()
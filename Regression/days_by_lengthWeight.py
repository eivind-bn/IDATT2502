import matplotlib.pyplot as plt
from torch import tensor, optim
from torch.nn.functional import mse_loss
import re


sample_days = []
sample_length = []
sample_weights = []


with open('Regression/day_length_weight.csv') as file:
    formatted = re.findall("(\\d+\\.\\d+e[+-]\\d+),(\\d+\\.\\d+e[+-]\\d+),(\\d\\.\\d+e[+-]\\d+)", file.read())

    for(day, length, weight) in formatted:
        sample_days.append(float(day))
        sample_length.append(float(length))
        sample_weights.append(float(weight))

sample_days = tensor(sample_days).reshape(-1, 1)
sample_length = tensor(sample_length).reshape(-1, 1)
sample_weights = tensor(sample_weights).reshape(-1, 1)


class LinearRegressionModel:

    def __init__(self):
        self.W1 = tensor([[0.0]], requires_grad=True)
        self.W2 = tensor([[0.0]], requires_grad=True)
        self.b = tensor([[0.0]], requires_grad=True)

    def f(self, arg0, arg1):
        return (arg0 @ self.W1) + (arg1 @ self.W2) + self.b

    def loss(self, arg0, arg1, result):
        return mse_loss(self.f(arg0, arg1), result)


model = LinearRegressionModel()
optimizer = optim.SGD([model.W1, model.W2, model.b], lr=0.0001)

for epoch in range(100_000):
    model.loss(sample_length, sample_weights, sample_days).backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"W1 = {model.W1}, W2 = {model.W2}, b = {model.b}, loss = {model.loss(sample_length, sample_weights, sample_days)}")


ax = plt.axes(projection="3d")
ax.set_xlabel('x = length')
ax.set_ylabel('y = weight')
ax.set_zlabel('z = days')

length_interval = tensor([[sample_length.min()], [sample_length.max()]])
weight_interval = tensor([[sample_weights.min()], [sample_weights.max()]])

ax.scatter(sample_length.flatten(), sample_weights.flatten(), sample_days.flatten())
ax.plot(length_interval.flatten(), weight_interval.flatten(), model.f(length_interval, weight_interval).detach().flatten(), color='orange', label='$f(x,y)=xW1+yW2+b$')

plt.legend()
plt.show()



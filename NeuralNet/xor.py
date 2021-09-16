
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional

train_in = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).reshape(-1, 2)
train_out = torch.tensor([[0.0], [1.0], [1.0], [0.0]]).reshape(-1, 1)


class XorModel:

    def __init__(self):

        self.W1 = torch.tensor([[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)],
                                [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]], requires_grad=True)
        self.W2 = torch.tensor([[random.uniform(-1.0, 1.0)], [random.uniform(-1.0, 1.0)]], requires_grad=True)
        self.b1 = torch.tensor([[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]], requires_grad=True)
        self.b2 = torch.tensor([[random.uniform(-1.0, 1.0)]], requires_grad=True)

    def f(self, x):
        return torch.sigmoid(torch.sigmoid(x @ self.W1 + self.b1) @ self.W2 + self.b2)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)


model = XorModel()


optimizer = torch.optim.SGD([model.b1, model.b2, model.W1, model.W2], lr=1)
for epoch in range(100_000):
    model.loss(train_in, train_out).backward()
    optimizer.step()
    optimizer.zero_grad()

print(f'W1 = {model.W1}, '
      f'W2 = {model.W2}, '
      f'b1 = {model.b1}, '
      f'b2 = {model.b2}, '
      f'loss = {model.loss(train_in.reshape(-1, 2), train_out)}')


fig = plt.figure()
ax = fig.gca(projection='3d')

plt.table(cellText=[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
          colWidths=[0.1] * 3,
          colLabels=["$x$", "$y$", "$f(x)$"],
          cellLoc="center",
          loc="lower right")

x = torch.arange(0, 1, 0.02)
y = torch.arange(0, 1, 0.02)

z = torch.empty(len(x),len(y), dtype=torch.double)

for i in range(len(x)):
    for j in range(len(y)):
        z[i,j] = float(model.f(torch.tensor([float(x[i]), float(y[j])])))

x,y = torch.meshgrid(x,y)
ax.plot_wireframe(x, y, z, color='darkred')


plt.show()

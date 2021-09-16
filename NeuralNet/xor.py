
import torch
import torch.nn.functional
import matplotlib.pyplot as plt
import numpy as np
import random


sampleIn = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).reshape(-1, 2)
sampleOut = torch.tensor([[0.0], [1.0], [1.0], [0.0]]).reshape(-1, 1)


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
    model.loss(sampleIn, sampleOut).backward()
    optimizer.step()
    optimizer.zero_grad()

print(f'W1 = {model.W1}, W2 = {model.W2}, b1 = {model.b1}, b2 = {model.b2}, loss = {model.loss(x_train.reshape(-1, 2), y_train)}')


fig = plt.figure()
ax = fig.gca(projection='3d')

plt.table(cellText=[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
          colWidths=[0.1] * 3,
          colLabels=["$x_1$", "$x_2$", "$f(x)$"],
          cellLoc="center",
          loc="lower right")



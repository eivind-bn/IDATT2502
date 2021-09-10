
import torch
import torch.nn.functional
import numpy as np
import matplotlib.pyplot as plt


sampleIn = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).reshape(-1, 2)
sampleOut = torch.tensor([[1.0], [1.0], [1.0], [0.0]]).reshape(-1, 1)


class NandModel:
    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x1, x2):
        return torch.sigmoid((x1 @ self.W[0]) + (x2 @ self.W[1]) + self.b)

    def logits(self, x1, x2):
        return ((x1 @ self.W[0]) + (x2 @ self.W[1]) + self.b).reshape(-1, 1)

    def loss(self, x1, x2, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x1, x2), y)


model = NandModel()
optimizer = torch.optim.SGD([model.b, model.W, model.W], 0.1)

for epoch in range(25_000):
    model.loss(sampleIn[:, 0].reshape(-1, 1),
               sampleIn[:, 1].reshape(-1, 1),
               sampleOut).backward()
    optimizer.step()
    optimizer.zero_grad()


print(f'W = {model.W}, b = {model.b}, loss = {model.loss(sampleIn[:, 0].reshape(-1, 1), sampleIn[:, 1].reshape(-1, 1), sampleOut)}')


x1_grid, x2_grid = np.meshgrid(np.linspace(-0.1, 1.1, 100), np.linspace(-0.1, 1.1, 100))
y_grid = np.empty([100, 100], dtype=np.double)

for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        tenseX = torch.tensor(float(x1_grid[i, j])).reshape(-1, 1)
        tenseY = torch.tensor(float(x2_grid[i, j])).reshape(-1, 1)
        y_grid[i, j] = model.f(tenseX, tenseY)


fig = plt.figure()
plot = fig.add_subplot(projection='3d')


plot.set_xlabel("$in_1$")
plot.set_ylabel("$in_2$")
plot.set_zlabel("$out$")


table = plt.table(cellText=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
                  colWidths=[0.1] * 3,
                  colLabels=["$in_1$", "$in_2$", "$out$"],
                  cellLoc="center",
                  loc="upper left")

plot.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")
plot.plot(sampleIn[:, 0].squeeze(),
          sampleIn[:, 1].squeeze(),
          sampleOut[:, 0].squeeze(),
          'o',
          color="blue")

plt.show()

import torch
import torch.nn.functional
import matplotlib.pyplot as plt


train_in = torch.tensor([[0.0], [1.0]]).reshape(-1, 1)
train_out = torch.tensor([[1.0], [0.0]]).reshape(-1, 1)


class NotModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = NotModel()
optimizer = torch.optim.SGD([model.b, model.W], 0.1)

for epoch in range(10_000):
    model.loss(train_in, train_out).backward()
    optimizer.step()
    optimizer.zero_grad()


print(f'W = {model.W}, '
      f'b = {model.b}, '
      f'loss = {model.loss(train_in, train_out)}')


plt.xlabel('in')
plt.ylabel('out')
plt.table(cellText=[[0, 1], [1, 0]],
          colWidths=[0.1] * 3,
          colLabels=["$in$", "$out$"],
          cellLoc="center",
          loc="lower left")


test_in = torch.arange(0.0, 1.0, 0.001).reshape(-1, 1)
test_out = model.f(test_in).detach()
plt.plot(test_in, test_out, color="orange")
plt.show()

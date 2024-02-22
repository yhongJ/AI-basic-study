import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype = torch.float32)

class XOR(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)
        self.activator = nn.Sigmoid()

    def forward(self, a):
        a = self.activator(self.layer1(a))
        a = self.activator(self.layer2(a))
        return a

model = XOR()
lossFunction = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 1)

for epoch in range(10000):
    outputs = model(x)
    loss = lossFunction(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'[epoch: {epoch + 1}], [loss: {loss.item()}]')

with torch.no_grad():

    print(f'Predicted: { model(x).round()}')
    print(f'Actual: {y}')







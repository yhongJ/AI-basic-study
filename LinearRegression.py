import torch
import torch.nn as nn
import torch.optim as optim


x = torch.tensor([[0], [1], [2], [3], [4], [5]], dtype = torch.float32)
y = torch.tensor([[1], [3], [5], [7], [9], [11]], dtype = torch.float32)

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float32))
    def forward(self, a):
        a = self.weights * a + self.bias
        return a

model = Linear()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(10000):
    output = model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f' Epoch: {epoch + 1}/ Loss: {loss.item()}')


with torch.no_grad():

    input_x = float(input())
    input_x_tensor = torch.tensor([input_x], dtype=torch.float)
    print(float(model(input_x_tensor)))
















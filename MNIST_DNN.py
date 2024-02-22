
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torchmetrics


import matplotlib.pyplot as plt
plt.style.use('seaborn')

torch.__version__


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])


trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
testset =  torchvision.datasets.MNIST(root = './data', train = False, download = False, transform = transform)

train_loader = DataLoader(trainset, batch_size = 128,  shuffle = True)
test_loader = DataLoader(testset, batch_size = 128, shuffle = True)


class DNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 128)
    self.fc2 = nn.Linear(128, 32 )
    self.fc3 = nn.Linear(32, 10)
    self.activator = nn.ReLU()

  def forward(self, a):
    a = a.view(a.size(0), -1)  #(batchsize, 28, 28) -> (batchsize, 28 *28)
    a = self.activator(self.fc1(a))
    a = self.activator(self.fc2(a))
    a = self.fc3(a)
    return a

model = DNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

for epoch in range(5):

  running_loss = 0

  for i, data in enumerate(train_loader, 0):

    images, labels = data
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  print(f'Epoch: {epoch + 1}/, Loss: {running_loss / len(train_loader)}')
  running_loss = 0


with torch.no_grad():

  correct = 0
  total = 0

  for i, data in enumerate(test_loader, 0):
    images, labels = data
    outputs = model(images)
    predicted = torch.argmax(outputs, dim = 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()


print(100 * correct / total)


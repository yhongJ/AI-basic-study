import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

torch.__version__


mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5)) ])  #전처리 과정

trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = mnist_transform)

testset =  torchvision.datasets.MNIST(root = './data', train = False, download = False, transform = mnist_transform)

train_loader = DataLoader(trainset, batch_size =128, shuffle = True, num_workers = 2 )

test_loader = DataLoader(testset, batch_size = 128, shuffle = False, num_workers = 2)

image, label = next(iter(train_loader))
print(image.shape, label.shape)

#t신경망구성

class Net(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(1, 6, 3, 1)
    self.conv2 = nn.Conv2d(6, 16, 3, 1)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)


  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x  = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1: ]
    num_features = 1
    for s in  size:
      num_features *= s

    return num_features


model = Net()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

for epoch in range(5):

  running_loss = 0

  for i, data in enumerate(train_loader, 0):
    inputs, labels = data

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print(f'Epoch: {epoch + 1}/, Loss: {running_loss / len(train_loader)}')
  running_loss = 0





correct = 0
total = 0

with torch.no_grad():
  for data in test_loader:
    images, labels = data
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()


print(100 * correct / total)

images_to_show = []
labels_to_show = []
predicted_to_show = []

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        if i == 10:  # 처음 10개 항목만 확인
            break
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        images_to_show.extend(images)
        labels_to_show.extend(labels)
        predicted_to_show.extend(predicted)

# 이미지를 플롯
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for idx, ax in enumerate(axes.ravel()):
    img = images_to_show[idx].squeeze()  # MNIST 이미지는 1x28x28이므로 squeeze 사용
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {labels_to_show[idx]}\nPredicted: {predicted_to_show[idx]}')


plt.show()


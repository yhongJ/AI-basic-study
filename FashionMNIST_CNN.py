import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

torch.__version__


fashionMNIST_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5)) ])

trainset = torchvision.datasets.FashionMNIST(root = './data', train = True, download = True, transform = fashionMNIST_transforms)

testset =  torchvision.datasets.FashionMNIST(root = './data', train = False, download = False, transform = fashionMNIST_transforms)

train_loader = torch.utils.data.DataLoader( trainset, batch_size = 100, shuffle = True, num_workers = 2)
test_loader = torch.utils.data.DataLoader( testset, batch_size = 100, shuffle = True, num_workers = 2)

labels_table = {0 : 'T-shirt',1 :  'Trouser', 2 : 'Pullover', 3 : 'Dress',4 :  'Coat',5 :  'Sandal', 6:  'Shirt', 7 : 'Sneaker', 8 :  'Bag', 9 :  'Ankle Boot'}

fig = plt.figure(figsize = (8, 8))
col = 5
row = 4
print(trainset)

for i in range(col * row):
  img_xy = np.random.randint(len(trainset))
  img = trainset[img_xy][0][0, :]
  fig.add_subplot(row, col, i + 1)
  plt.title(labels_table[trainset[img_xy][1]])
  plt.axis('off')
  plt.imshow(img, cmap = 'gray')

plt.show()


class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 10, 3, 1) #conv출력 크기: (input size - kernal size + 2 padding size) / stride  + 1  (mnist size = 784)
    self.conv2 = nn.Conv2d(10, 20, 3, 1)
    self.fc1 = nn.Linear(500, 100)
    self.fc2 = nn.Linear(100, 30)
    self.fc3 = nn.Linear(30, 10)
  def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #pooling 출력 크기: inputsize / kernal size
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



model = CNN()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

for epoch in range(5):

  running_loss = 0

  for i, data in enumerate(train_loader, 0):

    inputs, labels = data
    output = model(inputs)

    optimizer.zero_grad()

    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  print(f'epoch : {epoch + 1} / loss: {running_loss / len(train_loader) }')

correct = 0
total = 0
images_to_show = []
labels_to_show = []
predicted_to_show =[]

with torch.no_grad():
  for i, data in enumerate(test_loader, 0):
    images, labels = data
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1) #최댓값이 들어있는 tensor위치와 최댓값 중 최댓값을 반환
    if i == 0:
      images_to_show.extend(images)
      labels_to_show.extend(labels)
      predicted_to_show.extend(predicted)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print(100 * correct / total)

  fig, axes = plt.subplots(3, 5, figsize=(15, 10))
  for idx, ax in enumerate(axes.ravel()):
    img = images_to_show[idx].squeeze()
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {labels_table[labels_to_show[idx].item()]}\nPredicted: {labels_table[predicted_to_show[idx].item()]}') #값을 그대로 dictionary의 key로 사용할 수 없으므로 정수값으로 받기 위해 .item()사용


plt.show()





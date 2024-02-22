import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np

import cv2

import time

import os

import matplotlib.pyplot as plt


data_path = 'drive/MyDrive/catanddog/train'

transform =  transforms.Compose([transforms.Resize([256, 256]), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])

trainset = torchvision.datasets.ImageFolder(data_path, transform = transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size = 32, num_workers = 8, shuffle = True)

print(len(trainset))


labels_table = {0: 'cat', 1: 'dog'}

fig = plt.figure(figsize = (16, 24))

for i in range(24):
    sample_xy = np.random.randint(len(trainset))
    sample, label = trainset[sample_xy]
    ax = fig.add_subplot(4, 6, i + 1)  # 각 이미지에 대한 별도의 subplot 생성
    ax.set_title(labels_table[label])  # 이미지의 레이블을 제목으로 표시
    ax.axis('off')  # 축 숨기기
    ax.imshow(np.transpose(sample.numpy(), (1, 2, 0)))  # 이미지 표시


plt.subplots_adjust(bottom = 0.2, top = 0.6, hspace = 0)



model = models.resnet18(pretrained = True)

for params in model.parameters():
  params.requires_grad = False


model.fc = nn.Linear(512, 2)

for params in model.fc.parameters():
  params.requires_grad = True

criterion = nn.CrossEntropyLoss()



def train_model(model, dataloader, criterion, optimizer, epochs ):

  since  = time.time()
  accs = []
  losses = []
  best_acc = 0.0
  correct = 0

  for epoch in range(epochs):
    running_loss = 0
    running_corrects = 0

    for i, data in enumerate(dataloader, 0): #i가 0부터 start

      inputs, labels = data

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      _, predicted =  torch.max(outputs, 1)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() *  inputs.size(0)
      running_corrects += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)

    print(f'Epoch: {epoch + 1}/ Loss: {epoch_loss}/ Accuracy: {epoch_acc}')

    if epoch_acc > best_acc:
      best_acc = epoch_acc

    accs.append(epoch_acc)
    losses.append(epoch_loss)
    torch.save(model.state_dict(), os.path.join('drive/MyDrive/catanddog/', '{:02d}.pth'.format(epoch)))
    print()

  time_elapse = time.time() - since
  print(f'train completed in {time_elapse} s')
  print(f'best accuracy: {best_acc}')
  return accs, losses


params_to_update = []
for param in model.parameters():
  if param.requires_grad == True:
    params_to_update.append(param)


optimizer = optim.Adam(params_to_update)
trained_accs, trained_losses = train_model(model, train_loader, criterion, optimizer, 10)

plt.figure(figsize = (5, 5))
x = np.arange(0, 10, 1)
plt.plot(x, trained_accs, label = 'accuracy')
plt.plot(x, trained_losses, label = 'loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()



test_data_path = 'drive/MyDrive/catanddog/test'

testset = torchvision.datasets.ImageFolder(test_data_path, transform= transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, num_workers=8, shuffle = True)

correct = 0
total = 0
images_to_show = []
labels_to_show = []
predicted_to_show =[]

with torch.no_grad():
  for i, data in enumerate(test_loader, 0):
    images, labels = data
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    if(i == 0):
      images_to_show.extend(images)
      labels_to_show.extend(labels)
      predicted_to_show.extend(predicted)

    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print(100 * correct / total)

  fig, axes = plt.subplots(3, 5, figsize=(15, 10))
  for i, ax in enumerate(axes.ravel()):
    img = images_to_show[i].permute(1, 2, 0)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {labels_table[labels_to_show[i].item()]}\nPredicted: {labels_table[predicted_to_show[i].item()]}') #값을 그대로 dictionary의 key로 사용할 수 없으므로 정수값으로 받기 위해 .item()사용


plt.show()




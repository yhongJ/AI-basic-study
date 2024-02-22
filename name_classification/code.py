import torch
import torch.nn as nn
import torch.optim as optim

import unicodedata
import string
import os
import glob
import time
import random

import numpy as np

import matplotlib.pyplot as plt

data_path = 'drive/MyDrive/name_data/data/names/*.txt'
datas = glob.glob(data_path)
print(datas)

letters = string.ascii_letters
n_letters = len(letters)
print(letters)

def unicodeToAscii(s):
  return ''.join( #'어떤 것도 사이에 두지 않고 join
            c for c  in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn' and c in letters
  )

print(unicodeToAscii('Ślusàrski'))

category_lines = {}
all_categories = []

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n') #utf-8:유니코드 버전, 읽고, 공백제거, 줄바꿈단위로 쪼갬
    return [unicodeToAscii(line) for line in lines]

for filename in datas:
  category = os.path.splitext(os.path.basename(filename))[0]  #경로의 기본 이름. 즉 data/../filename.txt 중 filename, txt를 반환한 뒤, 그 첫번째인 filename을 반환
  all_categories.append(category)
  category_lines[category] = readLines(filename)

n_categories = len(all_categories)
n_total_data = 0
for i, data in enumerate(all_categories, 0):
  n_total_data += len(category_lines[data])

print(n_total_data)

def nameToTensor(name): #one-hot vector
    tensor = torch.zeros(len(name), 1, n_letters)
    for idx, letter in enumerate(name):
        tensor[idx][0][letters.find(letter)] = 1
    return tensor

print(nameToTensor('Yoon'))

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.i2o = nn.Linear(input_size + hidden_size, output_size)
    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.softmax = nn.LogSoftmax(dim = 1)

  def forward(self, input, hidden):
    combined = torch.cat((input, hidden), dim = 1)
    hidden = torch.tanh(self.i2h(combined))
    output = self.softmax(self.i2o(combined))

    return output, hidden

  def initHidden(self):
    return torch.zeros(1, self.hidden_size)

n_hidden = 128
model = RNN(n_letters, n_hidden, n_categories)

def categoryFromOutput(output):
  _, idx = torch.topk(output, 1) #output의 가장큰 1개의 값, 인덱스를 뽑아옴.
  category_idx = idx[0].item() #idx가 tensor로 반환되기 때문
  return category_idx

def choiceRandom(list_):
  return list_[random.randint(0, len(list_) - 1)]

def randomTrainData():
  category = choiceRandom(all_categories)
  name = choiceRandom(category_lines[category])
  category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long)
  name_tensor = nameToTensor(name)
  return category, name, category_tensor, name_tensor

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.005)

def train_model(category_tensor, name_tensor):
  hidden = model.initHidden()

  model.zero_grad()

  for i in range(name_tensor.size()[0]):
    output, hidden  = model(name_tensor[i], hidden)

  loss = criterion(output, category_tensor)
  loss.backward()
  optimizer.step()

  return output, loss.item()

since = time.time()
losses = []
current_loss = 0

for i in range(100000):
  category, name, category_tensor, name_tensor = randomTrainData()
  output, loss = train_model(category_tensor, name_tensor)
  current_loss += loss

  if i % 10000 == 0:
    print(f'{i + 1}th data/ {time.time() - since}s/ loss: {loss}/ name: {name}/ predicted: {all_categories[categoryFromOutput(output)]}/ country :{category}')

  if i % 500 == 0:
    losses.append(current_loss / 500)
    current_loss = 0

plt.figure()
plt.plot(losses)

correct = 0
total = 100

with torch.no_grad():
  for i in range(100):
    hidden  = model.initHidden()
    category, name, category_tensor, name_tensor = randomTrainData()
    for j in range(name_tensor.size()[0]):
      output, hidden = model(name_tensor[j], hidden)
    if categoryFromOutput(output) == category_tensor.item():
      correct += 1

print(f'accuracy: {correct / total * 100}')

inputName = input()
inputNameTensor = nameToTensor(inputName)

hidden = model.initHidden()
for i in range(inputNameTensor.size()[0]):
  output, hidden = model(inputNameTensor[i], hidden)

print(f'predicted: {all_categories[categoryFromOutput(output)]}')



import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from string import ascii_lowercase
import time
import math

import matplotlib.pyplot as plt

from sklearn.utils import shuffle


data = pd.read_csv('drive/MyDrive/name_gender_dataset.csv')
shuffle_data = shuffle(data)

print(shuffle_data.head(n = 10))
print(data.head(n = 10))
train_set = shuffle_data
test_set = data[ : 1000]

n_train = train_set.shape[0]
n_test = test_set.shape[0]
print(data.shape)
print(n_train)
print(n_test)

def nameNormalize(name):
  return name.lower()

letters = ascii_lowercase

def nameToTensor(name):
  name = nameNormalize(name)
  tensor = torch.zeros(len(name), 1, len(letters))
  for i in range(len(name)):
    tensor[i][0][letters.find(name[i])] = 1
  return tensor

print(nameToTensor('Yoon'))

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()

    self.hidden_size = hidden_size
    self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # (154, 128)
    self.i2o = nn.Linear(input_size + hidden_size, output_size) # (154, 2)
    self.softmax = nn.LogSoftmax(dim = 1) #(batchsize, data)중 data에 함수를 적용

  def forward(self, input, hidden):
    combined = torch.cat((input, hidden), dim = 1)
    hidden = torch.tanh(self.i2h(combined))
    output = self.softmax(self.i2o(combined))
    return hidden, output


  def init_hidden(self):
    return torch.zeros(1, self.hidden_size)

n_input = len(letters)
n_hidden = 200
n_ouput = 2 #man or woman

model = RNN(n_input, n_hidden, n_ouput)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.05)


def genderToTensor(c):
  if c == 'M':
    return torch.tensor([0])
  else:
    return torch.tensor([1])


def trainData(csv_data, idx):
  name = csv_data.loc[idx, 'Name']
  name_tensor = nameToTensor(name)
  gender = csv_data.loc[idx, 'Gender' ]
  gender_tensor = genderToTensor(gender)
  return name, name_tensor, gender, gender_tensor



def trainModel(name_tensor, gender_tensor):
  hidden = model.init_hidden()
  model.zero_grad()

  for i in range(name_tensor.size()[0]):
    hidden, output = model(name_tensor[i],hidden)

  loss = criterion(output, gender_tensor)
  loss.backward()
  optimizer.step()

  return output, loss.item()

since = time.time()
losses = []
current_loss = 0

def timeCal():
  s = time.time() - since
  m = math.floor(s / 60)
  s -= m * 60
  return s, m

def genderFromOutPut(output_tensor):
  _, idx = torch.topk(output_tensor, 1)
  return idx.item()

def idxToGender(n):
  if n == 0:
    return 'M'
  else:
    return 'F'

for i in range(n_train):
  name, name_tensor, gender, gender_tensor = trainData(train_set, i)
  output, loss = trainModel(name_tensor,  gender_tensor)
  current_loss += loss

  if i % 1000 == 0:
    losses.append(current_loss / 1000)
    current_loss = 0

  if i % 10000 == 0:
    s, m = timeCal()
    print(f'{i}th/ {m}m {s}s/ loss: {loss}/ name: {name}/ predicted: {idxToGender(genderFromOutPut(output))}/ gender: {gender}')


plt.figure()
plt.plot(losses)

correct = 0
total = n_test

with torch.no_grad():
  for i in range(total):
    hidden = model.init_hidden()
    name, name_tensor, gender, gender_tensor = trainData(test_set, i)
    for j in range(name_tensor.size()[0]):
      hidden, output = model(name_tensor[j], hidden)
    if genderFromOutPut(output) == gender_tensor.item():
      correct += 1

print(f'accuracy: {correct / total * 100}')


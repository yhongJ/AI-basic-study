import numpy as np
import math


T = 10000
noise_schedule = np.linspace(0.0001, 0.02, T)
alpha = 1 - noise_schedule
mean = []
variance = []
x = np.random.uniform(1, 10, (28, 28))
e = np.random.normal(0, 1, x.shape)
print("mean of initial x: ", end = '')
print(np.mean(x))
print("variance of initial x: ", end = '')
print(np.var(x))
for i in range(T):
    e = np.random.normal(0, 1, x.shape)
    x = x * math.sqrt(alpha[i]) + math.sqrt(1 - alpha[i]) * e
print("mean of later x: ", end = '')
print(np.mean(x))
print("variance of later x: ", end = '')
print(np.var(x))





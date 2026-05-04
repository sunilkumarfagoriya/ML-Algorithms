import numpy as np

size = np.array([12,23,34,45,56])
price = np.array([21,32,43,54,65])

w = np.random.randn()
b = np.random.randn()

i = 1000000
while(i):
    pred = w * size + b

    dw = np.mean((pred - price) * size)
    db = np.mean(pred - price)

    lr = 0.001

    w = w - lr * dw
    b = b - lr * db

    i -= 1

print(price)
print(pred)
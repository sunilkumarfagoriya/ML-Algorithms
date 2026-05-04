import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

w, b = 0, 0

i = 10000
while(i):
    pred = 1/(1+np.exp(-(w * X + b)))

    dw = np.mean((pred - y) * X)
    db = np.mean(pred - y)

    lr = 0.01

    w = w - lr * dw
    b = b - lr * db

    i -= 1

print(y)
print(pred)
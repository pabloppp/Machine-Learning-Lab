import numpy as np

np.random.seed(1234)

X = np.matrix([[4, 0], [2, 1]])
y = np.array([0, 1])  # 1: gato, 2: pato

W = np.random.rand(2, 1) - 0.5
b = 0


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def feed_forward(x, w, b):
    y = np.dot(x, w) + b
    y = sigmoide(y)
    return y


def loss(y, y_hat):
    return -(y_hat * np.log(y) + (1 - y_hat) * np.log(1 - y))


def back_prop(x, y, y_hat):
    grad_w = (1 / x.shape[1]) * np.dot(x, (np.resize(y_hat, [2, 1]) - y))
    grad_b = (1 / x.shape[1]) * np.sum(y_hat - y)

    return grad_w, grad_b


rate = 5

for i in range(0, 10):
    print("EPOCH {}".format(i))
    prediction = feed_forward(X, W, b)
    print("Predictions: {} {}".format(prediction[0], prediction[1]))
    print("Expected: {} {}".format(y[0], y[1]))

    e = loss(prediction, y)
    print("loss: {}".format(e))

    grad_w, grad_b = back_prop(X, prediction, y)

    W = W + grad_w * rate
    b = b + grad_b * rate

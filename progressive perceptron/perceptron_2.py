import numpy as np
import time

np.random.seed(1234)

X = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 0, 0, 1])  # XOR

n_hidden_nodes = 100

W1 = np.random.rand(2, n_hidden_nodes) - 0.5
W2 = np.random.rand(n_hidden_nodes, 1) - 0.5
b1 = np.zeros([1, n_hidden_nodes])
b2 = np.zeros([1, 1])


def relu(x):
    return np.maximum(x, 0)


def relu_grad(x):
    return np.minimum(np.maximum(x, 0), 1)


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoide_grad(x):
    return np.multiply(sigmoide(x), (1 - sigmoide(x)))


def feed_forward(x, w1, w2, b1, b2):
    l1 = np.dot(x, w1) + b1
    l1 = relu(l1)

    l2 = np.dot(l1, w2) + b2
    l2 = sigmoide(l2)

    return l1, l2


def loss(y, y_hat):
    return -(y_hat * np.log(y) + (1 - y_hat) * np.log(1 - y))


def back_prop(x, y2, y1, y_hat, w2, w1):
    grad_y2 = np.resize(y_hat, [4, 1]) - y2
    grad_w2 = (1 / w1.shape[1]) * np.dot(grad_y2.T, y1)
    grad_b2 = (1 / w1.shape[1]) * np.sum(grad_y2.T, axis=1)

    grad_y1 = np.multiply(np.dot(grad_y2, w2.T), relu_grad(y1))
    grad_w1 = (1 / x.shape[1]) * np.dot(grad_y1.T, x)
    grad_b1 = (1 / x.shape[1]) * np.sum(grad_y1.T, axis=1)

    return grad_w2.T, grad_b2.T, grad_w1.T, grad_b1.T


rate = 1

initial_time = time.time()
for i in range(0, 5000):
    print("EPOCH {}".format(i))
    y1, y2 = feed_forward(X, W1, W2, b1, b2)
    print("Predictions: {} {} {} {}".format(y2[0], y2[1], y2[2], y2[3]))
    print("Expected: {} {} {} {}".format(y[0], y[1], y[2], y[3]))

    e = loss(y2, y)
    print("loss: {}".format(e))

    grad_w2, grad_b2, grad_w1, grad_b1 = back_prop(X, y2, y1, y, W2, W1)

    W2 = W2 + grad_w2 * rate
    b2 = b2 + grad_b2 * rate
    W1 = W1 + grad_w1 * rate
    b1 = b1 + grad_b1 * rate

elapsed = time.time() - initial_time
print("Finished in {} ms".format(elapsed))

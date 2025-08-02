import numpy as np 
import matplotlib.pyplot as plt


a = [0, 0, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     1, 1, 1, 1, 1, 1,
     1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1]

b = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0]

c = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 1, 1, 0]

y = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]



plt.imshow(np.array(a).reshape(5, 6))

x = [np.array(a).reshape(1, 30), np.array(b).reshape(1, 30), np.array(c).reshape(1, 30)]

y = np.array(y)

print(x, "\n\n", y)


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def f_forward(x, w1, w2):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)

    z2 = a1.dot(w2)
    a2 = sigmoid(z2)
    return (a2)


def generate_wt(x, y):
    l = []
    for i in range(x * y):
        l.append(np.random.randn())
    return (np.array(l).reshape(x, y))


def loss(out, y):
    s = (np.square(out - y))
    s = (np.sum(s) / len(y))
    return (s)


def back_prop(x, y, w1, w2, alpha):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)

    z2 = a1.dot(w2)
    a2 = sigmoid(z2)

    d2 = (a2 - y)
    d1 = np.multiply(w2.dot(d2.T).T, a1 * (1 - a1))

    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)

    w1 = w1 - (alpha * (w1_adj))
    w2 = w2 - (alpha * (w2_adj))

    return (w1, w2)


def train(x, Y, w1, w2, alpha=0.01, epoch=10):
    acc = []
    losses = []
    for j in range(epoch):
        l = []
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2)
            l.append((loss(out, Y[i])))
            w1, w2 = back_prop(x[i], Y[i], w1, w2, alpha)
        print("Epochs:", j + 1, "acc:", (1 - (sum(l) / len(x))) * 100)
        acc.append((1 - (sum(l) / len(x))) * 100)
        losses.append(sum(l) / len(x))
    return (acc, losses, w1, w2)


def predict(x, w1, w2):
    Out = f_forward(x, w1, w2)
    label = np.argmax(Out)
    letters = ["A", "B", "C"]
    print("Image is of letter", letters[label])
    plt.imshow(x.reshape(5, 6))
    plt.show()
    return label



w1 = generate_wt(30, 5)
w2 = generate_wt(5, 3)
print(w1, "\n", w2)

acc, losses, w1, w2 = train(x, y, w1, w2, 0.1, 100)



plt.plot(acc)
plt.ylabel("Accuracy")
plt.xlabel("Epochs:")
plt.show()

plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Epochs:")
plt.show()

predict(x[0], w1, w2)
predict(x[1], w1, w2)
predict(x[2], w1, w2)
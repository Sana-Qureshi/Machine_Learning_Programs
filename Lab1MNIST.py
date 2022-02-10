from sklearn.linear_model import LogisticRegression  # for geting classifier
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')  # fetching data from net
print(mnist)
# we are storing two arrays in two variables
x, y = mnist['data'], mnist['target']
# print(x)
# print(y)
# this two array is not 2dimensional
# print(x.shape)  shape of x is (70000, 784) and y is 70000 but its one dimensional numpy arr
# we want 28 * 28 2d arr

some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()
#some_digit = x[784]
# some_digit_image = some_digit.reshape(28, 28)  # lets reshape to plot it
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
#     interpolation="nearest")
# plt.axis("off")
some_label = y.to_numpy()[36001]
print(some_label)

x_train, x_test = x.to_numpy()[:60000], x.to_numpy()[60000:]
y_train, y_test = y.to_numpy()[:60000], y.to_numpy()[60000:]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

# creating a 2 dtector
y_train = y_train.astype(np.int8)  # converting string vals to integer
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)
print(y_train_2)
print(y_test_2)

print(y_train)
clf = LogisticRegression(tol=0.1)
clf.fit(x_train, y_train_2)
print(clf.predict([some_digit]))

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
data_imgs = np.load("olivetti_faces.npy")
data_imgs.shape

type(data_imgs)
targets = np.load("olivetti_faces_target.npy")
targets.shape
type(targets)

data_imgs  # Image is 400X 64 X 64
data_imgs.shape
# 4.1 See an image
firstImage = data_imgs[0]
print("First image")
imshow(firstImage)


# 5.0 Flatten each image
data = data_imgs.reshape(data_imgs.shape[0], data_imgs.shape[1] * data_imgs.shape[2])  # 64 X 64 = 4096
# 5.1 Flattened 64 X 64 array
data.shape  # 400 X 4096

targets < 30  # Output is true/false
train = data[targets < 30]
test = data[targets >= 30]

n_faces = test.shape[0] // 10  # // is unconditionally "flooring division",
n_faces
face_ids = np.random.randint(0, 100, size=n_faces)
face_ids

test = test[face_ids, :]

n_pixels = data.shape[1]

X_train = train[:, :(n_pixels + 1) // 2]

y_train = train[:, n_pixels // 2:]

X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10,
                                       max_features=32,  # Out of 20000
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),  # Accept default parameters
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
}

# 9.1 Create an empty dictionary to collect prediction values
y_test_predict = dict()

# 10. Fit each model by turn and make predictions
#     Iterate over dict items. Each item is a tuple: ( name,estimator-object)s
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)  # fit() with instantiated object
    y_test_predict[name] = estimator.predict(X_test)  # Make predictions and save it in dict under key: name
    # Note that output of estimator.predict(X_test) is prediction for
    #  all the test images and NOT one (or one-by-one)
# 10.1 Just check
y_test_predict

# 10.2 Just check shape of one of them
y_test_predict['Ridge'].shape  # 5 X 2048

## Processing output
# 11. Each face should have this dimension
image_shape = (64, 64)

print("Images with same dimension 64*64")
plt.figure(figsize=(2 * n_faces * 2, 5))

j = 0
for i in range(n_faces):
    actual_face = test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Ridge'][i]))
    j = j + 1
    plt.subplot(4, 5, j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j + 1
    plt.subplot(4, 5, j)
    x = completed_face.reshape(image_shape)

    imshow(y)

plt.show()

# ## 12. For 'Extra trees' regression
# plt.figure(figsize=(2 * n_faces * 2, 5))
# j = 0
# for i in range(n_faces):
#     actual_face = test[i].reshape(image_shape)
#     completed_face = np.hstack((X_test[i], y_test_predict['Extra trees'][i]))
#     j = j + 1
#     plt.subplot(4, 5, j)
#     y = actual_face.reshape(image_shape)
#     x = completed_face.reshape(image_shape)
#     imshow(x)
#     j = j + 1
#     plt.subplot(4, 5, j)
#     x = completed_face.reshape(image_shape)
#     imshow(y)
#
# plt.show()

# ## 13. For 'Linear regression' regression
# plt.figure(figsize=(2 * n_faces * 2, 5))
# j = 0
# for i in range(n_faces):
#     actual_face = test[i].reshape(image_shape)
#     completed_face = np.hstack((X_test[i], y_test_predict['Linear regression'][i]))
#     j = j + 1
#     plt.subplot(4, 5, j)
#     y = actual_face.reshape(image_shape)
#     x = completed_face.reshape(image_shape)
#     imshow(x)
#     j = j + 1
#     plt.subplot(4, 5, j)
#     x = completed_face.reshape(image_shape)
#     imshow(y)
#
# plt.show()

# ## For '"K-nn' regression
# plt.figure(figsize=(2 * n_faces * 2, 5))
# j = 0
# for i in range(5):
#     actual_face = test[i].reshape(image_shape)
#     completed_face = np.hstack((X_test[i], y_test_predict['K-nn'][i]))
#     j = j + 1
#     plt.subplot(4, 5, j)
#     y = actual_face.reshape(image_shape)
#     x = completed_face.reshape(image_shape)
#     imshow(x)
#     j = j + 1
#     plt.subplot(4, 5, j)
#     x = completed_face.reshape(image_shape)
#     imshow(y)
#
# plt.show()

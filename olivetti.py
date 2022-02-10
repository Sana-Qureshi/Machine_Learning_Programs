import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow

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
plt.show()
# read npy
# import tensorflow
import numpy as np
from PIL import Image
src = '/media/ray/SSD/workspace/python/dataset/original/sketch_RNN/airplane/airplane.npy'
src = '/media/ray/SSD/workspace/python/dataset/original/sketch_RNN/airplane/airplane.npz'
src = '/media/ray/SSD/workspace/python/dataset/original/sketch_RNN/airplane/airplane/train.npy'
a = np.load(src, encoding='latin1')
s = 20
size = (28*s, 28*s)
def show_me(im_in):
    im_in = im_in.reshape(28, 28)
    img = Image.fromarray(im_in)
    img = img.resize(size, Image.ANTIALIAS)
    img.show()

# for i in range(10, 20):
#     show_me(a[i])

# print(a.shape)
# print(a[0].shape)
# print(a[0][0])
# for i in range(20):
#     print(a[i].shape)
# print(a[0])
for i in range(20):
    print(a[0][i][2])
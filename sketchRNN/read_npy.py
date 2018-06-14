# read npy
import numpy as np
from PIL import Image
src = '/media/ray/SSD/workspace/python/dataset/original/sketch_RNN/airplane/airplane.npy'

a = np.load(src)
s = 20
size = (28*s, 28*s)
def show_me(im_in):
    im_in = im_in.reshape(28, 28)
    img = Image.fromarray(im_in)
    img = img.resize(size, Image.ANTIALIAS)
    img.show()

for i in range(10, 20):
    show_me(a[i])
import numpy as np

src_filename = '/media/ray/SSD/workspace/python/dataset/save_here/1.npy'
a = np.load(src_filename)

# print(a)
# print(a.shape)
print(a[0])
print(a[0]['action'])
print(a[0]['state'])
import numpy as np

# a = np.random.random((2,3,4,5))
# print(a.shape)
# # a = np.transpose(a, (3, 1, 0, 2))
# a = a.transpose((3, 2, 1, 0))
# print(a.shape)
# np.savez_compressed('me.a', abc=[1,2,3], num=[10,11,12])
a = np.load('me.a.npz')
print(a['abc'])
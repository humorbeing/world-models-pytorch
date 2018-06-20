import numpy as np


a = [1,0,1]
b = [0,0,0]

aas = []
aas.append(a)
aas.append(b)
print(aas)
aas = np.array(aas)
print(aas.shape)

bbs = []
a = np.array(a)
b = np.array(b)

bbs.append(a[None,:])
bbs.append(b[None,:])
bbs = np.array(bbs)

print(bbs.shape)
bbs = np.concatenate(bbs)
print(bbs.shape)
print(aas)
print(bbs)
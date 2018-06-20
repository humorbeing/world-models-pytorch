import numpy as np
from matplotlib import pyplot as plt

for j in range(0,3):
    img = np.random.normal(size=(100,150))
    plt.figure(1); plt.clf()
    plt.imshow(img)
    plt.title('Number ' + str(j))
    plt.pause(3)


import numpy as  np
import matplotlib.pyplot as plt

b = np.zeros((50,50))
w = np.ones((50,50))

gornji = np.hstack((b, w))
donji = np.hstack((w, b))

slika = np.vstack((gornji,donji))

plt.imshow(slika,cmap="gray")
plt.show()

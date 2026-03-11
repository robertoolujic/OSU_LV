import numpy as  np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img[:,:,0].copy()
print(img)
print(img.shape)
print(img.dtype)
plt.figure()
cimg=img+np.full(img.shape,200)
print(img)
plt.imshow(cimg,cmap="gray")
cetvrtina = len(img)/4
cetvrtinaslike=cimg[[cetvrtina,cetvrtina*3],[]]
plt.imshow(cetvrtinaslike,cmap="gray")
plt.show()
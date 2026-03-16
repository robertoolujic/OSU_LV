import numpy as  np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img[:,:,0].copy()
plt.figure()
cimg=img+np.full(img.shape,100)
cimg=np.clip(cimg,0,255)
print(img)
print(cimg)
h,w = img.shape
cetvrtina = np.fliplr(np.rot90(cimg[0:(int)(h/2),(int)(w/2):w], 3))
plt.imshow(cetvrtina,cmap="gray")
plt.show()

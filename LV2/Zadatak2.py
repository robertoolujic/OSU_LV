import numpy as  np
import matplotlib.pyplot as plt

file = open("data.csv", "r")
linecount = len(file.readlines())-1

data = np.zeros((linecount, 3), np.float32)

linecount=0
file.seek(0)
info = file.readline().split(",")
for line in file:
    words = line.split(",")
    data[linecount][0]=words[0]
    data[linecount][1]=words[1]
    data[linecount][2]=words[2]
    linecount+=1
print("odrađeno je",linecount,"mjerenja.")
visina = data[:,1]
print("Minimalna visina:",visina.min())
print("Maksimalna visina:",visina.max())
print("Prosječna visina:",visina.mean())
Fvisina= data[data[:,0]==0, 1]
print("Minimalna visina žena:",Fvisina.min())
print("Maksimalna visina žena:",Fvisina.max())
print("Prosječna visina: žena",Fvisina.mean())

plt.scatter(data[:,2],data[:,1],marker =".")
plt.axis([0,250,0,250])
plt.xlabel("težina")
plt.ylabel("visina")
plt.title("omjer tezine i visine")
plt.show()

plt.scatter(data[::50,2],data[::50,1],marker =".")
plt.axis([0,250,0,250])
plt.xlabel("težina")
plt.ylabel("visina")
plt.title("omjer tezine i visine")
plt.show()
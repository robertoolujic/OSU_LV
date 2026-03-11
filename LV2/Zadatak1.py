import numpy as np
import matplotlib.pyplot as plt

line1=np.array([[1,3],[1,1]])
line2=np.array([[1,2],[1,2]])
line3=np.array([[2,3],[2,2]])
line4=np.array([[3,3],[1,2]])
plt.plot (line1[0],line1[1], 'r',
          line2[0],line2[1], 'g',
          line3[0],line3[1], 'b',
          line4[0],line4[1], 'k',
          linewidth =2 , marker =".", markersize =10 )
plt.axis([0,4,0,4])
plt.xlabel("x os")
plt.ylabel("y os")
plt.title("primjer")
plt.show()
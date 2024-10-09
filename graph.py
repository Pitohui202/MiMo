import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import csv
#def animate(i):
print(len([9]))
with open("reward_long_data.csv") as file:
    reader=csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
    rows = list(reader)
    print(rows)
    ar=rows[0][1:]
    #if len(rows)>-1:
        #ar=rows[0][1:]
        #ar2=rows[1][1:]
        #ar3=rows[2][1:]
        #ar3=np.fromstring(ar3[0],dtype=float,sep=',')
        #ar4=rows[3][1:]
        #ar5=rows[4][1:]
        #ar6=rows[5][1:]
        #ar4=np.fromstring(ar4[0],dtype=float,sep=',')
        #aar,aar2,aar3,aar4,aar5,aar6=[],[],[],[],[],[]
        #for i in range(len(ar)):
            #aar.append(sum(ar[:i])/(i+1))
        #for i in range(len(ar2)):
            #aar2.append(sum(ar2[:i])/(i+1))
        #for i in range(len(ar3)):
            #aar3.append(sum(ar3[:i])/(i+1))
        #for i in range(len(ar4)):
            #aar4.append(sum(ar4[:i])/(i+1))
        #for i in range(len(ar5)):
            #.append(sum(ar5[:i])/(i+1))
        #for i in range(len(ar6)):
            #aar6.append(sum(ar5[:i])/(i+1))
plt.clf()
plt.subplot(1,1,1)
plt.plot(ar,color="#AD92F5")
plt.ylabel("Average Reward")
#plt.plot(aar2,color="#09F67E")
#plt.plot(aar3,color="#FFF200")
#plt.plot(aar4,color="#F10E37")
#plt.plot(aar5,color="#9B009F")
#plt.plot(aar6,color="#016064")
#ani=FuncAnimation(plt.gcf(),animate,interval=1000)
plt.tight_layout()
plt.show()


import math

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import csv
#def animate(i):
with open("graph_data.csv") as file:
    reader=csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
    rows = list(reader)
    ar=rows[0]
    min=min(ar)
    max=1000
    ptt=sorted(ar)
    print(ptt)
    print(ar)
    for i in range(1,4):
        print(ar.index(ptt[i]))
    for i in range(1,4):
        print(ar.index(ptt[-i]))
    colors=['#ff0000','#ff8000','#ffff00','#80ff00','#2db300','#008080','#00ffff','#008080','#390df7','#7c788c','#bf00ff','#660066']
    currposx=-0.25
    currposy=0
    if len(rows)>-1:
        plt.clf()
        fig=plt.figure()
        ax = fig.add_subplot(111)
        i=0
        for a in ar:
            i+=1
            colorid=math.floor(((a-min)/(max-min))*9)
            if colorid>9:
                colorid=9
            rect=matplotlib.patches.Rectangle((currposx,currposy),0.01,0.01,color=colors[colorid])
            ax.add_patch(rect)
            if i%55==0:
                currposx=-0.25
                currposy+=0.01
            else:
                currposx+=0.01
        plt.xlabel("Initial ball x position")
        plt.ylabel("Initial ball y position")
        plt.gca().set_xlim([-0.5,0.5])
        plt.tight_layout()
        #plt.show()


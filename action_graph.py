import math
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import pylab
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import csv
#def animate(i):
def klassengenerierung(input):
    stri=""
    strarr=[]
    detect=False
    detect2=False
    detect3=False
    detect4=False
    for i in input:
        if i=="m" and not detect4:
            detect=True
        elif i=="e" and detect and not detect4:
            detect2=True
        elif i=="=" and detect2 and not detect4:
            detect3=True
        elif i=='"' and detect3 and not detect4:
            detect4=True
        elif detect4:
            if not i=='"':
                stri+=i
            else:
                strarr.append(stri)
                stri=""
                detect=False
                detect2=False
                detect3=False
                detect4=False
        else:
            detect=False
            detect2=False
            detect3=False
            detect4=False
    return strarr

avgload=[]
avgloads=[]
maxloadcount=[]
maxloadcounts=[]
tottime=[]
actuator_names=['act:hip_bend', 'act:hip_twist', 'act:hip_lean', 'act:head_swivel', 'act:head_tilt', 'act:head_tilt_side', 'act:left_eye_horizontal', 'act:left_eye_vertical', 'act:left_eye_torsional', 'act:right_eye_horizontal', 'act:right_eye_vertical', 'act:right_eye_torsional', 'act:right_shoulder_horizontal', 'act:right_shoulder_abduction', 'act:right_shoulder_internal', 'act:right_elbow', 'act:right_wrist_rotation', 'act:right_wrist_flexion', 'act:right_wrist_ulnar', 'act:right_fingers', 'act:left_shoulder_horizontal', 'act:left_shoulder_abduction', 'act:left_shoulder_internal', 'act:left_elbow', 'act:left_wrist_rotation', 'act:left_wrist_flexion', 'act:left_wrist_ulnar', 'act:left_fingers', 'act:right_hip_flex', 'act:right_hip_abduction', 'act:right_hip_rotation', 'act:right_knee', 'act:right_foot_flexion', 'act:right_foot_inversion', 'act:right_foot_rotation', 'act:right_toes', 'act:left_hip_flex', 'act:left_hip_abduction', 'act:left_hip_rotation', 'act:left_knee', 'act:left_foot_flexion', 'act:left_foot_inversion', 'act:left_foot_rotation', 'act:left_toes']
for an in actuator_names:
    maxloadcounts.append([])
    avgloads.append([])
fig,ax=plt.subplots()
for a in range(19):
    avgload=[]
    maxloadcount=[]
    actions=np.load('actions_{}.npy'.format(a))
    for a in actions[0]:
        avgload.append(0)
        maxloadcount.append(0)
    atc=0
    for at in actions:
        ac=0
        atc+=1
        for at2 in at:
            avgload[ac]+=abs(at2)
            if at2==3 or at2==-3:
                maxloadcount[ac]+=1
            ac+=1
    #ax.plot(np.transpose(actions))
    #plt.show()
    avgload=[a/atc for a in avgload]
    maxloadcount=[(m/atc)*100 for m in maxloadcount]
    for mlc in range(len(maxloadcounts)):
        maxloadcounts[mlc].append(maxloadcount[mlc])
    for alc in range(len(avgloads)):
        avgloads[alc].append(avgload[alc])
    tottime.append(atc)
maxloaddict=dict(zip(actuator_names,maxloadcounts))
avgloaddict=dict(zip(actuator_names,avgloads))
#sortmaxloaddict={key:value for key,value in sorted(maxloaddict.items(),key=itemgetter(1))}
#sortavgloaddict={key:value for key,value in sorted(avgloaddict.items(),key=itemgetter(1))}
plotmaxloadmin=[]
plotmaxloadmax=[]
plotmaxloadavg=[]
plotmaxloadtrueavg=[]
plotavgloadmin=[]
plotavgloadmax=[]
plotavgloadavg=[]
plotavgloadtrueavg=[]
for mlc in maxloadcounts:
    plotmaxloadmin.append(min(mlc))
    plotmaxloadmax.append(max(mlc)-sum(mlc)/float(len(mlc)))
    plotmaxloadavg.append(sum(mlc)/float(len(mlc))-min(mlc))
    plotmaxloadtrueavg.append(sum(mlc)/float(len(mlc)))
for alc in avgloads:
    plotavgloadmin.append(min(alc))
    plotavgloadmax.append(max(alc)-sum(alc)/float(len(alc)))
    plotavgloadavg.append(sum(alc)/float(len(alc))-min(alc))
    plotavgloadtrueavg.append(sum(alc)/float(len(alc)))

#plt.subplot(2,1,1)
#plt.yticks(color="w")
#frame=pylab.gca()
#plt.rcParams["font.size"]=6
#frame.axes.get_yaxis().set_visible(False)
#lbar=plt.barh(actuator_names,plotmaxloadmax,left=plotmaxloadtrueavg,color="#ff6600")
#plt.barh(actuator_names,plotmaxloadavg,left=plotmaxloadmin,color="#cc0000")
#plt.xlim([0,100])
#plt.margins(0,0)
#plt.bar_label(lbar,padding=2,labels=actuator_names)
#plt.title("% of time actuator is maxed out")
plt.subplot(1,1,1)
frame=pylab.gca()
frame.axes.get_yaxis().set_visible(False)
plt.ylabel=actuator_names
lbar2=plt.barh(actuator_names,plotavgloadmax,left=plotavgloadtrueavg,color="#00cc99")
plt.barh(actuator_names,plotavgloadavg,left=plotavgloadmin,color="#006600")
plt.xlim([0,3])
plt.margins(0,0)
plt.bar_label(lbar2,padding=2,labels=actuator_names)
plt.rcParams["font.size"]=6
plt.title("Average absolute actuator value")
plt.show()
print(maxloaddict)
print(avgloaddict)
print(tottime)
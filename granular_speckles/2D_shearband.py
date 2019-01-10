import numpy as np
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

def linear(x,m,b):
    return m*x+b

def mobilmean(block_size,array):
    return map(lambda x:np.mean(array[x:x+block_size]),range(len(array)-block_size))

matrices = np.load("dati/allCorrTimeMap_MinLoc_Sp10_NOCUTOFF.npy")
data  = [0,1,2,3,4,6,7]
print(data)
args = str(data).strip("[").strip("]").replace(" ","")
os.system("python2 trovaShearBand.py "+args.strip())
jumps_values = np.loadtxt(open("jump_values.dat","r"))

vel = np.array([0.05, 0.07, 0.1, 0.3, 0.5, 0.7,1.2,1.5])
speeds = vel[data]
matrices = matrices[data]
colors = cm.rainbow(np.linspace(0, 1, len(data)))
x_shift = 10
mobmean = 5

average_matrices = np.mean(matrices, axis=2)
average_matrices = np.array([list(mobilmean(mobmean,average_matrices[h,:]))
                                  for h in range(len(speeds))])
matrices = average_matrices.T/np.max(average_matrices,axis=1)
fig0, ax1 =plt.subplots(1,1)
ax1.pcolor(matrices)
c="r"
# plt.scatter([ x + 0.5 for x in range(len(data))], , c="r")
ax1.scatter([ x + 0.5 for x in range(len(data))],jumps_values[0,:],color=c)
ax1.scatter([ x + 0.5 for x in range(len(data))], jumps_values[0,:]+jumps_values[1,:]*0.5,color=c, marker=".")
ax1.scatter([ x + 0.5 for x in range(len(data))], jumps_values[0,:]-jumps_values[1,:]*0.5,color=c, marker=".")
fig0.savefig("CorrelationMap.pdf", ext="pdf", dpi=300)

times =([],[],[])
last_band= int(jumps_values[0,-1]+jumps_values[1,-1]*0.5)
first_band= int(jumps_values[0,0]+jumps_values[1,0]*0.5)
for enum in range(len(data)):
    # last_band= int(jumps_values[0,enum]+jumps_values[1,enum]*0.5)
    # first_band= int(jumps_values[0,enum]-jumps_values[1,enum]*0.5)
    # start_space= int(jumps_values[0,enum]+jumps_values[1,enum]*0.5)
    times[0].append(np.mean(matrices[:last_band,enum]))
    times[1].append(np.mean(matrices[last_band:first_band,enum]))
    times[2].append(np.mean(matrices[first_band:,enum]))
print(times)


color=["b","r","g"]
fig, ax2 = plt.subplots()
LABEL=["solid","shear band","fluid"]
for index in range(len(times)):
    ax2.scatter(speeds,times[index], c=color[index], marker="o", label=LABEL[index])
    popt, pvar = curve_fit(linear, speeds, times[index])
    ax2.plot(speeds,[x*popt[0]+popt[1] for x in speeds],c=color[index])
ax2.legend()

# fig.savefig("CorrelationTime.pdf", ext="pdf", dpi=300)
plt.show()


# clean_matrices = [list(mobilmean(mobmean,x)) for x in average_matrices]

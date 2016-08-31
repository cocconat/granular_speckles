import numpy as np
from coarsing import *
def importMatrix(args):
    fullPath="results/"+args.image_folder+"/full_correlation/"
    togePath="results/"+args.image_folder+"/all_together/"
    return np.load(fullPath+args.block_size+".npy"), np.load(togePath+args.block_size+".npy")

def plotCorrelation(data):
    plt.plot(range(len(data)),data)


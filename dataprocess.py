import numpy as np
from coarsing import *
def importMatrix(args):
    fullPath="results/"+args.image_folder+"/full_correlation/"
    togePath="results/"+args.image_folder+"/all_together/"
    return np.load(fullPath+args.block_size+".npy"), np.load(togePath+args.block_size+".npy")

def plotCorrelation(data):
    plt.plot(range(len(data)),data)

def space_correlate(matrix,shift):
    def column_correlation(matrix,shift,row):
            return np.sum(matrix[row,:-shift]*matrix[row,shift:])*1./(matrix.shape[1]-shift)
    if shift==0:
        return map( lambda x: np.mean(np.vectorize(np.square)(matrix[x,:])), range(matrix.shape[0]) )
    else:
        return map(lambda x: column_correlation(matrix,shift,x),range(matrix.shape[0]))

def space_correlation(matrix,time):
    mean=np.mean(matrix[:,:,time],axis=1)**2
    func=partial(space_correlate,matrix[:,:,time])
    pool=multiprocessing.Pool(5)
    correlation=np.array(pool.map(func,range(0,matrix.shape[1]/2)))
    return (correlation/mean/mean.shape[0]).transpose()


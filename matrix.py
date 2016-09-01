import math
import numpy as np
import os
from scipy import misc as smc
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing
import time, itertools
from functools import partial

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts)
        return result

    return timed



def spaceAveragedCorr(a):
    return np.mean(a[:,:,np.nonzero(a)[2]],axis=2)


def pixelStory(mat,raw,col):
    return mat[raw,col,:]


def plotMatrix(matrix,pause=20):
    '''
    plot a frame of time sequence matrix
    '''
    plot=Image.fromarray(matrix[:,:])
    plot.show()
    plt.pause(pause)
    plt.clf()

def plotStory(mat,raw,col,plt):
    '''
    plot the evolution of the value of a pixel during time.
    '''
    plt.plot(range(mat.shape[2]),mat[raw,col,:])
    plt.scatter(range(mat.shape[2]),mat[raw,col,:])
    plt.xlim(0,mat.shape[2])
    #plt.ylim(np.min(self.mat),np.max(self.mat))

    return plt


def histogram(self,raw,col,plt,mat,distribution=None,bins=10):
    '''
    plot the distribution of values of pixel along time evolution
    '''
    self.initMat(mat)
    if not distribution:
        distribution=self.pixelStory(raw,col)
    hist,bins = np.histogram(distribution, bins=bins)
    hist=[i*1./np.sum(hist) for i in hist]
    bin_width = 1
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=bin_width)
    plt.xlim(0,10)
    plt.ylim(0,1.5)
    return plt

def timeVariance(mat):
    return np.var(mat,axis=2)

def frame(mat,time):
    return mat[:,:,time]

def arrayVariance(mat,time):
	array=mat[:,:,time]
	n_elem=reduce(lambda x,y: x*y,array.shape)
	a=array.reshape(n_elem)
	variance=reduce(lambda x,y: x+y*y,a)
	return math.sqrt(variance)

def spaceVariance(mat):
	pool=multiprocessing.Pool(5)
	f=partial(arrayVariance,mat)
        return np.array(pool.map(f,xrange(mat.shape[-1])))

def correlate(array,shift):
    if shift==0: return np.sum(array*array)*1./(len(array))
    return np.sum(array[:-shift]*array[shift:])*1./(len(array)-shift)

def single_correlation(time,timepixel):
    mean=np.mean(timepixel)
    var=np.var(timepixel)
    if var > 100:
    #timepixel=timepixel-mean
            correlation=np.asarray(map(lambda x: correlate(timepixel-mean,x),range(0,time)))
            return correlation/var
    else:
        return np.zeros((time))

@timeit
def correlation(time,mat):
    '''
	measure correlation of a matrix for a maximum time (time)
	it returns:
        corrMatrix, np.nan_to_num(spaceAveragedCorr(time,arrays))
    '''
    hmat=mat.shape[0]
    lmat=mat.shape[1]
    pool=multiprocessing.Pool(processes=5)
    print "start correlation stack"
    coupleIter=(mat[r,c,:] for r,c in \
                itertools.product(range(mat.shape[0]),range(mat.shape[1])))
    f=partial(single_correlation,time)
    arrays=pool.map(f,coupleIter)
#        for i in range(self.mat.shape[0]):
#            a.append([self.single_correlation(i,j,time) for j in range(self.mat.shape[1])]):
#                print "sono al pixel {} {} di {}".format(i,j,self.mat.shape)
#                a.append(self.single_correlation(i,j,time))
    corrMatrix=np.zeros((hmat,lmat,int(time)))
    coupleIter= itertools.product(range(mat.shape[0]),range(mat.shape[1]))
    for r,c in coupleIter:
        corrMatrix[r,c,:]=arrays.pop(0)
    print "end correlation stack"
    return corrMatrix, np.nan_to_num(spaceAveragedCorr(corrMatrix))


class GetMatrix(object):
    def __init__(self,args):
        '''
        you need this class to import image,
        it process and give you a 3 dimensional array,
        2d-space and time
        '''
        self.folder=args.image_folder
        self.black=args.black
        self.mat=None

    def importImages(self):
        for fname in sorted(os.listdir(os.getcwd()+"/"+self.folder)):
            if fname.endswith(".png"):
                print fname
                yield os.path.join(os.getcwd()+"/"+self.folder, fname)


    @staticmethod
    def treesholdMatrix(matrix):
        '''
        this function flatten grey scale over black and white
        you decide by the option -bn
        '''
        matrix[matrix < 4]=0
        matrix[matrix >= 4]=255
        return matrix


    @staticmethod
    def imagesToArray(image_path):
        return smc.imread(image_path, flatten=True)


    @property
    def matrix(self):
        '''
        this is your matrix!!
        '''

        if self.mat==None:
            self.mat= self.stackImages()

        if self.black:
            self.mat=self.treesholdMatrix(self.mat)
        #self.normalize()
        return self.mat

    @timeit
    def stackImages(self):
        def countImages():
            count=0
            for fname in sorted(os.listdir(os.getcwd()+"/"+self.folder)):
                if fname.endswith(".png"):
                    count +=1
            return count
        matShape=firstImage.shape[0],firstImage.shape[1],countImages())
        print matShape
        array=np.zeros(matShape)
        for count, image in enumerate(self.importImages()):
            array[:,:,count]=self.imagesToArray(image)
        return array[:,:,:count]

    def normalize(self):
        self.mat=self.mat/np.max(self.mat)




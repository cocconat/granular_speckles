import math
import scipy
import numpy as np
import sys,os
from scipy import misc as smc
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import plot, ion, show
import multiprocessing
import time, itertools
from multiprocessing import Pool
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
    
        
        
def spaceAveragedCorr(time,arrays):
    notNone=(array for array in arrays if array.all())
    correlationList=np.stack(notNone,axis=0)
    return np.mean(correlationList,axis=0)
            

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

def plotFrame(frame,plt,matrix=None):
    '''
    plot a frame of time sequence matrix
    '''
    plot=Image.fromarray(matrix[:,:,frame])
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

def single_correlation(time,mat,position):
	raw,col=position
	timepixel=mat[raw,col,:]
	#var=np.var(timepixel)            
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
    #correlationMatrix=np.zeros(self.mat.shape)
 
    pool=multiprocessing.Pool(processes=5)
    print "start correlation stack" 
    coupleIter= itertools.product(range(mat.shape[0]),range(mat.shape[1]))
    f=partial(single_correlation,time,mat)
    arrays=pool.map(f,coupleIter)

#        for i in range(self.mat.shape[0]):
#            a.append([self.single_correlation(i,j,time) for j in range(self.mat.shape[1])]):
#                print "sono al pixel {} {} di {}".format(i,j,self.mat.shape)
#                a.append(self.single_correlation(i,j,time))         
    corrMatrix=np.stack(arrays,axis=0).reshape(hmat,lmat,int(time))
    print "end correlation stack" 
    return corrMatrix, np.nan_to_num(spaceAveragedCorr(time,arrays))

        
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
        return smc.imread(image_path, flatten=False, mode='L')


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
        for num, image in enumerate(self.importImages()):
            if num==0:
                array=self.imagesToArray(image)
            else:
                array=np.dstack((array,self.imagesToArray(image)))
        return 255-array
    
    def normalize(self):
        self.mat=self.mat/np.max(self.mat)
        
        
    

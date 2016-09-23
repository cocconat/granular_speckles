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

def corrTimeMap(mat):
	'''
	return the matrix of correlation time. Each entries is the
	correlation time associated to the evolution of the pixel with that
	coordinate.
	'''

	maxtime=mat.shape[2]

	def chooseFunc(a):
		if np.all(a==0):
			return maxtime

		else:
			return np.argmax(a<np.max(a)/2.)


	hmat=mat.shape[0]
	lmat=mat.shape[1]


	mat=mat.reshape(hmat*lmat,mat.shape[2])
	mat=np.asarray(map(lambda x: chooseFunc(x), mat))
	return mat.reshape(hmat,lmat)

def corrTimeMapEvolution(mat,interval,finalTime):
	'''
	needs a matrix of correlation function in input
	give the evolution of the correlation time map. Each temporal step
	is defined by interval argument.
	'''

	evolution=[]
	for i in range(0,finalTime/interval):
		a=correlation(interval,mat[:,:,i*interval:])
		evolution.append(corrTimeMap(a[0]))
	out=np.asarray(evolution)
	out=np.swapaxes(out,0,2)
	out=np.swapaxes(out,0,1)
	return 	out


def spaceAveragedCorr(mat):
    def non_zero_iter(a):
        for b in range(mat.shape[1]):
            if np.mean(mat[a,b,:])!=0:
                yield mat[a,b,:]
    myarray=np.zeros((mat.shape[0],mat.shape[2]))
    for a in range (mat.shape[0]):
        myarray[a,:]=np.mean(np.array([c for c in non_zero_iter(a)]),axis=0)
    print "#######################",mat.shape,myarray.shape

    return myarray



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
    if shift==0: return np.sum(array*1./(len(array))*array)
    return np.sum(array[:-shift]*1./(len(array)-shift)*array[shift:])
def single_correlation(time,cutoff,timepixel):
    mean=np.mean(timepixel)
    var=np.var(timepixel)
    if var > cutoff:
    #timepixel=timepixel-mean
            correlation=np.asarray(map(lambda x: correlate(timepixel-mean,x),range(0,time)))
            return correlation/var
    else:
        return np.zeros((time))

def other_correlation(time,cutoff,timepixel):
    mean=np.mean(timepixel)
    var=np.var(timepixel)
    if var > cutoff:
    #timepixel=timepixel-mean
            correlation=np.asarray(map(lambda x: correlate(timepixel,x),range(0,time)))
            return correlation/mean/mean
    else:
        return np.zeros((time))


@timeit
def correlation(time,mat,cutoff=5,function='chinasucks'):
    '''
	measure correlation of a matrix for a maximum time (time)
	it returns:
        corrMatrix, np.nan_to_num(spaceAveragedCorr(time,arrays))
    '''
    hmat=mat.shape[0]
    lmat=mat.shape[1]
    pool=multiprocessing.Pool(processes=7)
    print "start correlation stack"
    coupleIter=(mat[r,c,:] for r,c in itertools.product(range(mat.shape[0]),range(mat.shape[1])))
    if function=="chinasucks":
        print "chine science sucks"
        f=partial(china_correlation,time,cutoff)
    else:
        f=partial(single_correlation,time,cutoff)
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
        self.resize=args.resize

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
    def plotOne(image_path):
        image=smc.imread(image_path, flatten=True)
        plt.pcolor(image)
        plt.show()

    def imagesToArray(self,image_path):
        if self.resize:
            a = self.resize
            return smc.imread(image_path, mode='L')[a[0]:a[1],a[2]:a[3]]
        else:
            return smc.imread(image_path, mode='L')

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
        firstImage=self.imagesToArray(self.importImages().next())
        matShape=firstImage.shape[0],firstImage.shape[1],countImages()
        print matShape
        array=np.zeros(matShape)
        array[:,:,0]=np.array(firstImage,dtype=np.int8)
        for count, image in enumerate(self.importImages()):
            array[:,:,count]=np.array(self.imagesToArray(image),dtype=np.int8)
        return np.array(array[:,:,:count],dtype=np.int8)
    def normalize(self):
        self.mat=self.mat/np.max(self.mat)




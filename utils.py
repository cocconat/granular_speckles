from matrix import *
from matrix import GetMatrix
from coarsing import *
from matplotlib.pyplot import plot, ion, show
from video import *
import matplotlib.pyplot as plt

def timeit(method):
    '''this decorator measures operation time'''

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts)
        return result
    return timed


def explore(time_serie,start_col=0,start_raw=0,end_col=None,end_raw=None,pause=0):
    array=time_serie
    if not end_col:
        end_col=array.shape[1]
    if not end_raw:
        end_raw=array.shape[0]
    #varianceMatrix=time_serie.timeVariance()
    for i in reversed(range(start_raw,end_raw)):
        for j in reversed(range(start_col,end_col)):
            if pause:
                plt.figure(1)
                plt.title("here we are: {} ".format([i,j]))
                plt.subplot(211)
                plotStory(time_serie,i,j,plt=plt)
                plt.subplot(212)
                #plt.pcolor(varianceMatrix,cmap=plt.cm.Oranges)
                plt.scatter(j,i)
                plt.ylim((0,array.shape[0]))
                plt.xlim((0,array.shape[1]))
                plt.pause(pause)
                plt.clf()
            else:
                func(i,j)

def explore_time(time_serie,pause=0):
    '''explore time is a'''
    array=time_serie
    for i in range(array.shape[2]):
        if pause:
            plt.figure(1)
            plt.title("time is {}".format(i))
            plt.pcolor(array[:,:,i])
            plt.pause(pause)
            plt.clf()

#    plt.figure(1)
    #plt.annotate(" average is {}".format(center), xy=(0.5, 0.5), xycoords='axes fraction',
             #horizontalalignment='center', verticalalignment='center')

def timeDecorrelation(time_serie,path,args):
    if len(args)==1:
        args=a
        rgs[0]
    results=[coarseTime(time_serie,block) for block in args]
    deco=   [np.mean(TimeVariance(result)) for result in results]
    for enum, result in enumerate(results):
    	np.save(path+"/timeCooarse_"+str(enum), result)
    np.savetxt(path+"/time_deco.dat", deco)

#   coarseTime
def space(block_size,time_serie):
    time_serie=coarseSpace(time_serie,block_size)
    timeVariance=timeVariance(time_serie)
    spaceVariance=spaceVariance(time_serie)
    return time_serie, timeVariance, spaceVariance

def time (time_size,time_serie):
    time_serie=coarseTime(time_serie,time_size)
    timeVariance=timeVariance(time_serie)
    spaceVariance=spaceVariance(time_serie)
    return time_serie, timeVariance, spaceVariance

def blockIteration(timeserie,*args):
    if len(args)==1:
        args=args[0]
    for block in args:
        yield block,coarseSpace(block,timeserie)




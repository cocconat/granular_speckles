#!/usr/bin/python2.7
'''
trhda
'''
import numpy as np
import os
import matplotlib.pyplot as plt
from matrix import *
from coarsing import *
from matplotlib.pyplot import plot, ion, show
from video import *
import dataprocess
import histogram,fit,plati,utils
def getParser():
    import argparse
    ap = argparse.ArgumentParser("this program measure the correlation of a many frame video, then its  possible to performa a coarse graining over time or space dimension, it's a 2d project but it can be easily performed in 3d with little modification. \nThe algorithm implemented are top level for each stage and standard is parallelized over 5 procs. \n Needed library is Numpy, Scipy, multiprocessing , matplotlib, if you want to frame the video also opencv2 is needed")
    ap.add_argument("-bn", "--black", help="set black and white or greys",
                    action="store_true")
    ap.add_argument("-f", "--image_folder", help = "folder for png images to process")
    ap.add_argument("-b", "--block_size",help = "apply coarse graining, this is dimension for image reduction")
    ap.add_argument("-m", "--matrix_story", help = "colorful image for pixel evolution",action="store_true")
    ap.add_argument("-c", "--correlation", help = "measure the time correlation for each pixel of final matrix")
    ap.add_argument("-C", "--cutoff", help = "minimal variance ast to not be considered noise")
    ap.add_argument("-t", "--takealook", help = "colorful image for pixel evolution",action="store_true")
    ap.add_argument("-i", "--file_import", help = "folder for png images to process",action="store_true")
    ap.add_argument("-T", "--timecoarse", help = "time carsing and sigma analysis",action="store_true")
    ap.add_argument("-S", "--spacecoarse", help = "space coarsing and correlation matrix",action="store_true")
    ap.add_argument("-F", "--cor_function", help = "chose which correlation function you're using, the china or european styla",action="store_true")
    ap.add_argument("-v", "--videofile", help ="path for video file, long time required")
    ap.add_argument("-r", "--resize", type=int, nargs=4, help ="resize the image to center the lightspot")
    return ap

ion()

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



def main():
    args=getParser().parse_args()
    time_serie=None
    if not args.videofile==None:
        print "video frame is: ", get_frame_rate(args)
        print "so many frame {} to folder {}".format(videoToFrame(args),args.image_folder)

    if args.file_import:
        time_serie=2*np.load(args.image_folder+"/image_matrix"+".npy")
        if not time_serie==None :
            print "IMAGE-MATRIX  HAS BEEN CORRECTLY uploaded"
        else:
            raise "not correctly uploaded!!!"
    else:
        print "image acquisition in process..."
        time_serie=GetMatrix(args).matrix
        np.save(args.image_folder+"/image_matrix",time_serie)
        print "the starting matrix has shapes: {}".format(time_serie.shape)


    if args.takealook:
        #varianceMatrix=timeVariance(time_serie)
        #print "la somma della varianza della matrice di input e' {}".format(varianceMatrix.mean())
        explore_time(time_serie,pause=0.00001)

    #plotMatrix(varianceMatrix)
    #time_serie.plotFrame(19)
    #time_serie.plotFrame(frame=21,plt=plt).show()
    #explore(new_time_serie,'histogram',pause=0.00001)
    #explore(new_time_serie,'histogram',85,60,150,80,pause=0.00001)


    if not args.block_size:
	block_size=str(1)
    else: block_size=args.block_size

    resultPath="results"+"/"+args.image_folder
    if not os.path.exists(resultPath):
    	os.makedirs(resultPath)

    if args.timecoarse:
       #saving time decorrelation matrix
        timeDecorrelation(mcoarseSpace(time_serie,2),resultPath,range(1,50,1))

    if args.spacecoarse:
        fullPath="results/"+args.image_folder+"/full_correlation/"
        togePath="results/"+args.image_folder+"/all_together/"
        corrtime=args.correlation or 300
       #saving correlation for various coarse graining
        for block, reducedSerie in blockIteration(time_serie,[2,3,4,5,6,10,15,20,25]):
            full_core,all_together=correlation(int(corrtime),reducedSerie)
            mezza=dataprocess.mezzaltezza(all_together)
            np.save(togePath+block_size,all_togethe)
            np.save(fullPath+block_size,full_core)
            np.save(togePath+block_size+"mezzaltezza",mezza)
            del reducedSerie

    if args.matrix_story:
        explore_time(new_time_serie,'plotFrame',pause=0.00001)

    if args.correlation:
        full_core,all_together=correlation(int(args.correlation),time_serie,cutoff=args.cutoff,function=args.cor_function)
        #plt.plot(range(len(all_togheter)),all_togheter)
        #time_serie.plotStory(80,120,pause=20)
        mezza=dataprocess.mezzaltezza(all_together)
        fullPath="results/"+args.image_folder+"/full_correlation/"
        togePath="results/"+args.image_folder+"/all_together/"
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        if not os.path.exists(togePath):
            os.makedirs(togePath)
        np.save(fullPath+block_size,full_core)
        np.save(togePath+block_size,all_together)
        np.save(togePath+block_size+"mezzaltezza",mezza)

    #lista=new_time_serie.single_correlation(50,15,50)
    globals().update(locals())

#    explore(new_time_serie,0.00001)
if __name__=="__main__":
        main()








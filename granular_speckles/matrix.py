#! /usr/bin/python
# -*- coding: utf-8 -*-
# This file belongs to DWGranularSpeckles project.
# The software is realeased with MIT license.
import math
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing
import itertools
from functools import partial
from granular_speckles.utils import timeit


def corrTimeMap(matrix):
    '''
    return the matrix of correlation time. Each entries is the
    correlation time associated to the evolution of the pixel with that
    coordinate.
    '''

    maxtime = matrix.shape[2]

    def chooseFunc(a):
        if np.all(a == 0):
            return maxtime

        else:
            return np.argmax(a < np.max(a)/2.)

    hmat = matrix.shape[0]
    lmat = matrix.shape[1]

    matrix = matrix.reshape(hmat * lmat, matrix.shape[2])
    matrix = np.asarray(map(lambda x: chooseFunc(x), matrix))
    return matrix.reshape(hmat, lmat)


def corrTimeMapEvolution(matrix, interval, finalTime):
    '''
    needs a matrix of correlation function in input
    give the evolution of the correlation time map. Each temporal step
    is defined by interval argument.
    '''

    evolution = []
    for i in range(0, finalTime/interval):
        a = correlation(interval, matrix[:, :, i * interval:])
        evolution.append(corrTimeMap(a[0]))
    out = np.asarray(evolution)
    out = np.swapaxes(out, 0, 2)
    out = np.swapaxes(out, 0, 1)
    return out


def spaceAveragedCorr(mat):
    def non_zero_pixels(height) :
        pixels=[]
        for pixel in range(mat.shape[1]):
            if np.var(mat[height, pixel, :]) != 0:
                pixels.append(pixel)
        if not pixels:
            return np.zeros((1,1))
        else:
            return mat[[height]*len(pixels),pixels,:]


    myarray = np.zeros((mat.shape[0], mat.shape[2]))
    for a in range(mat.shape[0]):
        myarray[a, :] = np.mean(non_zero_pixels(a), axis=0)
    print ("matrix shape {}, myarray shape {}\
           ".format(mat.shape, myarray.shape))
    return myarray


def pixelStory(mat, raw, col):
    return mat[raw, col, :]


def plotMatrix(matrix, pause=20):
    '''
    plot a frame of time sequence matrix
    '''
    plot = plt.pcolor(matrix[:, :])
    plot.show()
    plt.pause(pause)
    plt.clf()


def plotStory(mat, raw, col, plt):
    '''
    plot the evolution of the value of a pixel during time.
    '''
    plt.plot(range(mat.shape[2]), mat[raw, col, :])
    plt.scatter(range(mat.shape[2]), mat[raw, col, :])
    plt.xlim(0, mat.shape[2])
    # plt.ylim(np.min(self.mat),np.max(self.mat))

    return plt


def histogram(self, raw, col, plt, mat, distribution=None, bins=10):
    '''
    plot the distribution of values of pixel along time evolution
    '''
    self.initMat(mat)
    if not distribution:
        distribution = self.pixelStory(raw, col)
    hist, bins = np.histogram(distribution, bins=bins)
    hist = [i*1./np.sum(hist) for i in hist]
    bin_width = 1
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=bin_width)
    plt.xlim(0, 10)
    plt.ylim(0, 1.5)
    return plt


def timeVariance(mat):
    return np.var(mat, axis=2)


def frame(mat, time):
    return mat[:, :, time]


def arrayVariance(mat, time):
    array = mat[:, :, time]
    n_elem = reduce(lambda x, y: x*y, array.shape)
    a = array.reshape(n_elem)
    variance = reduce(lambda x, y: x+y*y, a)
    return math.sqrt(variance)


def spaceVariance(mat):
    pool = multiprocessing.Pool(5)
    f = partial(arrayVariance, mat)
    return np.array(pool.map(f, xrange(mat.shape[-1])))


def correlate(array, shift):
    if shift == 0:
        return np.sum(array*1./(len(array))*array)
    return np.sum(array[:-shift]*1./(len(array)-shift)*array[shift:])


def single_correlation(time, cutoff, timepixel):
    mean = np.mean(timepixel)
    var = np.var(timepixel)
    if var > cutoff:
        # timepixel=timepixel-mean
        correlation = np.asarray(
            map(lambda x: correlate(timepixel-mean, x), range(0, time)))
        return correlation/var
    else:
        return np.zeros((time))


def other_correlation(time, cutoff, timepixel):
    mean = np.mean(timepixel)
    var = np.var(timepixel)
    if var > cutoff:
        # timepixel=timepixel-mean
        correlation = np.asarray(
            map(lambda x: correlate(timepixel, x), range(0, time)))
        return correlation/mean/mean
    else:
        return np.zeros((time))


@timeit
def correlation(time, matrix, cutoff=5, function='chinasucks'):
    '''
    Measure correlation of a matrix in third axis (time)

    The functions measure the correlation for all distances, from zero to time
    Parameters:
    ==========
    time: maximum time to measure
    matrix: matrix to process, type: np.ndarray(h,v,d)

    Returns:
    ========
    corrMatrix:
    np.nan_to_num(spaceAveragedCorr(time,arrays))
    '''
    hmat = matrix.shape[0]
    lmat = matrix.shape[1]
    pool = multiprocessing.Pool(processes=2)
    # print ("start correlation stack")
    coupleIter = (matrix[r, c, :] for r, c in itertools.product(
        range(hmat), range(lmat)))
    f = partial(single_correlation, time, cutoff)
    arrays = pool.map(f, coupleIter)
#        for i in range(self.mat.shape[0]):
#            a.append([self.single_correlation(i,j,time) for j
#                                       n range(self.mat.shape[1])]):
#                print "sono al pixel {} {} di {}".format(i,j,self.mat.shape)
#                a.append(self.single_correlation(i,j,time))
    corrMatrix = np.zeros((hmat, lmat, int(time)))
    coupleIter = itertools.product(range(matrix.shape[0]), range(matrix.shape[1]))
    for r, c in coupleIter:
        corrMatrix[r, c, :] = arrays.pop(0)
    print ("end correlation stack")
    return corrMatrix, np.nan_to_num(spaceAveragedCorr(corrMatrix))


class GetMatrix(object):
    def __init__(self, image_folder, resize=None, black=None):
        '''
        you need this class to import image,
        it process and give you a 3 dimensional array,
        2d-space and time
        '''
        self.folder = image_folder
        self.black = black
        self.mat = None
        self.resize = resize

    def importImages(self):
        for fname in sorted(os.listdir(os.getcwd()+"/"+self.folder)):
            if fname.endswith(".png"):
                print (fname)
                yield os.path.join(os.getcwd()+"/"+self.folder, fname)

    @staticmethod
    def treesholdMatrix(matrix):
        '''
        this function flatten grey scale over black and white
        you decide by the option -bn
        '''
        matrix[matrix < 4] = 0
        matrix[matrix >= 4] = 255
        return matrix

    @staticmethod
    def plotOne(image_path):
        image = Image.open(image_path).convert('LA')
        plt.pcolor(image)
        plt.show()

    def imagesToArray(self, image_path):
        def matrix_from_image(image_path):
            return np.asarray(Image.open(image_path).convert('LA'))[:,:,0]
        if self.resize:
            a = self.resize
            # TODO ensure is getting image as 8byte array
            return matrix_from_image(image_path)[a[0]:a[1], a[2]:a[3]]
        else:
            return matrix_from_image(image_path)

    @property
    def matrix(self):
        '''

        '''

        if self.mat is None:
            self.mat = self.stackImages()

        if self.black:
            self.mat = self.treesholdMatrix(self.mat)
        # self.normalize()
        return self.mat

    @timeit
    def stackImages(self):
        def countImages():
            count = 0
            for fname in sorted(os.listdir(os.getcwd()+"/"+self.folder)):
                if fname.endswith(".png"):
                    count += 1
            return count
        firstImage = self.imagesToArray(next(self.importImages()))
        matShape = firstImage.shape[0], firstImage.shape[1], countImages()
        print (matShape)
        array = np.zeros(matShape)
        array[:, :, 0] = np.array(firstImage, dtype=np.int8)
        for count, image in enumerate(self.importImages()):
            array[:, :, count] = np.array(
                self.imagesToArray(image), dtype=np.int8)
        return np.array(array[:, :, :count], dtype=np.int8)

    def normalize(self):
        self.mat = self.mat/np.max(self.mat)

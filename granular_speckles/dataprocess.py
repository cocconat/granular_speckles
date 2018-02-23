#! /usr/bin/python
# -*- coding: utf-8 -*-

# This file belongs to DWGranularSpeckles project.
# The software is realeased with MIT license.

import functools
import multiprocessing

import numpy as np
from scipy.signal import argrelextrema


def importMatrix(args):
    fullPath = "results/"+args.image_folder+"/full_correlation/"
    togePath = "results/"+args.image_folder+"/all_together/"
    return np.load(fullPath+args.block_size+".npy"), \
        np.load(togePath+args.block_size+".npy")


def space_correlate(matrix, shift):
    def column_correlation(matrix, shift, row):
        def check():
            if 0 <= row-shift < 120 and 0 <= row+shift < 120:
                return matrix[row+shift, shift:-shift] *\
                    matrix[row, shift:-shift] \
                    + matrix[row-shift, shift:-shift]\
                    * matrix[row, shift:-shift]
            else:
                return np.zeros((matrix.shape[1]-2*shift))
        return np.sum(matrix[row, shift:-shift]*matrix[row, :-2*shift] +
                      matrix[row, shift:-shift]*matrix[row, shift*2:] +
                      check())\
            * 1./(matrix.shape[1]-2*shift)/4
    if shift == 0:
        return map(lambda x: np.mean(np.vectorize(np.square)(matrix[x, :])),
                   range(matrix.shape[0]))
    else:
        return map(lambda x: column_correlation(matrix, shift, x),
                   range(matrix.shape[0]))


def space_correlation(matrix, time):
    mean = np.mean(matrix[:, :, time], axis=1)**2
    func = functools.partial(space_correlate, matrix[:, :, time])
    pool = multiprocessing.Pool(5)
    correlation = np.array(pool.map(func, range(0, matrix.shape[1]/10)))
    return np.nan_to_num(correlation/mean/mean.shape[0]).transpose()


def purify_row(matrix):
    return np.delete(matrix, np.where((np.all(matrix, axis=1) == 0) is True),
                     axis=0)


def halfheight(ave):
    z = []
    for count, a in enumerate(ave):
        z.append(np.argmax(a < np.max(a)/2.))
    return np.array(z)


def taulocalmin(ave):
    z = []
    for count, a in enumerate(ave):
        z.append(argrelextrema(a, np.less)[0][0])

    return np.array(z)
    

def spectrum(time_interval, sequence):
    return time_interval/(2*np.pi)*np.abs(np.fft.ifft(sequence \
                                    - np.mean(sequence))) \
                                    **2*len(sequence)
                                
def spectrum_matrix(matrix, time_interval):
    """
    Fourier transform of a matrix
    
    Parameters:
    ===========
    matrix is the set of indipendent signals to be processed in batch
    time_interval set the physics scale of the process, in second
    
    Returns:
    ========
    spectres: matrix with spectre of each signal
    freq:     Frequency related to matrix ascissa, in Hertz
    
    """
    pool = multiprocessing.Pool(4)
    vmat, hmat, time = matrix.shape
    func = partial(spectrum, time_interval) 
    sequences = [matrix[x,y,:] for x,y in product(np.arange(vmat),np.arange(hmat))]
    spectres = np.array(pool.map(func, sequences)).reshape(vmat,hmat,time)
    freq=np.fft.fftfreq(time,time_interval)
    return spectres, freq

def log_smooth_matrix(spectres, freq, minfreq, nbin):
    """
    Smooth signal with logaritmic binning for log scale plot
    """
    vmat, hmat, time = spectres.shape
    sequences = [spectres[x,y,:] for x,y in product(np.arange(vmat),np.arange(hmat))]
    results = []
    for seq in sequences:
        exp_time, smooth_seq = binLogMovAve5(freq, seq, minfreq, nbin)
        results.append(smooth_seq)
    return np.array(results).reshape(vmat, hmat, nbin), exp_time

def log_smooth(time, signal, minfreq, nbin):
    asc = asc[:int(len(asc)/2)]
    sig = sig[:int(len(sig)/2)]
    maxtime = len(sig)
    spec=[]
    exptime=[]
    logmaxtime=np.log(maxtime)
    logmintime=np.log(mintime)
    step=(logmaxtime-logmintime)/nbin
    for i in range(0,nbin):
        if(len(sig)>np.exp((i+1)*step)):
            exptime.append(np.mean(asc[np.exp(logmintime+i*step):np.exp(logmintime+(i+1)*step)]))
            spec.append(np.mean(sig[np.exp(logmintime+i*step):mintime+np.exp(logmintime+(i+1)*step)]))
        else:	
            exptime.append(np.mean(asc[np.exp(logmintime+i*step):len(asc)]))
            spec.append(np.mean(sig[np.exp(logmintime+i*step):len(sig)]))
            break
    return np.asarray(exptime), np.asarray(spec)
    
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_average_matrix(sequences, n):
    smooth_seq = []
    for seq in range(sequences.shape[0]):
        smoothed = moving_average(sequences[seq,:], n)
        smooth_seq.append(smoothed)
    return smooth_seq

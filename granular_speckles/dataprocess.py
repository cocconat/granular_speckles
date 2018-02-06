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

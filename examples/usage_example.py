#! /usr/bin/python
# -*- coding: utf-8 -*-
# This file belongs to DWGranularSpeckles project.
# The software is realeased with MIT license.

import numpy as np
import os

from granular_speckles import GetMatrix, correlation
from granular_speckles import halfheight
from granular_speckles import coarseSpace, coarseTime
import matplotlib.pyplot as plt

from granular_speckles import datavisual


# explore_time(time_serie)

if __name__=="__main__":
    #Set parameters:
    #%%
    corrtime = 25
    coarsing_space = 5
    coarsing_time = 2
    image_matrix = "testvideo/image_matrix.npy"
    #%%
    print("Importing matrix from file: {}".format(image_matrix))

    time_serie = np.load(image_matrix)

    if isinstance(time_serie, np.ndarray):
        print("image-matrix correctly uploaded")
    else:
        raise "image not correctly uploaded!!!"

    fullPath = "results/full_correlation/"
    togePath = "results/all_together/"

    if not os.path.isdir(fullPath):
        os.makedirs(fullPath)
    if not os.path.isdir(togePath):
        os.makedirs(togePath)

    #Corase the original matrix in space
    coarse_matrix = coarseSpace(time_serie, coarsing_space)
    #Corase the original matrix in time
    coarse_matrix = coarseTime(coarse_matrix, coarsing_time)

    #horizontal_average of correlation length is realized with non-zero pixels.
    full_core, horizontal_average = correlation(int(corrtime), coarse_matrix)
    mezza = halfheight(horizontal_average)

    fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)
    ax1.set_title("Correlation over time averaged \n on horizontal lines")
    ax1.pcolor(horizontal_average)
    ax2.set_title("Correlation is 0.5 \n at time:")
    ax1.set_ylabel("Depth(pixel)")
    ax2.set_xlabel("time (frame)")
    ax1.set_xlabel("time (frame)")
    ax2.scatter(mezza, range(len(mezza)))
    fig.tight_layout()
    plt.show()

    # datavisual.explore_time(time_serie)
    # datavisual.explore_time(coarse_matrix)
    # datavisual.explore_time(full_core)
    #%%

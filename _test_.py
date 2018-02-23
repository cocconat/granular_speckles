#! /usr/bin/python
# -*- coding: utf-8 -*-
# This file belongs to DWGranularSpeckles project.
# The software is realeased with MIT license.

import numpy as np
import os

from granular_speckles import GetMatrix, correlation
from granular_speckles import halfheight
from granular_speckles import timeDecorrelation, blockIteration, \
    explore_time, explore

image_matrix = "testvideo/image_matrix.npy"

print("Importing matrix from file: {}".format(image_matrix))

time_serie = np.load(image_matrix)

if isinstance(time_serie, np.ndarray):
    print("image-matrix correctly uploaded")
else:
    raise "image not correctly uploaded!!!"

# explore_time(time_serie)

block_size = 2

corrtime = 300

fullPath = "results/full_correlation/"
togePath = "results/all_together/"

if not os.path.isdir(fullPath):
    os.makedirs(fullPath)
if not os.path.isdir(togePath):
    os.makedirs(togePath)


for block, reducedSerie in blockIteration(time_serie,
                                          [2, 3, 6, 10, 15, 20, 25]):
    full_core, all_together = correlation(int(corrtime), reducedSerie)
    mezza = halfheight(all_together)
    np.save(togePath+block_size, all_together)
    np.save(fullPath+block_size, full_core)
    np.save(togePath+block_size+"mezzaltezza", mezza)
    del reducedSerie

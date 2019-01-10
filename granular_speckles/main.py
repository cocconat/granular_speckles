#! /usr/bin/python
# -*- coding: utf-8 -*-
# This file belongs to DWGranularSpeckles project.
# The software is realeased with MIT license.

import numpy as np
import os

from .cli_parser import getParser
from .coarsing import coarseSpace
from .dataprocess import halfheight
from .datavisual import timeDecorrelation, blockIteration, \
    explore_time, explore
from .matrix import GetMatrix, correlation
from .video import get_frame_rate, videoToFrame


def main():
    args = getParser().parse_args()
    time_series = None

    # import video file
    if args.videofile is not None:
        print("importing frames from video")
        print("video frame number : ", get_frame_rate(args.videofile))
        print("exporting video frames (num: {}) \
               to folder {}".format(videoToFrame(args), args.image_folder))

    # import previously processed video matrix
    if args.file_import:
        print("Importing matrix from file: {}".
              format(os.path.join(args.image_folder, "image_matrix" + ".npy")))
        time_series = np.load(os.path.join(
            args.image_folder, "image_matrix" + ".npy"))
        if isinstance(time_series, np.ndarray):
            print("image-matrix correctly uploaded")
        else:
            raise Exception("image not correctly uploaded!!!")
    else:
        print("image acquisition in process... requires time.")
        time_series = GetMatrix(args.image_folder,
                               args.resize,
                               args.black).matrix
        np.save(args.image_folder + "/image_matrix", time_series)
        print("imported matrix has shapes: {}".format(time_series.shape))

    if args.visualize:
        explore_time(time_series)

    # varianceMatrix=timeVariance(time_serie)
    # print "la somma della varianza della matrice di input e' {}".
    # format(varianceMatrix.mean())
    # plotMatrix(varianceMatrix)
    # time_serie.plotFrame(19)
    # time_serie.plotFrame(frame=21,plt=plt).show()
    # explore(new_time_serie,'histogram',pause=0.00001)
    # explore(new_time_serie,'histogram',85,60,150,80,pause=0.00001)

    if not args.block_size:
        block_size = str(1)
    else:
        block_size = args.block_size

    result_path = "results" + "/" + args.image_folder
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if args.timecoarse:
        timeDecorrelation(coarseSpace(time_series, 2),
                          result_path, range(1, 50, 1))

    if args.spacecoarse:
        full_path = "results/" + args.image_folder + "/full_correlation/"
        togePath = "results/" + args.image_folder + "/all_together/"
        corrtime = args.correlation or 300
        # saving correlation for various coarse graining
        for block, reducedSerie in blockIteration(time_series,
                                                  [2, 3, 6, 10, 15, 20, 25]):
            full_core, all_together = correlation(int(corrtime), reducedSerie)
            mezza = halfheight(all_together)
            np.save(togePath + block_size, all_together)
            np.save(full_path + block_size, full_core)
            np.save(togePath + block_size + "mezzaltezza", mezza)
            del reducedSerie

    if args.matrix_story:
        explore(time_series)

    if args.correlation:
        full_core, all_together = correlation(int(args.correlation),
                                              time_series, cutoff=args.cutoff,
                                              function=args.cor_function)
        # plt.plot(range(len(all_togheter)),all_togheter)
        # time_serie.plotStory(80,120,pause=20)
        mezza = halfheight(all_together)
        full_path = "results/" + args.image_folder + "/full_correlation/"
        togePath = "results/" + args.image_folder + "/all_together/"
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        if not os.path.exists(togePath):
            os.makedirs(togePath)
        np.save(full_path + block_size, full_core)
        np.save(togePath + block_size, all_together)
        np.save(togePath + block_size + "mezzaltezza", mezza)

    globals().update(locals())


if __name__ == "__main__":
    main()

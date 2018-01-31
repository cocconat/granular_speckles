
def getParser():
    import argparse
    ap = argparse.ArgumentParser("Utility to measure the correlation among \
                                 frames of diffusing wave spectroscopy videos.\
                                 It's possible to: \
                                 1. Coarse graining over time or space dimension\
                                 2. Move to 3D project \
                                 The algorithm implemented are higly optimized \
                                 It's parallelized over n procs. \
                                 Needs: Numpy, Scipy, multiprocessing, \
                                 matplotlib.\
                                 To frame the video also opencv2 is required")

    ap.add_argument("-bn", "--black", help="set black and white or greys",
                    action="store_true")
    ap.add_argument("-f", "--image_folder",
                    help="folder for png images to process")
    ap.add_argument("-b", "--block_size",
                    help="apply coarse graining, this is dimension for image reduction")
    ap.add_argument("-m", "--matrix_story",
                    help="colorful image for pixel evolution", action="store_true")
    ap.add_argument("-c", "--correlation",
                    help="measure the time correlation for each pixel of final matrix")
    ap.add_argument("-C", "--cutoff",
                    help="minimal variance ast to not be considered noise")
    ap.add_argument("-t", "--takealook",
                    help="colorful image for pixel evolution", action="store_true")
    ap.add_argument("-i", "--file_import",
                    help="folder for png images to process", action="store_true")
    ap.add_argument("-T", "--timecoarse",
                    help="time carsing and sigma analysis", action="store_true")
    ap.add_argument("-S", "--spacecoarse",
                    help="space coarsing and correlation matrix", action="store_true")
    ap.add_argument("-F", "--cor_function",
                    help="chose which correlation function you're using, the china or european styla", action="store_true")
    ap.add_argument("-v", "--videofile",
                    help="path for video file, long time required")
    ap.add_argument("-r", "--resize", type=int, nargs=4,
                    help="resize the image to center the lightspot")
    return ap

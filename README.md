# Granular Speckles
Python based analysis for speckels-based spectroscopy
Utility to measure the correlation among frames of diffusing wave spectroscopy videos.
The algorithm implementedi are optimized and parallelized over 'procs'.
Requirements: Numpy, Scipy, multiprocessing, matplotlib.
<!--To frame the video also opencv2 is required")-->

## How:
1. From video to frames
2. From frame to 8it matrices
3. Matrix normalization
4. Matrix Analysis and data
   visualization

## Matrix correlation analysis:
The matrix are coarse grained -to
reduce the sensor noise-  and the
correlation of the 2D frame is measured
with the respect of the temporal variable.
For each pixel there is
@TODO finire la descrizione






 `python granular speckles --help`
 `[-h] [-bn] [-f IMAGE_FOLDER] [-b BLOCK_SIZE] [-m] [-c CORRELATION] [-t] [-i] [-T] [-S] [-v VIDEOFILE]`


##### Optional arguments
	-h, --help            show this help message and exit
	-bn, --black          set black and white or greys
	-f IMAGE_FOLDER, --image_folder IMAGE_FOLDER
	                    folder for png images to process
	-b BLOCK_SIZE, --block_size BLOCK_SIZE
	                    apply coarse graining, this is dimension for image
	                    reduction
	-m, --matrix_story    colorful image for pixel evolution
	-c CORRELATION, --correlation CORRELATION
	                    measure the time correlation for each pixel of final
	                    matrix
	-t, --takealook       colorful image for pixel evolution
	-i, --file_import     folder for png images to process
	-T, --timecoarse      time carsing and sigma analysis
	-S, --spacecoarse     space coarsing and correlation matrix
	-v VIDEOFILE, --videofile VIDEOFILE
	                    path for video file, long time required
`

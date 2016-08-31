# granular_speckles
Python based analysis for many speckels spectroscopy


usage: this program measure the correlation of a many frame video, then its  possible to performa a coarse graining over time or space dimension, it's a 2d project but it can be easily performed in 3d with little modification. 
The algorithm implemented are top level for each stage and standard is parallelized over 5 procs. 
Needed library is Numpy, Scipy, multiprocessing , matplotlib, if you want to frame the video also opencv2 is needed
 
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

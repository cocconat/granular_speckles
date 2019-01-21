import os,gc
import numpy as np
import coarsing
for i in os.listdir(os.getcwd()):
    if "V" in i:
	s=np.load(i+"/image_matrix.npy")
        z=coarsing.coarseSpace(140,s)
	np.save(i+"/metapixel140",z)
	del s,z
	gc.collect()

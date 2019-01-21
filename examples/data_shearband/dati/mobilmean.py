import numpy as np
def mobilmean(block_size,array):
	return map(lambda x:np.mean(array[x:x+block_size]),range(len(array)-block_size))


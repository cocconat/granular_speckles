import numpy as np
def smlogyDer(array):
	a=[]
	for i in range(0,len(array)-1):
	    a.append((np.log(array[i+1])-np.log(array[i])))
	
	
	return np.asarray(a)

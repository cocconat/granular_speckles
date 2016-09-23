import numpy as np

def timeMeanVariance(mat):
	'''
	ritorna matrice delle intensita
	medie e delle varianze
	
	'''
	return np.mean(mat,axis=2),np.var(mat,axis=2)

def corrEuristic(mat):
	'''
	prende una matrice di correlazione ed 
	associa array di 1 ai pixel 'fermi'
	
	'''
	
	l=mat.shape[0]
	h=mat.shape[1]
	time=mat.shape[2]
	
	mat=mat.reshape(l*h,time)
	
	b=np.asarray(map(lambda x: chooseFuncPlat(x,time), mat))
	
	return b.reshape(l,h,time)
	

def chooseFuncPlat(a,maxtime):
	b=a
	tau=np.argmax(a<np.max(a)/2.)
	if tau <= 1:
		return np.ones(maxtime)
	else:
		return b


def intVSvar(mat):
	'''
	
	ritorna array 2D in cui la prima colonna e' l'intensita 
	ordinata la seconda e' la varianza associata
	
	'''
	a=timeMeanVariance(mat)
	a=np.column_stack((a[0].flatten(),a[1].flatten()))
	
	return np.sort(a.view('float64,float64'),order=['f0'],axis=0).view(np.float64)
	
	
		
	
	
	
	

#! /usr/bin/python
# -*- coding: utf-8 -*-
# This file belongs to DWGranularSpeckles project.
# The software is realeased with MIT license. 
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scifit
plt.ion()
def frequency_rank(matrix):
	l=[]
	def middle(array):
		l=[]
		for x in range(array.shape[0]-1):
			l.append(np.sum(array[x:x+2])/2)
		return l

	for row in range(matrix.shape[0]):
		z=scipy.histogram(matrix[row,:,:],bins=35)
		plt.semilogy()
		#plt.scatter(middle(z[1]),z[0])
		l.append((middle(z[1][:-5]),np.log(z[0][:-5])))
	return np.array(l)

			
def linear_fit(data):
	def func(x,a,b):
		return a*x+b
	return scifit.curve_fit(func,data[0],data[1])

def plot_fit(data_fit,data):
	plt.scatter(data[0],data[1])
	plt.plot(range(1,200,50),map(lambda x: data_fit[0][0]*x +data_fit[0][1], range(1,200,50))) 

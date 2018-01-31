# coding: utf-8

from .coarsing import CoarseMatrix
import numpy as np
a=np.random.random((10,14))
corse=CoarseMatrix(2,a.shape)
reduced=corse.coarseMatrix(a)
print "la matrice di aprtenza è la stessa? {}".format((corse.matrix==a).all())
buona=corse.matrixReduction()
if (buona == reduced).all():
	print "vaivaivai"
else:
	print buona

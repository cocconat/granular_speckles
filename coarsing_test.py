# coding: utf-8

import coarsing
import numpy as np
reload(coarsing)
a=np.random.random((10,14))
corse=coarsing.CoarseMatrix(2,a.shape)
reduced=corse.coarseMatrix(a)
print "la matrice di aprtenza è la stessa? {}".format((corse.matrix==a).all())
buona=corse.matrixReduction()
if (buona == reduced).all():
	print "vaivaivai"
else:
	print buona

z={}
for count,a in enumerate(ave):
	z[count]=np.argmax(a<np.max(a)/2.)

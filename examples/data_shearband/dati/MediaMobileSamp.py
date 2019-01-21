import numpy as np
def mmSamp(arr,bl):
    a=[]
    for i in range(bl,len(arr),bl):
        a.append(np.mean(arr[i-bl:i]))
        
    return a    

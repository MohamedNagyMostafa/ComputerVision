import numpy as np

s = np.array([-1, 0,0,1,1,1,0,-1,-1,0,1,0,0,-1])
t = np.array([1,1,0])

print(np.correlate(s,t,  "same"))
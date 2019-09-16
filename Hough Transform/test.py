from collections import defaultdict, Counter
import numpy as np
x = defaultdict(int)
x[1,1,1] = 100
x[1,1,1] += 100
x[1,1,2] = 50
x[1,2,1] =100
x[1,2,1] += 50
for (a,b,c), v in Counter(x).most_common():
	print(a,b,c, v)


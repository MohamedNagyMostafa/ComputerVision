import numpy as np
import cv2
from util import *
from collections import Counter
import matplotlib.pyplot as plt

img_or = read_RGBimage('images/car2.jpg')
img = to_gray(img_or)
img_blur = gaussian_blur(image=img, size=3, sigma=4)
img_edge = canny(image=img_blur, sigma=3.6, significantH=0.9, significantL=-0.7)
show_image(img_edge, type='gray', title='car')

indy, indx = np.where(img_edge > 0)

acc, r_values,av ,bv= hough_circle(image=img_edge, indx=indx, indy=indy, max_r=60, min_r=15, step= 10)
l = 40

for (a, b, r), v in Counter(acc).most_common(): 
	l -= 1
	print(b, a, r, r_values[0,r%r_values.shape[1]]  , v)
	img_or = cv2.circle(img_or, (b,a), r_values[0,r%r_values.shape[1]], color=(0,0,255))
	if l == 0:
		break

plt.imshow(av, bv)
plt.imshow(img_or)
plt.show()



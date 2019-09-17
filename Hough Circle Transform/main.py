import numpy as np
import cv2
from util import *
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import torch

img_rgb = read_RGBimage('images/bowling.jpg')
img_or = np.copy(img_rgb)
img = to_gray(img_or)
img_blur = gaussian_blur(image=img, size=3, sigma=4)
img_edge = canny(image=img_blur, sigma=0.8, lower=140, higher=200)
indy, indx = np.where(img_edge > 0)
max_x, min_x, max_y, min_y, max_r, min_r = np.max(indx), np.min(indx), np.max(indy), np.min(indy), 80, 35
acc, norm_x, norm_y, norm_r= hough_circle(
	image=img_edge, 
	indx=indx, indy=indy, 
	max_r=max_r, min_r=min_r,
	max_x= max_x, max_y=max_y,min_x=min_x, min_y=min_y)
l = 1
dic = defaultdict(int)
for i, a in enumerate(acc):
	for j, b in enumerate(a):
		for k, value in enumerate(b):
			dic[i,j,k] = value 
			

for (a, b, r), v in Counter(dic).most_common():
	if v != 0 and l != 0:
		print (b- norm_y, a - norm_x, r-norm_r, v)
		img_or = cv2.circle(img_or, ( a - norm_x,b- norm_y), r - norm_r, color=(255,0,0))
	else:
		break
	l -=1

show_images(row=2,column=2, images=[img_rgb, img, img_edge, img_or], titles=['Original', 'Gray', 'Canny Edges', 'Hough Circle'], types=['rgb', 'gray','gray', 'rgb'])

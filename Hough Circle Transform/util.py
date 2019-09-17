import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import pi
from collections import defaultdict, Counter

def show_images(row, column, images, titles, types):

	for ind, (image, title, type) in enumerate(zip(images, titles, types)):
		plt.subplot(row, column, ind+1)

		if type == 'rgb':
			plt.imshow(image,aspect='auto')
		elif type == 'gray':
			plt.imshow(image, cmap=type, aspect='auto')
		plt.title(title)
		plt.xticks([])
		plt.yticks([])
	plt.show()

def gaussian_blur(image, size, sigma):
	return cv2.GaussianBlur(image, (size,size), sigma)

def hough_circle(image, indx, indy, max_r, min_r, min_x, min_y, max_x, max_y):
	shift_x = min(max_r, min_x) - max(max_r, min_x)
	shift_y = min(max_r, min_y) - max(max_r, min_y)

	norm_x = -shift_x
	norm_y = -shift_y
	norm_r = -min_r

	shape_a = max_x + max_r + norm_x
	shape_b = max_y + max_r + norm_y
	shape_r = max_r - min_r

	acc = np.zeros((shape_a+1, shape_b+1, shape_r))
	print(shape_a, shape_b, shape_r)
	print(norm_x,norm_y, norm_r)
	theta_values = np.expand_dims(np.deg2rad(np.arange(0,360, 10)),0)
	radius_values = np.expand_dims((np.arange(min_r, max_r, 1)),0).T
	r = np.repeat(radius_values,len(theta_values),1)

	term_cos = np.dot(radius_values, np.cos(theta_values))
	term_sin = np.dot(radius_values, np.sin(theta_values))

	for x, y in zip(indx,indy):
		a = x - term_cos
		b = y - term_sin
		acc[a.astype(int) + norm_x,b.astype(int) + norm_y,r.astype(int) + norm_r] += 1
			
	return acc, norm_x, norm_y, norm_r

def read_RGBimage(src):
	image = cv2.imread(src)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def to_gray(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image, size, sigma):
	return cv2.GaussianBlur(image, (size,size), sigma)

def canny(image, sigma, lower, higher):
	mid_value = np.median(image)* sigma 
	return cv2.Canny(image, mid_value + lower, mid_value + higher)

def show_image(image, type, title='image'):
	if type == 'rgb':
		plt.imshow(image,aspect='auto')
	elif type == 'gray':
		plt.imshow(image, cmap=type,aspect='auto')
	plt.title(title)
	plt.xticks([])
	plt.yticks([])
	plt.show()

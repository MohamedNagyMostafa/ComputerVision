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

def canny(image, sigma, significantH=0.2, significantL= 0.2):
	mid_value = np.median(image)
	lower = int(max(0,mid_value * sigma - significantL * mid_value))
	higher = int(min(255,mid_value * sigma + significantH * mid_value))
	print(mid_value, lower, higher)	
	return cv2.Canny(image, lower, higher)


def topk(acc, k):
	a_values, b_values, r_values = np.array([]), np.array([]), np.array([])

	for it in Counter(acc):
		a_values = it[0]
		b_values = it[1]
		r_values = it[2]
		k -=1
		if k < 1:
			break

	return a_values, b_values, r_values

def gaussian_blur(image, size, sigma):
	return cv2.GaussianBlur(image, (size,size), sigma)

def hough_circle(image, indx, indy, max_r, min_r, step= 100):
	acc =defaultdict(int)
	theta_values = np.deg2rad([np.linspace(0,360, step)])
	r_values = np.expand_dims(np.arange(min_r, max_r+1, 1),0)
	r_num = r_values.shape[1]
	print(r_num)
	term_cos = np.dot(r_values.transpose(), np.cos(theta_values))
	print(term_cos.shape)
	term_sin = np.dot(r_values.transpose(), np.sin(theta_values))
	r = term_cos.shape[0] * term_cos.shape[1] * len(theta_values)
	av, bv = np.array([]), np.array([])
	for x, y in zip(indx, indy):
		a_values, b_values= x - term_cos.flatten(), y - term_sin.flatten()
		for i, (a, b) in enumerate(zip(a_values, b_values)):
			acc[int(a), int(b), i % r_num] +=1
			av = np.append(av, a)
			bv = np.append(bv,b)

			
	return acc, r_values, av, bv

def read_RGBimage(src):
	image = cv2.imread(src)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def to_gray(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image, size, sigma):
	return cv2.GaussianBlur(image, (size,size), sigma)

def canny(image, sigma, significantH=0.2, significantL= 0.2):
	mid_value = np.median(image)
	lower = int(max(0,mid_value * sigma - significantL * mid_value))
	higher = int(min(255,mid_value * sigma + significantH * mid_value))
	print(mid_value, lower, higher)	
	return cv2.Canny(image, lower, higher)

def show_image(image, type, title='image'):
	if type == 'rgb':
		plt.imshow(image,aspect='auto')
	elif type == 'gray':
		plt.imshow(image, cmap=type,aspect='auto')
	plt.title(title)
	plt.xticks([])
	plt.yticks([])
	plt.show()

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch 

def read_RGBimage(src):
	image = cv2.imread(src)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def to_gray(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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

def topK(container, k):
	x = np.zeros(k,dtype='int64')
	y = np.zeros(k,dtype='int64')
	top = torch.topk(torch.flatten(torch.from_numpy(container)), k)[0].numpy()

	for i in range(k):
		out = np.where(container == top[i])
		if len(out[0]) > 1:
			for yt, xt in zip(out[0], out[1]):
				x[i], y[i] = xt, yt
				i+=1
				if i == k:
					break
		else:
			y[i], x[i] = out

	return x, y

def hough_line(image, indx, indy):
	imgy, imgx = image.shape
	diag = np.sqrt(imgx**2+imgy**2)

	r_values = np.linspace(-diag, diag, 2*diag)
	theta_values = np.deg2rad(np.arange(-90.0, 91.0))

	cos_t = np.cos(theta_values)
	sin_t = np.sin(theta_values)
	num_theta = len(theta_values)

	accumulator = np.zeros((len(r_values) , len(theta_values)))	

	for xi, yi in zip(indx, indy):
		for theta in range(181):
			r = round(xi * cos_t[theta] + yi * sin_t[theta]) + diag
			accumulator[int(r), theta] +=1

	return accumulator, theta_values, r_values

def draw_hough_lines(image, indx, indy, thetas, rhos):
	#for rho, theta in zip(rhos, thetas):
	for x, y in zip(indx, indy):
		image = line_eq(image, thetas[x], rhos[y])

	return image

def line_eq(image, theta, r):
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	y_max, x_max, _ = image.shape

	if theta == 180 or theta == 0:
		x1, x2 = int(x), int(x)
		y1, y2 = 0, y_max
	elif theta == 90:
		y1, y2 = int(y), int(y)
		x1, x2 = 0, x_max

	else:
		m = -x/y
		k = r/np.sin(theta)
		x1 = 0
		y1 = int(x1 * m + k)
		x2 = x_max
		y2 = int(x2 * m + k)
	
		
	
	cv2.line(image, (x1, y1), (x2, y2), (0,0,255), 2)
	return image
# Testing
def line(image, theta, r):
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*r
	y0 = b*r
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	cv2.line(image, (x1, y1), (x2, y2), (0,0,255), 2)
	return image


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
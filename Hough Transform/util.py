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
	x_c, y_c = int(imgx/2.0), int(imgy/2.0)
	diag = np.sqrt(x_c**2+y_c**2)

	r_values = np.linspace(-diag, diag, 2*diag)
	theta_values = np.deg2rad(np.arange(-90.0, 91.0))

	cos_t = np.cos(theta_values)
	sin_t = np.sin(theta_values)
	num_theta = len(theta_values)

	accumulator = np.zeros((len(r_values) , len(theta_values)))	

	#Coordinate transformation 
	indx_d = indx - x_c
	indy_d = indy - y_c

	for xi, yi in zip(indx, indy):
		for theta in range(181):
			r = round(xi * cos_t[theta] + yi * sin_t[theta])
			accumulator[int(r), theta] +=1


	return accumulator, theta_values, r_values, (x_c, y_c)

def draw_hough_lines(image, rhos, thetas, x_c, y_c):
	for rho, theta in zip(rhos, thetas):
		image = line_coordinate(image, theta, rho,x_c, y_c)

	return image
def line_coordinatek(image, theta, rho,x_c, y_c):
	if theta == 90:
		x0_d = rho/np.sin(np.deg2rad(theta))
		y0_d = 0
		x1_d = x0_d
		y1_d = y_c

	elif theta == 0 and theta == 180:
		x0_d = 0
		y0_d = rho/np.cos(np.deg2rad(theta))
		x1_d = x_c
		y1_d = y0_d
		#Intersection by x-axis => y_d=0
	else:
		x0_d = rho/np.cos(np.deg2rad(theta))
		y0_d = 0
		#Intersection by y-axis => x_d=0
		y1_d = rho/np.sin(np.deg2rad(theta))
		x1_d = 0
		print(rho, theta, x0_d, y1_d)

	x0, x1, y0, y1 = np.int(np.round(x0_d + x_c)), np.int(np.round(x1_d + x_c)), np.int(np.round(y0_d + y_c)), np.int(np.round(y1_d + y_c))
	print(x0,y0,x1,y1)
	cv2.line(image, (x0, y0), (x1, y1), (0,0,255), 2)

	return image

def line_coordinate(image, theta, rho,x_c, y_c):
	y0 = int(np.round(rho*np.sin(np.deg2rad(theta)) + y_c))
	x0 = int(np.round(rho*np.cos(np.deg2rad(theta)) + x_c))
	m = y0/x0
	#Intersection with y = mx
	y1 = int(np.round(y0 +y0/x0 * 1000))
	x1 = int(np.round(x0 + x0/y0 * 1000))

	cv2.line(image, (x0, y0), (x1, y1), (0,0,255), 2)

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
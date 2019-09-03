import numpy as np
from util import *


image_rgb = read_RGBimage('images/streets.jpg')
image_gray = to_gray(image_rgb)
image_smoothing = gaussian_blur(image=image_gray, size=5, sigma=4)
image_edges = canny(image=image_smoothing, sigma=2, significantH=1.5, significantL= 0)
indy, indx = np.where(image_edges > 0)

accumulator, thetas, rhos = hough_line(image_edges, indx, indy)

#rescaling
indx, indy = topK(accumulator, 2)
print(indx, ' ' , indy)
image_copy = np.copy(image_rgb)

image_hough = draw_hough_lines(image_copy, indx, indy, thetas, rhos)

show_images(
	row=2,column=2,
	images=[image_rgb, image_edges, accumulator, image_hough], 
	titles=['original', 'Canny', 'Hough space', 'Hough Transform'],
	types=['rgb', 'gray','gray','rgb'])

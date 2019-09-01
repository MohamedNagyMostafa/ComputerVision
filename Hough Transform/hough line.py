import numpy as np
from util import *


image_rgb = read_RGBimage('images/test.png')
image_gray = to_gray(image_rgb)
image_smoothing = gaussian_blur(image=image_gray, size=5, sigma=4)
image_edges = canny(image=image_smoothing, sigma=1.0,significantL=0, significantH=0.8)
indy, indx = np.where(image_edges > 0)

accumulator, thetas, rhos, (x_c, y_c) = hough_line(image_edges, indx, indy)

#rescaling
indx, indy = topK(accumulator, 100)

image_copy = np.copy(image_rgb)

image_hough = draw_hough_lines(image_copy, indx, indy, x_c, y_c)

plt.imshow(image_hough)
plt.show()

	


plt.imshow(accumulator,aspect='auto', cmap='gray')
plt.plot(indx, indy, 'r')
plt.show()
'''
show_images(
	row=2, 
	column=2, 
	images=[image_rgb, image_gray, image_smoothing, image_edges], 
	titles=['original', 'gray', 'smoothing', 'edges'], 
	types=['rgb', 'gray', 'gray', 'gray']
	)
'''
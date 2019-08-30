import cv2
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, labels):
	count = len(images)
	for r, (i, l) in enumerate(zip(images, labels)):
		plt.subplot(1, count, r + 1)
		if l =='edge by diff':

			plt.imshow(i,cmap='gray', vmax=np.max(i), vmin=np.min(i))
		elif l == 'original':
			plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB), )
		else:
			plt.imshow(i, cmap='gray')
		plt.xticks([])
		plt.yticks([])
		plt.title(l)
	plt.show()


img = cv2.imread('images/lena.png', cv2.IMREAD_UNCHANGED)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sigma = 4
sigma_canny = 0.01
smoothed_img = cv2.GaussianBlur(img_gray, (5,5), sigma)
# Compute edges by image shifting
img_left = smoothed_img
img_left = smoothed_img[:,1:]
img_right = smoothed_img
img_right = smoothed_img[:,:-1]
img_diff = img_right - img_left
print(smoothed_img)
print(img_left)
print(img_right)
# Edge by canny
m_value = np.median(img_gray)
lower = int(max(0, (1-sigma_canny)*m_value))
higher = int(min(255, (1-sigma_canny)*m_value))

img_canny_edge = cv2.Canny(img_gray, lower, higher)
img_canny_edge_l = cv2.Laplacian(img_canny_edge, ddepth = cv2.CV_16S);
#Canny for smooth
m_value = np.median(smoothed_img)
lower = int(max(0, (1-sigma_canny)*m_value))
higher = int(min(255, (1-sigma_canny)*m_value))

img_smoothed_canny_edge = cv2.Canny(smoothed_img, lower, higher)

show_images([img, smoothed_img, img_diff, img_canny_edge, img_smoothed_canny_edge, img_canny_edge_l], ['original', 'smoothed', 'edge by diff', 'Canny', 'Canny Smoothed Img', 'lablacian Canny Img'])
'''
xx, yy = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, smoothed_img ,rstride=1, cstride=1, cmap=plt.cm.jet,linewidth=0)
plt.show()
'''
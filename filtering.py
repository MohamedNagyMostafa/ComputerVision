import cv2
import numpy as np
import matplotlib.pyplot as plt

img_me = cv2.imread('images/me.jfif', cv2.IMREAD_UNCHANGED)

kernal = np.ones((5,5), np.float32)/5**2
filtered_image = cv2.filter2D(img_me, -1,kernal)
blured_image = cv2.blur(img_me, (5,5))
blured_image_gaussian = cv2.GaussianBlur(img_me, (5,5),16)

plt.subplot(2,2,1)
plt.title('original')
plt.imshow(img_me)
plt.subplot(2,2,2)
plt.title('filtered')
plt.imshow(filtered_image)
plt.subplot(2,2,3)
plt.title('blur')
plt.imshow(blured_image)
plt.subplot(2,2,4)
plt.title('GaussianBlur')
plt.imshow(blured_image_gaussian)
plt.show()

noise_sigma = 12
x,y,z = img_me.shape
noise = np.random.randn(x,y,z) * noise_sigma
img_me_copy = np.copy(img_me)
img_me_noise = img_me_copy + noise
img_me_noise = img_me_noise/255
plt.title('noise image')
plt.imshow(img_me_noise)
plt.show()

#Remove noise
img_me_noise_gaussian = cv2.GaussianBlur(img_me_noise, (5,5), 2)

plt.subplot(1,2,1)
plt.title('noise image')
plt.imshow(img_me_noise)
plt.subplot(1,2,2)
plt.title('gaussian image')
plt.imshow(img_me_noise_gaussian)
plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt

img_me = cv2.imread('images/me.jfif', cv2.IMREAD_UNCHANGED)

sharp_filter = np.array([[0,0,0],[0,2,0],[0,0,0]]) #Impulse filter
blur_filter = np.ones((3,3), np.float32)* 1/(3**2)

sharped_image = cv2.filter2D(img_me, 0, sharp_filter, borderType=cv2.BORDER_REFLECT_101)
blured_image = cv2.filter2D(sharped_image, 0, blur_filter, borderType=cv2.BORDER_REFLECT_101)
blured__un_image = cv2.filter2D(img_me, 0, blur_filter, borderType=cv2.BORDER_REFLECT_101)
unsharp = sharped_image - blured__un_image

plt.subplot(1,3,1)
plt.title('sharp_filter')
plt.imshow(sharped_image)
plt.subplot(1,3,2)
plt.title('blured_image')
plt.imshow(blured_image)
plt.subplot(1,3,3)
plt.title('unsharp')
plt.imshow(unsharp)
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

#cv2.IMREAD_UNCHANGED
#cv2.IMREAD_COLOR
#cv2.IMREAD_GRAYSCALE
src= 'images/me.jfif'
title = 'my image'

#cv2.line(img, (0, 127), (255, 127), (0,255,0))
'''
#i = cv2.imshow(title, img)	
#cv2.waitKey(0)
#print(type(img))
# Red channel
plt.subplot(1,3,1)
plt.hist(img[:,:,0])
plt.title('red channel')
plt.xlim(0, 256)
plt.grid(True)
#Green channel
plt.subplot(1,3,2)
plt.hist(img[:,:,1])
plt.title('green channel')
plt.xlim(0, 256)
plt.grid(True)

#Blue Channel
plt.subplot(1,3,3)
plt.hist(img[:,:,2])
plt.title('blue channel')
plt.xlim(0, 256)
plt.grid(True)

plt.show()
'''
'''
img = cv2.imread('images/rgb.jpg', cv2.IMREAD_UNCHANGED)
img_redLayer = img[:,:,0]
img_greenLayer = img[:,:,1]
img_blueLayer = img[:,:,2]

horizontal_image = np.hstack((img_redLayer, img_greenLayer, img_blueLayer))

cv2.imshow('Channel', horizontal_image)
cv2.imshow('original', img)
cv2.waitKey(0)
'''


img_rgb =cv2.imread('images/rgb.jpg', cv2.IMREAD_UNCHANGED)
img_me =cv2.imread('images/me.jfif', cv2.IMREAD_UNCHANGED)

img_rgb_v, img_rgb_h, _ = img_rgb.shape
img_me_v, img_me_h, _ = img_me.shape

img_diff_v = np.abs(img_rgb_v - img_me_v)
img_diff_h = np.abs(img_rgb_h - img_me_h)

[img_me_new_v_s,  img_me_new_v_e]= [img_diff_v//2, img_diff_v//2] if img_diff_v/2.0 % 2 == 0 else [img_diff_v//2+ 1, img_diff_v//2] 
[img_me_new_h_s, img_me_new_h_e]= [img_diff_h//2, img_diff_h//2] if img_diff_h/2.0 % 2 == 0 else [img_diff_h//2+ 1, img_diff_h//2]

img_me_crop = img_me[img_me_new_v_s + 1:img_me_v - img_me_new_v_e + 1, img_me_new_h_s + 1:img_me_h - img_me_new_h_e + 1]

img_composed = img_me_crop + img_rgb

img_composed[img_composed > 255] = 255
cv2.imshow('cropped image', img_composed)
cv2.waitKey(0)
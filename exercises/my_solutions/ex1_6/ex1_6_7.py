import numpy as np
import cv2


image = cv2.imread('exercises/my_solutions/ex1/flower.jpg')

# BGR color space: yellow has low blue, high green/red
lower_bgr = (0, 120, 140)
upper_bgr = (130, 255, 255)
mask_bgr = cv2.inRange(image, lower_bgr, upper_bgr)
cv2.imwrite('exercises/my_solutions/ex1/1_6_7_mask_bgr.png', mask_bgr)

# HSV color space: yellow sits around hue 20-30 (OpenCV's 0-179 scale)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_hsv = (5, 50, 140)
upper_hsv = (35, 255, 255)
mask_hsv = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
cv2.imwrite('exercises/my_solutions/ex1/1_6_7_mask_hsv.png', mask_hsv)

# LAB color space: yellow has a high b channel (yellow-blue axis)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
lower_lab = (100, 100, 140)
upper_lab = (255, 160, 255)
mask_lab = cv2.inRange(image_lab, lower_lab, upper_lab)
cv2.imwrite('exercises/my_solutions/ex1/1_6_7_mask_lab.png', mask_lab)


import numpy as np
import cv2

image = cv2.imread('exercises/my_solutions/cat.jpg')

b, g, r = cv2.split(image)

cv2.imwrite('exercises/my_solutions/ex1/1_6_3_b_test.png', b)
cv2.imwrite('exercises/my_solutions/ex1/1_6_3_g_test.png', g)
cv2.imwrite('exercises/my_solutions/ex1/1_6_3_r_test.png', r)

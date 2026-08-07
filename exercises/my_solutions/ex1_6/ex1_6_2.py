import numpy as np
import cv2

# 1. Read image
image = cv2.imread('exercises/my_solutions/ex1/1_6_1_test.png')

# 2. Draw a line on the image
cv2.line(image, (40, 30), (20, 120), (255, 0, 255), 3)

# 3. Save the image with the drawn line as a new file
cv2.imwrite('exercises/my_solutions/ex1/1_6_2_test.png', image)
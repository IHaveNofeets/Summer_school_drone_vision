import cv2

# 1. Read image
image = cv2.imread('cat.jpg')

# 2. Blur the image with Gaussian Blur (ksize must be odd numbers, e.g., 15x15)
blurred_image = cv2.GaussianBlur(image, (63, 1), 0)

# 3. Save the blurred image as a new file
cv2.imwrite('blurred_cat.jpg', blurred_image)


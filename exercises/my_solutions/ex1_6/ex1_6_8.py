import cv2
import matplotlib.pyplot as plt

image = cv2.imread('exercises/my_solutions/ex1/flower.jpg')
b, g, r = cv2.split(image)

row = image.shape[0] // 2  # pick a row through the middle of the image

plt.figure()
plt.plot(g[row, :], color='green')
plt.title(f'Green channel intensity along row {row}')
plt.xlabel('Column (pixel)')
plt.ylabel('Green value')
plt.savefig('exercises/my_solutions/ex1/1_6_8_green.png')

plt.figure()
plt.plot(b[row, :], color='blue')
plt.title(f'Blue channel intensity along row {row}')
plt.xlabel('Column (pixel)')
plt.ylabel('Blue value')
plt.savefig('exercises/my_solutions/ex1/1_6_8_blue.png')

plt.figure()
plt.plot(r[row, :], color='red')
plt.title(f'Red channel intensity along row {row}')
plt.xlabel('Column (pixel)')
plt.ylabel('Red value')
plt.savefig('exercises/my_solutions/ex1/1_6_8_red.png')

plt.show()

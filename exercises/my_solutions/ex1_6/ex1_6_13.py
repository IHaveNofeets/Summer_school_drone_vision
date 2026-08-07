import cv2
import matplotlib.pyplot as plt

image = cv2.imread('exercises/my_solutions/ex1/1_6_11_distances.png', cv2.IMREAD_GRAYSCALE)

# Otsu's method picks the threshold automatically by minimising intra-class variance
otsu_value, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold: {otsu_value}")

cv2.imwrite('exercises/my_solutions/ex1/1_6_13_mask.png', mask)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Distance image')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title(f'Otsu segmentation (t={otsu_value:.1f})')
axes[1].axis('off')

plt.savefig('exercises/my_solutions/ex1/1_6_13_result.png')
plt.show()

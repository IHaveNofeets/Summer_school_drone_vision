import cv2

image = cv2.imread('exercises/my_solutions/ex1_7/IMG_20260806_143213204.jpg')  # Read the image from the specified path

print(image.shape)

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
otsu_value, mask = cv2.threshold(lab[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold: {otsu_value}")

delta = 5
mask = cv2.dilate(mask, None, iterations=delta)
mask = cv2.erode(mask, None, iterations=delta)
mask = cv2.dilate(mask, None, iterations=delta)

flood = mask.copy()
cv2.floodFill(flood, None, (0, 0), 255)  # fill the background reachable from the top-left corner
holes = cv2.bitwise_not(flood)  # what's left over is only the enclosed holes
mask = cv2.bitwise_or(mask, holes)  # fill those holes into the original mask

mask = cv2.erode(mask, None, iterations=delta*2)
mask = cv2.dilate(mask, None, iterations=delta)

cv2.imwrite('exercises/my_solutions/ex1_7/mask.png', mask)

image_with_mask = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite('exercises/my_solutions/ex1_7/image_with_mask.png', image_with_mask)

n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)

min_area = 500  # drop tiny noise blobs
annotated = image.copy()

for label in range(1, n_labels):  # label 0 is always the background, skip it
    x, y, w, h, area = stats[label] # the stats layout is (x, y, width, height, area)
    if area < min_area:
        continue
    cx, cy = centroids[label]
    print(f"Object {label}: bbox=({x},{y},{w},{h}) area={area} centroid=({cx:.1f},{cy:.1f})")
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.circle(annotated, (int(cx), int(cy)), 8, (0, 0, 255), -1)

cv2.imwrite('exercises/my_solutions/ex1_7/annotated.png', annotated)
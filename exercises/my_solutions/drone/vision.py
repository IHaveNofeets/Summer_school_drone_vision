import cv2
import numpy as np
import glob
import os
import re

def do_vision(image, path, index):
    # rotate image to be upright      
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(path + 'rot/img_' + str(index) + '.jpg', image)
    # print(image.shape)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    cv2.imwrite(path + 'grey/img_' + str(index) + '.jpg', lab[:,:,2])
    blurred = cv2.GaussianBlur(lab[:,:,2], (9, 9), 0)
    # otsu_value, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    value, mask = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)

    # print(f"Otsu threshold: {otsu_value}")
    #cv2.imshow("test",mask)
    cv2.imwrite(path + 'thresholding/img_' + str(index) + '.jpg', mask)
    

    delta = 5
    mask = cv2.dilate(mask, None, iterations=delta)
    mask = cv2.erode(mask, None, iterations=delta)
    mask = cv2.dilate(mask, None, iterations=delta)

    if(True):
        h, w = mask.shape[:2]
        # Pad the mask by 1 pixel on every side (equivalent to the +2 sized Mat)
        maskForFloodFill = np.zeros((h + 2, w + 2), dtype=np.uint8)
        maskForFloodFill[1:h + 1, 1:w + 1] = mask

        # Flood fill the background starting from the corner
        im_floodfill = maskForFloodFill.copy()
        cv2.floodFill(im_floodfill, None, (0, 0), 255)

        # Invert: this leaves only the holes
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # OR the original with the inverted fill to close the holes
        maskForFloodFill = maskForFloodFill | im_floodfill_inv

        # Crop back to the original size
        mask = maskForFloodFill[1:h + 1, 1:w + 1]
    else:
        flood = mask.copy()
        cv2.floodFill(flood, None, (0, 0), 255)  # fill the background reachable from the top-left corner
        holes = cv2.bitwise_not(flood)  # what's left over is only the enclosed holes
        mask = cv2.bitwise_or(mask, holes)  # fill those holes into the original mask

    mask = cv2.erode(mask, None, iterations=delta*2)
    mask = cv2.dilate(mask, None, iterations=delta)
    cv2.imwrite(path + 'masks/img_' + str(index) + '.jpg', mask)

    # Connected components analysis
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)

    min_area = 10000  # drop tiny noise blobs
    max_area = 200000
    max_perimeter = 2500  # drop huge blobs (like the whole image)
    annotated = image.copy()

    #for label in range(1, n_labels):  # label 0 is always the background, skip it
        #x, y, w, h, area = stats[label] # the stats layout is (x, y, width, height, area)
        #if area < min_area or area > max_area:
          #  continue
        # cx, cy = centroids[label]
        # print(f"Object {label}: bbox=({x},{y},{w},{h}) area={area} centroid=({cx:.1f},{cy:.1f})")
        # cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # cv2.circle(annotated, (int(cx), int(cy)), 8, (0, 0, 255), -1)

    annotated2 = annotated.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for contour in contours:
        if cv2.contourArea(contour) < min_area or cv2.contourArea(contour) > max_area or cv2.arcLength(contour, True) > max_perimeter:
            continue
        rect = cv2.minAreaRect(contour)  # ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(annotated2, [box], 0, (0, 255, 0), 3)

        M = cv2.moments(contour)
        hu_moments = cv2.HuMoments(M)
        print(f"Shape {i}")
        print(f" -Area={cv2.contourArea(contour)}")
        print(f" -Perimeter={cv2.arcLength(contour, True)}")
        print(f" -Centroid=({M['m10']/M['m00']:.1f},{M['m01']/M['m00']:.1f})")
        print(f" -Bounding box=({rect[0][0]:.1f},{rect[0][1]:.1f},{rect[1][0]:.1f},{rect[1][1]:.1f}) angle={rect[2]:.1f}")
        print(f" -Moments: {M}")
        print(f" -Hu Moments: {hu_moments.flatten()}")
        i += 1

    #cv2.imwrite('exercises/my_solutions/ex1_7/annotated2.png', annotated2)
    cv2.imwrite(path + '/annotated/img_' + str(index) + '.jpg', annotated2)

base_path = 'exercises/my_solutions/drone/'

for img_path in sorted(glob.glob(base_path + 'raw/img_*.jpg')):
    index = int(re.search(r'img_(\d+)\.jpg$', os.path.basename(img_path)).group(1))
    image = cv2.imread(img_path)
    do_vision(image, base_path, index)
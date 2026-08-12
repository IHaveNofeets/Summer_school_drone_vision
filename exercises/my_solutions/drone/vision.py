import cv2
import numpy as np
import glob
import os
import re

def load_gps_locations(path):
    locations = {}
    with open(path + 'gps_locations.log') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, lat, lon, alt, heading = line.split(',')
            locations[int(idx)] = (float(lat), float(lon), float(alt), float(heading))
    return locations

def make_combined(steps, path, index, target_height=500):
    resized = []
    for label, img in steps:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = target_height / h
        img = cv2.resize(img, (int(w * scale), target_height))
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        resized.append(img)
    combined = cv2.hconcat(resized)
    cv2.imwrite(path + 'combined/img_' + str(index) + '.jpg', combined)

def do_vision(image, path, index, gps_locations):
    # rotate image to be upright      
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(path + 'rot/img_' + str(index) + '.jpg', image)
    rot_step = image.copy()
    # print(image.shape)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    cv2.imwrite(path + 'grey/img_' + str(index) + '.jpg', lab[:,:,2])
    grey_step = lab[:,:,2].copy()
    blurred = cv2.GaussianBlur(lab[:,:,2], (9, 9), 0)
    # otsu_value, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    value, mask = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)

    # print(f"Otsu threshold: {otsu_value}")
    #cv2.imshow("test",mask)
    cv2.imwrite(path + 'thresholding/img_' + str(index) + '.jpg', mask)
    thresholding_step = mask.copy()


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
    mask_step = mask.copy()

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
        aspect_ratio = rect[1][0] / rect[1][1] if rect[1][1] != 0 else float('inf')
        animal_type = "Elephant" if aspect_ratio < 1 else "Folded Zebra" if aspect_ratio < 2.5 else "Rhino"

        label_pos = (int(box[:, 0].min()), max(int(box[:, 1].min()) - 10, 0))
        cv2.putText(annotated2, animal_type, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

        print(f"{animal_type} Seen:")
        print(f"- In img_{index}")
        if index in gps_locations:
            lat, lon, alt, heading = gps_locations[index]
            print(f" -Location: lat={lat}, lon={lon}, alt={alt}, heading={heading}")
        
        i += 1

    #cv2.imwrite('exercises/my_solutions/ex1_7/annotated2.png', annotated2)
    cv2.imwrite(path + '/annotated/img_' + str(index) + '.jpg', annotated2)

    make_combined([
        ("Camera", rot_step),
        ("Grey (LAB-b)", grey_step),
        ("Threshold", thresholding_step),
        ("Mask", mask_step),
        ("Annotated", annotated2),
    ], path, index)

base_path = 'exercises/my_solutions/drone/'
gps_locations = load_gps_locations(base_path)

for img_path in sorted(glob.glob(base_path + 'raw/img_*.jpg')):
    index = int(re.search(r'img_(\d+)\.jpg$', os.path.basename(img_path)).group(1))
    image = cv2.imread(img_path)
    do_vision(image, base_path, index, gps_locations)
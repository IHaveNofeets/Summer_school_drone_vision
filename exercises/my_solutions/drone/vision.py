import cv2
import numpy as np
import glob
import os
import re
import math

EARTH_RADIUS_M = 6378137.0
HORIZONTAL_FOV_DEG = 62.2  # along the camera's native sensor width
VERTICAL_FOV_DEG = 48.8    # along the camera's native sensor height
FIELD_ELEVATION_M = 14.0   # the field sits ~14m above sea level, but alt in the gps log is above sea level

def estimate_animal_latlon(cx, cy, img_w, img_h, drone_lat, drone_lon, alt_msl_m, heading_deg):
    alt_agl_m = alt_msl_m - FIELD_ELEVATION_M  # height above the field, not above sea level

    # the image was rotated 90 deg CCW to be upright, so in this (rotated) frame the
    # image width now spans the camera's native VERTICAL fov, and the image height
    # spans the native HORIZONTAL fov
    right_m = alt_agl_m * ((cx - img_w / 2) / (img_w / 2)) * math.tan(math.radians(VERTICAL_FOV_DEG / 2))
    forward_m = alt_agl_m * ((img_h / 2 - cy) / (img_h / 2)) * math.tan(math.radians(HORIZONTAL_FOV_DEG / 2))

    heading_rad = math.radians(heading_deg)
    north_m = forward_m * math.cos(heading_rad) - right_m * math.sin(heading_rad)
    east_m = forward_m * math.sin(heading_rad) + right_m * math.cos(heading_rad)

    dlat = (north_m / EARTH_RADIUS_M) * (180 / math.pi)
    dlon = (east_m / (EARTH_RADIUS_M * math.cos(math.radians(drone_lat)))) * (180 / math.pi)

    return drone_lat + dlat, drone_lon + dlon

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

    img_h, img_w = image.shape[:2]

    detections = []
    i = 0
    for contour in contours:
        if cv2.contourArea(contour) < min_area or cv2.contourArea(contour) > max_area or cv2.arcLength(contour, True) > max_perimeter:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bx <= 0 or by <= 0 or bx + bw >= img_w - 1 or by + bh >= img_h - 1:
            continue  # skip animals cut off by the image border
        rect = cv2.minAreaRect(contour)  # ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(annotated2, [box], 0, (0, 255, 0), 3)

        M = cv2.moments(contour)
        hu_moments = cv2.HuMoments(M)
        aspect_ratio = rect[1][0] / rect[1][1] if rect[1][1] != 0 else float('inf')
        animal_type = "Elephant" if aspect_ratio < 1 else "Folded Zebra" if aspect_ratio < 2.5 else "Rhino"

        label_pos = (int(box[:, 0].min()), max(int(box[:, 1].min()) - 10, 0))
        cv2.putText(annotated2, animal_type, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

        drone_lat, drone_lon, alt, heading = gps_locations[index]
        cx, cy = rect[0]
        animal_lat, animal_lon = estimate_animal_latlon(cx, cy, img_w, img_h, drone_lat, drone_lon, alt, heading)
        print(f"{animal_type} seen at location: lat={animal_lat:.7f}, lon={animal_lon:.7f}, alt={alt}, heading={heading}")
        detections.append({
            "type": animal_type,
            "lat": animal_lat,
            "lon": animal_lon,
            "index": index,
        })

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

    return detections

def write_map(detections, gps_locations, path):
    flight_path = [gps_locations[idx][:2] for idx in sorted(gps_locations)]
    colors = {"Rhino": "#e6194B", "Folded Zebra": "#3cb44b", "Elephant": "#4363d8"}

    markers_js = "\n".join(
        f'L.circleMarker([{d["lat"]}, {d["lon"]}], {{radius: 8, color: "{colors.get(d["type"], "#000")}", '
        f'fillOpacity: 0.8}}).addTo(map).bindPopup("{d["type"]} (img_{d["index"]})");'
        for d in detections
    )
    path_js = ", ".join(f"[{lat}, {lon}]" for lat, lon in flight_path)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>#map {{ height: 100vh; margin: 0; }} body {{ margin: 0; }}</style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
var map = L.map('map');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

var flightPath = L.polyline([{path_js}], {{color: 'grey', dashArray: '4 6'}}).addTo(map);
map.fitBounds(flightPath.getBounds());
map.setZoom(map.getZoom() + 2);

{markers_js}
</script>
</body>
</html>
"""
    with open(path + 'map.html', 'w') as f:
        f.write(html)

base_path = 'exercises/my_solutions/drone/'
gps_locations = load_gps_locations(base_path)
all_detections = []

for img_path in sorted(glob.glob(base_path + 'raw/img_*.jpg')):
    index = int(re.search(r'img_(\d+)\.jpg$', os.path.basename(img_path)).group(1))
    image = cv2.imread(img_path)
    all_detections.extend(do_vision(image, base_path, index, gps_locations))

write_map(all_detections, gps_locations, base_path)
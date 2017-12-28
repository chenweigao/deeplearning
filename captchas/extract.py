import glob
import os
import os.path
import sys

import cv2
import imutils

from datetime import datetime
SOURCE_FILE = "D://data//captcha_images"
EXTRACT_FILE = "D://data//captcha_test_files"

files = glob.glob(os.path.join(SOURCE_FILE, "*"))
counts = {}

for (i, file) in enumerate(files):
    sys.stdout.write("\r[INFO]:processing {} / {}".format(i + 1, len(files)))
    filename = os.path.basename(file)
    text = os.path.splitext(filename)[0]
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w / h > 1.25:
            half_width = int(w / 2)
            image_regions.append((x, y, half_width, h))
            image_regions.append((x + half_width, y, half_width, h))
        else:
            image_regions.append((x, y, w, h))

    if len(image_regions) != 4:
        continue

    image_regions = sorted(image_regions, key=lambda x: x[0])

    for bounding_box, letter_text in zip(image_regions, text):
        x, y, w, h = bounding_box
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
        # letter_image = gray[x - 2: x + w + 2, y - 2:y + h + 2]
        save_path = os.path.join(EXTRACT_FILE, letter_text)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        counts[letter_text] = count + 1
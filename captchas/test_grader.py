from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import datetime
import os

FILE_OUTPUT_PATH = "./"
def show_image(image_name):
    cv2.imshow("test", image_name)
    cv2.waitKey(0)
def save_image(image_name):
    cv2.imwrite("{}.png".format(datetime.datetime.now().strftime('%Y-%m-%d')), image_name, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    # cv2.imwrite(os.path.join(FILE_OUTPUT_PATH, image_name), image_name, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])    
    cv2.waitKey(0)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True,
                    help="path to the input image")
args = vars(parser.parse_args())
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

cnts = cv2.findContours(
    edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for contour in cnts:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            docCnt = approx
            break

# print(docCnt.shape, docCnt.reshape(4, 2), len(docCnt))
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# show_image(paper)
# show_image(warped)
thresh = cv2.threshold(
    warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# show_image(thresh)
# https://www.wikiwand.com/zh-cn/%E5%A4%A7%E6%B4%A5%E7%AE%97%E6%B3%95

# save_image(thresh)

cnts = cv2.findContours(
    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(cnts[0]),cnts[0].shape)
# (411, 329)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questioncnts = []

#solve for each contours
for contour in cnts:
    (x, y, w, h) = cv2.boundingRect(contour)
    ar = w / float(h)

    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questioncnts.append(contour)
    # the weights and height should more than 20 pixels
    # append the correct contours into []
image_contours = cv2.drawContours(paper, questioncnts, 3, (255, 0, 255))
questioncnts = contours.sort_contours(
    questioncnts, method="top-to-bottom")[0]
# len(questioncnts) == 25
correct = 0
for(q, i) in enumerate(np.arange(0, len(questioncnts), 5)):
    cnts = contours.sort_contours(questioncnts[i:i + 5])[0]
    bubbled = None
    for(j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

color = (0, 0, 255)
k = ANSWER_KEY[q]
if k == bubbled[1]:
    color = (0, 255, 0)
    correct += 1

cv2.drawContours(paper, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
print("[INFO] score : {:.2f}%".format(score))

cv2.putText(paper, "{:.2f}%".format(
    score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)

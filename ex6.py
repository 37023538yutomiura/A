import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8x.pt")
results = model.predict("ex3.jpg", classes=[0], conf=0.2)

img = cv2.imread("ex3.jpg")

for box in results[0].boxes:
    x1 = int(box.data[0][0]) +10
    y1 = int(box.data[0][1]) +10
    x2 = int(box.data[0][2]) -10
    y2 = int(box.data[0][3]) -10
    region = img[y1:y2, x1:x2]

    if region.size == 0:
        continue

    region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(region_hsv)

    # 色マスク
    blue_mask = (h > 105) & (h < 125) & (s > 80) & (s < 220) & (v > 40) & (v < 180)
    yellow_mask = (h > 22) & (h < 32) & (s > 160) & (v > 160)
    red_mask = (h > 160) & (h < 180) & (s > 100) & (v > 40) & (v < 150)
    white_mask = (h > 80) & (h < 90) & (s < 30) & (v > 60) & (v < 110)

    # カウント
    blue_count = np.count_nonzero(blue_mask)
    red_count = np.count_nonzero(red_mask)
    yellow_count = np.count_nonzero(yellow_mask)
    white_count = np.count_nonzero(white_mask)

    com_count = np.count_nonzero(blue_mask | white_mask | red_mask)

    region_area = (x2 - x1) * (y2 - y1)

    # 条件判定
    if yellow_count > 25:
        cv2.rectangle(img, (x1-10, y1-10), (x2+10, y2+10), (0, 0, 255), thickness=3)
    elif com_count / region_area > 0.03:
        cv2.rectangle(img, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 0), thickness=3)

# 結果表示
resized_img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
cv2.imshow("Result", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("othello.pt")

results = model.predict("ex4.jpg", conf=0.4)

img = cv2.imread("ex4.jpg")

for box in results[0].boxes:
    x1 = int(box.data[0][0]) 
    y1 = int(box.data[0][1]) 
    x2 = int(box.data[0][2]) 
    y2 = int(box.data[0][3]) 
    region = img[y1:y2, x1:x2]

    if region.size == 0:
        continue

    region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(region_hsv)

    # 色マスク
    
    white_mask = (s < 30) & (v > 200)
    black_mask = (s < 100) & (v < 60)

    # カウント
    w_count = np.count_nonzero(white_mask)
    b_count = np.count_nonzero(black_mask)


    region_area = (x2 - x1) * (y2 - y1)

    # 条件判定
    if w_count / region_area > 0.6:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
    elif b_count/ region_area > 0.1:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

# 結果表示
resized_img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
cv2.imshow("Result", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

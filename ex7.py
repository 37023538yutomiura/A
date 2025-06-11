import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("othello.pt")

results = model.predict("ex4.jpg", conf=0.99)

img = cv2.imread("ex4.jpg")

for box in results[0].boxes:
    x1 = int(box.data[0][0]) 
    y1 = int(box.data[0][1]) 
    x2 = int(box.data[0][2]) 
    y2 = int(box.data[0][3]) 
    
    # 条件判定
    if box.cls == 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
    elif box.cls ==  1:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

# 結果表示
resized_img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
cv2.imshow("Result", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

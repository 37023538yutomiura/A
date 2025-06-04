import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
results = model.predict("ex2.jpg", classes=[0], conf=0.1)

img = cv2.imread("ex2.jpg")

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    region = img[y1:y2, x1:x2]

    if region.size == 0:
        continue

    
    region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(region)

    
    blue_mask = (h > 100) & (h < 130) & (s > 90) & (v > 60) 
    blue_ratio = blue_mask.sum() / (region.shape[0] * region.shape[1])

    if blue_ratio > 0.01:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

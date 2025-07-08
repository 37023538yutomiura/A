import cv2
from ultralytics import YOLO

img = cv2.imread("ex3.jpg")

model = YOLO("best.pt")

results = model.predict(img, conf=0.03)

count_b = (results[0].boxes.cls == 1).sum().item()
count_w = (results[0].boxes.cls == 0).sum().item()
img = results[0].plot()



resized_img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
print("black"+str(count_b))
print("white"+str(count_w))
cv2.imshow("Result", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict("ex2.jpg")

boxes = results[0].boxes
for box in boxes:
    print(box.data)
print(results[0].names)

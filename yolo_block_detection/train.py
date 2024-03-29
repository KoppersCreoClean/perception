from ultralytics import YOLO

yolo = YOLO('yolov8n.pt')
yolo.train(data="./data/data.yaml", epochs=1)
valid_results = yolo.val()
print(valid_results) 
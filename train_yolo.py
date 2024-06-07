from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data='carla.yaml', epochs=200, imgsz=640, device=1)
model.export()
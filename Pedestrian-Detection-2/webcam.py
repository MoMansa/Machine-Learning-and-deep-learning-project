from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/detect/train3/weights/best.pt")

# Run inference on the source
results = model(source=0, stream=True)  
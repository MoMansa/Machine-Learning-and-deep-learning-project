from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/detect/train3/weights/best.pt")

# Define path to video file
source = "testimg/video"

# Run inference on the source
results = model(source, save=True) 
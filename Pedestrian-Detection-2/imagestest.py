from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/detect/train3/weights/best.pt")

# Define path to the image file
source = "testimg/image_processing20201113-4192-dxns9z.jpg"

# Run inference on the source
results = model(source, save=True) 

# View results
for r in results:
    print(r.probs)
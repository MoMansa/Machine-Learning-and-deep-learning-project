import torch
import matplotlib.pyplot as plt
from torchvision import transforms

def test_model(model, test_loader):
    # Variables to store results
    all_predictions = []
    all_labels = []

    # Disable gradient updates
    with torch.no_grad():
        for images, rois, labels in test_loader:
            # Pass images through the model
            outputs = model(images, rois)  # Model predictions

            # For each image, extract RoIs and predictions
            for i in range(len(images)):
                print(f"Predicted RoIs: {outputs[i]}")  # Display predicted RoIs
                print(f"True RoIs: {rois[i]}")  # Display true RoIs

                # Collect labels for evaluation
                all_predictions.append(outputs[i])
                all_labels.append(labels[i])

                # Display the image and its RoIs
                image = transforms.ToPILImage()(images[i])
                plt.imshow(image)
                plt.show()

    return all_predictions, all_labels

def compute_iou(pred_bbox, true_bbox):
    # Calculate Intersection over Union (IoU) between two bounding boxes
    x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox
    x1_true, y1_true, x2_true, y2_true = true_bbox

    # Calculate intersection area
    xi1 = max(x1_pred, x1_true)
    yi1 = max(y1_pred, y1_true)
    xi2 = min(x2_pred, x2_true)
    yi2 = min(y2_pred, y2_true)

    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate union area
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    union_area = pred_area + true_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def calculate_loss(predictions, labels):
    # Define a loss function (e.g., Mean Squared Error)
    loss_fn = torch.nn.MSELoss()
    
    # Calculate the loss between predictions and labels
    loss = loss_fn(predictions.float(), labels.float())
    
    return loss

# Main function to test the model and dataset
def main():
    from rcnn_model import load_model
    from coco_dataset import COCODataset
    
    from torch.utils.data import DataLoader
    
    # Load the pre-trained model from a file (replace 'model.pth' with your model file path)
    model_path = 'path/to/your/model.pth'
    model = load_model(model_path, num_classes=2)

    # Paths to COCO images and annotations
    image_folder = 'coco/images/val2017/'  # Use the validation set for testing
    annotation_file = 'coco/annotations/instances_val2017.json'

    # Transformations for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create an instance of the COCO dataset for pedestrians
    dataset = COCODataset(image_folder, annotation_file, transform, category_name='person')
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Call the test function and calculate IoU and loss
    all_predictions, all_labels = test_model(model, test_loader)

    # Calculate IoU for each prediction and label pair
    ious = [compute_iou(pred_bbox, true_bbox) for pred_bbox, true_bbox in zip(all_predictions, all_labels)]

    # Calculate the average IoU
    average_iou = sum(ious) / len(ious)

    # Calculate the loss function value
    loss_value = calculate_loss(torch.tensor(all_predictions), torch.tensor(all_labels))

    print(f"Average IoU: {average_iou}")
    print(f"Loss: {loss_value}")

if __name__ == "__main__":
    main()
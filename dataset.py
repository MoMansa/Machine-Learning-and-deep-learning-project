import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms

class COCODataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None, category_name='person'):
        self.image_folder = image_folder
        self.coco = COCO(annotation_file)
        self.transform = transform

        # Get the category ID (e.g., for pedestrians)
        self.category_id = self.coco.getCatIds(catNms=[category_name])[0]

        # Retrieve all image IDs corresponding to this category
        self.image_ids = self.coco.getImgIds(catIds=[self.category_id])

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Load image information
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_folder, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Load annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.category_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        # Extract RoIs (bounding boxes)
        rois = []
        for ann in anns:
            x, y, width, height = ann['bbox']
            rois.append([0, x, y, x + width, y + height])  # [batch_index, x1, y1, x2, y2]

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        rois = torch.tensor(rois, dtype=torch.float)

        # Label: 1 for pedestrian (or another value as needed)
        label = torch.tensor([1] * len(rois), dtype=torch.long)

        return image, rois, label

# Main function to test the dataset
def main():
    # Paths to COCO images and annotations
    image_folder = 'coco/images/train2017/'
    annotation_file = 'coco/annotations/instances_train2017.json'

    # Transformations for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create an instance of the COCO dataset for pedestrians
    dataset = COCODataset(image_folder, annotation_file, transform, category_name='person')

    # Test loading the first image
    image, rois, label = dataset[0]
    print(f"Image size: {image.shape}")
    print(f"RoIs: {rois}")
    print(f"Label: {label}")

if __name__ == "__main__":
    main()

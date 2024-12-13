## Pedestrian Detection with RCNN

This repository contains the implementation of a pedestrian detection model using a Region-based Convolutional Neural Network (RCNN) with the COCO dataset.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/MoMansa/pedestrian-detection-rcnn.git
cd pedestrian-detection-rcnn
pip install -r requirements.txt
```

## Usage

### Training
To train the model, run the `train.py` script. Make sure to adjust the paths to your dataset.

```bash
python train.py
```

### Testing
To test the model, run the `test_functions.py` script. Ensure that the path to the pre-trained model is correctly specified.

```bash
python test_functions.py
```

## Dataset
The model is trained and evaluated on the COCO dataset. You can download the COCO dataset from the official COCO website.

### Comparison with Yolo 
We also include a comparison of our RCNN model with the YOLO model. This comparison evaluates the performance of both models on the COCO dataset in terms of accuracy, speed, and Intersection over Union (IoU).

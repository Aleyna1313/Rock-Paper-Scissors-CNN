# Rock-Paper-Scissors Recognition using CNN

An experimental real-time hand gesture recognition project developed using **PyTorch and OpenCV**. The system aims to classify three hand gestures (Rock, Paper, Scissors) from live webcam input using a custom Convolutional Neural Network (CNN).

**Project Status:** Experimental Prototype / Under Development

## Technologies Used

- **Python:** Model development and data processing
- **PyTorch:** Custom CNN architecture, training and inference
- **OpenCV:** Real-time webcam capture and image processing
- **Torchvision:** Image transformations and data augmentation
- **NumPy / Pillow:** Image and numerical processing

## CNN Architecture

The model was implemented from scratch using PyTorch's `nn.Module`, without relying on a pretrained CNN architecture.

The network consists of:

- Three convolutional layers with 16, 32 and 64 output channels
- ReLU activation functions
- MaxPooling layers for spatial downsampling
- A fully connected classification head (4096 → 128 → 3)

**Input:** RGB images resized to 64 × 64 pixels

**Output:** Three gesture classes: `fist`, `palm`, `victory`

## Dataset and Preprocessing

The training dataset combines personally collected hand gesture images with images from an external dataset.

External dataset: [Rock Paper Scissors Dataset - Kaggle](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset)

The data preparation pipeline includes:

- Training, validation and test dataset separation
- Image resizing to 64 × 64
- Random rotation and brightness/contrast adjustments for training data
- Tensor conversion and normalization
- Batch-based loading using PyTorch DataLoader

## Model Training

The CNN is trained using:

- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Epochs:** 10

Validation loss and accuracy are monitored during training, and model weights are saved when validation accuracy improves.

## Real-Time Gesture Recognition

OpenCV is used to capture webcam frames and process a predefined Region of Interest (ROI).

The selected image region is converted from BGR to RGB, resized and normalized before being passed to the trained CNN.

The prediction and confidence score are displayed on the camera feed. Predictions below the configured 85% confidence threshold are labeled as `UNKNOWN`.

## Project Structure

```text
gesture_project/
├── src/
│   ├── collect_data.py
│   ├── dataset.py
│   ├── inference.py
│   ├── model.py
│   ├── split_data.py
│   └── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset, virtual environment and trained model weights are excluded from the repository.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Aleyna1313/Rock-Paper-Scissors-CNN.git
cd Rock-Paper-Scissors-CNN
python -m pip install -r requirements.txt
```

To run the project, the dataset must first be prepared and the CNN trained to generate model weights. The resulting model file is required for webcam inference.

## Current Limitations and Future Improvements

The model can recognize some gestures correctly under specific conditions. However, prediction performance varies depending on hand position within the camera frame.

Planned improvements include:

- Increasing the diversity of training images
- Improving generalization across different hand positions
- Experimenting with additional data augmentation techniques
- Evaluating model performance on unseen test data
- Improving robustness under different lighting and background conditions

This repository documents an ongoing learning and development project focused on CNN architecture design, model training and real-time computer vision.
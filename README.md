# Malaria Detection from Blood Cell Images Using a Custom CNN

## Overview

This project implements a custom Convolutional Neural Network (CNN) to
classify microscopic blood cell images as either Parasitized or Uninfected.

The objective is to demonstrate how deep learning and image classification
can be applied to a healthcare-related problem. The model is created and
trained from scratch without using a pretrained architecture.

## Problem Statement

Malaria is caused by parasites that infect red blood cells. Examination of
microscopic blood-smear images can help identify infected cells. Manual
examination can be time-consuming and depends on trained personnel.

This project investigates whether a custom CNN can learn visual features from
blood cell images and classify individual cells into two categories:

- Parasitized
- Uninfected

## Dataset

The project uses a publicly available collection of microscopic blood cell
images. Each image belongs to one of two classes:

1. Parasitized — cells containing visible malaria parasites
2. Uninfected — cells without visible malaria parasites

The dataset is divided into training, validation, and test sets. The test set
is kept separate and is used only for final model evaluation.

The dataset is not included in this repository because of its size. Follow
the dataset setup instructions below to download and organize it.

## Image Preprocessing

The following preprocessing steps are applied:

- Resize every image to 128 × 128 pixels
- Convert images to RGB format
- Normalize pixel values from 0–255 to 0–1
- Divide the dataset into training, validation, and test sets
- Apply data augmentation only to training images

Data augmentation may include:

- Small rotations
- Horizontal and vertical translations
- Zooming
- Horizontal flipping

Augmentation improves variation in the training data and helps reduce
overfitting.

## Model Architecture

A custom CNN is implemented instead of using a pretrained network.

The convolutional layers learn visual features such as cell boundaries,
textures, colour patterns, and parasite-like structures. Max-pooling layers
reduce the spatial dimensions of the feature maps. Dropout is used to reduce
overfitting.

The final layer contains one neuron with sigmoid activation because this is a
binary classification problem.

## Training Configuration

- Framework: TensorFlow/Keras
- Input size: 128 × 128 × 3
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Batch size: 32
- Maximum epochs: 25
- Output activation: Sigmoid
- Model checkpointing: Enabled
- Early stopping: Enabled
- Learning-rate reduction: Enabled

## Evaluation Metrics

The model is evaluated using:

- Accuracy: Overall proportion of correct predictions
- Precision: Proportion of predicted infected cells that are actually infected
- Recall: Proportion of infected cells correctly detected by the model
- F1-score: Harmonic mean of precision and recall
- Confusion matrix: Counts of correct and incorrect predictions by class

Recall is particularly relevant in this experiment because a false-negative
prediction means that a parasitized cell was classified as uninfected.

## Results

| Metric | Test Result |
|---|---:|
| Accuracy | 0.96 |
| Precision | 0.96 |
| Recall | 0.96 |
| F1-score | 0.96 |
| ROC-AUC | 0.9835 |

The repository also contains:

- Training and validation accuracy curves
- Training and validation loss curves
- Confusion matrix
- Sample predictions

## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/malaria-detection-cnn.git
cd malaria-detection-cnn
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Evaluate it:

```bash
python evaluate.py
```

Predict a single image:

```bash
python predict.py --image path/to/cell_image.png
```

To test on a new custom image (e.g., from your computer):

```bash
.\.venv\Scripts\python.exe predict.py --image "C:\Users\hp\Downloads\test_cell.png"
```

## Limitations

- Performance depends on the quality and diversity of the dataset.
- Images from different microscopes or laboratories may have different visual
  characteristics.
- A cell-level classification result is not equivalent to a patient-level
  malaria diagnosis.
- Dataset bias and image leakage can produce misleading results.
- The model has not been clinically validated.

## Disclaimer

This project is intended only for educational and research demonstration
purposes. It is not a medical device and must not be used for clinical
diagnosis or treatment decisions. Malaria diagnosis should be performed by
qualified healthcare professionals using approved procedures.

## Conclusion

This project demonstrates the implementation of a custom CNN for classifying
microscopic blood cell images as parasitized or uninfected. The CNN learns
image features directly from the training data and is evaluated on a separate
test set using multiple classification metrics.

Future improvements could include testing additional CNN architectures,
improving dataset splitting, performing external validation, and using model
interpretability techniques such as Grad-CAM.

## Author

Mohit Anuragi  
Roll Number: 2023KUCP1155

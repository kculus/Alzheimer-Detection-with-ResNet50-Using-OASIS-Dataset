# Alzheimer's Detection with ResNet50 Using OASIS Dataset

This repository contains a comprehensive implementation for detecting the stages of Alzheimer's Disease using deep learning. The project employs the **ResNet50** architecture on the **OASIS Alzheimer's MRI Dataset**, with a focus on classifying brain MRI images into four stages of Alzheimer's progression.

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Model Architecture](#model-architecture)
4. [Training Details](#training-details)
5. [Results and Evaluation](#results-and-evaluation)
6. [References and Acknowledgements](#references-and-acknowledgements)

---

## **Project Overview**

Alzheimer's is a progressive brain disorder that impacts memory and cognitive functions. Early detection is critical for effective management. This project focuses on classifying MRI brain scans into the following categories:

1. **Mild Dementia**
2. **Moderate Dementia**
3. **Non-Demented**
4. **Very Mild Dementia**

Using the power of deep learning and transfer learning, the ResNet50 model is fine-tuned to achieve high accuracy in classification. The training process ensures the model generalizes well to unseen data.

---

## **Dataset Information**

We used the **OASIS Alzheimer's Detection**, which is available on Kaggle. The dataset contains MRI images that are preprocessed and categorized into four classes based on Alzheimer's progression.

- **Classes**:
  - Mild Dementia: 5002 images
  - Moderate Dementia: 488 images
  - Non-Demented: 67,200 images
  - Very Mild Dementia: 13,700 images

- **Dataset Source**: [OASIS Dataset on Kaggle](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data)
- **Preprocessing Steps**:
  - Images resized to 496x248 pixels
  - Normalization applied to scale pixel values
  - Data split into training (80%) and validation (20%) sets

---

## **Model Architecture**

The model uses the **ResNet50** architecture, a deep residual learning framework that is pre-trained on ImageNet. The final fully connected layer is replaced to match the number of classes in this task.

- **Base Model**: ResNet50
- **Modifications**:
  - Final fully connected layer replaced with a layer of size 4 (corresponding to the 4 classes)
  - Weighted Cross-Entropy Loss used to handle class imbalance
- **Optimizer**: Adam Optimizer
- **Learning Rate**: 0.001 with a step decay scheduler

---

## **Training Details**

### **Steps in Training**:
1. **Data Augmentation**:
   - Random horizontal flips and rotations were applied to diversify the training data.
2. **Weighted Loss Function**:
   - Class weights were calculated to handle class imbalance, ensuring the model pays attention to underrepresented classes.
3. **Early Stopping**:
   - The training was stopped when no improvement was observed in validation accuracy for 3 consecutive epochs.

### **Key Hyperparameters**:
- Batch Size: 32
- Number of Epochs: 7
- Learning Rate: 0.001

---

## **Results and Evaluation**

### **Training and Validation Accuracy**:
- The model achieved a **validation accuracy of 99.05%** during training.

### **Confusion Matrix**:
- Detailed class-wise performance was analyzed using a confusion matrix. 

### **Validation Loss**:
- The lowest validation loss recorded was **0.0380**.

---

## References and Acknowledgements

- **Dataset**: The dataset used in this project is the OASIS Alzheimer's MRI dataset, which can be accessed on Kaggle:
  [OASIS Dataset on Kaggle](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data)

- **Kaggle Notebook**: The original notebook containing the complete code, results, and visualizations can be accessed here:
  [View Notebook on Kaggle](https://www.kaggle.com/code/kasmcanulus/alzheimer-detection-with-resnet50)
---

### Note

This repository contains the modularized version of the original Kaggle notebook. The code here can be run **directly on Kaggle** if the dataset path is updated appropriately. Simply copy the repository files to your Kaggle notebook and ensure the dataset is loaded into the `/kaggle/input` directory.



# Using Pre-trained Vision Models with PyTorch

This project evaluates multiple pre-trained convolutional neural networks from `torchvision.models` on image classification tasks using the CIFAR-10 dataset. The aim is to compare model performance and training behavior in a unified setup.

---

## 🎯 Objective

- Load and adapt pre-trained models (ResNet18, VGG16, MobileNetV2)
- Modify their final layers to fit CIFAR-10 (10 classes)
- Train each model for 5 epochs
- Evaluate and compare performance based on test accuracy

---

## 🧠 Models Used

- ✅ ResNet18
- ✅ VGG16
- ✅ MobileNetV2

All models were modified by replacing the final fully connected (FC) layer.

---

## 🧪 Dataset

- **Dataset**: CIFAR-10
- **Input shape**: Resized to 224×224
- **Transforms**:
  - `transforms.Resize((224, 224))`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean, std)`

---

## ⚙️ Training Configuration

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 5
- **Device**: GPU when available (e.g., Kaggle, Colab)

---

## 📁 Files

- `assignment-3.ipynb`: Full implementation and results
- `deep_learning_assignment.pdf`: Project requirements and instructions

---

## 📌 Notes

- All training loops include tqdm progress bars
- Each model was trained independently and fairly
- Reproducible using PyTorch and torchvision

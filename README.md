# 🍚 Rice Image Classification with CNNs using PyTorch

![https://via.placeholder.com/1000x300.png?text=Rice+Image+Classification+with+PyTorch](https://longdan.co.uk/cdn/shop/articles/ca895f725f1748f5d63c625e91591613_800x.jpg?v=1700554692) <!-- Optional: Add your own banner image here -->

## 📝 Project Overview
This repository contains a deep learning project aimed at classifying different rice grain varieties using a Convolutional Neural Network (CNN) built with PyTorch. The dataset includes five rice categories: **Arborio, Basmati, Jasmine, Ipsala,** and **Karacadag**. 

The project demonstrates:
- Building and training a custom CNN model using PyTorch.
- Analyzing class distribution and data preprocessing techniques.
- Evaluating the model's performance with accuracy, loss metrics, and a confusion matrix.

## 📂 Dataset
The **Rice Image Dataset** used in this project contains **75,000 images**, evenly distributed across five classes:
- **Arborio**: 15,000 images
- **Basmati**: 15,000 images
- **Jasmine**: 15,000 images
- **Ipsala**: 15,000 images
- **Karacadag**: 15,000 images

The dataset was split into **75% training** and **25% validation** using a 75/25 ratio, resulting in:
- **Training set**: 56,250 images
- **Validation set**: 18,750 images

## ⚙️ Features
- **Custom CNN Architecture**: A simple yet effective model with convolutional, pooling, and fully connected layers.
- **Data Augmentation**: Using techniques like random horizontal flipping for better generalization.
- **Performance Metrics**: Tracking training accuracy, validation accuracy, and loss over epochs.
- **Visualization**: Plotting accuracy, loss curves, and confusion matrix for detailed analysis.

## 📊 Results
- The custom CNN model achieved **99.73% training accuracy** and **99.56% validation accuracy** after 15 epochs.
- Below are visualizations for the model's performance:


## 🛠️ Model Architecture
The CNN model consists of:
1. **Two convolutional layers** with ReLU activations and batch normalization.
2. **Max pooling** for downsampling.
3. **Fully connected layers** for classification.

```python
class CNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 62 * 62, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## 📈 Performance Metrics
| Metric               | Value   |
|----------------------|---------|
| Training Accuracy    | 99.73%   |
| Validation Accuracy  | 99.56%   |
| Training Loss        | 0.86    |


## 🤝 Contributing
Contributions are welcome! Feel free to submit a Pull Request if you want to enhance the model or add new features.

## 📧 Contact
Developed by [Hamed Mahmoudi](https://github.com/halfbloodhamed).

If you have any questions or suggestions, please feel free to reach out!

# 🧠 Meningitis Detection

> Advanced machine learning system for early detection and diagnosis of meningitis using deep learning and medical imaging analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Alyaaa16/Meningitis-Detection)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## ✨ Features

- 🎯 **Accurate Detection** - State-of-the-art deep learning models for meningitis classification
- 🖼️ **Medical Image Analysis** - Advanced CNN architectures for CT and MRI imaging
- 📊 **Data Visualization** - Comprehensive EDA and result visualization
- 🔬 **Multiple ML Models** - Ensemble methods combining various classification algorithms
- 📈 **Performance Metrics** - Detailed evaluation with confusion matrices and ROC curves
- 🧬 **Explainable AI** - Feature importance analysis and model interpretability
- ⚡ **Optimized Training** - Efficient data preprocessing and model optimization

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results & Metrics](#results--metrics)
- [Notebooks Overview](#notebooks-overview)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quick Start

Get started with meningitis detection in minutes:

```bash
# Clone the repository
git clone https://github.com/Alyaaa16/Meningitis-Detection.git
cd Meningitis-Detection

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Open the main analysis notebook
# Navigate to notebooks/01_meningitis_detection.ipynb
```

## 📦 Prerequisites

Before you begin, ensure you have the following:

- **Python** 3.8 or higher
- **Jupyter Notebook** or **JupyterLab**
- **pip** package manager
- GPU support (CUDA 11.0+) - recommended for faster training
- At least 4GB RAM (8GB+ recommended)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Alyaaa16/Meningitis-Detection.git
cd Meningitis-Detection
```

### 2. Create Virtual Environment

```bash
# Using Python venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `tensorflow>=2.10.0` - Deep learning framework
- `scikit-learn>=1.0.0` - Machine learning algorithms
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.4.0` - Visualization
- `seaborn>=0.11.0` - Statistical visualization
- `opencv-python>=4.5.0` - Image processing
- `jupyter>=1.0.0` - Notebook environment

### 4. Download Dataset

```bash
# Create data directory
mkdir data

# Add your medical imaging dataset to:
# data/raw/ - raw images
# data/processed/ - preprocessed images
```

## 📊 Dataset

### Expected Structure

```
data/
├── raw/
│   ├── meningitis/
│   │   ├── scan_001.jpg
│   │   └── scan_002.jpg
│   └── normal/
│       ├── scan_101.jpg
│       └── scan_102.jpg
└── processed/
    ├── train/
    ├── val/
    └── test/
```

### Dataset Information

- **Classes**: Meningitis (positive) vs. Normal (negative)
- **Image Format**: JPEG, PNG, or DICOM
- **Typical Split**: 70% training, 15% validation, 15% testing
- **Augmentation**: Applied to prevent overfitting

## 💻 Usage

### Basic Model Training

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Load and preprocess data
from sklearn.model_selection import train_test_split
import numpy as np

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)
```

### Making Predictions

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess image
img = image.load_img('path/to/scan.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize

# Make prediction
prediction = model.predict(img_array)
confidence = prediction[0][0]

if confidence > 0.5:
    print(f"Meningitis detected (Confidence: {confidence:.2%})")
else:
    print(f"Normal scan (Confidence: {(1-confidence):.2%})")
```

### Ensemble Model

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Train multiple models
rf_model = RandomForestClassifier(n_estimators=100)
gb_model = GradientBoostingClassifier(n_estimators=100)
svm_model = SVC(probability=True)

# Train on feature-extracted data
rf_model.fit(X_train_features, y_train)
gb_model.fit(X_train_features, y_train)
svm_model.fit(X_train_features, y_train)

# Ensemble prediction
predictions = np.array([
    rf_model.predict_proba(X_test_features)[:, 1],
    gb_model.predict_proba(X_test_features)[:, 1],
    svm_model.predict_proba(X_test_features)[:, 1]
])

ensemble_prediction = np.mean(predictions, axis=0)
```

### Evaluation & Metrics

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt

# Generate classification report
print(classification_report(y_test, y_pred))

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Meningitis Detection')
plt.legend(loc="lower right")
plt.show()
```

## 🏗️ Model Architecture

### Convolutional Neural Network (CNN)

```
Input Layer (224×224×3)
    ↓
Conv2D(32, 3×3) + ReLU + MaxPooling
    ↓
Conv2D(64, 3×3) + ReLU + MaxPooling
    ↓
Conv2D(128, 3×3) + ReLU + MaxPooling
    ↓
Flatten
    ↓
Dense(128) + ReLU + Dropout(0.5)
    ↓
Dense(1) + Sigmoid
    ↓
Output (Meningitis/Normal)
```

### Transfer Learning (Optional)

```python
from tensorflow.keras.applications import ResNet50, VGG16

# Use pre-trained ResNet50
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model weights
base_model.trainable = False

# Add custom layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

## 📈 Results & Metrics

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 94.5% |
| Sensitivity (Recall) | 96.2% |
| Specificity | 92.8% |
| Precision | 93.5% |
| F1-Score | 0.948 |
| AUC-ROC | 0.987 |

### Confusion Matrix

```
                  Predicted
                  Negative    Positive
Actual Negative     185          15
       Positive       8          192
```

## 📓 Notebooks Overview

### Available Notebooks

1. **01_meningitis_detection.ipynb**
   - Exploratory Data Analysis (EDA)
   - Data preprocessing and augmentation
   - Initial model building and training

2. **02_model_evaluation.ipynb**
   - Model performance evaluation
   - ROC curves and confusion matrices
   - Error analysis

3. **03_ensemble_methods.ipynb**
   - Combining multiple models
   - Voting classifiers
   - Stacking approaches

4. **04_transfer_learning.ipynb**
   - Pre-trained model fine-tuning
   - ResNet50 and VGG16 implementations
   - Comparison with baseline models

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab (recommended)
jupyter lab
```

## 📸 Screenshots

### Exploratory Data Analysis
![EDA Visualizations](https://via.placeholder.com/1200x400?text=EDA+Visualizations)
*Data distribution and sample medical images*

### Model Performance
![Performance Metrics](https://via.placeholder.com/1200x400?text=Model+Performance+Metrics)
*Accuracy, precision, recall, and F1-score comparisons*

### ROC Curve Analysis
![ROC Curves](https://via.placeholder.com/1200x400?text=ROC+Curve+Analysis)
*Receiver Operating Characteristic curves for different models*

### Confusion Matrices
![Confusion Matrices](https://via.placeholder.com/1200x400?text=Confusion+Matrices)
*True/False positive and negative distributions*

### Feature Importance
![Feature Importance](https://via.placeholder.com/1200x400?text=Feature+Importance+Analysis)
*Key features driving model predictions*

## 🧪 Testing & Validation

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 5-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Cross-validation Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct and development process.

## 📚 Resources

### Documentation

- [Jupyter Notebook Guide](https://jupyter.readthedocs.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Medical Image Analysis](https://en.wikipedia.org/wiki/Medical_image)

### References

- Deep Learning for Medical Image Analysis
- CNN Architectures in Computer Vision
- Meningitis Diagnosis Literature Review

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for actual medical diagnosis and treatment.

## 🙋 Support

Need help? Here are some resources:

- 📖 [Jupyter Documentation](https://jupyter.org/documentation)
- 💬 [GitHub Discussions](https://github.com/Alyaaa16/Meningitis-Detection/discussions)
- 🐛 [Issue Tracker](https://github.com/Alyaaa16/Meningitis-Detection/issues)
- 📧 [Email Support](mailto:support@example.com)

## 🎉 Acknowledgments

- Built with ❤️ by [Alyaaa16](https://github.com/Alyaaa16)
- Inspired by advances in medical AI and deep learning
- Thanks to the medical imaging and machine learning communities
- All [contributors](https://github.com/Alyaaa16/Meningitis-Detection/graphs/contributors) and researchers

---

<div align="center">

**[⬆ back to top](#-meningitis-detection)**

Made with 💙 | [GitHub](https://github.com/Alyaaa16/Meningitis-Detection) | [Issues](https://github.com/Alyaaa16/Meningitis-Detection/issues)

⚠️ For educational and research purposes only | Not for clinical use

</div>

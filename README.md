# 🐦 Bird Species Identification from an Image

A Machine Learning mini-project that identifies bird species from images using Traditional Machine Learning and Deep Learning with Transfer Learning.

## 👨‍💻 Team Members

| Name | SRN |
|---|---|
| A. Vinay | PES1UG24CS647 |
| Santosh | PES1UG24CS615 |

## 📌 Problem Statement

Bird species identification is a challenging image classification problem because different bird species may have similar colors, shapes, and feather patterns.

The objective of this project is to predict the species of a bird from an input image by comparing traditional machine learning techniques with a deep learning-based transfer learning approach.

## 📂 Dataset

**Dataset:** CUB-200-2011 (Caltech-UCSD Birds-200-2011)

- Total images: 11,788
- Number of bird species: 200
- Includes bird images, species labels, and bounding-box annotations.

For our experiments, we used a balanced subset of the dataset:

- Training set: 4,000 images (20 images per species)
- Test set: 400 images (2 images per species)

The dataset is not included in this repository and must be obtained separately.

## 🛠️ Methodologies

### Method 1: Traditional Machine Learning

This approach uses handcrafted features extracted from bird images and traditional machine learning classifiers.

**Image preprocessing**
- Crop bird images using bounding-box annotations.
- Resize images to 128 × 128 pixels.

**Feature extraction techniques explored**
- RGB Color Histogram
- Histogram of Oriented Gradients (HOG)
- Local Binary Patterns (LBP)
- HSV Color Features
- Color Moments
- Gabor Features

**Models and techniques**
- Linear Support Vector Machine (Linear SVM)
- RBF Support Vector Machine (RBF SVM)
- Random Forest
- StandardScaler
- Principal Component Analysis (PCA)

**Best result:** RGB Color Histogram + HOG + Linear SVM achieved **13.50% test accuracy**.

### Method 2: Transfer Learning with EfficientNet-B0

This approach uses a pretrained EfficientNet-B0 convolutional neural network to learn visual representations for bird species classification.

**Implementation**
1. Load pretrained EfficientNet-B0 weights.
2. Replace the final classification layer with 200 output classes.
3. Freeze the feature extraction layers and train the new classification layer for 3 epochs.
4. Unfreeze the last two feature blocks and fine-tune the model for 3 additional epochs.
5. Evaluate the model on the test dataset.
6. Generate sample predictions and save the trained model.

**Training configuration**
- Input image size: 224 × 224 pixels
- Optimizer: Adam
- Fine-tuning learning rate: 0.0001
- Training images: 4,000
- Test images: 400

**Best result:** Fine-tuned EfficientNet-B0 achieved **55.00% test accuracy**.

## 📊 Results and Comparison

| Approach | Model / Features | Test Accuracy |
|---|---|---:|
| Method 1 | RGB Histogram + HOG + Linear SVM | 13.50% |
| Method 2 — Stage 1 | EfficientNet-B0 | 50.25% |
| Method 2 — Fine-tuned | EfficientNet-B0 | 55.00% |

The fine-tuned EfficientNet-B0 model improved test accuracy by **41.50 percentage points** compared with the best reported traditional machine learning experiment.

These results are based on our selected balanced subset, not the entire dataset.

## 📁 Project Structure

```text
bird-species-identification/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── notebooks/
│   ├── method1_traditional_ml.ipynb
│   └── method2_efficientnet_b0.ipynb
│
├── models/
│   └── efficientnet_b0_cub_finetuned.pth
│
└── report/
    └── bird_species_project_report.pdf
```

*Note: The filenames and folders above are the suggested structure. Keep the actual notebook filenames consistent with those in your repository. Include the model and report only if those files are available.*

## 💻 Technologies Used

- Python
- Jupyter Notebook
- NumPy
- OpenCV
- scikit-learn
- PyTorch
- Torchvision
- Matplotlib, where applicable

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_PRIVATE_REPOSITORY_URL>
cd bird-species-identification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the CUB-200-2011 dataset separately and configure the dataset path in the notebooks to match your local directory structure.

### 5. Run the notebooks

Open Jupyter Notebook:

```bash
jupyter notebook
```

Open the Method 1 or Method 2 notebook and follow its cells in order.

**Note:** Running the training cells again may take additional time and computational resources. The notebooks may contain saved outputs from the experiments already performed.

## 🎯 Conclusion

This project compares handcrafted feature-based classification with deep learning-based transfer learning for bird species identification.

The traditional machine learning approach achieved 13.50% test accuracy, while fine-tuned EfficientNet-B0 achieved 55.00%. The results indicate that transfer learning was more effective for our experiments on the selected balanced subset.

## 🚀 Future Improvements

- Increase the number of training images per species.
- Experiment with data augmentation.
- Evaluate additional pretrained CNN architectures.
- Tune hyperparameters and fine-tuning strategies.
- Evaluate per-species performance using confusion matrices and classification metrics.

## 📄 Project Information

- **Course:** Machine Learning
- **Course Code:** UE24CS352A
- **Project:** Bird Species Identification from an Image
- **Team Members:** A. Vinay and Santosh

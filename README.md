# Deep Learning Project Group 11
Automated Vulnerability Detection in Source Code Using Deep Representation Learning

Dataset Google Drive Link: https://drive.google.com/drive/folders/1X1KaFmMeyDEzOm9pmkaFm0sePfSh2p4k?usp=sharing


# Automated Vulnerability Detection Using CNN and Ensemble Learning

## 📚 Project Overview
This project focuses on detecting software vulnerabilities automatically from C/C++ source code. We apply deep learning techniques, specifically a Convolutional Neural Network (CNN) architecture, to extract features from tokenized code functions. To enhance performance, we explore using Random Forest classifiers on CNN-extracted features, and experiment with hyperparameter tuning to optimize model performance.

The project consists of three main stages:
- **SVD_1:** CNN model + Random Forest ensemble classification.
- **SVD_2:** Improved CNN model (adjusted hyperparameters) + Random Forest ensemble.
- **SVD_3:** CNN direct end-to-end classification without ensemble.

---

## 📁 Dataset
- **Source:** Lexed and tokenized C/C++ functions collected from the SATE IV Juliet Test Suite, Debian repositories, and GitHub repositories.
- **Size:** ~1,094,129 functions.
- **Labeling:** Binary classification for vulnerability presence (e.g., CWE-119, CWE-120, CWE-469, CWE-476, CWE-other).
- **Split:** 
  - 80% Training
  - 10% Validation
  - 10% Testing

Each function is converted into a sequence of token IDs using a vocabulary extracted from the source dataset.

---

## 🏗️ Methodology

- **CNN Model:**
  - Embedding layer to convert token IDs into dense vector representations.
  - 1D Convolutional layers to learn local syntactic patterns.
  - MaxPooling and Dropout layers to prevent overfitting.
  - Dense layers for feature abstraction.
  - Final Dense layer for binary output (sigmoid activation).

- **Random Forest Classifier (SVD_1 and SVD_2):**
  - Features are extracted from the penultimate Dense layer (before final sigmoid).
  - Random Forest is trained on CNN-extracted features for improved robustness.

- **Hyperparameter Tuning (SVD_2):**
  - Experimented with kernel size, number of convolution filters, embedding dimension, dropout rate, and dense layer sizes.

- **Direct CNN Classification (SVD_3):**
  - CNN is trained end-to-end, predicting vulnerabilities directly from source code tokens.

---

## ⚙️ Notebooks and Their Purpose

| Notebook | Purpose |
|:--|:--|
| `CNN.ipynb` | Build and train a basic CNN model on tokenized functions; direct binary classification. |
| `CNN_hyperparameters.ipynb` | Explore different CNN hyperparameter settings and retrain to observe improvements. |
| `CNN_RF.ipynb` | Extract CNN features and train a Random Forest classifier to improve final classification performance. |

---

## 🚀 How to Run the Code

1. **Setup Environment:**
   - Install required libraries:
     ```bash
     pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
     ```

2. **Run Notebooks Sequentially:**
   - **Step 1:** Open `CNN.ipynb`
     - Load the dataset.
     - Tokenize and preprocess data.
     - Build and train the base CNN model.
     - Evaluate direct classification performance.
   - **Step 2:** Open `CNN_hyperparameters.ipynb`
     - Modify and experiment with CNN hyperparameters.
     - Retrain the CNN and observe improvements in validation/test performance.
   - **Step 3:** Open `CNN_RF.ipynb`
     - Use CNN feature extractor outputs.
     - Train a Random Forest classifier.
     - Evaluate combined CNN+RF model.

3. **Important Notes:**
   - Ensure tokenization and input vector preparation match across notebooks.
   - GPU usage is highly recommended for CNN training.

---

## 📊 Results Summary

| Metric | SVD_1 (CNN + RF) | SVD_2 (Improved CNN + RF) | SVD_3 (Direct CNN) |
|:--|:--|:--|:--|
| Accuracy | 97.02% | 96.65% | 90.71% |
| Precision | 35.66% | 33.19% | 37.02% |
| Recall | 68.23% | 72.76% | 61.71% |
| F1-Score | 46.84% | 45.59% | 46.27% |
| ROC AUC | 0.9452 | 0.9437 | 0.8831 |
| MCC | 0.4802 | 0.4776 | 0.4317 |

- **SVD_1** achieved the best overall balance between precision and recall.
- **SVD_2** slightly improved recall but with minor trade-offs.
- **SVD_3** had higher precision but lower overall AUC and MCC.

---

## 📈 Visualizations
- PCA (Principal Component Analysis) applied to CNN features to visualize feature separation between vulnerable and non-vulnerable samples.
- Confusion matrices and ROC curves generated for model evaluation.
<img src ="CNN acc and loss.png>
---

## 🧠 Future Work
- Fine-tune Random Forest hyperparameters (depth, number of trees).
- Explore transformer-based models like CodeBERT for source code understanding.
- Extend classification to multi-label or multi-class vulnerabilities.

---

# 🙌 Acknowledgments
- This project is inspired by Russell et al., "Automated Vulnerability Detection in Source Code using Deep Representation Learning," ICMLA 2018.

---

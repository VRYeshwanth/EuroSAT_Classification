# 🛰️ EuroSAT Land-Use Classification

A PyTorch-based image classification project using the **EuroSAT RGB dataset** to explore convolutional neural networks, transfer learning, and fine-tuning. The project compares a custom CNN trained from scratch with an ImageNet-pretrained **ResNet-18**, first using it as a frozen feature extractor and then fine-tuning its deeper layers.

## 📋 Project Overview

The main goal of this project was to understand:

- How CNNs learn image features from scratch
- How pretrained models can be reused for a new classification task
- Feature extraction using a frozen pretrained backbone
- Fine-tuning pretrained layers
- The importance of train/validation/test separation
- Model evaluation using accuracy, precision, recall, and confusion matrices

Three different approaches were trained and compared:

1. **Custom CNN** — trained completely from scratch
2. **ResNet-18 with a frozen backbone** — only a new classifier was trained
3. **ResNet-18 with fine-tuning** — the final ResNet block (`layer4`) and classifier were trained

---

## 🗺️ Dataset

The project uses the **EuroSAT RGB dataset**, which contains satellite images belonging to 10 different land-use and land-cover classes. The RGB version contains **27,000 images**, with each image having a resolution of `64 × 64`.

### Classes

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

### Dataset Split

The original dataset was divided into:

- **80% Training**
- **10% Validation**
- **10% Test**

The split was performed independently for each class to maintain approximately the same class distribution across the three sets. A fixed random seed (`42`) was used to make the split reproducible.

The test set was kept completely separate during model development and was only used for the final evaluation.

---

## 📁 Project Structure

```text
EuroSAT_Classification/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── EuroSAT_RGB/
│   └── Original EuroSAT RGB dataset
│
├── images/
│   ├── Eurosat_Classes.png
│   └── evaluation/
│       ├── CustomCNN Confusion Matrix.png
│       ├── TL (Fine-Tuned) Confusion Matrix.png
│       └── TL (Frozen Backbone) Confusion Matrix.png
│
├── models/
│   ├── CustomCNNModel.pt
│   ├── TL_FineTune_Backbone.pt
│   └── TL_Frozen_Backbone.pt
│
├── notebooks/
│   ├── 00. data_exploration.ipynb
│   ├── 01. CustomCNN.ipynb
│   ├── 02. TransferLearning.ipynb
│   ├── 03. TL_Fine_Tuning.ipynb
│   └── 04. Evaluation.ipynb
│
├── README.md
├── requirements.txt
└── utils.py
```

---

# 1. Custom CNN

The first model was a custom convolutional neural network trained entirely from scratch.

The network contains multiple convolutional blocks with:

- Convolution layers
- Batch Normalization
- ReLU activations
- Max Pooling
- Dropout
- Fully connected layers

Since the model was initialized randomly, it had to learn the visual features of satellite images directly from the EuroSAT training data.

### Training & Validation

- Final Training Accuracy: **96.79%**
- Best Validation Accuracy: **96.96%**

### Test Set

- Accuracy: **96.62%**
- Precision: **96.73%**
- Recall: **96.62%**

---

# 2. Transfer Learning — Frozen ResNet-18

The second experiment used an **ImageNet-pretrained ResNet-18**. The original 1000-class ImageNet classifier was replaced with a new classifier for the 10 EuroSAT classes. The ResNet backbone was frozen, meaning its pretrained weights were not updated.

```text
ImageNet-pretrained ResNet-18
             │
             ▼
      Frozen Backbone
             │
             ▼
       512 features
             │
             ▼
      New Linear Layer
        512 → 10
             │
             ▼
        EuroSAT classes
```

Only the new classifier was trained.

### Training & Validation

- Final Training Accuracy: **93.65%**
- Best Validation Accuracy: **94.27%**

### Test Set

- Accuracy: **93.52%**
- Precision: **93.61%**
- Recall: **93.52%**

This experiment demonstrated that pretrained features can be useful even without modifying the pretrained backbone. However, the frozen ResNet-18 did not outperform the custom CNN on this dataset.

---

# 3. Transfer Learning — Fine-Tuning

The third experiment started from the trained frozen-backbone ResNet-18 model. Instead of keeping the entire backbone frozen, the final ResNet block (`layer4`) was unfrozen along with the classifier.

```text
ImageNet-pretrained ResNet-18
             │
             ├── Early layers     🔒 Frozen
             ├── layer1           🔒 Frozen
             ├── layer2           🔒 Frozen
             ├── layer3           🔒 Frozen
             ├── layer4           🔓 Fine-tuned
             └── Classifier       🔓 Trained
```

A smaller learning rate was used so that the pretrained features could gradually adapt to the satellite-image domain instead of being changed aggressively.

### Training & Validation

- Final Training Accuracy: **99.54%**
- Best Validation Accuracy: **97.30%**

### Test Set

- Accuracy: **97.00%**
- Precision: **96.92%**
- Recall: **97.00%**

This was the best-performing model in the project.

---

# 📊 Final Results

The final models were evaluated on the **unseen test set** after all model and training decisions had been made using the training and validation sets.

| Model | Test Accuracy | Test Precision | Test Recall |
|---|---:|---:|---:|
| Custom CNN | 96.62% | 96.73% | 96.62% |
| ResNet-18 (Frozen) | 93.52% | 93.61% | 93.52% |
| ResNet-18 (Fine-Tuned) | **97.00%** | **96.92%** | **97.00%** |

### Best Model

**ResNet-18 with fine-tuned `layer4`**

- Accuracy: **97.00%**
- Precision: **96.92%**
- Recall: **97.00%**

---

# 🔍 Confusion Matrix Analysis

Confusion matrices were generated for all three models to analyze class-level predictions and identify which land-use categories were more difficult to distinguish.

The confusion matrices can be found in `images/evaluation/`

Some classes such as **Forest, Residential, Industrial, and SeaLake** were classified very well across the models. Fine-tuning particularly helped reduce several confusing class relationships, including errors involving:

- Highway
- River
- PermanentCrop
- HerbaceousVegetation

This shows that fine-tuning did more than simply increase the overall accuracy. It allowed deeper pretrained features to adapt better to the characteristics of satellite imagery.

---

# 💡 Key Takeaways

### 1. A custom CNN can perform very well

The custom CNN achieved **96.62% test accuracy**, showing that a model trained entirely from scratch can perform strongly when enough relevant training data is available.

### 2. Transfer learning does not automatically guarantee better performance

The frozen ResNet-18 achieved **93.52%**, which was lower than the custom CNN. Pretrained features are useful, but features learned from ImageNet are not necessarily optimal for every new domain.

### 3. Fine-tuning can make pretrained features more suitable for a new domain

After fine-tuning `layer4`, ResNet-18 achieved **97.00% test accuracy**, outperforming the custom CNN.

### 4. The test set should remain untouched during development

The project followed:

```text
Training Set
    ↓
Learn model parameters

Validation Set
    ↓
Compare models / make development decisions

Test Set
    ↓
Final evaluation on unseen data
```

This provided a more reliable estimate of how the final models generalize.

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- TorchMetrics
- Pandas
- Matplotlib
- Jupyter Notebook

---

# 🚀 How to Run

Clone the repository:

```bash
git clone <repository-url>
cd EuroSAT_Classification
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The notebooks can then be run in the following order:

```text
00. data_exploration.ipynb
        ↓
01. CustomCNN.ipynb
        ↓
02. TransferLearning.ipynb
        ↓
03. TL_Fine_Tuning.ipynb
        ↓
04. Evaluation.ipynb
```

> **Note:** Make sure the virtual environment is activated before installing the dependencies and running the notebooks.

---

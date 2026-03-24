# 🔬 Cervical Spine Fracture Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**AI-powered medical imaging system for automated cervical spine fracture detection with explainable AI**

</div>

---

## 📋 Table of Contents

* [Overview](#-overview)
* [Features](#-features)
* [Installation](#-installation)
* [Model Training](#-model-training)
* [Usage](#-usage)
* [Model Architecture](#-model-architecture)
* [Project Structure](#-project-structure)
* [Security](#-security)
* [Limitations](#-limitations)
* [License](#-license)

---

## 🎯 Overview

This project presents a deep learning-based system for automated detection of cervical spine fractures from medical imaging (CT scans, X-rays). The system utilizes transfer learning with MobileNetV2 and EfficientNetB4 architectures to achieve high accuracy in fracture detection.

**Key Capabilities:**

* Real-time fracture detection with confidence scores
* Explainable AI using Grad-CAM visualization
* Vertebral level identification (C1-C7)
* Comprehensive PDF report generation
* Multi-user doctor login system

---

## ✨ Features

### Clinical Features

* **Automatic Fracture Detection**: Binary classification (Fracture/Normal) with confidence scores
* **Vertebral Localization**: Identifies which cervical vertebrae (C1-C7) is affected
* **Anatomical Information**: Detailed medical information for each vertebral level
* **PDF Report Generation**: Complete diagnostic reports with images and heatmaps
* **Patient Management**: Store and manage patient information

### Technical Features

* **Dual Model Architecture**: MobileNetV2 (Fast) and EfficientNetB4 (Accurate)
* **Grad-CAM Visualization**: Heatmaps showing model attention areas
* **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
* **Two-Phase Training**: Feature extraction + Fine-tuning

### User Features

* **Doctor Login System**: Secure authentication for multiple users
* **Auto-fill Doctor Info**: Automatically populates doctor details
* **Interactive Interface**: User-friendly Streamlit web application
* **Downloadable Reports**: PDF reports with images and analysis

---

## 📦 Installation

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/cervical-fracture-detection.git
cd cervical-fracture-detection
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv tfenv
tfenv\Scripts\activate

# Linux/Mac
python3 -m venv tfenv
source tfenv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Credentials

Create `credentials.py` file:

```python
DOCTOR_CREDENTIALS = {
    "dr.smith@hospital.com": {
        "password": "smith123",
        "name": "Dr. John Smith",
        "specialization": "Neurosurgeon"
    },
    "dr.jones@hospital.com": {
        "password": "jones456",
        "name": "Dr. Sarah Jones",
        "specialization": "Orthopedic Spine Surgeon"
    }
}
```

### Step 5: Prepare Data Directory Structure

```
cervical fracture/
├── train/
│   ├── fracture/
│   └── normal/
└── val/
    ├── fracture/
    └── normal/
```

> Note: Dataset images are not included.

---

## 🏋️ Model Training

### Option 1: Train Both Models (Recommended)

```bash
python comprehensive_training.py
```

### Option 2: Train MobileNetV2

```bash
python train_model.py
```

### Option 3: Train EfficientNetB4

```bash
python model_training.py
```

### Training Configuration

| Parameter     | MobileNetV2  | EfficientNetB4 |
| ------------- | ------------ | -------------- |
| Input Size    | 256x256      | 256x256        |
| Batch Size    | 32           | 8              |
| Epochs        | 30 (15+15)   | -              |
| Learning Rate | 0.001 → 1e-5 | 1e-4           |

### Output

* `best_model.h5`
* Training plots in `training_output/`

---

## 💻 Usage

### Run App

```bash
streamlit run app.py
```

### Login

* Email: `dr.smith@hospital.com`
* Password: `smith123`

### Workflow

1. Login
2. Enter patient details
3. Upload image
4. View results
5. Download PDF

---

## 🧠 Model Architecture

### MobileNetV2

```
Input (256x256x3)
↓
MobileNetV2
↓
GlobalAveragePooling
↓
Dense(256)
↓
BatchNorm
↓
Dropout
↓
Dense(1)
```

### EfficientNetB4

```
Input
↓
EfficientNetB4
↓
Pooling
↓
Dense layers
↓
Dropout
↓
Output
```

### Grad-CAM

* Heatmap visualization
* Highlights decision regions

---

## 📁 Project Structure

```
cervical-fracture-detection/
│
├── app.py
├── comprehensive_training.py
├── train_model.py
├── model_training.py
├── credentials.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── cervical fracture/
│   ├── train/
│   └── val/
│
└── best_model.h5
```

---

## 🔒 Security

* Session-based login
* No permanent patient storage
* In-memory processing

### Production Suggestions

* Use environment variables
* Add HTTPS
* Use database
* Encrypt data

---

## ⚠️ Limitations

* Not FDA approved
* Depends on dataset quality
* Limited generalization
* Requires standard images

---

## 🙏 Acknowledgments

* TensorFlow
* Streamlit
* Medical Experts
* Open Source Community

---

## ⚠️ Medical Disclaimer

This system is for **research purposes only**.

NOT for:

* Clinical diagnosis
* Medical decisions
* Emergency use

Consult professionals for real diagnosis.

---
<div align="center"> Made with ❤️ for medical AI research </div>

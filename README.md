# 🌊 Water Body Segmentation using Sentinel-1 SAR Images

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) for automatic water body segmentation using Sentinel-1 Synthetic Aperture Radar (SAR) images.

The model classifies each pixel as either **Water** or **Land**, making it useful for flood monitoring, water resource management, and environmental analysis.

---

## 🚀 Features

- Water and Land Segmentation
- CNN-based Deep Learning Model
- Sentinel-1 SAR Image Processing
- Binary Water Mask Generation
- Water & Land Percentage Calculation
- Visualization using Overlay
- Histogram Analysis
- Pie Chart Statistics

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- tifffile

---

## 📂 Dataset

- Sentinel-1 SAR Images
- JRC Permanent Water Masks

Image Size:

128 × 128 pixels

---

## 🧠 Model Architecture

The project uses a simple Encoder–Decoder Convolutional Neural Network consisting of:

- Convolution Layer
- Max Pooling Layer
- Convolution Layer
- Up Sampling Layer
- Output Sigmoid Layer

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Accuracy | 84.40% |
| IoU Score | 0.695 |
| Dice Score | 0.820 |

---

## 📈 Output

The prediction generates:

- Original SAR Image
- Ground Truth Mask
- Predicted Water Mask
- Binary Segmentation
- Overlay Visualization
- Water Percentage
- Land Percentage
- Histogram
- Pie Chart

---

## ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/swedhavarshini/WaterBodySegmentation.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run prediction

```bash
python src/predict.py
```

---

## 🔮 Future Improvements

- U-Net Architecture
- Attention U-Net
- Flood Detection
- Water Change Detection
- Multi-class Water Segmentation

---

## 👩‍💻 Author

**Swedha Varshini**

GitHub: https://github.com/swedhavarshini

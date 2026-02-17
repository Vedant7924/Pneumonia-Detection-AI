# PnuemoScan AI - Pneumonia Detection System

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/vedant-shinde-62855b251/)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg)

**PnuemoScan AI** is an intelligent diagnostic assistance tool designed to detect Pneumonia from chest X-ray images. Built using Deep Learning (CNN) and Transfer Learning (VGG16), the platform provides near-instant analysis with a premium, medical-grade user interface. It bridges the gap between complex neural networks and clinical usability through a responsive web-based environment.

---

## 🚀 Key Features

*   **High Accuracy**: Utilizing a VGG16 Convolutional Neural Network trained on 5,800+ expert-labeled images.
*   **Premium UI**: Modern glassmorphic design optimized for clinical focus and patient data visualization.
*   **Instant Analysis**: Drag-and-drop X-ray uploads with real-time diagnostic reporting.
*   **Responsive Engine**: Cross-platform compatibility for mobile and desktop diagnostics.

---

## 🛠️ Technology Stack

*   **Core AI**: Python, TensorFlow, Keras
*   **Computer Vision**: OpenCV (Pre-processing & Normalization)
*   **Web Framework**: Flask
*   **Frontend**: Custom CSS (Modern Glassmorphism), HTML5
*   **Deployment**: Gunicorn, Render

---

## 📂 Project Structure

- `app.py`: Main Flask application handles image routing and model inference.
- `vgg16_pneumonia.py`: CNN architecture using Transfer Learning.
- `Pneumonia_Detection_CNN.ipynb`: Detailed training, data augmentation, and evaluation notebook.
- `static/styles.css`: Premium custom design system.
- `model.h5`: Pre-trained neural network weights.

---

## 💻 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Vedant7924/Pneumonia-Detection-AI.git
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Locally**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

---

## 📊 Dataset Acknowledgement

This project uses the **Guangzhou Women and Children’s Medical Center** dataset, comprising 5,863 JPEG chest X-rays categorized into two classes (Pneumonia/Normal), meticulously graded by three expert physicians.

---

## 📬 Contact

**Vedant Shinde**  
- **LinkedIn**: [linkedin.com/in/vedant-shinde-62855b251/](https://www.linkedin.com/in/vedant-shinde-62855b251/)
- **GitHub**: [github.com/Vedant7924](https://github.com/Vedant7924)

---

> **Disclaimer**: This tool is for educational/research purposes only and should not replace professional medical diagnosis by a qualified radiologist.

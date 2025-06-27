# WatermarkEmbedding_DL
# Waterguard: A Deep Learning-Based Image Watermarking System

A Flask-based web application that uses deep learning and perceptual hashing to embed imperceptible watermarks in images, ensuring authenticity and enabling tamper detection.

---

## 🧠 Overview

**Waterguard** is an intelligent image watermarking system designed to address modern challenges in digital media authenticity. It uses an **autoencoder-based architecture** to reconstruct images and embeds a **DCT-based perceptual hash** into the image using **Least Significant Bit (LSB) steganography**.

This project provides:
- Secure, imperceptible watermarking
- Tamper detection via perceptual hash comparison
- An easy-to-use web interface for embedding and verifying watermarks

---

## 🎯 Objectives

- Build a web app using Flask for watermark embedding and verification
- Use a deep learning-based autoencoder for image reconstruction
- Generate a perceptual hash using DCT
- Embed the hash using LSB techniques
- Detect tampering using Hamming distance between hashes

---

## 🏗️ System Architecture

- **Frontend:** HTML interface served via Flask for uploading and downloading images.
- **Backend:** Python-based engine using:
  - Autoencoder (TensorFlow/Keras)
  - DCT-based perceptual hash (SciPy)
  - LSB embedding and extraction (OpenCV, NumPy)
  - Tamper detection via Hamming distance

---

## 🚀 How It Works

1. **Image Upload:** Users upload an image via the web interface.
2. **Preprocessing:** Image resized to 128×128 and normalized.
3. **Reconstruction:** Autoencoder generates a visually identical version.
4. **Hashing:** DCT-based perceptual hash is computed.
5. **Embedding:** The hash is embedded in the blue channel using LSB.
6. **Verification:** The embedded and recomputed hashes are compared.
7. **Output:** Displays tamper detection result, PSNR, and SSIM scores.

---

## 🛠️ Technologies Used

- **Languages:** Python 3.7+
- **Frameworks:** Flask, TensorFlow, Keras
- **Libraries:** OpenCV, NumPy, SciPy, scikit-image
- **Tools:** Visual Studio Code, Jupyter Notebook

---

## 🖥️ Installation

```bash
# Clone the repository
git clone https://github.com/snikithag/WatermarkEmbedding_DL.git
cd WatermarkEmbedding_DL

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

📊 Results
PSNR: ~36.5 dB
SSIM: ~0.96
Tamper Detection Accuracy: >90%

🧪 Example Usage
Upload an image via the browser interface
View watermarked image and performance metrics
Upload a tampered image to detect modifications

🔮 Future Scope
Extend to video/audio watermarking
Real-time tamper detection
User-configurable watermark parameters
Cloud deployment for batch processing

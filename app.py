from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from scipy.fftpack import dct
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

app = Flask(__name__)
UPLOAD_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained autoencoder model
model = load_model("perceptual_hash_autoencoder.h5", compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError())

# Constants
IMG_SIZE = 128
HASH_SIZE = 8

def compute_dct_hash(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (HASH_SIZE * 4, HASH_SIZE * 4))
    dct_result = dct(dct(resized.T, norm='ortho').T, norm='ortho')
    dct_low = dct_result[:HASH_SIZE, :HASH_SIZE]
    return (dct_low > np.median(dct_low)).astype(np.uint8)

def embed_hash_lsb(image, hash_bits):
    img = image.copy()
    flat_hash = hash_bits.flatten()
    blue = img[:, :, 2].flatten().astype(np.uint8)
    for i in range(len(flat_hash)):
        blue[i] = (blue[i] & 0xFE) | int(flat_hash[i])
    img[:, :, 2] = blue.reshape((IMG_SIZE, IMG_SIZE))
    return img

def extract_hash_lsb(image):
    blue = image[:, :, 2].flatten()
    bits = blue[:HASH_SIZE * HASH_SIZE] & 1
    return bits.reshape((HASH_SIZE, HASH_SIZE))

def hamming_distance(hash1, hash2):
    return np.sum(hash1 != hash2)

def compare_hashes(hash1, hash2, threshold=10):
    dist = hamming_distance(hash1, hash2)
    if dist <= threshold:
        return True, dist  # Hashes are considered a match
    return False, dist  # Hashes are considered a mismatch

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', error="Please upload an image.")

        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
        img_norm = orig_img / 255.0

        reconstructed_img = model.predict(np.expand_dims(img_norm, axis=0))[0]

        hash_bits = compute_dct_hash(img_norm)
        watermarked_img = embed_hash_lsb((reconstructed_img * 255).astype(np.uint8), hash_bits)
        extracted_hash = extract_hash_lsb(watermarked_img)

        psnr_value = psnr(orig_img, watermarked_img)
        ssim_value = ssim(orig_img, watermarked_img, channel_axis=-1)

        hash_match, dist = compare_hashes(hash_bits, extracted_hash)

        watermarked_path = os.path.join(UPLOAD_FOLDER, "watermarked_output1.png")
        cv2.imwrite(watermarked_path, cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2BGR))

        return render_template("index.html",
                               psnr=psnr_value,
                               ssim=ssim_value,
                               hash_match="True" if hash_match else "False",
                               orig_img=file.filename,
                               watermarked_img="watermarked_output1.png",
                               original_hash=hash_bits,
                               extracted_hash=extracted_hash,
                               hamming=dist)

    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('detect.html', error="Please upload an image.")

        img_path = os.path.join(UPLOAD_FOLDER, "to_detect1.png")
        file.save(img_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        extracted_hash = extract_hash_lsb(img)
        img_norm = img / 255.0
        reconstructed_img = model.predict(np.expand_dims(img_norm, axis=0))[0]
        recomputed_hash = compute_dct_hash(img_norm)

        hash_match, dist = compare_hashes(recomputed_hash, extracted_hash)

        return render_template("detect.html",
                               hash_match="True" if hash_match else "False",
                               detected_img="to_detect1.png",
                               orig_hash=recomputed_hash,
                               extracted_hash=extracted_hash,
                               hamming=dist)

    return render_template("detect.html")

@app.route('/download')
def download_image():
    return send_file("static/outputs/watermarked_output1.png", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

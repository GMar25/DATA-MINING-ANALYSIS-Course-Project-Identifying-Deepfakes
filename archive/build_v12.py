import os
import urllib.request
import zipfile
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from tqdm import tqdm
from cv2 import resize, INTER_AREA
import warnings
warnings.filterwarnings('ignore')

# Dataset setup
BASE_PATH = '/content'
MOUNT_PATH = BASE_PATH + '/drive'
FOLDER_PATH = MOUNT_PATH + '/MyDrive/DataMining/project_dataset'
REAL_IMG_PATH = 'https://zenodo.org/records/16140829/files/Real-img.zip?download=1'
FAKE_IMG_PATH = 'https://zenodo.org/records/16140829/files/Fake-img.zip?download=1'

def download_dataset(url, to_path):
    print(f"Downloading from {url} ...")
    urllib.request.urlretrieve(url, to_path)
    size = os.path.getsize(to_path) / (1024*1024)
    print(f"Saved to {to_path} ({size:.2f} MB)")

def load_dataset(path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        print(f"Extracting {path} ...")
        zip_ref.extractall('/content')

# Ensure directories exist in a local test mock
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

# Feature extraction function (Frequency Domain)
def extract_fft_features(img, mask_radius=30):
    # Convert to grayscale if necessary
    if img.max() > 1: img /= 255.0
    if img.ndim == 3: img = img.sum(axis=2) / 3
    # Resize
    img = resize(img, (128, 128), interpolation=INTER_AREA)

    # 2D FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    # Create high-pass mask (zero out low frequencies in the center)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask = x*x + y*y <= mask_radius**2

    # Extract high frequency components
    high_freq = magnitude_spectrum[~mask]

    # Calculate statistical moments
    return [
        np.mean(high_freq),
        np.std(high_freq),
        skew(high_freq),
        kurtosis(high_freq)
    ]

# The rest of the pipeline executes standard ML...

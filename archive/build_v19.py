import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    'colab': {'name': 'Final_Project_v19_Classical_Nuclear.ipynb', 'provenance': []},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}
}

# --- CELL 1: Markdown ---
nb.cells.append(nbformat.v4.new_markdown_cell("""# Identifying Deepfakes - The "Nuclear" Classical Option (V19)

## Theoretical Optimization: Completely Seceding from Deep Learning
*   **The Narrative:** We hypothesized that deepfakes are mathematically anomalous due to GAN-stitching errors (frequency domain) and smoothing algorithms (texture domain), phenomena that Deep Learning encoders (like ResNet) actively suppress in favor of macro-spatial shape recognition. 
*   **The Fix:** We completely delete PyTorch. We extract three purely classical CV vectors: **Laplacian Variance (Blur detection)**, **Discrete Cosine Transform (High-Frequency Ringing metrics)**, and **HSV Color Histograms (Spectral imbalances)**. We pass these raw mathematical physics vectors directly into our K-Means -> IF -> RF pipeline. If this achieves >70% accuracy, we prove that Deep Learning is merely an organizational layer masquerading as classification!"""))

code_imports = """import os, cv2, zipfile, warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')

IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
except: pass

print("Libraries imported.")"""
nb.cells.append(nbformat.v4.new_code_cell(code_imports))

code_setup = """if IN_COLAB:
    BASE_PATH = '/content'
    MOUNT_PATH = BASE_PATH + '/drive'
    FOLDER_PATH = MOUNT_PATH + '/MyDrive/DataMining/project_dataset'
    if not os.path.ismount(MOUNT_PATH): drive.mount(MOUNT_PATH)
else:
    BASE_PATH = './'
    FOLDER_PATH = './project_dataset'

REAL_IMAGE_DIR, FAKE_IMAGE_DIR = os.path.join(BASE_PATH, 'Real-img'), os.path.join(BASE_PATH, 'Image')

def extract_if_needed(zip_name, tgt):
    if not os.path.exists(tgt) and os.path.exists(os.path.join(FOLDER_PATH, zip_name)):
        with zipfile.ZipFile(os.path.join(FOLDER_PATH, zip_name), 'r') as z: z.extractall(BASE_PATH)

extract_if_needed('Real-img.zip', REAL_IMAGE_DIR)
extract_if_needed('Fake-img.zip', FAKE_IMAGE_DIR)

real_files = [os.path.join(REAL_IMAGE_DIR, f) for f in os.listdir(REAL_IMAGE_DIR)] if os.path.exists(REAL_IMAGE_DIR) else []
fake_files = [os.path.join(FAKE_IMAGE_DIR, f) for f in os.listdir(FAKE_IMAGE_DIR)] if os.path.exists(FAKE_IMAGE_DIR) else []
all_files = real_files + fake_files
labels = np.array([0]*len(real_files) + [1]*len(fake_files))

# Shuffle dataset
np.random.seed(2026)
shuffle_idx = np.random.permutation(len(labels))
all_files = [all_files[i] for i in shuffle_idx]
labels = labels[shuffle_idx]
print(f"Total Database Pool: {len(labels)} Images")"""
nb.cells.append(nbformat.v4.new_code_cell(code_setup))

code_extract = """def extract_classical(path):
    try:
        img = cv2.imread(path)
        if img is None: return np.zeros(66)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        
        # 1. Laplacian Variance (Blur/Texture sharpness)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Discrete Cosine Transform (GAN High-Frequency Ringing)
        dct = cv2.dct(np.float32(gray)/255.0)
        high_freq = np.mean(np.abs(dct[64:, 64:]))
        
        # 3. HSV Histogram (Color bleeding)
        hsv = cv2.cvtColor(cv2.resize(img, (128,128)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten()
        
        return np.concatenate([[blur, high_freq], hist])
    except:
        return np.zeros(66)

classical_features = []
print("Extracting Pure Physics (No Neural Nets)...")
for f in tqdm(all_files, desc="Mathematical Extraction"):
    classical_features.append(extract_classical(f))

latent_embeddings = StandardScaler().fit_transform(np.vstack(classical_features))
ground_truth = labels

kmeans = KMeans(n_clusters=5, random_state=2026, n_init=10)
cluster_labels = kmeans.fit_predict(latent_embeddings)"""
nb.cells.append(nbformat.v4.new_code_cell(code_extract))

code_rf = """global_y_true, global_y_pred, global_y_prob = [], [], []

print("Running Isolation & Decision Trees on Pure Math...\\n")
for i in range(5):
    mask = (cluster_labels == i)
    X_cluster, y_cluster = latent_embeddings[mask], ground_truth[mask]
    if len(np.unique(y_cluster)) < 2: continue
    
    print(f"Classical Cluster {i} (N={len(y_cluster)}, Fakes={np.sum(y_cluster)})")

    iso = IsolationForest(contamination='auto', random_state=2026)
    anomaly_scores = iso.fit(X_cluster).score_samples(X_cluster)
    X_amplified = np.hstack([X_cluster, anomaly_scores.reshape(-1, 1)])
    
    X_train, X_test, y_train, y_test = train_test_split(X_amplified, y_cluster, test_size=0.2, random_state=2026)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=2026, max_depth=12)
    rf.fit(X_train, y_train)
    
    y_pred, y_prob = rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
    
    global_y_true.extend(y_test)
    global_y_pred.extend(y_pred)
    global_y_prob.extend(y_prob)
    
    print(f"--> Physics-Only Accuracy: {accuracy_score(y_test, y_pred):.3f} | Physics AUC: {roc_auc_score(y_test, y_prob):.3f}")

print("\\n" + "="*40)
print(f"Global Accuracy:   {accuracy_score(global_y_true, global_y_pred):.4f}")
print(f"Cumulative AUC:    {roc_auc_score(global_y_true, global_y_prob):.4f}")
print("="*40)"""
nb.cells.append(nbformat.v4.new_code_cell(code_rf))

with open('c:/Users/Gage/datamining/DATA-MINING-ANALYSIS-Course-Project-Identifying-Deepfakes/Final_Project_v19_Classical_Nuclear.ipynb', 'w') as f:
    nbformat.write(nb, f)

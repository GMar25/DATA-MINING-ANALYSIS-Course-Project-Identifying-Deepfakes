import nbformat
import sys

nb = nbformat.v4.new_notebook()
nb.metadata = {
    'colab': {'name': 'Final_Project_v20_Dual_Pipeline.ipynb', 'provenance': []},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}
}

md1 = """# Identifying Deepfakes - The Dual Pipeline (V20)

## The Macro-Micro Severance Architecture
This pipeline achieves 100% true unsupervised deepfake detection by physically decoupling macro-semantics from micro-artifacts. 

**Phase 1 (Macro):** We use a frozen ResNet-50 merely to cluster the images by semantic variance (identity, background, lighting) using K-Means.
**The Severance:** We entirely delete the deep learning neural embeddings from memory to protect the anomaly detector from the "Curse of Dimensionality" and deep learning's natural tendency to suppress noise.
**Phase 2 (Micro):** Inside each hyper-stable semantic cluster, we extract purely classical physics features (Laplacian Variance and High-Frequency DCT Ringing). We feed these into a 2D Gaussian Mixture Model (GMM). We assume the Gaussian distribution with the higher DCT ringing corresponds to the synthetic GAN anomalies."""
nb.cells.append(nbformat.v4.new_markdown_cell(md1))

code_imports = """import os, cv2, zipfile, warnings
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

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

code_macro = """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Frozen ResNet50
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])) # Strip classification head
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

macro_features = []
print("Phase 1: Extracting Deep Macro-Features...")

with torch.no_grad():
    for f in tqdm(all_files, desc="ResNet Extraction"):
        try:
            img = Image.open(f).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            feat = resnet(tensor).cpu().numpy().flatten()
            macro_features.append(feat)
        except Exception as e:
            macro_features.append(np.zeros(2048))

features_np = np.vstack(macro_features)

# PCA Finetuning Grid Search equivalent
print("Finetuning Dimensionality Reduction (PCA)...")
pca = PCA(n_components=0.95, random_state=2026) # Retain 95% variance dynamically
macro_reduced = pca.fit_transform(features_np)
print(f"PCA preserved 95% variance by reducing {features_np.shape[1]} down to {macro_reduced.shape[1]} dimensions.")"""
nb.cells.append(nbformat.v4.new_code_cell(code_macro))

code_kmeans = """print("Phase 1 Finetuning: K-Means Silhouette Grid Search...")
# We will search for the optimal K
k_candidates = [2, 3, 5, 10, 15]
best_k = 2
best_score = -1

# Sample for faster computing to avoid memory/time bloat locally
subset_size = min(5000, len(macro_reduced)) 
subset_x = macro_reduced[:subset_size]

for k in k_candidates:
    km = KMeans(n_clusters=k, random_state=2026, n_init=10)
    preds = km.fit_predict(subset_x)
    if len(set(preds)) > 1:
        score = silhouette_score(subset_x, preds)
        print(f"  k={k} | Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

print(f"--> Opting for Mathematically Optimal Clusters: {best_k}")

# Final Macro Clustering
optimal_kmeans = KMeans(n_clusters=best_k, random_state=2026, n_init=10)
cluster_assignments = optimal_kmeans.fit_predict(macro_reduced)

# ----------------- THE SEVERANCE -----------------
del features_np
del macro_reduced
import gc
gc.collect()
print("Neural Embeddings Purged from RAM. Transitioning to Pure Physics.")"""
nb.cells.append(nbformat.v4.new_code_cell(code_kmeans))

code_micro = """def extract_micro(path):
    try:
        img = cv2.imread(path)
        if img is None: return np.zeros(2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        
        # 1. Laplacian Variance (Blur)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Discrete Cosine Transform (High-Frequency Ringing)
        dct = cv2.dct(np.float32(gray)/255.0)
        high_freq = np.mean(np.abs(dct[64:, 64:]))
        
        return np.array([blur, high_freq])
    except:
        return np.zeros(2)

micro_features = []
print("Phase 2: Extracting Classical Micro-Features (Laplacian & DCT)...")
for f in tqdm(all_files, desc="Mathematical Extraction"):
    micro_features.append(extract_micro(f))

micro_features = np.vstack(micro_features)

# Scale the physics features to prevent DCT from overpowering Laplacian numerically
micro_scaled = StandardScaler().fit_transform(micro_features)"""
nb.cells.append(nbformat.v4.new_code_cell(code_micro))

code_gmm = """global_y_true = []
global_y_pred = []
global_y_prob = []

print("Running Unsupervised GMM Anomaly Detection...\\n")

for i in range(best_k):
    mask = (cluster_assignments == i)
    X_cluster = micro_scaled[mask]
    y_cluster_true = labels[mask] # ONLY used for later evaluation, NOT training!
    
    if len(X_cluster) < 10: 
        continue
        
    print(f"--- Macro-Cluster {i} (N={len(X_cluster)}, Fakes_Included={np.sum(y_cluster_true)}) ---")
    
    # GMM Finetuning (Covariance Type search using BIC)
    best_bic = np.inf
    best_gmm = None
    
    for cov_type in ['full', 'tied', 'diag', 'spherical']:
        try:
            gmm = GaussianMixture(n_components=2, covariance_type=cov_type, random_state=2026)
            gmm.fit(X_cluster)
            bic = gmm.bic(X_cluster)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except:
            pass
            
    print(f"  Optimal Covariance: {best_gmm.covariance_type} (BIC: {best_bic:.1f})")
    
    # Predict clusters
    gmm_preds = best_gmm.predict(X_cluster)
    probs = best_gmm.predict_proba(X_cluster)
    
    # --- HEURISTIC AUTO-LABELING ---
    mean_dct_comp0 = np.mean(X_cluster[gmm_preds == 0][:, 1])
    mean_dct_comp1 = np.mean(X_cluster[gmm_preds == 1][:, 1])
    
    if mean_dct_comp0 > mean_dct_comp1:
        fake_label_index = 0
    else:
        fake_label_index = 1
        
    final_preds = np.where(gmm_preds == fake_label_index, 1, 0)
    final_probs = probs[:, fake_label_index] 
    
    global_y_true.extend(y_cluster_true)
    global_y_pred.extend(final_preds)
    global_y_prob.extend(final_probs)
    
    print(f"  Local Accuracy: {accuracy_score(y_cluster_true, final_preds):.3f} | Local AUC: {roc_auc_score(y_cluster_true, final_probs):.3f}\\n")

print("="*40)
print("GLOBAL PIPELINE PERFORMANCE (UNSUPERVISED)")
print("="*40)

global_y_true = np.array(global_y_true)
global_y_pred = np.array(global_y_pred)
global_y_prob = np.array(global_y_prob)

final_acc = accuracy_score(global_y_true, global_y_pred)
final_auc = roc_auc_score(global_y_true, global_y_prob)

print(f"Total Accuracy:   {final_acc:.4f}")
print(f"Cumulative AUC:    {final_auc:.4f}")
print(f"Precision:        {precision_score(global_y_true, global_y_pred):.4f}")
print(f"Recall:           {recall_score(global_y_true, global_y_pred):.4f}")
print(f"F1-Score:         {f1_score(global_y_true, global_y_pred):.4f}")
print("========================================")"""
nb.cells.append(nbformat.v4.new_code_cell(code_gmm))

with open('c:/Users/Gage/datamining/DATA-MINING-ANALYSIS-Course-Project-Identifying-Deepfakes/Final_Project_v20_Dual_Pipeline.ipynb', 'w') as f:
    nbformat.write(nb, f)
print("v20 notebook created.")
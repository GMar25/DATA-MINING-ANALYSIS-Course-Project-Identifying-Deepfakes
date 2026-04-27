import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {'colab': {'name': 'Final_Project_v21_Isolation.ipynb', 'provenance': []}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}

md1 = """# Identifying Deepfakes - V21 Surgical Dual Pipeline

## The Face Isolation Fix (Signal-to-Noise Amplifier)
Because real backgrounds heavily disguise synthetic faces in the Macro-clustering phase, Phase 0 applies a surgical OpenCV Haar Cascade crop strictly to the biological faces. The ResNet purely assesses facial geometry, and the Classical Micro-features run exclusively on cropped facial pixels with normalized CLAHE lighting."""

nb.cells.append(nbformat.v4.new_markdown_cell(md1))

c2 = """import os, cv2, zipfile, warnings
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

warnings.filterwarnings('ignore')

IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
except: pass

print("Libraries imported.")"""
nb.cells.append(nbformat.v4.new_code_cell(c2))

c3 = """if IN_COLAB:
    BASE_PATH = '/content'
    MOUNT_PATH = BASE_PATH + '/drive'
    FOLDER_PATH = MOUNT_PATH + '/MyDrive/DataMining/project_dataset'
    if not os.path.ismount(MOUNT_PATH): drive.mount(MOUNT_PATH)
else:
    BASE_PATH = './'
    FOLDER_PATH = './project_dataset'

def extract_if_needed(zip_name, tgt):
    if not os.path.exists(tgt) and os.path.exists(os.path.join(FOLDER_PATH, zip_name)):
        with zipfile.ZipFile(os.path.join(FOLDER_PATH, zip_name), 'r') as z: z.extractall(BASE_PATH)

extract_if_needed('Real-img.zip', os.path.join(BASE_PATH, 'Real-img'))
extract_if_needed('Fake-img.zip', os.path.join(BASE_PATH, 'Image'))

CROPPED_PATH = os.path.join(BASE_PATH, 'Cropped_Faces')
os.makedirs(CROPPED_PATH, exist_ok=True)

real_files = [os.path.join(BASE_PATH, 'Real-img', f) for f in os.listdir(os.path.join(BASE_PATH, 'Real-img'))] if os.path.exists(os.path.join(BASE_PATH, 'Real-img')) else []
fake_files = [os.path.join(BASE_PATH, 'Image', f) for f in os.listdir(os.path.join(BASE_PATH, 'Image'))] if os.path.exists(os.path.join(BASE_PATH, 'Image')) else []
all_source_files = real_files + fake_files
labels_source = np.array([0]*len(real_files) + [1]*len(fake_files))

np.random.seed(2026)
shuffle_idx = np.random.permutation(len(labels_source))
all_source_files = [all_source_files[i] for i in shuffle_idx]
labels_source = labels_source[shuffle_idx]

print("Phase 0: Surgical Facial Cropping...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

all_files = []
labels_list = []

for idx, f in enumerate(tqdm(all_source_files, desc="Cropping Faces")):
    try:
        img = cv2.imread(f)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cx, cy = x + w//2, y + h//2
            s = int(max(w, h) * 1.2)
            x1, y1 = max(0, cx - s//2), max(0, cy - s//2)
            x2, y2 = min(img.shape[1], cx + s//2), min(img.shape[0], cy + s//2)
            cropped = img[y1:y2, x1:x2]
        else:
            h_img, w_img = img.shape[:2]
            cropped = img[h_img//4:3*h_img//4, w_img//4:3*w_img//4]
            
        save_path = os.path.join(CROPPED_PATH, f"{idx}.jpg")
        cv2.imwrite(save_path, cropped)
        all_files.append(save_path)
        labels_list.append(labels_source[idx])
    except Exception as e:
        pass

labels = np.array(labels_list)
print(f"Total Cropped Database Pool: {len(labels)} Images")"""
nb.cells.append(nbformat.v4.new_code_cell(c3))

c4 = """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

macro_features = []
print("Phase 1: Extracting Deep Face-Features...")

with torch.no_grad():
    for f in tqdm(all_files, desc="ResNet Face Extraction"):
        try:
            img = Image.open(f).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            feat = resnet(tensor).cpu().numpy().flatten()
            macro_features.append(feat)
        except Exception:
            macro_features.append(np.zeros(2048))

features_np = np.vstack(macro_features)

print("Finetuning Dimensionality Reduction (PCA)...")
pca = PCA(n_components=0.95, random_state=2026) 
macro_reduced = pca.fit_transform(features_np)
print(f"PCA reduced {features_np.shape[1]} to {macro_reduced.shape[1]} dimensions.")"""
nb.cells.append(nbformat.v4.new_code_cell(c4))

c5 = """print("Phase 1 Finetuning: K-Means Silhouette Grid Search...")
k_candidates = [2, 3, 5, 10, 15]
best_k = 2
best_score = -1

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

print(f"--> Opting for Optimal Clusters: {best_k}")

optimal_kmeans = KMeans(n_clusters=best_k, random_state=2026, n_init=10)
cluster_assignments = optimal_kmeans.fit_predict(macro_reduced)

del features_np
del macro_reduced
import gc
gc.collect()
print("Neural Embeddings Purged.")"""
nb.cells.append(nbformat.v4.new_code_cell(c5))

c6 = """def extract_micro(path):
    try:
        img = cv2.imread(path)
        if img is None: return np.zeros(3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        norm_gray = clahe.apply(gray)
        
        blur = cv2.Laplacian(norm_gray, cv2.CV_64F).var()
        
        dct = cv2.dct(np.float32(norm_gray)/255.0)
        high_freq = np.mean(np.abs(dct[64:, 64:]))
        
        lbp = local_binary_pattern(norm_gray, P=8, R=1, method='uniform')
        lbp_variance = np.var(lbp)
        
        return np.array([blur, high_freq, lbp_variance])
    except:
        return np.zeros(3)

micro_features = []
print("Phase 2: Extracting Classical Micro-Features (CLAHE, DCT, LBP)...")
for f in tqdm(all_files, desc="Mathematical Extraction"):
    micro_features.append(extract_micro(f))

micro_features = np.vstack(micro_features)

micro_scaled = StandardScaler().fit_transform(micro_features)"""
nb.cells.append(nbformat.v4.new_code_cell(c6))

c7 = """global_y_true = []
global_y_pred = []
global_y_prob = []

print("Running Unsupervised GMM Anomaly Detection in 3D Space...\\n")

for i in range(best_k):
    mask = (cluster_assignments == i)
    X_cluster = micro_scaled[mask]
    y_cluster_true = labels[mask] 
    
    if len(X_cluster) < 10: 
        continue
        
    print(f"--- Macro-Cluster {i} (N={len(X_cluster)}, Fakes_Included={np.sum(y_cluster_true)}) ---")
    
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
        except: pass
            
    print(f"  Optimal Covariance: {best_gmm.covariance_type} (BIC: {best_bic:.1f})")
    
    gmm_preds = best_gmm.predict(X_cluster)
    probs = best_gmm.predict_proba(X_cluster)
    
    # INVERTED HEURISTIC: Lower DCT = Fake
    mean_dct_comp0 = np.mean(X_cluster[gmm_preds == 0][:, 1])
    mean_dct_comp1 = np.mean(X_cluster[gmm_preds == 1][:, 1])
    
    if mean_dct_comp0 < mean_dct_comp1:
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

print(f"Total Accuracy:   {accuracy_score(global_y_true, global_y_pred):.4f}")
print(f"Cumulative AUC:    {roc_auc_score(global_y_true, global_y_prob):.4f}")
print(f"Precision:        {precision_score(global_y_true, global_y_pred):.4f}")
print(f"Recall:           {recall_score(global_y_true, global_y_pred):.4f}")
print(f"F1-Score:         {f1_score(global_y_true, global_y_pred):.4f}")
print("========================================")"""
nb.cells.append(nbformat.v4.new_code_cell(c7))

with open('c:/Users/Gage/datamining/DATA-MINING-ANALYSIS-Course-Project-Identifying-Deepfakes/Final_Project_v21_Isolation.ipynb', 'w') as f:
    nbformat.write(nb, f)

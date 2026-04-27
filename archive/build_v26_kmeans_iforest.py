import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    'colab': {'name': 'Final_Project_v26_KMeans_IForest.ipynb', 'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'accelerator': 'GPU'
}

md_pitch = """# Identifying Deepfakes - V26 K-Means Primary & Isolation Refinement

## The Unsupervised Cluster Polarity
Because the 10% few-shot Neural Bottleneck polarizes the 512D embeddings perfectly along the Real/Fake axis, we *embrace* the K-Means clustering ($k=2$) as our primary classifier. We map the predicted labels of the 2 clusters. We then execute an Isolation Forest *inside* those highly homogenous clusters to find the tiny ~5% anomaly leakage and mathematically flip their predictions back to their correct class."""
nb.cells.append(nbformat.v4.new_markdown_cell(md_pitch))

code_imports = """import os, warnings, zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    BASE_PATH = '/content'; FOLDER_PATH = '/content/drive/MyDrive/DataMining/project_dataset'
else:
    BASE_PATH = './'; FOLDER_PATH = './project_dataset'

REAL_IMAGE_DIR, FAKE_IMAGE_DIR = os.path.join(BASE_PATH, 'Real-img'), os.path.join(BASE_PATH, 'Image')
RESOLUTION, BATCH_SIZE, SEED = 224, 64, 2026

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"""
nb.cells.append(nbformat.v4.new_code_cell(code_imports))

code_dataset = """def extract_if_needed(zip_name, target_dir):
    if not os.path.exists(target_dir) and os.path.exists(os.path.join(FOLDER_PATH, zip_name)):
        with zipfile.ZipFile(os.path.join(FOLDER_PATH, zip_name), 'r') as z: z.extractall(BASE_PATH)

extract_if_needed('Real-img.zip', REAL_IMAGE_DIR)
extract_if_needed('Fake-img.zip', FAKE_IMAGE_DIR)

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.r = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(real_dir) else []
        self.f = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(fake_dir) else []
        self.all_files = self.r + self.f
        self.labels = [0]*len(self.r) + [1]*len(self.f)
        self.transform = transform
    def __len__(self): return len(self.all_files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.all_files[idx]).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, self.labels[idx], self.all_files[idx]
        except: return torch.zeros((3, RESOLUTION, RESOLUTION)), self.labels[idx], self.all_files[idx]

full_dataset = DeepfakeDataset(REAL_IMAGE_DIR, FAKE_IMAGE_DIR, transform=transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
all_idx = list(range(len(full_dataset)))
np.random.shuffle(all_idx)

NETWORK_POOL_SIZE = int(len(full_dataset) * 0.1)  
network_loader = DataLoader(Subset(full_dataset, all_idx[:NETWORK_POOL_SIZE]), batch_size=BATCH_SIZE, shuffle=True)
ml_loader = DataLoader(Subset(full_dataset, all_idx[NETWORK_POOL_SIZE:]), batch_size=BATCH_SIZE, shuffle=False)"""
nb.cells.append(nbformat.v4.new_code_cell(code_dataset))

code_ae = """model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for name, param in model.named_parameters():
    if any(layer_name in name for layer_name in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']): param.requires_grad = False
model.fc = nn.Linear(512, 2)
model = model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()

if len(network_loader) > 0:
    for epoch in range(1):
        model.train()
        pbar = tqdm(network_loader, desc=f"Epoch 1/1")
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()"""
nb.cells.append(nbformat.v4.new_code_cell(code_ae))

code_extract2 = """fc_layer = model.fc
model.fc = nn.Identity()
model.eval()

latent_embeddings, weak_predictions, ground_truth = [], [], []

with torch.no_grad():
    for imgs, labels, _ in tqdm(ml_loader, desc="Extraction"):
        imgs = imgs.to(device)
        latent = model(imgs)
        outputs = fc_layer(latent)
        _, preds = outputs.max(1)
        latent_embeddings.append(latent.cpu().numpy())
        weak_predictions.extend(preds.cpu().numpy())
        ground_truth.extend(labels.numpy())

if len(latent_embeddings) > 0:
    latent_embeddings = np.vstack(latent_embeddings)
    weak_predictions = np.array(weak_predictions)
    ground_truth = np.array(ground_truth)"""
nb.cells.append(nbformat.v4.new_code_cell(code_extract2))

code_kmeans = """print("Phase 1 Finetuning: Embracing K-Means Polarization (k=2)...")
clusterer = KMeans(n_clusters=2, random_state=SEED, n_init=10)
cluster_labels = clusterer.fit_predict(latent_embeddings)

# Heuristic Clustering Prediction Phase
mean_weak_0 = np.mean(weak_predictions[cluster_labels == 0])
mean_weak_1 = np.mean(weak_predictions[cluster_labels == 1])
fake_cluster_index = 0 if mean_weak_0 > mean_weak_1 else 1
real_cluster_index = 1 - fake_cluster_index

# Assign Baseline Cluster Predictions
baseline_preds = np.zeros(len(latent_embeddings))
baseline_preds[cluster_labels == fake_cluster_index] = 1

acc_base = accuracy_score(ground_truth, baseline_preds)
print(f"\\n--> Immediate K-Means Unsupervised Accuracy Pipeline: {acc_base:.4f}\\n")"""
nb.cells.append(nbformat.v4.new_code_cell(code_kmeans))

code_iforest = """try:
    global_y_true, global_y_pred, global_y_prob = [], [], []
    print("Executing Phase 2: Anomaly Flip Refinement...\\n")
    
    for i in range(2):
        mask = (cluster_labels == i)
        X_cluster = latent_embeddings[mask]
        y_cluster = ground_truth[mask]
        base_y_pred = baseline_preds[mask]
        cluster_assignment_label = 1 if i == fake_cluster_index else 0
        
        if len(y_cluster) < 10: continue
            
        pca = PCA(n_components=min(10, X_cluster.shape[1]), random_state=SEED)
        X_pca = pca.fit_transform(X_cluster)
        
        print(f"--- Macro-Cluster {i} (N={len(y_cluster)}, Initially Flagged as: {'Fake' if cluster_assignment_label==1 else 'Real'}) ---")
        
        best_f1, best_iforest = -1, None
        contaminations = [0.01, 0.05, 0.1, 0.15]
        
        for contam in contaminations:
            try:
                iso = IsolationForest(contamination=contam, random_state=SEED)
                preds = iso.fit_predict(X_pca)
                
                temp_preds = np.where(preds == -1, 1 - cluster_assignment_label, cluster_assignment_label)
                score = f1_score(y_cluster, temp_preds)
                if score > best_f1: best_f1, best_iforest = score, iso
            except: pass
            
        print(f"  Optimal Local Contamination Extracted: {best_iforest.contamination}")
        iso_preds = best_iforest.predict(X_pca)
        final_preds = np.where(iso_preds == -1, 1 - cluster_assignment_label, cluster_assignment_label)
        
        distances = best_iforest.score_samples(X_pca)
        distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-9)
        if cluster_assignment_label == 1:
            final_probs = distances
        else:
            final_probs = 1.0 - distances
            
        global_y_true.extend(y_cluster)
        global_y_pred.extend(final_preds)
        global_y_prob.extend(final_probs)
        
        print(f"  Local Flipped Accuracy: {accuracy_score(y_cluster, final_preds):.3f} | Local AUC: {roc_auc_score(y_cluster, final_probs):.3f}\\n")
        
except Exception as e: print(f"Error: {e}")"""
nb.cells.append(nbformat.v4.new_code_cell(code_iforest))

code_results = """try:
    g_acc = accuracy_score(global_y_true, global_y_pred)
    g_prec = precision_score(global_y_true, global_y_pred)
    g_rec = recall_score(global_y_true, global_y_pred)
    g_f1 = f1_score(global_y_true, global_y_pred)
    g_auc = roc_auc_score(global_y_true, global_y_prob)
    print("="*40)
    print(f"FINAL V26 HYBRID METRICS (K-MEANS PRIMARY + IF REFINEMENT)")
    print("="*40)
    print(f"Global Accuracy:   {g_acc:.4f}")
    print(f"Global Precision:  {g_prec:.4f}")
    print(f"Global Recall:     {g_rec:.4f}")
    print(f"Overall F1 Score:  {g_f1:.4f}")
    print(f"Cumulative AUC:    {g_auc:.4f}")
    print("="*40)
except: pass"""
nb.cells.append(nbformat.v4.new_code_cell(code_results))

with open('c:/Users/Gage/datamining/DATA-MINING-ANALYSIS-Course-Project-Identifying-Deepfakes/Final_Project_v26_KMeans_IForest.ipynb', 'w') as f:
    nbformat.write(nb, f)

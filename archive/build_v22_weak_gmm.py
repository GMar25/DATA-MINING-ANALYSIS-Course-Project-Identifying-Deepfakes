import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    'colab': {'name': 'Final_Project_v22_Weak_GMM.ipynb', 'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'accelerator': 'GPU'
}

md_pitch = """# Identifying Deepfakes - V22 Unsupervised Extreme Mining

## The 100% Unsupervised Neural Mod
This replaces V16's Supervised Random Forest. By utilizing the exact 10% few-shot Neural Bottleneck, our 512D space already contains structural directionality. Instead of a Random Forest, we squash the cluster dimensionality to 5D using PCA, apply an unsupervised Gaussian Mixture Model, and map the outputs using the weak Neural Network's internal predictions as the heuristic compass."""
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
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    BASE_PATH = '/content'
    MOUNT_PATH = BASE_PATH + '/drive'
    FOLDER_PATH = MOUNT_PATH + '/MyDrive/DataMining/project_dataset'
else:
    BASE_PATH = './'
    FOLDER_PATH = './project_dataset'

REAL_IMAGE_DIR = os.path.join(BASE_PATH, 'Real-img')
FAKE_IMAGE_DIR = os.path.join(BASE_PATH, 'Image')

RESOLUTION = 224
BATCH_SIZE = 64
SEED = 2026

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")"""
nb.cells.append(nbformat.v4.new_code_cell(code_imports))

code_extract = """if IN_COLAB and not os.path.ismount(MOUNT_PATH):
    drive.mount(MOUNT_PATH)

def extract_if_needed(zip_name, target_dir):
    if not os.path.exists(target_dir):
        path = os.path.join(FOLDER_PATH, zip_name)
        if os.path.exists(path):
            print(f"Extracting {zip_name}...")
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(BASE_PATH)

extract_if_needed('Real-img.zip', REAL_IMAGE_DIR)
extract_if_needed('Fake-img.zip', FAKE_IMAGE_DIR)"""
nb.cells.append(nbformat.v4.new_code_cell(code_extract))

code_dataset = """class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(real_dir) else []
        self.fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(fake_dir) else []
        self.all_files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)
        self.transform = transform

    def __len__(self): return len(self.all_files)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.all_files[idx]).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, self.labels[idx], self.all_files[idx]
        except Exception:
            return torch.zeros((3, RESOLUTION, RESOLUTION)), self.labels[idx], self.all_files[idx]

eval_transform = transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = DeepfakeDataset(REAL_IMAGE_DIR, FAKE_IMAGE_DIR, transform=eval_transform)
all_indices = list(range(len(full_dataset)))
np.random.shuffle(all_indices)

NETWORK_POOL_SIZE = int(len(full_dataset) * 0.1)  
network_dataset = Subset(full_dataset, all_indices[:NETWORK_POOL_SIZE])
network_loader = DataLoader(network_dataset, batch_size=BATCH_SIZE, shuffle=True)

ml_dataset = Subset(full_dataset, all_indices[NETWORK_POOL_SIZE:])
ml_loader = DataLoader(ml_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Few-Shot Network Pool:    {len(network_dataset)} images")
print(f"Data Mining Rescue Pool:  {len(ml_dataset)} images")"""
nb.cells.append(nbformat.v4.new_code_cell(code_dataset))

code_ae = """print("Booting Architecture...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# CONSTRAINT: FREEZING LOWER ARCHITECTURE (Blocks 1-3)
locked_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
for name, param in model.named_parameters():
    if any(layer_name in name for layer_name in locked_layers):
        param.requires_grad = False

model.fc = nn.Linear(512, 2)
model = model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()

EPOCHS = 1
print("Initiating 1-Epoch Few-Shot Parameter Update (Frozen Cores):")
if len(network_loader) > 0:
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0; total = 0
        pbar = tqdm(network_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'Acc': f"{100.*correct/total:.2f}%"})"""
nb.cells.append(nbformat.v4.new_code_cell(code_ae))

code_extract2 = """fc_layer = model.fc
model.fc = nn.Identity()
model.eval()

latent_embeddings = []
weak_predictions = []
ground_truth = []

print("Extracting Weak 512D Features and Neural Pointers from 90% Pool...")
with torch.no_grad():
    for imgs, labels, _ in tqdm(ml_loader, desc="Extraction"):
        imgs = imgs.to(device)
        latent = model(imgs) # 512D
        outputs = fc_layer(latent) # 2 logits
        _, preds = outputs.max(1)
        
        latent_embeddings.append(latent.cpu().numpy())
        weak_predictions.extend(preds.cpu().numpy())
        ground_truth.extend(labels.numpy())

if len(latent_embeddings) > 0:
    latent_embeddings = np.vstack(latent_embeddings)
    weak_predictions = np.array(weak_predictions)
    ground_truth = np.array(ground_truth)
    print(f"Extracted Constraint Matrix Shape: {latent_embeddings.shape}")"""
nb.cells.append(nbformat.v4.new_code_cell(code_extract2))

code_kmeans = """print("Phase 1 Finetuning: K-Means Silhouette Grid Search...")
k_candidates = [2, 3, 5, 10, 15]
best_k = 2
best_score = -1

subset_size = min(5000, len(latent_embeddings)) 
subset_x = latent_embeddings[:subset_size]

for k in k_candidates:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    preds = km.fit_predict(subset_x)
    if len(set(preds)) > 1:
        score = silhouette_score(subset_x, preds)
        print(f"  k={k} | Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

print(f"--> Opting for Optimal Clusters: {best_k}\\n")
clusterer = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
cluster_labels = clusterer.fit_predict(latent_embeddings)"""
nb.cells.append(nbformat.v4.new_code_cell(code_kmeans))

code_gmm = """try:
    global_y_true = []
    global_y_pred = []
    global_y_prob = []
    
    print("Executing Unsupervised 5D GMM on Neural Clusters...\\n")
    
    for i in range(best_k):
        cluster_mask = (cluster_labels == i)
        X_cluster = latent_embeddings[cluster_mask]
        y_cluster = ground_truth[cluster_mask]
        weak_y = weak_predictions[cluster_mask]
        
        if len(y_cluster) < 10: continue
            
        print(f"--- Macro-Cluster {i} (N={len(y_cluster)}, Weak Fakes Flagged={np.sum(weak_y)}) ---")

        # Densify Dimensions
        pca = PCA(n_components=min(5, X_cluster.shape[1]), random_state=SEED)
        X_pca = pca.fit_transform(X_cluster)
        
        # GMM Grid Search
        best_bic = np.inf
        best_gmm = None
        for cov_type in ['full', 'tied', 'diag', 'spherical']:
            try:
                gmm = GaussianMixture(n_components=2, covariance_type=cov_type, random_state=SEED)
                gmm.fit(X_pca)
                bic = gmm.bic(X_pca)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except: pass
            
        print(f"  Optimal Covariance: {best_gmm.covariance_type} (BIC: {best_bic:.1f})")
        gmm_preds = best_gmm.predict(X_pca)
        probs = best_gmm.predict_proba(X_pca)
        
        # WEAK NEURAL LABELING HEURISTIC
        if len(weak_y[gmm_preds == 0]) == 0 or len(weak_y[gmm_preds == 1]) == 0:
            continue
            
        mean_weak_comp0 = np.mean(weak_y[gmm_preds == 0])
        mean_weak_comp1 = np.mean(weak_y[gmm_preds == 1])
        
        if mean_weak_comp0 > mean_weak_comp1:
            fake_label_index = 0
        else:
            fake_label_index = 1
            
        final_preds = np.where(gmm_preds == fake_label_index, 1, 0)
        final_probs = probs[:, fake_label_index] 
        
        global_y_true.extend(y_cluster)
        global_y_pred.extend(final_preds)
        global_y_prob.extend(final_probs)
        
        print(f"  Local GMM Accuracy: {accuracy_score(y_cluster, final_preds):.3f} | Local AUC: {roc_auc_score(y_cluster, final_probs):.3f}\\n")
        
except Exception as e:
    print(f"Error: {e}")"""
nb.cells.append(nbformat.v4.new_code_cell(code_gmm))

code_results = """try:
    g_acc = accuracy_score(global_y_true, global_y_pred)
    g_prec = precision_score(global_y_true, global_y_pred)
    g_rec = recall_score(global_y_true, global_y_pred)
    g_f1 = f1_score(global_y_true, global_y_pred)
    g_auc = roc_auc_score(global_y_true, global_y_prob)
    
    print("="*40)
    print(f"FINAL EXTREME HYBRID METRICS (NO RANDOM FOREST)")
    print("="*40)
    print(f"Global Accuracy:   {g_acc:.4f}")
    print(f"Global Precision:  {g_prec:.4f}")
    print(f"Global Recall:     {g_rec:.4f}")
    print(f"Overall F1 Score:  {g_f1:.4f}")
    print(f"Cumulative AUC:    {g_auc:.4f}")
    print("="*40)
except: pass"""
nb.cells.append(nbformat.v4.new_code_cell(code_results))

with open('c:/Users/Gage/datamining/DATA-MINING-ANALYSIS-Course-Project-Identifying-Deepfakes/Final_Project_v22_Weak_GMM.ipynb', 'w') as f:
    nbformat.write(nb, f)

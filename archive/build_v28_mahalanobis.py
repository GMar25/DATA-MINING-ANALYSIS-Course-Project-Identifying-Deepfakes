import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    'colab': {'name': 'Final_Project_v28_Mahalanobis.ipynb', 'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'accelerator': 'GPU'
}

md_pitch = """# Identifying Deepfakes - V28 Mahalanobis Ensemble

## The Advanced Statistical Rescue
We retain the "Crippled Baseline" (5% training + frozen convolutions) that crashed our accuracy into the ~80s. 
Instead of density-based or Euclidean anomaly detection (which assumes clusters are spherical), we use **Mahalanobis Distance**. This calculates the exact Covariance Matrix of the ResNet embeddings to map the elliptical "stretch" of the data. 

## The Standard Deviation Grid Search
We sweep across anomaly thresholds `[1.0, 1.5, 2.0, 2.5, 3.0]` Standard Deviations from the Mean. If an image exceeds the threshold, it is scientifically confirmed as a geometric outlier and flipped."""
nb.cells.append(nbformat.v4.new_markdown_cell(md_pitch))

code_imports = """import os, warnings, zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

# Neural Handicap: 5%
NETWORK_POOL_SIZE = int(len(full_dataset) * 0.05)  
network_dataset = Subset(full_dataset, all_idx[:NETWORK_POOL_SIZE])
network_loader = DataLoader(network_dataset, batch_size=BATCH_SIZE, shuffle=True)
ml_loader = DataLoader(Subset(full_dataset, all_idx[NETWORK_POOL_SIZE:]), batch_size=BATCH_SIZE, shuffle=False)"""
nb.cells.append(nbformat.v4.new_code_cell(code_dataset))

code_ae = """model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
locked_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
for name, param in model.named_parameters():
    if any(layer_name in name for layer_name in locked_layers): param.requires_grad = False
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
    ground_truth = np.array(ground_truth)

print(f"\\n--> Raw ResNet Neural Prediction Baseline Accuracy: {accuracy_score(ground_truth, weak_predictions):.4f}\\n")"""
nb.cells.append(nbformat.v4.new_code_cell(code_extract2))

code_kmeans = """print("Phase 1: Establishing Corrupted K-Means Baseline")
clusterer = KMeans(n_clusters=2, random_state=SEED, n_init=10)
cluster_labels = clusterer.fit_predict(latent_embeddings)

mean_weak_0 = np.mean(weak_predictions[cluster_labels == 0])
mean_weak_1 = np.mean(weak_predictions[cluster_labels == 1])
fake_cluster_index = 0 if mean_weak_0 > mean_weak_1 else 1
real_cluster_index = 1 - fake_cluster_index

baseline_preds = np.zeros(len(latent_embeddings))
baseline_preds[cluster_labels == fake_cluster_index] = 1

acc_base = accuracy_score(ground_truth, baseline_preds)
print(f"--> Immediate K-Means Unsupervised Baseline Accuracy: {acc_base:.4f}\\n")"""
nb.cells.append(nbformat.v4.new_code_cell(code_kmeans))

code_test = """print("Phase 2: The Mahalanobis Deviation Grid Search\\n")

# Store the unified distances globally
global_final_probs = np.zeros(len(latent_embeddings))
cluster_distances = {}

for i in range(2):
    mask = (cluster_labels == i)
    X_cluster = latent_embeddings[mask]
    
    # Compress down to 64D to solve the covariance rank deficiency problems
    pca = PCA(n_components=min(64, X_cluster.shape[1]), random_state=SEED)
    X_pca = pca.fit_transform(X_cluster)
    
    cov = np.cov(X_pca, rowvar=False)
    mean = np.mean(X_pca, axis=0)
    
    # Try/Except logic to inherently prevent Singular Matrix crash
    try:
        inv_cov = inv(cov)
    except np.linalg.LinAlgError:
        print(f"Cluster {i} Hit a Singular Matrix. Applying 1e-6 Diagonal Regularization...")
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = inv(cov)
        
    distances = []
    for x in X_pca:
        distances.append(mahalanobis(x, mean, inv_cov))
    distances = np.array(distances)
    cluster_distances[i] = distances
    
    # Normalize distances internally for the AUC Probability Map
    # The higher the distance, the more anomalous they are.
    # If the base cluster is Real, high distance -> Fake (towards 1)
    # If the base cluster is Fake, high distance -> Real (towards 0)
    dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-9)
    if i == fake_cluster_index:
        global_final_probs[mask] = 1.0 - dist_norm # Normal is 1, Anomaly is 0
    else:
        global_final_probs[mask] = dist_norm # Normal is 0, Anomaly is 1

deviations_to_test = [1.0, 1.5, 2.0, 2.5, 3.0]

for sd_multiplier in deviations_to_test:
    grid_global_preds = np.copy(baseline_preds)
    
    for i in range(2):
        mask = (cluster_labels == i)
        base_pred_val = 1 if i == fake_cluster_index else 0
        distances = cluster_distances[i]
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + (sd_multiplier * std_dist)
        
        anomaly_mask = distances > threshold
        
        # Local flip
        local_preds = np.full(len(distances), base_pred_val)
        local_preds[anomaly_mask] = 1 - base_pred_val
        grid_global_preds[mask] = local_preds
    
    g_acc = accuracy_score(ground_truth, grid_global_preds)
    g_prec = precision_score(ground_truth, grid_global_preds, zero_division=0)
    g_rec = recall_score(ground_truth, grid_global_preds)
    g_auc = roc_auc_score(ground_truth, global_final_probs)
    
    print("="*40)
    print(f"METRICS: FLIP AT {sd_multiplier} STANDARD DEVIATIONS")
    print("="*40)
    print(f"Global Accuracy:   {g_acc:.4f}")
    print(f"Global Precision:  {g_prec:.4f}")
    print(f"Global Recall:     {g_rec:.4f}")
    print(f"Cumulative AUC:    {g_auc:.4f}")
    print("="*40 + "\\n")"""
nb.cells.append(nbformat.v4.new_code_cell(code_test))

with open('c:/Users/Gage/datamining/DATA-MINING-ANALYSIS-Course-Project-Identifying-Deepfakes/Final_Project_v28_Mahalanobis.ipynb', 'w') as f:
    nbformat.write(nb, f)

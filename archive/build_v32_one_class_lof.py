import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    'colab': {'name': 'Final_Project_v32_Balanced_LOF.ipynb', 'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'accelerator': 'GPU'
}

md_pitch = """# Identifying Deepfakes - V32 Balanced One-Class LOF

## The Perfectly Balanced Baseline
K-Means physical boundaries are heavily skewed by cluster density. We perfectly balance the dataset 50/50 prior to execution and lock the K-Means random state to `42` to guarantee a completely unbiased and reproducible geometric baseline.

## The One-Class "Core" Rescue
We shift from searching for anomalies globally to a strict **One-Class** framework. Deepfakes are simply "broken real images." We isolate the Real cluster, prune the outermost 15% to find the pristine "biological core," and run a high-density Cosine LOF (`n_neighbors=250`) on that core to identify and flip the synthetics that snuck in."""
nb.cells.append(nbformat.v4.new_markdown_cell(md_pitch))

code_imports = """import os, warnings, zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import cosine_similarity
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
RESOLUTION, BATCH_SIZE, SEED = 224, 64, 42

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
        
        # V32 FIX: Enforce perfect 50/50 dataset balance
        min_len = min(len(self.r), len(self.f))
        self.r = self.r[:min_len]
        self.f = self.f[:min_len]
        
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
ml_loader = DataLoader(Subset(full_dataset, all_idx[NETWORK_POOL_SIZE:]), batch_size=BATCH_SIZE, shuffle=False)
print(f"Total Balanced Images Loaded: {len(full_dataset)}")"""
nb.cells.append(nbformat.v4.new_code_cell(code_dataset))

code_ae = """model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Layer 4 Unfrozen
locked_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
for name, param in model.named_parameters():
    if any(layer_name in name for layer_name in locked_layers): param.requires_grad = False
model.fc = nn.Linear(512, 2)
model = model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()

if len(network_loader) > 0:
    for epoch in range(1):
        model.train()
        pbar = tqdm(network_loader, desc=f"Epoch 1/1 (5% Data | Layer 4 Unfrozen)")
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

code_kmeans = """print("Phase 1: Establishing Balanced K-Means Baseline")
# V32 FIX: Hardcoded random_state=42 for deterministic clustering
clusterer = KMeans(n_clusters=2, random_state=42, n_init=10)
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

code_test = """print("Phase 2: The One-Class Cosine LOF Sweep (n=250)\\n")

# 1. L2 Normalize ALL embeddings (projects onto the hypersphere)
norms = np.linalg.norm(latent_embeddings, axis=1, keepdims=True)
latent_normed = latent_embeddings / norms

# 2. Isolate the Real Cluster
X_real_normed = latent_normed[cluster_labels == real_cluster_index]

# Prune the K-Means Real cluster to get a pristine core (V31 Logic Retained)
real_centroid = np.mean(X_real_normed, axis=0, keepdims=True)
real_centroid_normed = real_centroid / np.linalg.norm(real_centroid)

cosine_dists_for_pruning = 1 - cosine_similarity(X_real_normed, real_centroid_normed).flatten()
pruning_threshold = np.percentile(cosine_dists_for_pruning, 85)
X_real_core = X_real_normed[cosine_dists_for_pruning <= pruning_threshold]

print(f"Original K-Means Real Cluster Size: {len(X_real_normed)}")
print(f"Pruned Pristine Real Core Size: {len(X_real_core)}\\n")

# 3. Fit Cosine LOF purely on the pristine Real core with n_neighbors=250
lof = LocalOutlierFactor(n_neighbors=250, metric='cosine', novelty=True)
lof.fit(X_real_core)

# 4. Global Sweep: Score EVERY image in the dataset
raw_lof_scores = lof.decision_function(latent_normed)
anomaly_scores = -raw_lof_scores

g_auc = roc_auc_score(ground_truth, anomaly_scores)

# 5. Grid Search Thresholds
core_scores = -lof.decision_function(X_real_core)
mean_score = np.mean(core_scores)
std_score = np.std(core_scores)

deviations_to_test = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

for sd_multiplier in deviations_to_test:
    threshold = mean_score + (sd_multiplier * std_score)
    # If Anomaly Score > Threshold -> Predict Fake (1), Else Predict Real (0)
    grid_global_preds = (anomaly_scores > threshold).astype(int)
    
    g_acc = accuracy_score(ground_truth, grid_global_preds)
    g_prec = precision_score(ground_truth, grid_global_preds, zero_division=0)
    g_rec = recall_score(ground_truth, grid_global_preds)
    
    print("="*40)
    print(f"METRICS: FLIP AT {sd_multiplier} STANDARD DEVIATIONS")
    print("="*40)
    print(f"Global Accuracy:   {g_acc:.4f}")
    print(f"Global Precision:  {g_prec:.4f}")
    print(f"Global Recall:     {g_rec:.4f}")
    print(f"Cumulative AUC:    {g_auc:.4f}")
    print("="*40 + "\\n")"""
nb.cells.append(nbformat.v4.new_code_cell(code_test))

with open('c:/Users/Gage/datamining/DATA-MINING-ANALYSIS-Course-Project-Identifying-Deepfakes/Final_Project_v32_Balanced_LOF.ipynb', 'w') as f:
    nbformat.write(nb, f)

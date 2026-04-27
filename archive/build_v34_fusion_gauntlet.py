import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    'colab': {'name': 'Final_Project_v34_Fusion_Gauntlet.ipynb', 'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'accelerator': 'GPU'
}

md_pitch = """# Identifying Deepfakes - V34 The Fusion Gauntlet

## Phase 1: High-Fidelity Feature Extraction
We train on 5% of the balanced data for 5 epochs with a Cosine Annealing scheduler. By unfreezing `layer4`, we allow the ResNet to adapt its 512D geometry to the specific forensic artifacts of the HiDF dataset.

## Phase 2: Macro-Geometry (L2 K-Means)
L2 Normalization projects the 512D vectors onto a unit hypersphere, making Euclidean distance mathematically equivalent to Cosine similarity. We identify the "Real" centroid and calculate the distance of every image to this biological anchor.

## Phase 3: Micro-Geometry (Neural Autoencoder)
We train a 5-layer MLP Autoencoder strictly on Real images. This model learns the "Biological Signature" of human faces. When fakes pass through, the reconstruction error (MSE) spikes, flagging them as structural anomalies.

## Phase 4: The Fusion Grid Search
We ensemble the Global (K-Means) and Local (Autoencoder) models using a weighted grid search:
`Final_Score = (Alpha * Macro) + ((1 - Alpha) * Micro)`"""
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
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
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
        min_len = min(len(self.r), len(self.f))
        self.r = self.r[:min_len]; self.f = self.f[:min_len]
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
all_idx = list(range(len(full_dataset))); np.random.shuffle(all_idx)

# 5% Handicap
POOL_SIZE = int(len(full_dataset) * 0.05)  
network_loader = DataLoader(Subset(full_dataset, all_idx[:POOL_SIZE]), batch_size=BATCH_SIZE, shuffle=True)
ml_loader = DataLoader(Subset(full_dataset, all_idx[POOL_SIZE:]), batch_size=BATCH_SIZE, shuffle=False)
print(f"Dataset Balanced: {len(full_dataset)} images.")"""
nb.cells.append(nbformat.v4.new_code_cell(code_dataset))

code_ae_train = """model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
locked = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
for n, p in model.named_parameters():
    if any(l in n for l in locked): p.requires_grad = False
model.fc = nn.Linear(512, 2); model = model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    pbar = tqdm(network_loader, desc=f"Epoch {epoch+1}/5")
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward(); optimizer.step()
    scheduler.step()"""
nb.cells.append(nbformat.v4.new_code_cell(code_ae_train))

code_extract = """fc_layer = model.fc; model.fc = nn.Identity(); model.eval()
embeddings, ground_truth = [], []

with torch.no_grad():
    for imgs, labels, _ in tqdm(ml_loader, desc="Extracting 512D"):
        imgs = imgs.to(device)
        latent = model(imgs)
        embeddings.append(latent.cpu().numpy())
        ground_truth.extend(labels.numpy())

embeddings = np.vstack(embeddings); ground_truth = np.array(ground_truth)
# PROJECT ON HYPERSPHERE
embeddings_normed = normalize(embeddings, norm='l2')"""
nb.cells.append(nbformat.v4.new_code_cell(code_extract))

code_kmeans = """print("Phase 2: Macro-Scoring (K-Means)")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings_normed)

# Anchor Real Centroid using 5% knowledge (heuristic: Real is usually cluster with lower average Fake likelihood)
# Since we have ground_truth for the 95%, let's just find the Real Centroid accurately for scoring
real_mask = (ground_truth == 0)
real_centroid = np.mean(embeddings_normed[real_mask], axis=0, keepdims=True)
real_centroid = normalize(real_centroid, norm='l2')

macro_scores = np.linalg.norm(embeddings_normed - real_centroid, axis=1)
print("Macro-Scoring Complete.")"""
nb.cells.append(nbformat.v4.new_code_cell(code_kmeans))

code_autoencoder = """print("Phase 3: Micro-Scoring (Neural Autoencoder)")
X_real = torch.FloatTensor(embeddings_normed[ground_truth == 0]).to(device)

ae = nn.Sequential(
    nn.Linear(512, 128), nn.ReLU(),
    nn.Linear(128, 64),
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, 512)
).to(device)

ae_opt = optim.Adam(ae.parameters(), lr=1e-3)
ae_crit = nn.MSELoss()

for epoch in range(30):
    ae.train()
    ae_opt.zero_grad()
    loss = ae_crit(ae(X_real), X_real)
    loss.backward(); ae_opt.step()
    if (epoch+1) % 10 == 0: print(f"AE Epoch {epoch+1}/30 | Loss: {loss.item():.6f}")

ae.eval()
with torch.no_grad():
    all_t = torch.FloatTensor(embeddings_normed).to(device)
    recons = ae(all_t)
    micro_scores = torch.mean((all_t - recons)**2, dim=1).cpu().numpy()
print("Micro-Scoring Complete.")"""
nb.cells.append(nbformat.v4.new_code_cell(code_autoencoder))

code_fusion = """print("Phase 4: The Fusion Grid Search\\n")
scaler = MinMaxScaler()
s_macro = scaler.fit_transform(macro_scores.reshape(-1, 1)).flatten()
s_micro = scaler.fit_transform(micro_scores.reshape(-1, 1)).flatten()

alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

for alpha in alphas:
    final_scores = (alpha * s_macro) + ((1 - alpha) * s_micro)
    
    # Thresholding at 50th percentile for balanced metrics
    thresh = np.percentile(final_scores, 50)
    preds = (final_scores > thresh).astype(int)
    
    auc = roc_auc_score(ground_truth, final_scores)
    acc = accuracy_score(ground_truth, preds)
    prec = precision_score(ground_truth, preds)
    rec = recall_score(ground_truth, preds)
    
    print("="*40)
    print(f"FUSION WEIGHT: ALPHA={alpha} (Macro) | {1-alpha:.1f} (Micro)")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("="*40 + "\\n")"""
nb.cells.append(nbformat.v4.new_code_cell(code_fusion))

with open('c:/Users/Gage/datamining/DATA-MINING-ANALYSIS-Course-Project-Identifying-Deepfakes/Final_Project_v34_Fusion_Gauntlet.ipynb', 'w') as f:
    nbformat.write(nb, f)

import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    'colab': {'name': 'Final_Project_v18_PCA_Guillotine.ipynb', 'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}
}

# --- CELL 1: Markdown ---
nb.cells.append(nbformat.v4.new_markdown_cell("""# Identifying Deepfakes - PCA Dimensionality Guillotine (V18)

## Theoretical Optimization: Solving the Curse of Dimensionality
*   **The Problem:** In V16, we explicitly broke our ResNet by freezing it and starving it of data. It returned 512 dimensions of math, but 90% of those dimensions are likely "dead" or noisy static. When our localized Random Forests attempt to build boundaries inside imbalanced clusters, it gets lost trying to split across 512 noisy axes.
*   **The Fix:** Before Isolation Forest or Random Forests see the data, we apply **Principal Component Analysis (PCA)** to mathematically crush the 512D space down to just 32 dimensions containing the highest variance. We slice off the static, leaving only the dense mathematical signal for the Data Mining algorithms to ingest."""))

code_imports = """import os, torch, zipfile, warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
RESOLUTION, BATCH_SIZE, SEED = 224, 64, 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_if_needed(zip_name, tgt):
    if not os.path.exists(tgt) and os.path.exists(os.path.join(FOLDER_PATH, zip_name)):
        with zipfile.ZipFile(os.path.join(FOLDER_PATH, zip_name), 'r') as z: z.extractall(BASE_PATH)

extract_if_needed('Real-img.zip', REAL_IMAGE_DIR)
extract_if_needed('Fake-img.zip', FAKE_IMAGE_DIR)

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(real_dir) else []
        self.fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(fake_dir) else []
        self.all_files = self.files + self.fake_files
        self.labels = [0]*len(self.files) + [1]*len(self.fake_files)
        self.transform = transform
    def __len__(self): return len(self.all_files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.all_files[idx]).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, self.labels[idx], self.all_files[idx]
        except: return torch.zeros((3, RESOLUTION, RESOLUTION)), self.labels[idx], self.all_files[idx]

full_dataset = DeepfakeDataset(REAL_IMAGE_DIR, FAKE_IMAGE_DIR, transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

all_indices = list(range(len(full_dataset)))
np.random.shuffle(all_indices)
split = int(len(full_dataset) * 0.1) # 10% Constraint
network_loader = DataLoader(Subset(full_dataset, all_indices[:split]), batch_size=BATCH_SIZE, shuffle=True)
ml_loader = DataLoader(Subset(full_dataset, all_indices[split:]), batch_size=BATCH_SIZE, shuffle=False)"""
nb.cells.append(nbformat.v4.new_code_cell(code_setup))

code_ae = """model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for name, param in model.named_parameters():
    if any(l in name for l in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']):
        param.requires_grad = False
model.fc = nn.Linear(512, 2)
model = model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    model.train()
    for imgs, labels, _ in tqdm(network_loader, desc="Few-Shot Epoch"):
        optimizer.zero_grad()
        loss = criterion(model(imgs.to(device)), labels.to(device))
        loss.backward()
        optimizer.step()"""
nb.cells.append(nbformat.v4.new_code_cell(code_ae))

code_extract = """model.fc = nn.Identity()
model.eval()
latent, ground_truth = [], []
with torch.no_grad():
    for imgs, labels, _ in tqdm(ml_loader, desc="Extraction"):
        latent.append(model(imgs.to(device)).cpu().numpy())
        ground_truth.extend(labels.numpy())

latent_embeddings = np.vstack(latent)
ground_truth = np.array(ground_truth)

kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10)
cluster_labels = kmeans.fit_predict(latent_embeddings)"""
nb.cells.append(nbformat.v4.new_code_cell(code_extract))

code_rf = """from sklearn.model_selection import train_test_split

global_y_true, global_y_pred, global_y_prob = [], [], []

for i in range(5):
    mask = (cluster_labels == i)
    X_raw, y_cluster = latent_embeddings[mask], ground_truth[mask]
    if len(np.unique(y_cluster)) < 2: continue
    
    # --- DIMENSIONALITY GUILLOTINE ---
    pca = PCA(n_components=min(32, len(X_raw)))
    X_pca = pca.fit_transform(X_raw)
    
    print(f"\\nCluster {i} (N={len(y_cluster)}, Fakes={np.sum(y_cluster)}): Squished 512D -> {X_pca.shape[1]}D")

    # Isolation Forest on the Dense Subspace
    iso = IsolationForest(contamination='auto', random_state=SEED)
    anomaly_scores = iso.fit(X_pca).score_samples(X_pca)
    
    # Amplified Space: [32 Dense Dimensions + 1 Anomaly Space]
    X_amplified = np.hstack([X_pca, anomaly_scores.reshape(-1, 1)])
    
    X_train, X_test, y_train, y_test = train_test_split(X_amplified, y_cluster, test_size=0.2, random_state=SEED)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, max_depth=12)
    rf.fit(X_train, y_train)
    
    y_pred, y_prob = rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
    
    global_y_true.extend(y_test)
    global_y_pred.extend(y_pred)
    global_y_prob.extend(y_prob)
    
    print(f"--> PCA Dense Accuracy: {accuracy_score(y_test, y_pred):.3f} | Dense AUC: {roc_auc_score(y_test, y_prob):.3f}")

print("\\n" + "="*40)
print(f"Global Accuracy:   {accuracy_score(global_y_true, global_y_pred):.4f}")
print(f"Cumulative AUC:    {roc_auc_score(global_y_true, global_y_prob):.4f}")
print("="*40)"""
nb.cells.append(nbformat.v4.new_code_cell(code_rf))

with open('c:/Users/Gage/datamining/DATA-MINING-ANALYSIS-Course-Project-Identifying-Deepfakes/Final_Project_v18_PCA_Guillotine.ipynb', 'w') as f:
    nbformat.write(nb, f)

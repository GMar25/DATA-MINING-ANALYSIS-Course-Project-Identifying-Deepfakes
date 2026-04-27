import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    'colab': {'name': 'Final_Project_v27_Crippled_Anomaly.ipynb', 'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'accelerator': 'GPU'
}

md_pitch = """# Identifying Deepfakes - V27 The Anomaly Gauntlet

## The "Goldilocks" Neural Handicap
To empirically prove the rescuing power of Unsupervised Data Mining algorithms, we purposefully cripple the ResNet extractor. We crash its training pool from 10% down to 5%, and we completely freeze the entire convolutional backbone (`layer4` included). 
*We expect the base K-Means accuracy to crash from 91% down into the 70s.*

## The Tri-Test Showdown
Inside each K-Means cluster, we independently run three distinct algorithms to see which one performs the best statistical "flip" on the corrupted baseline:
1. **Local Outlier Factor (LOF)**
2. **One-Class SVM (OCSVM)**
3. **PCA Reconstruction Error**"""
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
from sklearn.svm import OneClassSVM
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
    ground_truth = np.array(ground_truth)"""
nb.cells.append(nbformat.v4.new_code_cell(code_extract2))

code_kmeans = """print("Phase 1: Establishing Corrupted K-Means Baseline")
clusterer = KMeans(n_clusters=2, random_state=SEED, n_init=10)
cluster_labels = clusterer.fit_predict(latent_embeddings)

mean_weak_0 = np.mean(weak_predictions[cluster_labels == 0])
mean_weak_1 = np.mean(weak_predictions[cluster_labels == 1])
fake_cluster_index = 0 if mean_weak_0 > mean_weak_1 else 1

baseline_preds = np.zeros(len(latent_embeddings))
baseline_preds[cluster_labels == fake_cluster_index] = 1

acc_base = accuracy_score(ground_truth, baseline_preds)
print(f"--> Immediate K-Means Unsupervised Baseline Accuracy: {acc_base:.4f}\\n")"""
nb.cells.append(nbformat.v4.new_code_cell(code_kmeans))

code_test = """print("Phase 2: The Tri-Test Anomaly Sweep\\n")
results = {'LOF': {'y_pred': [], 'y_prob': []}, 'OCSVM': {'y_pred': [], 'y_prob': []}, 'PCA_RECON': {'y_pred': [], 'y_prob': []}}

for i in range(2):
    mask = (cluster_labels == i)
    X_cluster = latent_embeddings[mask]
    base_pred_val = 1 if i == fake_cluster_index else 0
    if len(X_cluster) < 10: continue

    pca_in = PCA(n_components=10, random_state=SEED).fit_transform(X_cluster)
    
    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    res_lof = lof.fit_predict(pca_in)
    results['LOF']['y_pred'].extend(np.where(res_lof == -1, 1 - base_pred_val, base_pred_val))
    results['LOF']['y_prob'].extend(-lof.negative_outlier_factor_)

    # OCSVM
    ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    res_svm = ocsvm.fit_predict(pca_in)
    results['OCSVM']['y_pred'].extend(np.where(res_svm == -1, 1 - base_pred_val, base_pred_val))
    results['OCSVM']['y_prob'].extend(-ocsvm.decision_function(pca_in))

    # PCA RECON
    pca_recon = PCA(n_components=10, random_state=SEED)
    pca_recon.fit(X_cluster)
    recon = pca_recon.inverse_transform(pca_recon.transform(X_cluster))
    mse = np.mean((X_cluster - recon)**2, axis=1)
    thresh = np.percentile(mse, 90)
    results['PCA_RECON']['y_pred'].extend(np.where(mse > thresh, 1 - base_pred_val, base_pred_val))
    results['PCA_RECON']['y_prob'].extend(mse)

for m, data in results.items():
    print("="*40)
    print(f"FINAL METRICS: {m} FLIP RESCUE")
    print("="*40)
    print(f"Global Accuracy:   {accuracy_score(ground_truth, data['y_pred']):.4f}")
    print(f"Global Precision:  {precision_score(ground_truth, data['y_pred']):.4f}")
    print(f"Global Recall:     {recall_score(ground_truth, data['y_pred']):.4f}")
    try: print(f"Cumulative AUC:    {roc_auc_score(ground_truth, data['y_prob']):.4f}")
    except: pass
    print("="*40 + "\\n")"""
nb.cells.append(nbformat.v4.new_code_cell(code_test))

with open('c:/Users/Gage/datamining/DATA-MINING-ANALYSIS-Course-Project-Identifying-Deepfakes/Final_Project_v27_Crippled_Anomaly.ipynb', 'w') as f:
    nbformat.write(nb, f)

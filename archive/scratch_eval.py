import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from skimage.feature import hog
import cv2

BASE_PATH = './'
REAL_IMAGE_DIR = os.path.join(BASE_PATH, 'project_dataset', 'Real-img')
FAKE_IMAGE_DIR = os.path.join(BASE_PATH, 'project_dataset', 'Image')

# To be safe against local file paths
if not os.path.exists(REAL_IMAGE_DIR) and os.path.exists(os.path.join(BASE_PATH, 'Real-img')):
    REAL_IMAGE_DIR = os.path.join(BASE_PATH, 'Real-img')
if not os.path.exists(FAKE_IMAGE_DIR) and os.path.exists(os.path.join(BASE_PATH, 'Image')):
    FAKE_IMAGE_DIR = os.path.join(BASE_PATH, 'Image')

class QuickDataset(Dataset):
    def __init__(self, real_dir, fake_dir):
        self.real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))][:500] if os.path.exists(real_dir) else []
        self.fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))][:500] if os.path.exists(fake_dir) else []
        self.all_files = self.real_files + self.fake_files
        self.labels = [0]*len(self.real_files) + [1]*len(self.fake_files)
    def __len__(self): return len(self.all_files)
    def __getitem__(self, idx):
        return self.all_files[idx], self.labels[idx]

if __name__ == '__main__':
    dataset = QuickDataset(REAL_IMAGE_DIR, FAKE_IMAGE_DIR)
    print(f"Loaded {len(dataset)} images")

    # Method 1: HOG Features
    print("\\nTesting HOG + RF...")
    hog_features = []
    labels = []
    for path, lbl in dataset:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            h = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            hog_features.append(h)
            labels.append(lbl)
        except: pass
    
    if len(hog_features) > 0:
        X = np.array(hog_features)
        y = np.array(labels)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        print(f"HOG + RF Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

    # Method 2: ResNet-18 FINE TUNED END-TO-END vs Pretrained Extract
    print("\\nTesting ResNet18 Supervised Fine-Tuning (Just 2 epochs to see if it learns)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class PTDataset(Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            path, lbl = self.ds[idx]
            img = Image.open(path).convert('RGB')
            return transform(img), lbl
            
    pt_ds = PTDataset(dataset)
    train_size = int(0.8 * len(pt_ds))
    test_size = len(pt_ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(pt_ds, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    for epoch in range(2):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
    
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(lbls.numpy())
    
    print(f"ResNet-18 End-to-End Accuracy: {accuracy_score(trues, preds):.4f}")

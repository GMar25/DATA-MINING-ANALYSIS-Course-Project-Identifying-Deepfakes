import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_PATH = './'
REAL_IMAGE_DIR = os.path.join(BASE_PATH, 'Real-img')
FAKE_IMAGE_DIR = os.path.join(BASE_PATH, 'Image')

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_files = []
        if os.path.exists(real_dir):
            self.real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.fake_files = []
        if os.path.exists(fake_dir):
            self.fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        self.all_files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.all_files[idx]).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, self.labels[idx]
        except:
            return torch.zeros((3, 224, 224)), self.labels[idx]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = DeepfakeDataset(REAL_IMAGE_DIR, FAKE_IMAGE_DIR, transform=transform)

# Sample 4000 images for quick test
indices = np.random.choice(len(dataset), 4000, replace=False)
subset = Subset(dataset, indices)
loader = DataLoader(subset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = models.resnet18(pretrained=True).to(device)
model.fc = torch.nn.Identity() # Remove classification head to get 512D embeddings
model.eval()

embeddings, labels = [], []
with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        feats = model(imgs)
        embeddings.append(feats.cpu().numpy())
        labels.extend(lbls.numpy())

X = np.vstack(embeddings)
y = np.array(labels)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"ResNet18 + Global Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")

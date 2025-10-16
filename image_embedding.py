import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ============================================================
# LOAD YOUR DATA
# ============================================================
npz_path = r"C:\Users\matan\.cache\kagglehub\datasets\saurabhbagchi\ship-and-iceberg-images\versions\1\input_data.npz"
npz = np.load(npz_path)
print("Loaded arrays:", npz.files)

# Choose which split you want to embed
X = npz["X_train"]
y = npz["Y_train"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# SETUP DEVICE & MODEL
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load pretrained ResNet18 and remove classification head
model = models.resnet18(weights='DEFAULT')
model.fc = nn.Identity()     # get 512-dim embeddings instead of logits
model = model.to(device)
model.eval()

# ============================================================
# PREPROCESS & BATCH EMBEDDING LOOP
# ============================================================
# Convert numpy → torch tensor
images = torch.tensor(X, dtype=torch.float32)

# If shape is (N, 75, 75, 3), reorder to (N, 3, 75, 75)
if images.shape[-1] == 3:
    images = images.permute(0, 3, 1, 2)

batch_size = 64
all_embeddings = []

# Normalization constants (ImageNet)
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

with torch.no_grad():
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)

        # Resize to 224x224 for ResNet input
        batch_resized = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalize
        batch_resized = (batch_resized - mean) / std

        # Extract features
        feats = model(batch_resized)
        all_embeddings.append(feats.cpu().numpy())

        print(f"Processed batch {i//batch_size + 1}/{(len(images)//batch_size) + 1}")

# Combine all batches
embeddings = np.vstack(all_embeddings)
print("Embedding extraction complete!")
print("Embeddings shape:", embeddings.shape)

# ============================================================
# SAVE EMBEDDINGS
# ============================================================
np.save("embeddings.npy", embeddings)
np.save("labels.npy", y)
print("Saved 'embeddings.npy' and 'labels.npy' successfully.")

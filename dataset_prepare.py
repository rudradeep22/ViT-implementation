import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from collections import Counter
from utils import plot_training_batch
from config import TRAIN_SIZE, VAL_SIZE, BATCH_SIZE

dataset_path = os.path.join("datasets", "dataset")
dataset_classes = os.listdir(dataset_path)
for dataset_class in dataset_classes:
    num_files = len([x for x in os.listdir(os.path.join(dataset_path, dataset_class))])
    print(f"Weather type: {dataset_class}, Num_images: {num_files}")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),        
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(                           
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
print("Datasets constructed")

idx_to_class = {val: key for key, val in full_dataset.class_to_idx.items()}

total = len(full_dataset)
train_size = int(TRAIN_SIZE * total)
val_size   = int(VAL_SIZE * total)
test_size  = total - train_size - val_size

train_set, val_set, test_set = random_split(
    full_dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
print(f"Splits -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

train_indices = train_set.indices     
train_labels  = [full_dataset.targets[i] for i in train_indices]

label_counts  = Counter(train_labels)
class_weights = {cls: 1.0/count for cls, count in label_counts.items()}
sample_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),  # one epoch still = train_size
    replacement=True                  # Oversampling
)

batch_size = BATCH_SIZE
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)
print("Oversampled train_loader ready, plus val/test loaders")

dataiter = iter(train_loader)
images, labels = next(dataiter)

print("Plotting smaple images for training")
plot_training_batch(images, labels, idx_to_class)

loader_path = os.path.join("datasets", "loaders")
if not os.path.exists(loader_path):
    os.makedirs(loader_path)

torch.save(train_loader, os.path.join(loader_path, 'train_loader.pth'))
torch.save(val_loader, os.path.join(loader_path, 'val_loader.pth'))
torch.save(test_loader, os.path.join(loader_path, 'test_loader.pth'))

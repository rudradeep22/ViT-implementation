import torch
import torchmetrics
import os
from config import NUM_CLASSES, IMG_SIZE, PATCH_SIZE, EMBED_DIM, DEPTH, MLP_DIM, NUM_HEADS
from models.vit_model import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_state_dict = torch.load("saved_models/vit_model.pth")
model = ViT(
    img_size=IMG_SIZE, 
    patch_size=PATCH_SIZE, 
    num_classes=NUM_CLASSES, 
    embed_dim=EMBED_DIM, 
    depth=DEPTH, 
    mlp_dim=MLP_DIM, 
    num_heads=NUM_HEADS)
model.to(device)
model.load_state_dict(model_state_dict)
model.eval()

test_loader = torch.load(os.path.join("datasets", "loaders", "test_loader.pth"), weights_only=False)

test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
test_accuracy.reset()

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        test_accuracy.update(outputs, targets) 

total_test_accuracy = test_accuracy.compute() 
print(f"Test Accuracy: {total_test_accuracy.item():.4f}")
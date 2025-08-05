import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchsummary import summary
from models.vit_model import ViT   
import os
from utils import plot_training_loss
from config import IMG_SIZE, PATCH_SIZE, NUM_CLASSES, EMBED_DIM, DEPTH, MLP_DIM, NUM_HEADS, NUM_EPOCHS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

train_loader = torch.load(os.path.join("datasets", "loaders", "train_loader.pth"), weights_only=False)
val_loader = torch.load(os.path.join("datasets", "loaders", "val_loader.pth"), weights_only=False)

model = ViT(
    img_size=IMG_SIZE, 
    patch_size=PATCH_SIZE, 
    num_classes=NUM_CLASSES, 
    embed_dim=EMBED_DIM, 
    depth=DEPTH, 
    mlp_dim=MLP_DIM, 
    num_heads=NUM_HEADS)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

train_losses = []
val_losses = []

val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)

summary(model, (3, IMG_SIZE, IMG_SIZE))

for epoch in range(NUM_EPOCHS):
    model.train()

    total_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    val_loss_sum = 0
    val_accuracy.reset() 

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss_sum += loss.item()
            val_accuracy.update(outputs, targets) 

    avg_val_loss = val_loss_sum / len(val_loader)
    epoch_val_accuracy = val_accuracy.compute() 

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    # if (epoch+1)%10 == 0:
    print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")

torch.save(model.state_dict(), f"saved_models/vit_model.pth")

epochs = list(range(1, NUM_EPOCHS+1))

plot_training_loss(NUM_EPOCHS, train_losses, val_losses)

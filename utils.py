import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt
import torch

def plot_training_batch(images, labels, idx_to_class):
    fig, axs = plt.subplots(6, 6, figsize=(8, 8))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i < len(images):  
            picture = torch.permute(images[i], (1, 2, 0))
            title = idx_to_class[labels[i].item()]
            ax.imshow(picture)
            ax.set_title(title)
            ax.axis('off')  
        else:
            fig.delaxes(ax) 

    plt.tight_layout()  
    print("Visualising training batch")
    plt.show()

def plot_training_loss(NUM_EPOCHS, train_losses, val_losses):
    epochs = list(range(1, NUM_EPOCHS+1))
    plt.plot(epochs, train_losses, 'b', label='Training loss') 
    plt.plot(epochs, val_losses, 'r', label='Validation loss') 
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
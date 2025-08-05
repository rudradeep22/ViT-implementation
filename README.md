# Vision Transformer (ViT) Implementation from Scratch

This repository contains a PyTorch implementation of the Vision Transformer (ViT) model from the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). The implementation is built from scratch and trained on a custom multi-class classification task.

The `notebooks/` directory contains a Jupyter notebook that contains the implementation of the ViT model.

The model is trained and evaluated on the [Weather Dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) from Kaggle.

## Performance

The model achieves the following performance metrics on the test set when trained for 50 epochs:
- **Best Accuracy**: 68.25%

## Model Configuration

The model can be configured using the following parameters (found in `config.py`):

```python
IMG_SIZE = 64         # Input image size
PATCH_SIZE = 8        # Size of each image patch
NUM_CLASSES = 11      # Number of output classes
EMBED_DIM = 128       # Embedding dimension
DEPTH = 4             # Number of transformer blocks
MLP_DIM = 512         # Hidden layer size in MLP
NUM_HEADS = 4         # Number of attention heads
BATCH_SIZE = 64       # Training batch size
```

### Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset (sample) using the script:
   ```bash
   chmod +x setup_dataset.sh
   ./setup_dataset.sh
   ```

3. Prepare dataset loaders using the script:
   ```bash
   python3 dataset_prepare.py
   ```

### Training the Model

To train the model with default configuration:
```bash
python3 train.py
```

### Evaluation

To evaluate the trained model on the test set:
```bash
python3 test.py
```
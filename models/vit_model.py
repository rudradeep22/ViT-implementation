import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=8, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=3,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)              
        x = x.flatten(2)             
        x = x.transpose(1, 2).contiguous()
        return x

class InputEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, img_size=64):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        x = self.patch_embed(x)  
        B = x.shape[0]
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)          
        x = x + self.pos_embedding                    
        return x

class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, mlp_dim, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = x.transpose(0, 1).contiguous()
        x, _ = self.mha(x, x, x)
        x = x.transpose(0, 1).contiguous()
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x

class ViT(nn.Module):
    def __init__(self, patch_size, img_size, num_classes, embed_dim, depth, mlp_dim, num_heads):
        super().__init__()
        self.input_embedding = InputEmbedding(patch_size, embed_dim, img_size)
        encoders = []
        for _ in range(depth):
            encoders.append(TransformerEncoderBlock(mlp_dim, embed_dim, num_heads))
    
        self.encoder = nn.Sequential(*encoders)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.lin = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.encoder(x)
        x = self.norm3(x)
        cls_tokens = x[:, 0]
        logits = self.lin(cls_tokens) 
        return logits
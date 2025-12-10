import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        assert self.num_heads * self.head_dim == self.embed_dim, "width must be divisible by heads"

        self.q_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.k_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.v_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.out_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x) # [batch_size, num_patches, embed_dim]
        q = q.reshape(B, N, self.num_heads, self.head_dim) # [batch_size, num_patches, num_heads, head_dim]
        q = q.transpose(1,2) # [batch_size, num_heads, num_patches, head_dim]

        k = self.k_proj(x) # [batch_size, num_patches, embed_dim]
        k = k.reshape(B,N,self.num_heads, self.head_dim) # [batch_size, num_patches, num_heads, head_dim]
        k = k.transpose(1,2) # [batch_size, num_heads, num_patches, head_dim]

        v = self.v_proj(x) # [batch_size, num_patches, embed_dim]
        v = v.reshape(B,N,self.num_heads, self.head_dim) # [batch_size, num_patches, num_heads, head_dim]
        v = v.transpose(1,2) # [batch_size, num_heads, num_patches, head_dim]

        # ============ MANUAL ATTENTION IMPLEMENTATION (commented for reference) ============
        # attn_scores = torch.matmul(q, k.transpose(2,3)) # [batch_size, num_heads, num_patches, num_patches]
        # attn_scores = attn_scores / self.head_dim**0.5 # [batch_size, num_heads, num_patches, num_patches]
        # attn_scores = torch.softmax(attn_scores, dim=-1) # [batch_size, num_heads, num_patches, num_patches]
        # output = torch.matmul(attn_scores, v) # [batch_size, num_heads, num_patches, head_dim]
        # ===================================================================================

        # OPTIMIZED: Using PyTorch's scaled_dot_product_attention (Flash Attention)
        output = F.scaled_dot_product_attention(q, k, v)

        output = output.transpose(1,2) # [batch_size, num_patches, num_heads, head_dim]
        output = output.flatten(2) # [batch_size, num_patches, embed_dim]
        output = self.out_proj(output) # [batch_size, num_patches, embed_dim]
        return output

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.input_size = embed_dim
        self.hidden_size = self.input_size * 4
        self.output_size = self.input_size
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim, eps, num_heads):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=eps)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=eps)
        self.attn = Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp = MLP(embed_dim=embed_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.attn(x)
        x = x + residual
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


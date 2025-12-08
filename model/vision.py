import torch
from torch import nn
from nano_clip.config import config

class PatchEmbedding(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=3, # we are dealing with RGB images
            out_channels = cfg.vision_width,
            kernel_size = cfg.patch_size,
            stride = cfg.patch_size
        )
    
    def forward(self, x):

        x = self.proj(x) # [batch_size, channels, image_height, image_width] -> [batch_size, vision_width, num_patches_H, num_patches_W]
        x = x.flatten(2) # [batch_size, vision_width, num_patches_H, num_patches_W] -> [batch_size, vision_width, num_patches]
        x = x.transpose(1,2) # [batch_size, vision_width, num_patches] -> [batch_size, num_patches, vision_width]
        return x


class VisionTransformer(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.patch_embed = PatchEmbedding(cfg) # the embedding for our batch
        self.cls_token = nn.Parameter(torch.zeros((1,1,cfg.vision_width))) # the classification token
        self.positional_embedding = nn.Parameter(torch.randn(1, (cfg.image_size // cfg.patch_size)**2 +1, cfg.vision_width))
        self.final_layer_norm = nn.LayerNorm(cfg.vision_width, eps=cfg.vision_layer_norm_eps)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.vision_layers)])

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.positional_embedding
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        return x[:,0,:] # we only want to return the CLS token as output, in order to compare with the text

class Attention(nn.Module):
    def __init__(self, cfg:Config):
        super().__init__()
        self.embed_dim = cfg.vision_width
        self.num_heads = cfg.vision_heads
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

        attn_scores = torch.matmul(q, k.transpose(2,3)) # [batch_size, num_heads, num_patches, num_patches]
        attn_scores = attn_scores / self.head_dim**0.5 # [batch_size, num_heads, num_patches, num_patches]
        attn_scores = torch.softmax(attn_scores, dim=-1) # [batch_size, num_heads, num_patches, num_patches]

        v = self.v_proj(x) # [batch_size, num_patches, embed_dim]
        v = v.reshape(B,N,self.num_heads, self.head_dim) # [batch_size, num_patches, num_heads, head_dim]
        v = v.transpose(1,2) # [batch_size, num_heads, num_patches, head_dim]
        output = torch.matmul(attn_scores, v) # [batch_size, num_heads, num_patches, head_dim]
        output = output.transpose(1,2) # [batch_size, num_patches, num_heads, head_dim]
        output = output.flatten(2) # [batch_size, num_patches, embed_dim]
        output = self.out_proj(output) # [batch_size, num_patches, embed_dim]
        return output

class MLP(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.input_size = cfg.vision_width
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
    def __init__(self, cfg : Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.vision_width, eps=cfg.vision_layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(cfg.vision_width, eps=cfg.vision_layer_norm_eps)
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

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


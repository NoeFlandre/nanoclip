import torch
from torch import nn
from nano_clip.config import config
from nano_clip.utils import Block

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
        self.blocks = nn.ModuleList([Block(embed_dim=cfg.vision_width, eps=cfg.vision_layer_norm_eps) for _ in range(cfg.vision_layers)])

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


import torch
from torch import nn
from nano_clip.config import Config
from nano_clip.model.text import TextTransformer
from nano_clip.model.vision import VisionTransformer
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.vision_encoder = VisionTransformer(cfg)
        self.text_encoder = TextTransformer(cfg)
        self.temperature = nn.Parameter(torch.ones([])*torch.log(torch.tensor(1/0.07)))
        self.vision_proj = nn.Linear(in_features=cfg.vision_width, out_features=cfg.shared_dim)
        self.text_proj = nn.Linear(in_features=cfg.text_width, out_features=cfg.shared_dim)

    def forward(self, image, text):
        image_features = self.vision_encoder(image) # [batch_size, vision_width]
        image_features = self.vision_proj(image_features) # [batch_size, shared_dim]
        text_features = self.text_encoder(text) # [batch_size, text_width]
        text_features = self.text_proj(text_features) # [batch_size, shared_dim]
        text_features, image_features = F.normalize(text_features, dim=-1), F.normalize(image_features, dim=-1) #similarity should matter only for the direction of the vector, not their magnitude
        similarity = image_features @ text_features.T
        similarity = similarity * torch.exp(self.temperature)
        return similarity
import torch
from torch import nn
from config import Config
from .text import TextTransformer
from .vision import VisionTransformer
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.vision_encoder = VisionTransformer(cfg)
        self.text_encoder = TextTransformer(cfg)
        # Initialize temperature to log(1/0.07) ≈ 2.66 following CLIP paper
        # This means exp(temperature) ≈ 14.3, which properly scales similarity scores
        self.temperature = nn.Parameter(torch.tensor(2.6593))
        self.vision_proj = nn.Linear(in_features=cfg.vision_width, out_features=cfg.shared_dim)
        self.text_proj = nn.Linear(in_features=cfg.text_width, out_features=cfg.shared_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image, text):
        N = image.shape[0]
        device = image.device
        labels = torch.arange(N).to(device)
        image_features = self.vision_encoder(image) # [batch_size, vision_width]
        image_features = self.vision_proj(image_features) # [batch_size, shared_dim]
        text_features = self.text_encoder(text) # [batch_size, text_width]
        text_features = self.text_proj(text_features) # [batch_size, shared_dim]
        text_features, image_features = F.normalize(text_features, dim=-1), F.normalize(image_features, dim=-1) #similarity should matter only for the direction of the vector, not their magnitude
        similarity = image_features @ text_features.T
        similarity = similarity * torch.exp(self.temperature)
        loss_T = self.loss_fn(similarity.T, labels)
        loss_I = self.loss_fn(similarity, labels)
        average_loss = (loss_T + loss_I) / 2
        return similarity, average_loss
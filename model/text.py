import torch
from torch import nn
from config import Config
from utils import Block

class TextTransformer(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.max_length = cfg.max_length
        self.vocab_size = cfg.vocab_size
        self.text_width = cfg.text_width
        self.token_embedding = nn.Embedding(self.vocab_size, self.text_width)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.max_length+1, self.text_width)) # [batch_size, max_length, text_width]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.text_width))
        self.blocks = nn.ModuleList([Block(embed_dim=self.text_width, eps=cfg.text_layer_norm_eps, num_heads=cfg.text_heads) for _ in range(cfg.text_layers)])
        self.final_layer_norm = nn.LayerNorm(cfg.text_width, eps=cfg.text_layer_norm_eps)

    def forward(self, x):
        x = self.token_embedding(x)
        B= x.shape[0]
        cls_token = self.cls_token.expand(B,-1,-1)
        x = torch.concat((cls_token, x), dim=1)
        N = x.shape[1]
        positional_embedding = self.positional_embedding[:,:N,:]
        x += positional_embedding
        for block in self.blocks :
            x = block(x)
        x = self.final_layer_norm(x)
        return x[:,0,:]
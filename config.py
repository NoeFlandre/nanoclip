from dataclasses import dataclass

class Config:

    # 1) Vision Encoder Parameters (ViT-Tiny)
    image_size : int = 224
    patch_size : int = 16
    vision_width : int = 192
    vision_layers : int = 12
    vision_heads : int = 3
    vision_layer_norm_eps = 1e-6

    # 2) Text Encoder Parameters (Nano-Transformer)
    vocab_size : int = 30000
    text_width : int = 256
    text_layers : int = 8
    text_heads : int = 4

    # 3) Training Parameters
    batch_size : int = 512
    learning_rate : float : 1e-3
    epochs : int = 30
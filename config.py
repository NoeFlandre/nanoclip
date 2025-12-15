from dataclasses import dataclass

@dataclass
class Config:

    # 1) Vision Encoder Parameters (ViT-Tiny)
    image_size : int = 224 # this is the resolution of our image
    patch_size : int = 16 # this is the size of a patch unit within the image, each patch produces one embedding
    vision_width : int = 192 # this is the embedding size within the vision encoder
    vision_layers : int = 12 # this is the number of transformer block we are using within the vision encoder
    vision_heads : int = 3 # this is the number of heads we are using inside the multi head attention in the vision encoder
    vision_layer_norm_eps : float = 1e-6 # this is the epsilon we are using in the layer normalization for numerical stability in the vision encoder

    # 2) Text Encoder Parameters (Nano-Transformer)
    vocab_size : int = 49408 # this is the vocabulary size of our text encoder (the number of unique tokens)
    text_width : int = 256 # this is the embedding dimension within our text encoder
    text_layers : int = 8 # this is the number of transformer block we have within our text encoder
    text_heads : int = 4 # this is the number of attention heads we have in our multi head attention within the text encoder
    max_length : int = 77 # this is the maximum number of embeddings in a sequence
    text_layer_norm_eps : float = 1e-6 # the epsilon we use for numerical stability in our layer normalization in the text encoder

    # 3) Shared Parameters
    shared_dim : int = 512 # this is the dimension of the common shared space between text and space

    # 4) Training Parameters
    batch_size : int = 256 # size of our batch (fits well in 12GB VRAM)
    learning_rate : float = 1e-4 # learning rate (2x original, safer with warmup + clipping)
    epochs : int = 10 # more epochs to see convergence
    loss_type : str = "contrastive"
    gradient_clipping = True
    warmup_steps_percentage = 0.10 # 10% warmup for smoother start
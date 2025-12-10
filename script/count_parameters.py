import torch
from model.clip import CLIP
from config import Config

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

config = Config()
model = CLIP(config)

total_parameters = count_parameters(model)
vision_parameters = count_parameters(model.vision_encoder)
text_parameters = count_parameters(model.text_encoder)
text_embedding_layer_parameters = count_parameters(model.text_encoder.token_embedding)

print("Model Statistics")
print(f"Total number of parameters : {total_parameters}")
print(f"Vision parameters : {vision_parameters}")
print(f"Text parameters : {text_parameters}")
print(f"Text Embedding Layer (dictionary lookup (vocab_size, embed_dim)) {text_embedding_layer_parameters}")
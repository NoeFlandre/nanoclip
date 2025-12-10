import torch
from config import Config
from model.clip import CLIP

images = torch.randn(4,3,224,224)
text = torch.randint(0,1000,(4,10))

model = CLIP(Config())

similarity, loss = model(images,text)
print(f"Similarity shape : {similarity.shape} | Loss shape : {loss.shape}")

import torch
from PIL import Image 
from torchvision import transforms
from dataset import CLIPDataset
from model.tokenizer import CLIPTokenizer

# we create a random red dummy image for testing purposes
Image.new('RGB', (100,100), color='red').save("data/test_image.jpg")

tokenizer = CLIPTokenizer()

transform_pipeline = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

dataset = CLIPDataset(
    captions = ["a photo of a red square"],
    image_paths = ["data/test_image.jpg"],
    transforms = transform_pipeline,
    tokenizer = tokenizer
)

image, token_ids = dataset[0]

print(f"Image shape : {image.shape}")
print(f"Tokens shape : {token_ids.shape}")
print(f"Five first tokens : {token_ids[:5]}")
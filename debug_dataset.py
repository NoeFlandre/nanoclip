import torch
from PIL import Image 
from torchvision.datasets import CIFAR10

class CIFAR10CLIPDataset(CIFAR10):
    def __init__(self, train, root, transform, tokenizer, return_labels=False):
        super().__init__(root=root, train=train, download=True, transform=None)
        self.clip_transform = transform
        self.tokenizer = tokenizer
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.return_labels = return_labels

    def __getitem__(self, index):
        image, label_idx = super().__getitem__(index)

        if self.clip_transform:
            image = self.clip_transform(image)

        if self.return_labels:
            return image, label_idx 
            
        class_name = self.classes[label_idx]
        caption = f"a photo of a {class_name}"

        tokens = self.tokenizer(caption)

        return image, tokens
import torch
from PIL import Image
from torch.utils.data import Dataset

class CLIPDataset(Dataset):
    def __init__(self, captions, image_paths, transforms, tokenizer):
        """
        image_paths : A list of file paths to images (e.g ["data/dog.jpg", ...])
        captions : A list of captions for each image (e.g ["a photo of a dog", ...])
        transforms : A torchvision transform pipeline to resize / normalize the images
        tokenizer : A function which is converting a text string to a tensor of integers
        
        """
        self.captions = captions
        self.image_paths = image_paths
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __len__(self):
        # return the number of samples
        return len(self.captions)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx] # loads the image
        image = Image.open(image_path).convert("RGB") # converts the image to RGB to make sure we always have 3 channels 
        if self.transforms:
            image = self.transforms(image) # we transform the image through the torchvision pipeline

        caption = self.captions[idx] # we retrieve the caption
        tokens = self.tokenizer(caption) # we convert the caption in tokens
        return image, tokens 
        

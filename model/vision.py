import torch
from torch import nn
from config import Config
from utils import Block

class PatchEmbedding(nn.Module):

    """ 
    This class is taking an image as input and is outputing the corresponding embeddings. 

    Input : [batch_size, channels, height, width]
    Output = [batch_size, num_patches, embedding_dimension]

    We use a 2D Convolution to perform this. We use a kernel of the size of each patch along with a stride of this exact same size
    in order to make sure that there is no overlapping. 
    """
    def __init__(self, cfg : Config):
        super().__init__() # call the parent class's __init__
        self.proj = nn.Conv2d(
            in_channels=3, # we are dealing with RGB images so there are 3 channels
            out_channels = cfg.vision_width, # this is the number of channels we are outputting and in our case it corresponds to the dimension of the embeddings
            kernel_size = cfg.patch_size, # our kernel is perfectly matching our patch size
            stride = cfg.patch_size # the kernel is sliding exactly over each patch to make sure there is no overlapping
        )
    
    def forward(self, x):

        x = self.proj(x) # [batch_size, channels, image_height, image_width] -> [batch_size, vision_width, num_patches_H, num_patches_W] this is the output of our convolution
        x = x.flatten(2) # [batch_size, vision_width, num_patches_H, num_patches_W] -> [batch_size, vision_width, num_patches] we collapse the last two dimensions (which are multiplied together), in order to get the total number of patches
        x = x.transpose(1,2) # [batch_size, vision_width, num_patches] -> [batch_size, num_patches, vision_width] here we transpose to respect the input dimension expected by the transformer
        return x


class VisionTransformer(nn.Module):
    """
    This class is definiting a Vision Transformer. It handles the data taken as input with the classification token and the positional embedding.
    This class is implementing the layers of block as well as the final layer normalization before outputing the classification token. 
    
    """
    def __init__(self, cfg : Config):
        super().__init__() # call the parent class's __init__
        self.patch_embed = PatchEmbedding(cfg) # the embeddings for our batch
        self.cls_token = nn.Parameter(torch.zeros((1,1,cfg.vision_width))) # the classification token which we initialized as zeros
        self.positional_embedding = nn.Parameter(torch.randn(1, (cfg.image_size // cfg.patch_size)**2 +1, cfg.vision_width)) # the positional embeddings which is initialized at random
        self.final_layer_norm = nn.LayerNorm(cfg.vision_width, eps=cfg.vision_layer_norm_eps) # the final layer normalization which comes after the very end of the block layers
        self.blocks = nn.ModuleList([Block(embed_dim=cfg.vision_width, eps=cfg.vision_layer_norm_eps, num_heads=cfg.vision_heads) for _ in range(cfg.vision_layers)]) # the different layers of blocks

    def forward(self, x):
        B = x.shape[0] # we retrieve the batch size
        x = self.patch_embed(x) # from our input image we retrieve their corresponding embeddings
        cls_token = self.cls_token.expand(B,-1,-1) # we define our classification token which we expand across the batch 
        x = torch.cat((cls_token, x), dim=1) # we concatenate the cls token and our patch embeddings
        x = x + self.positional_embedding # we add the positional embedding
        for block in self.blocks: # we iterate across all blocks
            x = block(x) # the data is goig through each block
        x = self.final_layer_norm(x) # we normalized one last time
        return x[:,0,:] # we only want to return the CLS token as output, in order to compare with the text


import torch
from model.tokenizer import CLIPTokenizer
from model.clip import CLIP
from dataset import CLIPDataset
from torch.utils.data import DataLoader
from config import Config 
from torchvision import transforms
from debug_dataset import CIFAR10CLIPDataset

def main():
    cfg = Config()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else :
        device = "cpu"
    
    print(f"Device : {device}")

    tokenizer = CLIPTokenizer()
    model = CLIP(cfg).to(device)

    transform_pipeline = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    dataset = CIFAR10CLIPDataset(
        transform=transform_pipeline,
        tokenizer=tokenizer,
        root="data",
        train=True
    )

    dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)

    train_loss_values = []

    for epoch in range(cfg.epochs):
        model.train()
        max_batch = 5
        for batch, (X, Y) in enumerate(dataloader):
            if batch < max_batch :
                print(f"Batch : {batch} / {len(dataloader)}")
                X, Y = X.to(device), Y.to(device)
                similarity, train_loss = model(image=X, text=Y)
                train_loss_values.append(train_loss.item())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
        print(f"Epoch : {epoch} | Training Loss : {train_loss_values[-1]} ")

if __name__ == "__main__":
        main()
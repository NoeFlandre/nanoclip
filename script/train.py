import torch
from model.tokenizer import CLIPTokenizer
from model.clip import CLIP
from dataset import CLIPDataset
from torch.utils.data import DataLoader
from config import Config 
from torchvision import transforms
from debug_dataset import CIFAR10CLIPDataset
import time
import os 
from utils import ZeroShotEvaluator
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "logs.txt")
    with open(log_file, "w") as f:
        pass
    
    cfg = Config()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else :
        device = "cpu"

    print(f"Device : {device}")
    with open(log_file, "a") as f:
        f.write(f"Device : {device}\n")    

    tokenizer = CLIPTokenizer()
    model = CLIP(cfg).to(device)
    if device == "cuda":
        model = torch.compile(model)

    transform_pipeline = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    train_dataset = CIFAR10CLIPDataset(
        transform=transform_pipeline,
        tokenizer=tokenizer,
        root="data",
        train=True
    )

    test_dataset = CIFAR10CLIPDataset(
        transform = transform_pipeline,
        tokenizer=tokenizer,
        root="data",
        train=False,
        return_labels=True
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    evaluator = ZeroShotEvaluator(model, tokenizer, CIFAR10_CLASSES, device)

    print(f"Training samples {len(train_dataloader.dataset)}")
    with open(log_file, "a") as f:
        f.write(f"Training samples {len(train_dataloader.dataset)}\n")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.learning_rate)

    train_loss_values = []

    for epoch in range(cfg.epochs):
        model.train()
       # max_batch = 5
       # count = 0
        for batch, (X, Y) in enumerate(train_dataloader):
           # count+=1
           # if count > max_batch :
           #     break
            t0 = time.time()
            X, Y = X.to(device), Y.to(device)
            similarity, train_loss = model(image=X, text=Y)
            train_loss_value = train_loss.item()
            train_loss_values.append(train_loss_value)
            optimizer.zero_grad()
            train_loss.backward()
            norm = 0
            for param in model.parameters():
                norm+=(torch.norm(param.grad))**2
            norm = torch.sqrt(norm)
            optimizer.step()
            t1 = time.time()
            dt = (t1-t0)*1000 #ms
            print(f"Batch : {batch} / {len(train_dataloader)} | Time {dt:.0f} ms | Train Loss : {train_loss_value:.4f} | Grad Norm : {norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"Batch : {batch} / {len(train_dataloader)} | Time {dt:.0f} ms | Train Loss : {train_loss_value:.4f} | Grad Norm : {norm:.4f}\n")
        accuracy = evaluator.evaluate(test_dataloader)
        model.train()
        print(f"Epoch : {epoch} | Training Loss : {train_loss_values[-1]:.4f} | Accuracy on test set : {accuracy:.4f} ")
        with open(log_file, "a") as f:
            f.write(f"Epoch : {epoch} | Training Loss : {train_loss_values[-1]:.4f}\n ")
if __name__ == "__main__":
        main()
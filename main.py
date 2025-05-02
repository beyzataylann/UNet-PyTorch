import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from Segmentation.dataset import Dataset

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience=5):
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for img, mask in tqdm(train_loader):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == mask).sum().item()
            total += mask.numel()
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                output = model(img)
                loss = criterion(output, mask)
                val_loss += loss.item()
                pred = (torch.sigmoid(output) > 0.5).float()
                val_correct += (pred == mask).sum().item()
                val_total += mask.numel()
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = Dataset("/content/drive/MyDrive/bitirmeprojesi/pdata")
    val_dataset = Dataset("/content/drive/MyDrive/bitirmeprojesi/pdata", test=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    model = UNet(3, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=50, patience=7)
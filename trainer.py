import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from src.models.model import SimpleCNN
from src.datas.dataloader import create_dataloaders
from utils import save_checkpoint, load_checkpoint

# --- Cấu hình huấn luyện ---
DATA_DIR = 'data'
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
CHECKPOINT_FILE = "weather_best.pth.tar"
RESUME_TRAINING = True

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    dataloaders, class_names = create_dataloaders(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)

    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    best_val_loss = float('inf')

    if RESUME_TRAINING and os.path.exists(CHECKPOINT_FILE):
        completed_epoch = load_checkpoint(CHECKPOINT_FILE, model, optimizer)
        start_epoch = completed_epoch 
        
        print(f"Resume training từ epoch {start_epoch + 1}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        loop = tqdm(dataloaders['train'], leave=True)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

            loop.set_postfix(loss=loss.item())

        epoch_train_loss = train_loss / len(dataloaders['train'].dataset)
        epoch_train_acc = train_corrects.double() / len(dataloaders['train'].dataset)
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(dataloaders['val'].dataset)
        epoch_val_acc = val_corrects.double() / len(dataloaders['val'].dataset)
        print(f"Validation Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        if epoch_val_loss < best_val_loss:
            print(f"Validation loss giảm ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Đang lưu model...")
            best_val_loss = epoch_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'class_to_idx': dataloaders['train'].dataset.class_to_idx
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_FILE)

    print("\nHoàn tất quá trình huấn luyện!")

if __name__ == '__main__':
    main()
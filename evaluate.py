import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.models.model import SimpleCNN
from src.datas.dataloader import create_dataloaders

# --- Cấu hình ---
CHECKPOINT_FILE = "weather_best.pth.tar"
DATA_DIR = 'data'
BATCH_SIZE = 32

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Đang tải model từ {CHECKPOINT_FILE}")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    class_to_idx = checkpoint['class_to_idx']
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    
    dataloaders, _ = create_dataloaders(DATA_DIR, BATCH_SIZE)
    test_loader = dataloaders['test']
    
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Báo cáo Phân loại (Classification Report) ---")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='BuGn')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    print("Đã lưu ma trận nhầm lẫn vào file 'confusion_matrix.png'")


if __name__ == '__main__':
    evaluate()
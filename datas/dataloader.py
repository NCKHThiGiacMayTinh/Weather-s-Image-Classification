# dataloader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def create_dataloaders(data_dir, batch_size):
    """
    Tạo các đối tượng DataLoader cho tập train, validation và test.
    Args:
        data_dir (str): Đường dẫn đến thư mục 'data' chứa train/val/test.
        batch_size (int): Kích thước của mỗi batch.
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    # Sử dụng ImageFolder để tự động đọc ảnh và gán nhãn từ cấu trúc thư mục
    image_datasets = {x: datasets.ImageFolder(f'{data_dir}/{x}', data_transforms[x])
                      for x in ['train', 'val', 'test']}

    # Tạo các DataLoader
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val', 'test']}

    class_names = image_datasets['train'].classes
    
    print(f"Các lớp được tìm thấy: {class_names}")
    print(f"Số lớp: {len(class_names)}")

    return dataloaders, class_names
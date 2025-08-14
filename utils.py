import torch

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Lưu checkpoint."""
    print("=> Đang lưu checkpoint")
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer):
    """
    Tải checkpoint từ một đường dẫn file.
    Sẽ tải trạng thái của model và optimizer.
    """
    print(f"=> Đang tải checkpoint từ '{filepath}'")
    checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint['epoch']
import os
import shutil
from sklearn.model_selection import train_test_split
import glob

print("Bắt đầu quá trình phân chia dữ liệu...")

# --- Cấu hình ---
SOURCE_DIR = 'dataset'
DEST_DIR = 'data'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

if os.path.exists(DEST_DIR):
    print(f"Thư mục '{DEST_DIR}' đã tồn tại. Xóa để tạo lại...")
    shutil.rmtree(DEST_DIR)

os.makedirs(os.path.join(DEST_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, 'val'), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, 'test'), exist_ok=True)

weather_classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for wc_class in weather_classes:
    class_path = os.path.join(SOURCE_DIR, wc_class)
    
    os.makedirs(os.path.join(DEST_DIR, 'train', wc_class), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, 'val', wc_class), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, 'test', wc_class), exist_ok=True)
    
    all_files = glob.glob(os.path.join(class_path, '*.*'))
    
    train_files, val_test_files = train_test_split(
        all_files, 
        test_size=(1 - TRAIN_RATIO), 
        random_state=RANDOM_STATE
    )

    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_files, test_files = train_test_split(
        val_test_files, 
        test_size=relative_test_ratio, 
        random_state=RANDOM_STATE
    )
    
    print(f"Lớp '{wc_class}': {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    def copy_files(files, split_name):
        for f in files:
            shutil.copy(f, os.path.join(DEST_DIR, split_name, wc_class, os.path.basename(f)))
            
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

print("\nPhân chia dữ liệu hoàn tất!")
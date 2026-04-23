import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import io

# Cấu hình hiển thị tiếng Việt trên Terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 1. TẢI DỮ LIỆU ĐÃ TIỀN XỬ LÝ
data_path = r"C:\Users\PC\Study\NL-ATTT\ember2018\ember_cleaned_step1.npz"
print("1. Đang tải dữ liệu gốc từ:", data_path)
start_time = time.time()

data = np.load(data_path)
X_full = data['X']
y_full = data['y']

print(f"   -> Đã tải xong! Tổng số dữ liệu: {X_full.shape[0]:,} mẫu, {X_full.shape[1]} đặc trưng.")

# ==========================================
# 2. CHIA TẬP TRAIN (80%) VÀ TEST (20%)
print("\n2. Tiến hành cắt dữ liệu...")
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, 
    test_size=0.2, 
    stratify=y_full, 
    random_state=42
)

print(f"   -> Tập Huấn luyện (Train): {X_train.shape[0]:,} mẫu.")
print(f"   -> Tập Kiểm thử (Test):  {X_test.shape[0]:,} mẫu.")

# ==========================================
# 3. LƯU THÀNH CÁC FILE RIÊNG BIỆT
print("\n3. Đang lưu ra các file nén (.npz)...")

train_file = 'ember_train.npz'
test_file = 'ember_test.npz'

np.savez_compressed(train_file, X=X_train, y=y_train)
np.savez_compressed(test_file, X=X_test, y=y_test)

print(f"\nHoàn tất toàn bộ sau {(time.time() - start_time):.2f} giây!")
print(f"Đã tạo thành công:\n   - {train_file}\n   - {test_file}")
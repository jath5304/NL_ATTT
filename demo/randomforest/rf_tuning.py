import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import time
import sys
import io

# Cấu hình hiển thị tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. TẢI DỮ LIỆU ĐÃ TIỀN XỬ LÝ
data_path = r"C:\Users\PC\Study\NL-ATTT\ember2018\ember_cleaned_step1.npz"
print("1. Đang tải dữ liệu từ:", data_path)
data = np.load(data_path)
X_full = data['X']
y_full = data['y']

print(f"   -> Dữ liệu gốc: {X_full.shape[0]:,} mẫu x {X_full.shape[1]} chiều")

# 2. LẤY MẪU PHỤ (SUB-SAMPLING) ĐỂ CHẠY THỬ NGHIỆM CHO NHANH
# Lấy ra 50.000 mẫu, dùng stratify=y_full để đảm bảo tỷ lệ Lành tính/Mã độc vẫn là 50/50
print("\n2. Đang trích xuất tập mẫu nhỏ (50,000 mẫu) để dò tham số...")
X_sample, _, y_sample, _ = train_test_split(
    X_full, y_full, 
    train_size=50000, 
    stratify=y_full, 
    random_state=42
)
print(f"   -> Kích thước tập thử nghiệm: {X_sample.shape[0]:,} mẫu")

# 3. THIẾT LẬP VÀ CHẠY GRID SEARCH
print("\n3. Bắt đầu Grid Search Cross-Validation...")
start_time = time.time()

param_grid = {
    'max_depth': [10, 20, 30],
    'n_estimators': [100, 250, 500, 750]
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf_model, 
    param_grid=param_grid, 
    cv=3,                # Chia dữ liệu làm 3 phần để đánh giá chéo
    scoring='accuracy',  # Đánh giá bằng độ chính xác
    verbose=2,           # In log tiến trình ra Terminal
    n_jobs=-1            # Dùng tất cả nhân CPU để chạy song song
)

# Bắt đầu huấn luyện thử nghiệm
grid_search.fit(X_sample, y_sample)

print(f"\nQuá trình tìm kiếm hoàn tất sau {(time.time() - start_time)/60:.2f} phút!")
print(f"Tham số tối ưu nhất tìm được: {grid_search.best_params_}")
print(f"Độ chính xác cao nhất: {grid_search.best_score_:.4f}")

# 4. TRÍCH XUẤT ĐIỂM SỐ ĐỂ VẼ ĐỒ THỊ HEATMAP
# Đây chính là bước trích xuất ra các con số 0.945, 0.950... mà bạn thắc mắc
scores = grid_search.cv_results_['mean_test_score']

# Chuyển đổi mảng điểm số 1D thành ma trận 2D
scores_matrix = scores.reshape(len(param_grid['max_depth']), len(param_grid['n_estimators']))
df_scores = pd.DataFrame(scores_matrix, index=param_grid['max_depth'], columns=param_grid['n_estimators'])

# 5. VẼ ĐỒ THỊ
print("\n4. Đang vẽ đồ thị báo cáo...")
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(
    df_scores, annot=True, fmt=".3f", cmap="Blues", 
    cbar_kws={'label': 'Validation Accuracy'}, ax=ax
)

ax.set_title('Random Forest: Grid Search (Depth vs Estimators)', fontsize=14)
ax.set_xlabel('Số lượng cây (n_estimators)', fontsize=12)
ax.set_ylabel('Độ sâu tối đa (max_depth)', fontsize=12)

# Tự động đóng khung đỏ vào ô có điểm số cao nhất
best_row_idx = param_grid['max_depth'].index(grid_search.best_params_['max_depth'])
best_col_idx = param_grid['n_estimators'].index(grid_search.best_params_['n_estimators'])

rect = patches.Rectangle(
    (best_col_idx, best_row_idx), 1, 1, 
    linewidth=4, edgecolor='red', facecolor='none'
)
ax.add_patch(rect)
plt.text(
    best_col_idx + 0.5, best_row_idx + 0.55, 
    'Selected\n(Best)', 
    ha='center', va='center', color='red', fontsize=10, fontweight='bold'
)

plt.tight_layout()
plt.savefig('rf_tuning_results.png', dpi=300)
print("   -> Đã lưu biểu đồ thành 'rf_tuning_results.png'.")
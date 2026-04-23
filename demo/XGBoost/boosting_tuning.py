import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
import time
import sys
import io

# Cấu hình hiển thị tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
#TẢI DỮ LIỆU
data_path = r"C:\Users\PC\Study\NL-ATTT\ember2018\ember_cleaned_step1.npz"
print("1. Đang tải dữ liệu...")
data = np.load(data_path)
X_full = data['X']
y_full = data['y']

# Lấy 50,000 mẫu để Grid Search chạy nhanh hơn
print("2. Đang trích xuất 50,000 mẫu để thử nghiệm...")
X_sample, _, y_sample, _ = train_test_split(
    X_full, y_full, 
    train_size=50000, 
    stratify=y_full, 
    random_state=42
)

# ==========================================
# 2. THIẾT LẬP VÀ CHẠY GRID SEARCH
print("\n3. Bắt đầu Grid Search cho XGBoost...")
start_time = time.time()

# Lưới tham số dựa trên biểu đồ của bạn
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [6, 8, 10, 12]
}

# CHỌN MÔ HÌNH: 
# Cố định số lượng cây n_estimators=300 (hoặc 500) để tập trung dò Learning Rate và Depth
# model = lgb.LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1)

# NẾU DÙNG XGBOOST, BỎ COMMENT DÒNG DƯỚI ĐÂY:
model = xgb.XGBClassifier(n_estimators=300, tree_method='hist', random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    cv=3,                # Kiểm chứng chéo 3-fold
    scoring='accuracy',  
    verbose=2,
    n_jobs=1
)

grid_search.fit(X_sample, y_sample)

print(f"\n✅ Tìm kiếm hoàn tất sau {(time.time() - start_time)/60:.2f} phút!")
print(f"🌟 Tham số tối ưu nhất: {grid_search.best_params_}")
print(f"🌟 Độ chính xác cao nhất: {grid_search.best_score_:.4f}")

# ==========================================
# 3. TRÍCH XUẤT KẾT QUẢ, VẼ HEATMAP
print("\n4. Đang vẽ Dashboard kết quả (Màu xanh lá)...")

# Trích xuất mảng điểm số
scores = grid_search.cv_results_['mean_test_score']

# Chuyển thành ma trận 2D: Hàng là learning_rate, Cột là max_depth
scores_matrix = scores.reshape(len(param_grid['learning_rate']), len(param_grid['max_depth']))
df_scores = pd.DataFrame(
    scores_matrix, 
    index=param_grid['learning_rate'], 
    columns=param_grid['max_depth']
)

# Thiết lập vẽ đồ thị
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(10, 7))

# Vẽ Heatmap với tông màu Xanh lá (YlGn = Yellow to Green)
sns.heatmap(
    df_scores, 
    annot=True, 
    fmt=".3f", 
    cmap="YlGn", 
    cbar_kws={'label': 'Validation Accuracy'}, 
    ax=ax
)

ax.set_title('XGBoost/LightGBM: Learning Rate vs Complexity', fontsize=14)
ax.set_xlabel('Độ phức tạp cây (Max Depth / Tương đương Num_leaves)', fontsize=12)
ax.set_ylabel('Tốc độ học (Learning Rate)', fontsize=12)

# Tự động tìm vị trí của tham số tốt nhất để vẽ khung đỏ
best_lr = grid_search.best_params_['learning_rate']
best_depth = grid_search.best_params_['max_depth']

best_row_idx = param_grid['learning_rate'].index(best_lr)
best_col_idx = param_grid['max_depth'].index(best_depth)

# Vẽ khung đỏ đánh dấu
rect = patches.Rectangle(
    (best_col_idx, best_row_idx), 1, 1, 
    linewidth=4, edgecolor='red', facecolor='none'
)
ax.add_patch(rect)

# Viết chữ "Optimal Region"
plt.text(
    best_col_idx + 0.5, best_row_idx + 0.55, 
    'Optimal\nRegion', 
    ha='center', va='center', color='red', fontsize=10, fontweight='bold'
)

plt.tight_layout()
chart_name = 'boosting_tuning_heatmap.png'
plt.savefig(chart_name, dpi=300)
print(f"   -> Đã lưu biểu đồ thành '{chart_name}'.")
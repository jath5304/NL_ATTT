import ember
from ember.features import PEFeatureExtractor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data_dir = r"C:\Users\PC\Study\NL-ATTT\ember2018"

print("--- BẮT ĐẦU PHÂN TÍCH VÀ LỌC PHƯƠNG SAI ---")

# 1. TẢI VÀ SÀNG LỌC DỮ LIỆU
X_train, y_train, _, _ = ember.read_vectorized_features(data_dir)
valid_mask = y_train != -1
X_filtered = X_train[valid_mask]

# Lấp đầy NaN bằng 0.0 để tránh lỗi toán học khi tính phương sai
X_filtered = np.nan_to_num(X_filtered, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

# 2. TẠO BẢN ĐỒ ĐẶC TRƯNG TỰ ĐỘNG (Dynamic Feature Map)
print("\n2. Đang đọc cấu trúc đặc trưng tự động từ EMBER...")
extractor = PEFeatureExtractor(feature_version=2)
feature_map = {}
current_pos = 0

for feature in extractor.features:
    group_name = feature.name  
    group_dim = feature.dim    
    feature_map[group_name] = (current_pos, current_pos + group_dim)
    current_pos += group_dim


# --- PHẦN 1: TÍNH TOÁN VÀ VẼ ĐỒ THỊ PHÂN TÍCH PHƯƠNG SAI ---
print("\n3. Đang tính toán phương sai toàn diện để vẽ đồ thị...")

var_stats = []
group_names = []

for group, (start, end) in feature_map.items():
    group_data = X_filtered[:, start:end]
    # Tính phương sai cho từng cột trong nhóm
    variances = np.var(group_data, axis=0)
    
    # Phân loại phương sai
    zero_var = np.sum(variances == 0)
    low_var = np.sum((variances > 0) & (variances < 0.01))
    medium_var = np.sum((variances >= 0.01) & (variances < 0.1))
    high_var = np.sum(variances >= 0.1)
    
    mean_var = np.mean(variances)
    low_var_percent = ((zero_var + low_var) / (end - start)) * 100
    
    var_stats.append({
        'Group': group,
        'Zero Var': zero_var,
        'Low Var (<0.01)': low_var,
        'Medium Var (<0.1)': medium_var,
        'High Var (>=0.1)': high_var,
        'Mean Variance': mean_var,
        '% Low Variance': low_var_percent
    })
    group_names.append(group)

df_var = pd.DataFrame(var_stats)

# Vẽ Dashboard
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Phân tích phương sai toàn diện (Variance Analysis)', fontsize=16, fontweight='bold')

# Biểu đồ 1: Stacked Bar - Phân loại mức độ phương sai (Top)
ax1 = plt.subplot(2, 1, 1)
# ✅ ĐÃ SỬA LỖI KEYERROR Ở ĐÂY (Thêm cặp ngoặc vuông)
df_var.set_index('Group')[['Zero Var', 'Low Var (<0.01)', 'Medium Var (<0.1)', 'High Var (>=0.1)']].plot(
    kind='bar', stacked=True, color=['#ff6b6b', '#ffe66d', '#a8e6cf', '#4bc0c0'], ax=ax1, edgecolor='black'
)
ax1.set_title('Phân loại features theo mức độ phương sai', fontweight='bold')
ax1.set_ylabel('Số lượng features')
ax1.set_xlabel('')
plt.xticks(rotation=45)

# Biểu đồ 2: Bar ngang - % Features variance thấp (Bottom Left)
ax2 = plt.subplot(2, 2, 3)
sns.barplot(x='% Low Variance', y='Group', data=df_var, palette='husl', edgecolor='black', ax=ax2)
ax2.axvline(20, color='orange', linestyle='--', label='20% threshold')
ax2.axvline(50, color='red', linestyle='--', label='50% threshold')
ax2.set_title('% Features variance thấp (<0.01)', fontweight='bold')
ax2.set_xlabel('Tỷ lệ features variance thấp (%)')
ax2.set_ylabel('')
for i, v in enumerate(df_var['% Low Variance']):
    ax2.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
ax2.legend()

# Biểu đồ 3: Bar ngang - Phương sai trung bình (Bottom Right)
ax3 = plt.subplot(2, 2, 4)
sns.barplot(x='Mean Variance', y='Group', data=df_var, color='#ffe66d', edgecolor='black', ax=ax3)
ax3.set_xscale('symlog') # Dùng log scale vì chênh lệch quá lớn
ax3.set_title('Mức độ phân tán dữ liệu (Mean Variance)', fontweight='bold')
ax3.set_xlabel('Phương sai trung bình (Log scale)')
ax3.set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('variance_analysis_dashboard.png', dpi=300)
print("   -> Đã lưu biểu đồ thành 'variance_analysis_dashboard.png'.")


# --- PHẦN 2: THỰC THI CHIẾN LƯỢC TỐI ƯU HÓA (TUNING) ---
print("\n4. Bắt đầu thực thi tối ưu hóa ngưỡng phương sai (Feature Tuning)...")

processed_blocks = {}

# A. Nhóm Histogram & Entropy: Lọc hằng số tuyệt đối (Var = 0)
for g in ['histogram', 'byteentropy']:
    start, end = feature_map[g]
    data_g = X_filtered[:, start:end]
    variances = np.var(data_g, axis=0) # Dùng np.var chống tràn RAM
    processed_blocks[g] = data_g[:, variances > 0.0]
    print(f"   [{g.upper()}] Giữ Var > 0: Từ {data_g.shape[1]} -> {processed_blocks[g].shape[1]} features.")

# B. Nhóm Imports: Lọc ngưỡng 0.005
start, end = feature_map['imports']
data_imports = X_filtered[:, start:end]
variances_imports = np.var(data_imports, axis=0)
processed_blocks['imports'] = data_imports[:, variances_imports >= 0.005]
print(f"   [IMPORTS] Lọc Var >= 0.005: Từ {data_imports.shape[1]} -> {processed_blocks['imports'].shape[1]} features.")

# C. Nhóm Exports: Giữ Top 10 phổ biến nhất
start, end = feature_map['exports']
data_exports = X_filtered[:, start:end]
non_zero_counts = np.count_nonzero(data_exports, axis=0)
top_10_indices = np.argsort(non_zero_counts)[-10:]
processed_blocks['exports'] = data_exports[:, top_10_indices]
print(f"   [EXPORTS] Giữ Top 10 phổ biến: Từ {data_exports.shape[1]} -> {processed_blocks['exports'].shape[1]} features.")

print("\nHoàn tất bước Tối ưu hóa đặc trưng! Đồ thị đã được lưu.")
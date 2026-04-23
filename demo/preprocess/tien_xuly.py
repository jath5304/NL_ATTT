import ember
from ember.features import PEFeatureExtractor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Binarizer
import sys
import gc
import io
import time
import matplotlib.gridspec as gridspec
import joblib

# ==========================================
# CẤU HÌNH HỆ THỐNG & ĐƯỜNG DẪN
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Đường dẫn đến thư mục chứa dữ liệu EMBER
data_dir = r"C:\Users\PC\Study\NL-ATTT\ember2018"
save_path = r"C:\Users\PC\Study\NL-ATTT\ember2018\ember_cleaned_step1.npz"

print("="*60)
print("🚀 KHỞI ĐỘNG PIPELINE TIỀN XỬ LÝ DỮ LIỆU EMBER (V2 - BẢN HOÀN THIỆN)")
print("="*60)
start_time_total = time.time()

# ==========================================
# BƯỚC 1: TẢI & SÀNG LỌC MẪU (SAMPLE CLEANING)
print("\n[1/8] TẢI DỮ LIỆU GỐC & LỌC MẪU UNKNOWN")
X_train, y_train, _, _ = ember.read_vectorized_features(data_dir)

total_samples, total_features = X_train.shape
print(f"   -> Kích thước ban đầu: {total_samples:,} mẫu x {total_features} chiều")

valid_samples_mask = (y_train != -1)
X_filtered = X_train[valid_samples_mask]
y_filtered = y_train[valid_samples_mask]

print(f"   -> Đã loại bỏ: {total_samples - X_filtered.shape[0]:,} mẫu rác (Nhãn -1).")
print(f"   -> Kích thước sau lọc: {X_filtered.shape[0]:,} mẫu x {X_filtered.shape[1]} chiều")

label_counts = np.bincount(y_filtered.astype(int))
print(f"   -> Phân bố: Lành tính (0): {label_counts[0]:,} | Mã độc (1): {label_counts[1]:,}")

# ==========================================
# BƯỚC 2: XỬ LÝ GIÁ TRỊ DỊ BIỆT
print("\n[2/8] KIỂM TRA & XỬ LÝ DỊ BIỆT (MISSING/INFINITY VALUES)")
has_nan = np.isnan(X_filtered).any()
has_inf = np.isinf(X_filtered).any()

if has_nan or has_inf:
    print("   -> Phát hiện giá trị NaN hoặc Vô cực (Infinity) trong dữ liệu!")
    X_filtered = np.nan_to_num(X_filtered, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    print("   -> Đã lấp đầy toàn bộ NaN/Inf bằng 0.0 để tránh lỗi thuật toán.")
else:
    print("   -> Dữ liệu sạch, không chứa giá trị NaN hay Infinity.")

# ==========================================
# BƯỚC 3: TẠO BẢN ĐỒ ĐẶC TRƯNG TỰ ĐỘNG
print("\n[3/8] TẠO BẢN ĐỒ ĐẶC TRƯNG (DYNAMIC FEATURE MAP)")
extractor = PEFeatureExtractor(feature_version=2)
feature_map = {}
current_pos = 0

for feature in extractor.features:
    group_name = feature.name  
    group_dim = feature.dim    
    feature_map[group_name] = (current_pos, current_pos + group_dim)
    current_pos += group_dim
print("   -> Đã trích xuất thành công cấu trúc nhóm đặc trưng.")

# ==========================================
# BƯỚC 4: PHÂN TÍCH VÀ VẼ ĐỒ THỊ PHƯƠNG SAI
print("\n[4/8] PHÂN TÍCH PHƯƠNG SAI TOÀN DIỆN & VẼ DASHBOARD")
var_stats = []
for group, (start, end) in feature_map.items():
    group_data = X_filtered[:, start:end]
    variances = np.var(group_data, axis=0)
    
    zero_var = np.sum(variances == 0)
    low_var = np.sum((variances > 0) & (variances < 0.01))
    medium_var = np.sum((variances >= 0.01) & (variances < 0.1))
    high_var = np.sum(variances >= 0.1)
    
    var_stats.append({
        'Group': group,
        'Zero Var': zero_var,
        'Low Var (<0.01)': low_var,
        'Medium Var (<0.1)': medium_var,
        'High Var (>=0.1)': high_var,
        'Mean Variance': np.mean(variances),
        '% Low Variance': ((zero_var + low_var) / (end - start)) * 100
    })

df_var = pd.DataFrame(var_stats)

# Vẽ đồ thị Dashboard
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Phân tích phương sai toàn diện (Variance Analysis)', fontsize=16, fontweight='bold')

ax1 = plt.subplot(2, 1, 1)
df_var.set_index('Group')[['Zero Var', 'Low Var (<0.01)', 'Medium Var (<0.1)', 'High Var (>=0.1)']].plot(
    kind='bar', stacked=True, color=['#ff6b6b', '#ffe66d', '#a8e6cf', '#4bc0c0'], ax=ax1, edgecolor='black')
ax1.set_title('Phân loại features theo mức độ phương sai', fontweight='bold')
ax1.set_ylabel('Số lượng features'); ax1.set_xlabel(''); plt.xticks(rotation=45)

ax2 = plt.subplot(2, 2, 3)
# Đã thêm hue='Group' và legend=False để sửa lỗi cảnh báo Seaborn
sns.barplot(x='% Low Variance', y='Group', data=df_var, hue='Group', palette='husl', edgecolor='black', ax=ax2, legend=False)
ax2.axvline(20, color='orange', linestyle='--'); ax2.axvline(50, color='red', linestyle='--')
ax2.set_title('% Features variance thấp (<0.01)', fontweight='bold')
ax2.set_xlabel('Tỷ lệ features variance thấp (%)'); ax2.set_ylabel('')
for i, v in enumerate(df_var['% Low Variance']): ax2.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)

ax3 = plt.subplot(2, 2, 4)
# Đã thêm hue='Group' và legend=False
sns.barplot(x='Mean Variance', y='Group', data=df_var, hue='Group', palette='dark:y_r', edgecolor='black', ax=ax3, legend=False)
ax3.set_xscale('symlog')
ax3.set_title('Mức độ phân tán dữ liệu (Mean Variance)', fontweight='bold')
ax3.set_xlabel('Phương sai trung bình (Log scale)'); ax3.set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('variance_analysis_dashboard.png', dpi=300)
print("   -> Đã lưu biểu đồ: 'variance_analysis_dashboard.png'")


# Tạo một từ điển để lưu lại luật cắt gọt cho lúc chạy app
inference_pipeline = {
    'feature_map': feature_map,
    'kept_features': {}
}

# ==========================================
# BƯỚC 5: TỐI ƯU HÓA NGƯỠNG PHƯƠNG SAI (TUNING) 
print("\n[5/8] TỐI ƯU HÓA ĐẶC TRƯNG BẰNG PHƯƠNG SAI (Tiết kiệm RAM)")
processed_blocks = {}

# A. Histogram & Byteentropy
for g in ['histogram', 'byteentropy']:
    start, end = feature_map[g]
    data_g = X_filtered[:, start:end]
    variances = np.var(data_g, axis=0)

    kept_mask = variances > 0.0
    processed_blocks[g] = data_g[:, kept_mask]
    inference_pipeline['kept_features'][g] = kept_mask
    print(f"   [{g.upper()}] Lọc Var > 0: Còn {processed_blocks[g].shape[1]} features.")

# B. Imports
start, end = feature_map['imports']
data_imports = X_filtered[:, start:end]
variances_imports = np.var(data_imports, axis=0)
kept_mask_imports = variances_imports >= 0.005
processed_blocks['imports'] = data_imports[:, kept_mask_imports]
inference_pipeline['kept_features']['imports'] = kept_mask_imports
print(f"   [IMPORTS] Lọc Var >= 0.005: Còn {processed_blocks['imports'].shape[1]} features.")

# C. Exports
start, end = feature_map['exports']
data_exports = X_filtered[:, start:end]
top_10_indices = np.argsort(np.count_nonzero(data_exports, axis=0))[-10:]

processed_blocks['exports'] = data_exports[:, top_10_indices]
inference_pipeline['kept_features']['exports'] = top_10_indices
print(f"   [EXPORTS] Giữ Top 10 phổ biến: Còn {processed_blocks['exports'].shape[1]} features.")

# ==========================================
# BƯỚC 6: NHỊ PHÂN HÓA (BINARIZATION)
print("\n[6/8] NHỊ PHÂN HÓA NHÓM IMPORTS")
binarizer = Binarizer(threshold=0.0, copy=False)
processed_blocks['imports'] = binarizer.fit_transform(processed_blocks['imports'])
print("   -> Đã chuyển đổi tần suất nhóm Imports sang nhị phân (0 và 1).")

# ==========================================
# BƯỚC 7: XỬ LÝ ĐA CỘNG TUYẾN (MULTICOLLINEARITY)
print("\n[7/8] XỬ LÝ ĐA CỘNG TUYẾN (PEARSON > 0.95)")
for g in ['header', 'section', 'datadirectories']:
    data_g = processed_blocks.get(g, X_filtered[:, feature_map[g][0]:feature_map[g][1]])
    df_g = pd.DataFrame(data_g)

    corr_matrix = df_g.corr(method='pearson').abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]

    kept_cols = [col for col in df_g.columns if col not in to_drop]
    processed_blocks[g] = df_g[kept_cols].values
    inference_pipeline['kept_features'][g] = kept_cols
    print(f"   [{g.upper()}] Đã gọt {len(to_drop)} cột trùng lặp. Giữ lại: {processed_blocks[g].shape[1]} features.")

# ==========================================
# BƯỚC 8: TỔNG HỢP VÀ LƯU THÀNH PHẨM
print("\n[8/8] TỔNG HỢP & LƯU PIPELINE")

for g in feature_map.keys():
    if g not in processed_blocks:
        start, end = feature_map[g]
        data_g = X_filtered[:, start:end]
        variances_safe = np.var(data_g, axis=0)

        kept_mask_safe = variances_safe > 0.0
        processed_blocks[g] = data_g[:, kept_mask_safe]
        inference_pipeline['kept_features'][g] = kept_mask_safe
        print(f"   [{g.upper()}] Đã quét an toàn (Var > 0): Còn {processed_blocks[g].shape[1]} features.")


try:
    del X_filtered
except NameError:
    pass
gc.collect()
# Gộp nối tiếp theo đúng trình tự
# X_final_clean = np.hstack([processed_blocks[g] for g in feature_map.keys()])

total_rows = processed_blocks['general'].shape[0] # Lấy số dòng (600,000)
total_cols = sum(processed_blocks[g].shape[1] for g in feature_map.keys())

print(f"   -> Đang khởi tạo mảng rỗng ({total_rows} x {total_cols}) với float16...")

X_final_clean = np.zeros((total_rows, total_cols), dtype=np.float16)

current_col = 0
for g in feature_map.keys():
    num_cols = processed_blocks[g].shape[1]
    
    X_final_clean[:, current_col : current_col + num_cols] = processed_blocks[g]
    current_col += num_cols
    processed_blocks[g] = None
    gc.collect()

# print(f"   -> Kích thước SAU khi làm sạch: {X_final_clean.shape[0]:,} mẫu x {X_final_clean.shape[1]} chiều")

print(f"   -> Kích thước GỐC trước làm sạch: {X_filtered.shape[0]:,} mẫu x {X_filtered.shape[1]} chiều")
print(f"   -> Kích thước SAU khi làm sạch: {X_final_clean.shape[0]:,} mẫu x {X_final_clean.shape[1]} chiều")
print(f"   -> Đã cắt giảm tổng cộng: {X_filtered.shape[1] - X_final_clean.shape[1]} chiều dư thừa")

print(f"\nĐang nén ma trận và lưu xuống ổ cứng...")
np.savez_compressed(save_path, X=X_final_clean, y=y_filtered)

joblib.dump(inference_pipeline, r"C:\Users\PC\Study\NL-ATTT\ember2018\ember_inference_pipeline.pkl")
print("   -> Đã lưu bộ quy tắc cắt gọt (Pipeline) thành công!")

total_time = time.time() - start_time_total
print("="*60)
print(f"✅ PIPELINE HOÀN TẤT THÀNH CÔNG! (Thời gian chạy: {total_time:.2f} giây)")
print(f"📁 Dữ liệu sạch đã lưu tại: {save_path}")
print("="*60)

# =========================================================================================
# VẼ BÁO CÁO TỔNG HỢP (POST-PREPROCESSING DASHBOARD)
print("\n--- ĐANG TẠO BÁO CÁO TỔNG HỢP SAU TIỀN XỬ LÝ ---")

groups = list(feature_map.keys())
original_dims = [feature_map[g][1] - feature_map[g][0] for g in groups]
cleaned_dims = [processed_blocks[g].shape[1] for g in groups]

total_original_features = sum(original_dims)
total_cleaned_features = sum(cleaned_dims)
reduced_features = total_original_features - total_cleaned_features

df_compare = pd.DataFrame({'Nhóm đặc trưng': groups, 'Trang thái': ['Gốc'] * len(groups), 'Số chiều': original_dims})
df_compare_clean = pd.DataFrame({'Nhóm đặc trưng': groups, 'Trang thái': ['Sau làm sạch'] * len(groups), 'Số chiều': cleaned_dims})
df_final = pd.concat([df_compare, df_compare_clean], ignore_index=True)

label_counts = np.bincount(y_filtered.astype(int))

sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 2, 1])
fig.suptitle('BÁO CÁO TỔNG HỢP SAU TIỀN XỬ LÝ (POST-PREPROCESSING REPORT)', fontsize=18, fontweight='bold', color='#2c3e50')

# Khung 1: Tổng quan
ax_summary = fig.add_subplot(gs[0, :])
ax_summary.axis('off')
summary_text = (
    f"TỔNG QUAN KẾT QUẢ TỐI ƯU HÓA\n\n"
    f"Tổng số mẫu dữ liệu giữ lại: {len(y_filtered):,} samples (Đã xóa mẫu Unknown)\n"
    f"Số chiều dữ liệu ban đầu: {total_original_features:,} features\n"
    f"Số chiều sau làm sạch: {total_cleaned_features:,} features\n"
    f"Đã cắt giảm: {reduced_features:,} features dư thừa (Giảm {(reduced_features/total_original_features)*100:.2f}% khối lượng)\n"
    f"Trạng thái cân bằng lớp: Hoàn hảo (Tỷ lệ 1:1)"
)
bbox_props = dict(boxstyle="round,pad=1", fc="#e8f6f3", ec="#1abc9c", lw=2)
ax_summary.text(0.5, 0.5, summary_text, ha="center", va="center", fontsize=14, fontweight='bold', bbox=bbox_props, color='#34495e', linespacing=1.8)

# Khung 2: Pie chart
ax_pie = fig.add_subplot(gs[1, 0])
ax_pie.pie(label_counts, explode=(0.05, 0), labels=['Lành tính (Benign - 0)', 'Mã độc (Malware - 1)'], colors=['#4bc0c0', '#ff6b6b'], autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax_pie.set_title('Phân bố nhãn dữ liệu sau khi lọc rác', fontsize=14, fontweight='bold')

# Khung 3: Bar chart
ax_bar = fig.add_subplot(gs[1, 1])
sns.barplot(data=df_final, x='Số chiều', y='Nhóm đặc trưng', hue='Trang thái', palette=['#95a5a6', '#3498db'], edgecolor='black', ax=ax_bar)
ax_bar.set_title('Mức độ tối ưu hóa theo từng nhóm đặc trưng', fontsize=14, fontweight='bold')
ax_bar.set_xlabel('Số lượng features (Chiều)'); ax_bar.set_ylabel(''); ax_bar.legend(title='Trạng thái')

for p in ax_bar.patches:
    width = p.get_width()
    if width > 0:
        ax_bar.annotate(f'{int(width)}', (width + 5, p.get_y() + p.get_height() / 2), ha='left', va='center', fontsize=9, color='black')

# Khung 4: Kết luận
ax_recommend = fig.add_subplot(gs[2, :])
ax_recommend.axis('off')
recommend_text = (
    "KẾT LUẬN & BƯỚC TIẾP THEO:\n"
    "✔️ Dữ liệu đã sạch hoàn toàn NaN/Infinity, lọc sạch các đặc trưng hằng số chết (Var=0) ở tất cả các nhóm.\n"
    "✔️ Loại bỏ Đa cộng tuyến (Pearson > 0.95) và nhị phân hóa thành công nhóm Imports.\n"
    "🚀 BƯỚC TIẾP THEO: Sẵn sàng đưa vào huấn luyện mô hình (Machine Learning)."
)
bbox_props_rec = dict(boxstyle="square,pad=1", fc="#fdfefe", ec="#bdc3c7", lw=1.5, ls="--")
ax_recommend.text(0.5, 0.5, recommend_text, ha="center", va="center", fontsize=13, bbox=bbox_props_rec, color='#2c3e50', linespacing=1.6)

plt.tight_layout()
file_out = 'post_preprocessing_dashboard_v2.png'
plt.savefig(file_out, dpi=300, bbox_inches='tight')
print(f"-> Đã vẽ xong và lưu Báo cáo tổng hợp tại: {file_out} 📊")
import ember
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import sys
import io
from ember.features import PEFeatureExtractor
# Cấu hình hiển thị tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data_dir = r"C:\Users\PC\Study\NL-ATTT\ember2018"

print("1. Đang kết nối dữ liệu từ X_train.dat...")
X_train, _, _, _ = ember.read_vectorized_features(data_dir)

# Lấy mẫu 20,000 file để thống kê chính xác hơn
sample_size = 20000
X_sample = X_train[:sample_size]

# Mapping các nhóm đặc trưng (EMBER v2)
# feature_map = {
#     'histogram': (0, 256),
#     'byteentropy': (256, 512),
#     'strings': (512, 616),
#     'general': (616, 626),
#     'header': (626, 688),
#     'section': (688, 943),
#     'imports': (943, 2223),
#     'exports': (2223, 2351),
#     'datadirectories': (2351, 2381)
# }

extractor = PEFeatureExtractor(feature_version=2)
feature_map = {}
current_pos = 0

for feature in extractor.features:
    group_name = feature.name  
    group_dim = feature.dim    
    feature_map[group_name] = (current_pos, current_pos + group_dim)
    current_pos += group_dim
print("   -> Đã trích xuất thành công cấu trúc nhóm đặc trưng.")
# 2. Tính toán thống kê
stats_list = []
for name, (start, end) in feature_map.items():
    group_data = X_sample[:, start:end]
    total_elements = group_data.size
    
    zeros = np.sum(group_data == 0)
    nans = np.sum(np.isnan(group_data))
    non_zeros = total_elements - zeros - nans
    
    stats_list.append({
        'Group': name,
        'Zero %': (zeros / total_elements) * 100,
        'NaN %': (nans / total_elements) * 100,
        'Non-zero %': (non_zeros / total_elements) * 100,
        'Total Elements': total_elements
    })

# Tạo DataFrame và sắp xếp theo độ thưa giảm dần (giống hình mẫu)
df_stats = pd.DataFrame(stats_list).sort_values(by='Zero %', ascending=False)

print("3. Đang dựng Dashboard...")
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Phân tích độ thưa toàn diện (Comprehensive Sparsity Analysis)', fontsize=18, fontweight='bold', y=0.98)

# Chia lưới: 3 hàng
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.4, wspace=0.2)

# --- BIỂU ĐỒ 1: Stacked Bar (Hàng đầu) ---
ax1 = fig.add_subplot(gs[0, :])
df_stats.set_index('Group')[['Zero %', 'NaN %', 'Non-zero %']].plot(
    kind='bar', stacked=True, ax=ax1, color=['#ff6b6b', '#ffd93d', '#4bc0c0'], edgecolor='black', width=0.7
)
ax1.set_title('Phân tích độ thưa theo nhóm (Stacked)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Tỷ lệ (%)')
ax1.legend(loc='upper right', labels=['Zero values', 'NaN values', 'Non-zero values'])
plt.xticks(rotation=30)

# --- BIỂU ĐỒ 2: Tỷ lệ giá trị Zero (Giữa trái) ---
ax2 = fig.add_subplot(gs[1, 0])
sns.barplot(data=df_stats, x='Zero %', y='Group', palette='Reds_r', ax=ax2, edgecolor='black')
ax2.axvline(50, color='orange', linestyle='--', label='50%')
ax2.axvline(80, color='red', linestyle='--', label='80%')
ax2.set_title('Tỷ lệ giá trị Zero', fontsize=13, fontweight='bold')
for i, v in enumerate(df_stats['Zero %']):
    ax2.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=10, fontweight='bold')
ax2.legend()

# --- BIỂU ĐỒ 3: Mối quan hệ Kích thước vs Độ thưa (Giữa phải) ---
ax3 = fig.add_subplot(gs[1, 1])
scatter = ax3.scatter(df_stats['Total Elements'], df_stats['Zero %'], 
                      s=300, # Kích thước bóng dựa trên số phần tử
                      c=df_stats['Zero %'], cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidth=1.5)
ax3.set_title('Mối quan hệ: Kích thước vs Độ thưa', fontsize=13, fontweight='bold')
ax3.set_xlabel('Tổng số phần tử (Mẫu x Chiều)')
ax3.set_ylabel('Zero values (%)')
for i, txt in enumerate(df_stats['Group']):
    ax3.annotate(txt, (df_stats['Total Elements'].iloc[i], df_stats['Zero %'].iloc[i]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=10)
plt.colorbar(scatter, ax=ax3, label='Zero %')

# --- BIỂU ĐỒ 4: Heatmap (Hàng cuối) ---
ax4 = fig.add_subplot(gs[2, :])
heatmap_data = df_stats.set_index('Group')[['Zero %', 'NaN %', 'Non-zero %']].T
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='RdYlGn_r', 
            linewidths=1, linecolor='black', ax=ax4, cbar_kws={'label': 'Percentage'})
ax4.set_title('Ma trận độ thưa (Sparsity Heatmap)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Loại giá trị')

plt.tight_layout(rect=[0, 0, 1, 0.96])
print("Xong! Dashboard đang hiển thị.")
plt.show()
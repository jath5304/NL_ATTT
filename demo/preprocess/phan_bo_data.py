import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import ember
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Đường dẫn thư mục chứa dữ liệu
data_dir = r"C:\Users\PC\Study\NL-ATTT\ember2018"

print("1. Đang đọc dữ liệu nhãn (y) từ các file .dat...")
_, y_train, _, y_test = ember.read_vectorized_features(data_dir)

print("2. Đang tổng hợp dữ liệu để vẽ biểu đồ...")
# Gom dữ liệu Train và Test lại
df_train = pd.DataFrame({'Label': y_train, 'Split': 'Train'})
df_test = pd.DataFrame({'Label': y_test, 'Split': 'Test'})
df = pd.concat([df_train, df_test], ignore_index=True)

# Ánh xạ nhãn số sang chữ
label_map = {-1: 'Unknown', 0: 'Benign', 1: 'Malware'}
df['Label_Name'] = df['Label'].map(label_map)

# Tính toán các con số thống kê
counts = df['Label_Name'].value_counts()
total_samples = len(df)

# Bảng màu giống trong ảnh của bạn
colors = {'Unknown': '#8fd9d6', 'Benign': '#48c9b0', 'Malware': '#ff6b6b'}

print("3. Bắt đầu vẽ Dashboard...")
# Thiết lập kích thước tổng thể
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Phân tích phân bố lớp dữ liệu (Class Distribution Analysis)', fontsize=16, fontweight='bold')

# Chia lưới (Grid) cho Dashboard: 3 hàng, 3 cột
gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1.2, 0.6], hspace=0.4)

# ==========================================
# 1. Biểu đồ tròn (Góc trái trên)
# ==========================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.pie(counts, labels=[f"{idx}\n{val:,}" for idx, val in counts.items()], 
        autopct='%1.1f%%', startangle=90, colors=[colors[key] for key in counts.index])
ax1.set_title('Phân bố tổng thể')

# ==========================================
# 2. Biểu đồ cột (Giữa trên)
# ==========================================
ax2 = fig.add_subplot(gs[0, 1])
sns.countplot(data=df, x='Label_Name', order=['Unknown', 'Benign', 'Malware'], 
              palette=colors, ax=ax2, edgecolor='black')
ax2.set_title('Số lượng theo nhãn')
ax2.set_ylabel('Số lượng mẫu')
ax2.set_xlabel('')
# Thêm số liệu lên đầu cột
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontweight='bold')

# ==========================================
# 3. Tỷ lệ mất cân bằng (Góc phải trên)
# ==========================================
ax3 = fig.add_subplot(gs[0, 2])
benign_count = counts.get('Benign', 0)
malware_count = counts.get('Malware', 0)
ratio = benign_count / malware_count if malware_count > 0 else 0

ax3.barh(['Imbalance\nRatio'], [ratio], color='#ff6b6b', edgecolor='black')
ax3.axvline(x=1.5, color='green', linestyle='--', linewidth=2, label='Ngưỡng tốt (1.5)')
ax3.set_xlim(0, 1.6)
ax3.set_title('Tỷ lệ mất cân bằng')
ax3.set_xlabel('Ratio')
ax3.text(ratio / 2, 0, f'{ratio:.2f}:1', ha='center', va='center', 
         color='white', fontweight='bold', fontsize=14)
ax3.legend()

# ==========================================
# 4. Phân bố theo Train/Test Split (Hàng giữa)
# ==========================================
ax4 = fig.add_subplot(gs[1, :])
sns.countplot(data=df, x='Split', hue='Label_Name', hue_order=['Unknown', 'Benign', 'Malware'],
              palette=colors, ax=ax4, edgecolor='black')
ax4.set_title('Phân bố nhãn theo Train/Test Split')
ax4.set_ylabel('Số lượng mẫu')
ax4.set_xlabel('Dataset Split')
for p in ax4.patches:
    if p.get_height() > 0:
        ax4.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9)

# ==========================================
# 5. Bảng thống kê chi tiết (Hàng dưới cùng)
# ==========================================
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off') # Ẩn trục tọa độ đi để vẽ bảng

# Chuẩn bị dữ liệu cho bảng
table_data = [
    ['Unknown', f"{counts.get('Unknown', 0):,}", f"{(counts.get('Unknown', 0)/total_samples)*100:.2f}%"],
    ['Benign', f"{counts.get('Benign', 0):,}", f"{(counts.get('Benign', 0)/total_samples)*100:.2f}%"],
    ['Malware', f"{counts.get('Malware', 0):,}", f"{(counts.get('Malware', 0)/total_samples)*100:.2f}%"],
    ['TOTAL', f"{total_samples:,}", '100.00%']
]

table = ax5.table(cellText=table_data, colLabels=['Nhãn', 'Số lượng', 'Tỷ lệ (%)'], 
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# Tô màu cho bảng giống ảnh
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#48c9b0')
        cell.set_text_props(color='white', fontweight='bold')
    elif row == 4: # Hàng TOTAL
        cell.set_facecolor('#ffeaa7')
        cell.set_text_props(fontweight='bold')

plt.tight_layout()
print("4. Hoàn tất! Đang hiển thị Dashboard...")
plt.show()
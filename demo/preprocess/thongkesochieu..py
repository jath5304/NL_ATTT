import ember
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import sys
import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data_dir = r"C:\Users\PC\Study\NL-ATTT\ember2018"

print("1. Đang truy xuất cấu trúc file X_train.dat...")
X_train, _, _, _ = ember.read_vectorized_features(data_dir)

# Lấy tổng số chiều thực tế từ file
total_actual_dims = X_train.shape[1]
print(f"Tổng số chiều phát hiện trong file: {total_actual_dims}")

# 2. Định nghĩa cấu trúc nhóm
feature_sizes = {
    'imports': 1280,
    'byteentropy': 256,
    'histogram': 256,
    'section': 255,
    'exports': 128,
    'strings': 104,
    'header': 62,
    'datadirectories': 30,
    'general': 10
}

# Chuyển thành DataFrame để xử lý vẽ đồ thị
df = pd.DataFrame({
    'Feature_Group': list(feature_sizes.keys()),
    'Dimensions': list(feature_sizes.values())
})

# Tính toán các tỷ lệ dựa trên số liệu thực tế từ file
df['Percentage'] = (df['Dimensions'] / total_actual_dims) * 100
df['Cumulative'] = df['Percentage'].cumsum()

print("3. Đang tạo Dashboard phân tích chiều dữ liệu...")

# --- KHỞI TẠO FIGURE ---
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f'Phân tích chiều dữ liệu (Dimensionality Analysis)\n', 
             fontsize=16, fontweight='bold', y=0.96)

gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.35)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(df)))

# --- 1. BIỂU ĐỒ CỘT (Hàng trên) ---
ax1 = fig.add_subplot(gs[0, :])
bars = ax1.bar(df['Feature_Group'], df['Dimensions'], color=colors, edgecolor='black', alpha=0.9)
ax1.set_ylabel('Số chiều', fontweight='bold')
ax1.set_title(f'Số chiều các nhóm đặc trưng (Tổng: {total_actual_dims:,} chiều)', fontsize=13, pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Gán nhãn số lượng và % lên đầu cột
for bar, pct in zip(bars, df['Percentage']):
    height = bar.get_height()
    ax1.annotate(f'{int(height)}\n({pct:.1f}%)',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontweight='bold', fontsize=9)

# --- 2. BIỂU ĐỒ TRÒN (Dưới trái) ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.pie(df['Dimensions'], labels=df['Feature_Group'], autopct='%1.1f%%', 
        startangle=140, colors=colors, explode=[0.05]*len(df),
        wedgeprops={'edgecolor': 'white', 'linewidth': 1})
ax2.set_title('Tỷ lệ % các nhóm', fontweight='bold')

# --- 3. BIỂU ĐỒ TÍCH LŨY (Dưới phải) ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(df['Feature_Group'], df['Cumulative'], marker='o', color='#ff4d4d', linewidth=2.5, markersize=8)
ax3.fill_between(df['Feature_Group'], df['Cumulative'], color='#ff4d4d', alpha=0.2)
ax3.axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% threshold')

ax3.set_ylabel('Tỷ lệ tích lũy (%)', fontweight='bold')
ax3.set_title('Phân bố tích lũy', fontweight='bold')
ax3.set_ylim(0, 105)
ax3.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
ax3.legend(loc='upper left')

# Gán nhãn % tích lũy
for i, txt in enumerate(df['Cumulative']):
    ax3.annotate(f'{txt:.1f}%', (i, df['Cumulative'][i]), 
                 xytext=(0, 8), textcoords="offset points", ha='center', fontweight='bold', color='#c0392b')

plt.tight_layout(rect=[0, 0, 1, 0.94])
print("4. Hoàn tất! Đang hiển thị biểu đồ...")
plt.show()
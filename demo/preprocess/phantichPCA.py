import ember
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import sys
import io

# Ép Terminal hiển thị tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data_dir = r"C:\Users\PC\Study\NL-ATTT\ember2018"

print("1. Đang tải và chuẩn bị dữ liệu (Lấy mẫu 5,000 file)...")
X_train, y_train, _, _ = ember.read_vectorized_features(data_dir)

# Lọc bỏ nhãn Unknown (-1) và lấy mẫu
mask = y_train != -1
X_filtered = X_train[mask][:5000]
y_filtered = y_train[mask][:5000]

print("2. Đang chuẩn hóa dữ liệu (Standardization)...")
# PCA nhạy cảm với thang đo, nên bắt buộc phải đưa về cùng phương sai
X_scaled = StandardScaler().fit_transform(X_filtered)

print("3. Đang thực hiện thuật toán PCA (Tính toán 50 thành phần chính)...")
pca = PCA(n_components=50)
pca_features = pca.fit_transform(X_scaled)

# Tính toán các chỉ số variance
exp_var = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(exp_var)

# Xác định số PC cần thiết cho các ngưỡng
n80 = np.argmax(cum_var >= 80) + 1
n90 = np.argmax(cum_var >= 90) + 1
n95 = np.argmax(cum_var >= 95) + 1

# --- BẮT ĐẦU VẼ DASHBOARD ---
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Phân tích PCA và khả năng giảm chiều (PCA & Dimensionality Reduction)', fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.3)

# 1. Phương sai giải thích (Hàng đầu)
ax1 = fig.add_subplot(gs[0, :])
ax1.bar(range(20), exp_var[:20], alpha=0.7, color='#4bc0c0', label='Individual', edgecolor='black')
ax1.plot(range(20), cum_var[:20], marker='o', color='#ff6b6b', label='Cumulative', linewidth=2)
ax1.set_title('Phương sai giải thích bởi PCA (Top 20 components)', fontweight='bold')
ax1.set_ylabel('Explained Variance (%)')
ax1.set_xlabel('Principal Component')
ax1.legend()
for i, v in enumerate(cum_var[:5]): # Hiện số cho 5 cột đầu
    ax1.text(i, v + 2, f"{v:.1f}%", ha='center', color='red', fontweight='bold')

# 2. PCA: PC1 vs PC2 (Hàng giữa - Trái)
ax2 = fig.add_subplot(gs[1, 0])
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=y_filtered, 
                palette={0: '#4bc0c0', 1: '#ff6b6b'}, alpha=0.5, ax=ax2)
ax2.set_title(f'PCA: PC1 vs PC2', fontweight='bold')
ax2.set_xlabel(f'PC1 ({exp_var[0]:.1f}%)')
ax2.set_ylabel(f'PC2 ({exp_var[1]:.1f}%)')

# 3. PCA: PC2 vs PC3 (Hàng giữa - Giữa)
ax3 = fig.add_subplot(gs[1, 1])
sns.scatterplot(x=pca_features[:, 1], y=pca_features[:, 2], hue=y_filtered, 
                palette={0: '#4bc0c0', 1: '#ff6b6b'}, alpha=0.5, ax=ax3)
ax3.set_title(f'PCA: PC2 vs PC3', fontweight='bold')
ax3.set_xlabel(f'PC2 ({exp_var[1]:.1f}%)')
ax3.set_ylabel(f'PC3 ({exp_var[2]:.1f}%)')

# 4. PCA 3D Visualization (Hàng giữa - Phải)
ax4 = fig.add_subplot(gs[1, 2], projection='3d')
ax4.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], 
            c=y_filtered, cmap=plt.cm.coolwarm, alpha=0.4)
ax4.set_title('PCA 3D Visualization', fontweight='bold')
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_zlabel('PC3')

# 5. Scree Plot (Dưới - Trái)
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(range(1, 21), exp_var[:20], 'go-', linewidth=2)
ax5.set_title('Scree Plot', fontweight='bold')
ax5.set_ylabel('Explained Variance (%)')
ax5.set_xlabel('Component Number')

# 6. Components đạt ngưỡng (Dưới - Giữa)
ax6 = fig.add_subplot(gs[2, 1])
thresholds = ['80%', '90%', '95%']
values = [n80, n90, n95]
bars = ax6.bar(thresholds, values, color=['#4bc0c0', '#ffd93d', '#ff6b6b'], edgecolor='black')
ax6.set_title('Components để đạt ngưỡng variance', fontweight='bold')
ax6.set_ylabel('Số components cần thiết')
for bar in bars:
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', 
             ha='center', va='bottom', fontweight='bold')

# 7. Bảng tổng kết (Dưới - Phải)
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
table_data = [
    ['Mục tiêu', 'Số PC', 'Giảm chiều'],
    ['80% variance', n80, f'{((2381-n80)/2381)*100:.1f}%'],
    ['90% variance', n90, f'{((2381-n90)/2381)*100:.1f}%'],
    ['95% variance', n95, f'{((2381-n95)/2381)*100:.1f}%'],
    ['Chiều gốc', 2381, '0%']
]
table = ax7.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
print("4. Xong! Đang hiển thị Dashboard PCA...")
plt.show()
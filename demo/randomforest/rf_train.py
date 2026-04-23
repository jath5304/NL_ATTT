import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import time
import sys
import io
import joblib

# Ép Terminal hiển thị tiếng Việt chuẩn UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 1. TẢI DỮ LIỆU ĐÃ CHIA (TRAIN & TEST)
# ==========================================
print("\n[1/4] ĐANG TẢI DỮ LIỆU...")
# Đảm bảo bạn đã có 2 file này từ bước chạy split_data.py trước đó
data_train = np.load('ember_train.npz')
X_train, y_train = data_train['X'], data_train['y']

data_test = np.load('ember_test.npz')
X_test, y_test = data_test['X'], data_test['y']

print(f"   -> Tập Train: {X_train.shape[0]:,} mẫu")
print(f"   -> Tập Test:  {X_test.shape[0]:,} mẫu")

# ==========================================
# 2. KHỞI TẠO & HUẤN LUYỆN MÔ HÌNH VỚI THAM SỐ TỐI ƯU
# ==========================================
print("\n[2/4] BẮT ĐẦU HUẤN LUYỆN RANDOM FOREST...")
# 🔧 Thay đổi các tham số này bằng những con số tốt nhất mà bạn đã tìm được
rf_model = RandomForestClassifier(
    n_estimators=500,        
    max_depth=30,         
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,         # Cố định random_state để kết quả luôn giống nhau nếu chạy lại
    n_jobs=-1
)

start_time = time.time()
rf_model.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"   -> Huấn luyện xong trong: {train_time:.2f} giây")

# ==========================================
# 3. ĐÁNH GIÁ TRÊN TẬP TEST
# ==========================================
print("\n[3/4] ĐANG ĐÁNH GIÁ MÔ HÌNH...")
test_start = time.time()
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
test_time = time.time() - test_start

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("-" * 40)
print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {prec*100:.2f}%")
print(f"Recall    : {rec*100:.2f}%")
print(f"F1-Score  : {f1*100:.2f}%")
print(f"ROC AUC   : {auc:.4f}")
print("-" * 40)

# ==========================================
# 4. VẼ DASHBOARD BÁO CÁO & LƯU MÔ HÌNH
# ==========================================
print("\n[4/4] ĐANG XUẤT BÁO CÁO VÀ LƯU MÔ HÌNH...")

# 1. Vẽ Dashboard
cm = confusion_matrix(y_test, y_pred)
importances = rf_model.feature_importances_
fpr, tpr, _ = roc_curve(y_test, y_prob)

sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2)
fig.suptitle(f'BÁO CÁO HIỆU NĂNG RANDOM FOREST (FINAL MODEL)', fontsize=18, fontweight='bold', color='#27ae60')

# - Biểu đồ 1: Confusion Matrix
ax_cm = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm, annot_kws={"size": 14},
            xticklabels=['Benign (0)', 'Malware (1)'], yticklabels=['Benign (0)', 'Malware (1)'])
ax_cm.set_title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
ax_cm.set_ylabel('Thực tế')
ax_cm.set_xlabel('Dự đoán')

# - Biểu đồ 2: ROC Curve
ax_roc = fig.add_subplot(gs[0, 1])
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Đường cong ROC', fontsize=14, fontweight='bold')
ax_roc.legend(loc="lower right")

# - Biểu đồ 3: Feature Importances
ax_fi = fig.add_subplot(gs[1, :])
top_n = 20
top_indices = np.argsort(importances)[::-1][:top_n]
feature_names = [f"Feature_{i}" for i in top_indices]

sns.barplot(x=importances[top_indices], y=feature_names, ax=ax_fi, palette='magma', hue=feature_names, legend=False)
ax_fi.set_title(f'Top {top_n} Đặc trưng quan trọng nhất', fontsize=14, fontweight='bold')
ax_fi.set_xlabel('Mức độ quan trọng (Gini Importance)')

plt.tight_layout()
dashboard_path = 'rf_final_dashboard.png'
plt.savefig(dashboard_path, dpi=300)
print(f"   -> Đã lưu biểu đồ: {dashboard_path}")

# 2. Lưu mô hình (Export Model)
model_path = 'final_rf_model.pkl'
joblib.dump(rf_model, model_path)
print(f"   -> Đã lưu mô hình: {model_path} (Sẵn sàng cho Streamlit)")

print("="*60)
print("HOÀN TẤT TOÀN BỘ QUY TRÌNH!")
print("="*60)
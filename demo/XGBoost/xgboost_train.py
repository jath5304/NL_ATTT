import numpy as np
import xgboost as xgb
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

print("="*70)
print(" FINAL TRAINING XGBOOST - TỐI ƯU HÓA DMATRIX VÀ HISTOGRAM")
print("="*70)

# ==========================================
# 1. TẢI DỮ LIỆU ĐÃ CHIA TỪ TRƯỚC
# ==========================================
print("\n[1/4] ĐANG TẢI DỮ LIỆU TRAIN & TEST...")
start_time = time.time()

# Tải dữ liệu đã chia (đảm bảo 2 file này nằm cùng thư mục)
data_train = np.load('ember_train.npz')
X_train, y_train = data_train['X'], data_train['y']

data_test = np.load('ember_test.npz')
X_test, y_test = data_test['X'], data_test['y']

print(f"   -> Tập Huấn luyện (Train): {X_train.shape[0]:,} mẫu")
print(f"   -> Tập Kiểm thử (Test):  {X_test.shape[0]:,} mẫu")

# ==========================================
# 2. CHUYỂN ĐỔI SANG DMATRIX & HUẤN LUYỆN
# ==========================================
print("\n[2/4] CHUYỂN ĐỔI DMATRIX VÀ BẮT ĐẦU HUẤN LUYỆN...")
dmatrix_start = time.time()

# DMatrix giúp tối ưu tốc độ và bộ nhớ
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

print(f"   -> Tạo DMatrix xong. Thời gian: {time.time() - dmatrix_start:.2f}s")

# 🔧 Thông số tối ưu bạn đã chọn
params = {
    'objective': 'binary:logistic',  
    'eval_metric': 'auc',            
    'tree_method': 'hist',           
    'learning_rate': 0.05,           
    'gamma': 0.1,                    
    'alpha': 0.1,
    'lambda': 1.0,
    'max_depth': 8,                  
    'seed': 42,
    'nthread': -1                    
}

print("   -> Đang huấn luyện (Early Stopping = 10 vòng)...")
train_start = time.time()

evals = [(dtrain, 'train'), (dtest, 'eval')]

# Huấn luyện mô hình
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,             
    evals=evals,
    early_stopping_rounds=10,         
    verbose_eval=50                   
)

print(f"   -> Huấn luyện xong! Thời gian: {time.time() - train_start:.2f}s")
print(f"   -> Mô hình chốt ở vòng thứ: {xgb_model.best_iteration}")

# ==========================================
# 3. ĐÁNH GIÁ HIỆU NĂNG TRÊN TẬP TEST
# ==========================================
print("\n[3/4] ĐANG ĐÁNH GIÁ MÔ HÌNH...")

# Dự đoán xác suất
y_prob = xgb_model.predict(dtest)
# Chuyển đổi xác suất thành nhãn (ngưỡng 0.5)
y_pred = (y_prob > 0.5).astype(int)

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
print("\n[4/4] ĐANG VẼ BÁO CÁO VÀ LƯU MÔ HÌNH...")

sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2)
fig.suptitle('BÁO CÁO HIỆU NĂNG XGBOOST (FINAL MODEL)', fontsize=18, fontweight='bold', color='#c0392b')

# ---- Biểu đồ 1: Ma trận nhầm lẫn ----
ax_cm = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax_cm, annot_kws={"size": 14},
            xticklabels=['Benign (0)', 'Malware (1)'], yticklabels=['Benign (0)', 'Malware (1)'])
ax_cm.set_title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
ax_cm.set_ylabel('Thực tế')
ax_cm.set_xlabel('Dự đoán')

# ---- Biểu đồ 2: Đường cong ROC ----
ax_roc = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax_roc.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Đường cong ROC', fontsize=14, fontweight='bold')
ax_roc.legend(loc="lower right")

# ---- Biểu đồ 3: Đặc trưng quan trọng (Đã mở khóa) ----
ax_fi = fig.add_subplot(gs[1, :])

# Lấy dict feature importances từ XGBoost
importance_dict = xgb_model.get_score(importance_type='gain')
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

top_n = 20
top_features = [x[0].replace('f', 'Feature_') for x in sorted_importance[:top_n]]
top_scores = [x[1] for x in sorted_importance[:top_n]]

# Thêm hue và legend=False để tương thích với Seaborn mới
sns.barplot(x=top_scores, y=top_features, ax=ax_fi, palette='rocket', hue=top_features, legend=False)
ax_fi.set_title(f'Top {top_n} Đặc trưng quan trọng nhất (XGBoost)', fontsize=14, fontweight='bold')
ax_fi.set_xlabel('Mức độ quan trọng (Gain)')

plt.tight_layout()
dashboard_path = 'xgb_final_dashboard.png'
plt.savefig(dashboard_path, dpi=300)
print(f"   -> Đã lưu biểu đồ: {dashboard_path}")

# LƯU MÔ HÌNH ĐỂ SỬ DỤNG TRÊN ỨNG DỤNG WEB
model_path = 'final_xgb_model.pkl'
joblib.dump(xgb_model, model_path)
print(f"   -> Đã lưu mô hình: {model_path} (Sẵn sàng cho Streamlit)")

print("="*70)
print(f"✅ HOÀN TẤT TOÀN BỘ! (Tổng thời gian: {time.time() - start_time:.2f} giây)")
print("="*70)
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import time
import sys
import io
import joblib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("🚀 FINAL TRAINING LIGHTGBM - TỐI ƯU HÓA VỚI LGB.DATASET")
print("="*70)

# ==========================================
# 1. TẢI DỮ LIỆU ĐÃ CHIA TỪ TRƯỚC
print("\n[1/4] ĐANG TẢI DỮ LIỆU TRAIN & TEST...")
start_time = time.time()

# Tải dữ liệu đã chia
data_train = np.load('ember_train.npz')
X_train, y_train = data_train['X'], data_train['y']

data_test = np.load('ember_test.npz')
X_test, y_test = data_test['X'], data_test['y']

print(f"   -> Tập Huấn luyện (Train): {X_train.shape[0]:,} mẫu")
print(f"   -> Tập Kiểm thử (Test):  {X_test.shape[0]:,} mẫu")

# ==========================================
# 2. CHUYỂN ĐỔI SANG DATASET & HUẤN LUYỆN
print("\n[2/4] CHUYỂN ĐỔI DATASET VÀ BẮT ĐẦU HUẤN LUYỆN...")
dataset_start = time.time()

# Cấu trúc lgb.Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

print(f"   -> Tạo Dataset xong. Thời gian: {time.time() - dataset_start:.2f}s")

# Thông số tối ưu
params = {
    'objective': 'binary',        
    'metric': 'binary_logloss',              
    'boosting_type': 'gbdt',      
    'learning_rate': 0.05,         
    'num_leaves': 64,             
    'max_depth': 8,
    'n_estimators': 500,              
    'feature_fraction': 0.8,      
    'n_jobs': -1,                 
    'seed': 42,                   
    'verbose': -1                 
}

print("   -> Đang huấn luyện")
train_start = time.time()

# Khởi tạo biến lưu trữ quá trình học để vẽ biểu đồ
evals_result = {}

# Khai báo Callbacks để theo dõi và dừng sớm
callbacks = [
    lgb.early_stopping(stopping_rounds=10, verbose=True),
    lgb.log_evaluation(period=10), # In kết quả mỗi 10 vòng để dễ theo dõi
    lgb.record_evaluation(evals_result) # LƯU TRỮ KẾT QUẢ TỪNG VÒNG
]

# Huấn luyện mô hình
lgb_model = lgb.train(
    params=params,
    train_set=train_data,
    num_boost_round=1000,                
    valid_sets=[train_data, test_data],  
    valid_names=['train', 'eval'],
    callbacks=callbacks
)

print(f"\n   -> Huấn luyện xong! Thời gian: {time.time() - train_start:.2f}s")
print(f"   -> Mô hình chốt ở vòng thứ: {lgb_model.best_iteration}")

# ==========================================
# 3. ĐÁNH GIÁ HIỆU NĂNG TRÊN TẬP TEST
print("\n[3/4] ĐANG ĐÁNH GIÁ MÔ HÌNH...")

# Dự đoán xác suất
y_prob = lgb_model.predict(X_test)
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
print("\n[4/4] ĐANG VẼ BÁO CÁO VÀ LƯU MÔ HÌNH...")

sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})
# Mở rộng kích thước ảnh để chứa 4 biểu đồ
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2)
fig.suptitle('BÁO CÁO HIỆU NĂNG LIGHTGBM (FINAL MODEL)', fontsize=18, fontweight='bold', color='#2980b9')

# ---- Biểu đồ 1: Ma trận nhầm lẫn ----
ax_cm = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, annot_kws={"size": 14},
            xticklabels=['Benign (0)', 'Malware (1)'], yticklabels=['Benign (0)', 'Malware (1)'])
ax_cm.set_title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
ax_cm.set_ylabel('Thực tế')
ax_cm.set_xlabel('Dự đoán')

# ---- Biểu đồ 2: Đường cong ROC ----
ax_roc = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax_roc.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Đường cong ROC', fontsize=14, fontweight='bold')
ax_roc.legend(loc="lower right")

# ---- Biểu đồ 3: Đặc trưng quan trọng ----
ax_fi = fig.add_subplot(gs[1, 0])
importances = lgb_model.feature_importance(importance_type='gain')
top_n = 20
top_indices = np.argsort(importances)[::-1][:top_n]
top_features = [f"Feature_{i}" for i in top_indices]
top_scores = importances[top_indices]

sns.barplot(x=top_scores, y=top_features, ax=ax_fi, palette='mako', hue=top_features, legend=False)
ax_fi.set_title(f'Top {top_n} Đặc trưng quan trọng', fontsize=14, fontweight='bold')
ax_fi.set_xlabel('Mức độ quan trọng (Gain)')

# ---- Biểu đồ 4: Quá trình huấn luyện (Learning Curve) ----
ax_lc = fig.add_subplot(gs[1, 1])
train_auc = evals_result['train']['auc']
eval_auc = evals_result['eval']['auc']
epochs = range(len(train_auc))

ax_lc.plot(epochs, train_auc, label='Train AUC', color='teal', lw=2)
ax_lc.plot(epochs, eval_auc, label='Test AUC', color='orange', lw=2)
# Đánh dấu vòng lặp tốt nhất
if lgb_model.best_iteration > 0:
    ax_lc.axvline(x=lgb_model.best_iteration - 1, color='red', linestyle='--', label=f'Best Iteration ({lgb_model.best_iteration})')

ax_lc.set_title('Quá trình Huấn luyện (Learning Curve)', fontsize=14, fontweight='bold')
ax_lc.set_xlabel('Vòng lặp (Boosting Round)')
ax_lc.set_ylabel('ROC AUC Score')
ax_lc.legend(loc='lower right')

plt.tight_layout()
dashboard_path = 'lgb_final_dashboard_with_learning_curve.png'
plt.savefig(dashboard_path, dpi=300)
print(f"   -> Đã lưu biểu đồ: {dashboard_path}")

# LƯU MÔ HÌNH
model_path = 'final_lgb_model.pkl'
joblib.dump(lgb_model, model_path)
print(f"   -> Đã lưu mô hình: {model_path} (Sẵn sàng cho Streamlit)")

print("="*70)
print(f"HOÀN TẤT TOÀN BỘ! (Tổng thời gian: {time.time() - start_time:.2f} giây)")
print("="*70)
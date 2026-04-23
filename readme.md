# 🛡️ AI Antivirus: Malware Detection using Machine Learning

Dự án nghiên cứu và triển khai hệ thống phát hiện mã độc (Malware) dựa trên phương pháp học máy (Machine Learning) sử dụng tập dữ liệu đặc trưng tĩnh EMBER 2018. Hệ thống có khả năng phân tích các tệp tin thực thi Windows (.exe, .dll) và đưa ra dự đoán về mức độ nguy hại với độ chính xác cực cao.

## 🚀 Tính năng chính

* **Phân tích tĩnh (Static Analysis):** Trích xuất đặc trưng từ tệp tin PE mà không cần thực thi tệp, đảm bảo an toàn tuyệt đối cho hệ thống quét.
* **Pipeline tiền xử lý thông minh:** Tự động lọc và chuẩn hóa từ 2.381 đặc trưng thô xuống còn 2.064 đặc trưng cốt lõi.
* **Đa mô hình tiên tiến:** Tích hợp các thuật toán mạnh mẽ nhất hiện nay: XGBoost, LightGBM và Random Forest.
* **Giao diện Web trực quan:** Xây dựng bằng Streamlit, hỗ trợ kéo thả tệp tin và trả kết quả thời gian thực.
* **Cơ chế Cache:** Tối ưu hóa tốc độ nạp mô hình, giúp quá trình quét diễn ra gần như tức thì.


## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python 3.8+
* **Học máy:** Scikit-learn, XGBoost, LightGBM, Numpy, Pandas.
* **Trích xuất đặc trưng:** Lief, Ember Feature Extractor.
* **Giao diện:** Streamlit.
* **Lưu trữ mô hình:** Joblib.

## 📁 Cấu trúc thư mục

```text
├── LightGBM/                     
    ├──final_lgb_model.pkl
    └── lightgbm_train.py
├── randomforest
    ├──rf_tuning.py
    └── rf_train.py
├── XGBoost
    ├── boosting_tuning.py
    └── xgboost_train.py
├── dataset ember/                       # Tài liệu liên quan đến tập dữ liệu EMBER
    ├──__init__.py
    └── reatures.py
├── preprocess/
│   ├── feature_extractor.py    # Lõi trích xuất đặc trưng từ file PE
    ├── dothua_data.py
    ├── extract_data.py
    ├── phan_bo_data.py
    ├── phantichPCA.py
    ├── split_data.py
    ├── thongkesochieu.py
    ├── tien_xuly.py
    ├── variance_analysis.py
│   └── ember_inference_pipeline.pkl 
├── app.py                      # File chạy ứng dụng giao diện Web 
└── readme.md                   # Hướng dẫn dự án

##⚙️ Cài đặt và Sử dụng
## Thu thập và Xử lý Dữ liệu thô
* **Nguồn dữ liệu:** [EMBER 2018 Dataset](https://github.com/elastic/ember) (1 triệu mẫu PE files).
* **Trích xuất:** Sử dụng thư viện `LIEF` để parse cấu trúc file PE.
* **Sàng lọc:** Loại bỏ các mẫu không có nhãn (Unlabeled, nhãn -1), chỉ giữ lại mẫu Lành tính (0) và Độc hại (1).

### 2 Kỹ thuật Đặc trưng (Feature Engineering)
Xử lý làm sạch và tối ưu hóa **2381 đặc trưng** đầu vào:
* **Lọc phương sai (Variance Threshold):** Loại bỏ các đặc trưng hằng số (Constant) và phương sai thấp (< 0.005) để giảm nhiễu.
* **Bảo toàn thông tin quan trọng:** Giữ nguyên toàn bộ nhóm đặc trưng **Byte Histogram** và **Byte Entropy** vì tính phân loại cao.
* **Xử lý tương quan:** Loại bỏ các đặc trưng có độ tương quan cao (> 0.95) trong nhóm Header/Section để tránh đa cộng tuyến.
* **Chuẩn hóa:** Áp dụng `StandardScaler` cho mô hình Mạng nơ-ron (MLP).

### 3 Huấn luyện Mô hình (Model Training)
Triển khai huấn luyện 03 thuật toán với các chiến lược tối ưu riêng biệt:

| Mô hình | Chiến lược tối ưu |
| :--- | :--- |
| **Random Forest** | Sử dụng chiến lược *Progressive Training* (Tăng dần số cây từ 100 -> 1000). |
| **XGBoost** | Cấu hình `tree_method='hist'` để tăng tốc trên dữ liệu lớn. |
| **LightGBM** | Áp dụng chiến lược *Leaf-wise growth*, tối ưu hóa tốc độ và bộ nhớ. |

### 4 Đánh giá và So sánh (Evaluation)
* Sử dụng tập kiểm thử độc lập (20% dữ liệu).
* Đánh giá dựa trên 4 chỉ số: **Accuracy, Precision, Recall, F1-Score**.
* Ưu tiên chỉ số **Recall** (Tỷ lệ phát hiện) để giảm thiểu bỏ sót mã độc.

### 5 Chạy ứng dụng
Sau khi cài đặt xong, khởi chạy giao diện Web bằng lệnh:


## 🚀 Hướng dẫn Cài đặt & Thực thi

Vui lòng tuân thủ đúng trình tự sau để đảm bảo luồng dữ liệu (Data Pipeline) hoạt động chính xác.

### Giai đoạn 1: Xử lý Dữ liệu


1. **Trích xuất đặc trưng**
```bash
    python preprocess/feature_extractor.py
```


2. **chuyển đổi dữ liệu sang vector**
```bash
    python preprocess/extract_data.py
```

3 **Trực quan hóa**:
xem các biểu đồ:
thống kê số chiều
```bash
    python preprocess/thongkesochieu.py
```

phân bố data
```bash
    python preprocess/phan_bo_data.py
```

độ thưa dữ liệu
```bash
    python preprocess/dothua_data.py
```

phân tích PCA
```bash
    python preprocess/phantichPCA.py
```

4 **Tiền xử lý**
```bash
    python preprocess/tien_xuly.py
```

### Huấn luyện mô hình
```bash
# Huấn luyện Random Forest
python randomforest/rf_train.py

# Huấn luyện LightGBM
python LightGBM/lightgbm_train.py


# Huấn luyện XGBoost
python XGBoost/xgboost_train.py

### Giai đoạn 3: Cấu hình & Chạy Ứng dụng

1.  **Cập nhật Model:** Mở file `App.py`, tìm dòng khai báo đường dẫn model và thay thế bằng đường dẫn tới file `.pkl` tốt nhất vừa huấn luyện (ví dụ: `LightGBM/best_lightgbm_model.pkl`).

2.  **Khởi chạy:**
```bash
    python -m streamlit run app.py
```
---
3. **kiểm tra**
kéo thả file .exe hoặc .dll vào ô

bấm quét file ngay và chờ kết quả
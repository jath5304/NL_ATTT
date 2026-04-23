import sys
import io
import numpy as np


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'object'):
    np.object = np.object_

import lief

class DummyLiefError(Exception): 
    pass

missing_errors = ['bad_format', 'bad_file', 'pe_error', 'parser_error', 'read_out_of_bound']
for err in missing_errors:
    if not hasattr(lief, err):
        setattr(lief, err, DummyLiefError)

if hasattr(lief, 'pe') and not hasattr(lief.pe, 'BadDOSHeader'):
    lief.pe.BadDOSHeader = DummyLiefError

# ==========================================
from sklearn.feature_extraction import FeatureHasher

original_transform = FeatureHasher.transform

def patched_transform(self, raw_X):
    # Chuyển dữ liệu đầu vào thành list
    X_list = list(raw_X)
    # Nếu có dữ liệu và phần tử đầu tiên là chuỗi (string) -> Đây là mảng 1 chiều
    if len(X_list) > 0 and isinstance(X_list[0], str):
        X_list = [X_list] # Bọc lại thành mảng 2 chiều để lừa Scikit-Learn
    return original_transform(self, X_list)

FeatureHasher.transform = patched_transform

import ember
import os

def extract_features(file_path):
    """
    Hàm đọc file PE (.exe, .dll) và trích xuất ra mảng 2381 đặc trưng chuẩn EMBER 2018
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    print(f"[*] Đang phân tích file: {os.path.basename(file_path)}...")
    
    # Khởi tạo bộ trích xuất của EMBER
    extractor = ember.PEFeatureExtractor(2)
    
    # Đọc file nhị phân
    with open(file_path, "rb") as f:
        file_data = f.read()
        
    # Trích xuất đặc trưng
    raw_features = extractor.feature_vector(file_data)
    
    # Định dạng lại thành mảng 2D cho mô hình AI: (1, 2381)
    features_array = np.array(raw_features, dtype=np.float32).reshape(1, -1)
    
    return features_array

# ==========================================
# CHẠY TEST
if __name__ == "__main__":
    test_file = r"C:\Windows\System32\calc.exe" 
    
    try:
        features = extract_features(test_file)
        print("TRÍCH XUẤT THÀNH CÔNG!")
        print(f"   -> Kích thước mảng: {features.shape}")
        print(f"   -> Một vài giá trị đầu tiên: {features[0][:5]}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
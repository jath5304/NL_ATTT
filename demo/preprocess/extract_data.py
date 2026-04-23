import sys
import io
import os
import ember
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import multiprocessing

orig_transform = FeatureHasher.transform

def patched_transform(self, raw_X):
    if self.input_type == "string":
        if isinstance(raw_X, str):
            raw_X = [[raw_X]]
        else:
            raw_list = list(raw_X)
            if len(raw_list) == 0:
                raw_X = [[]] 
            elif isinstance(raw_list[0], (str, np.str_, np.character)):
                raw_X = [raw_list] 
            else:
                raw_X = raw_list 
    return orig_transform(self, raw_X)

FeatureHasher.transform = patched_transform
# -------------------------------------------------------------

def main():
    # hiển thị tiếng Việt
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    data_dir = r"C:\Users\PC\Study\NL-ATTT\ember2018" 
    ember_version = 2

    print(f"Đang kiểm tra thư mục: {data_dir}")

    try:
        print("Bắt đầu đọc JSON và chuyển đổi sang Vector")
        ember.create_vectorized_features(data_dir, ember_version)
        print("\nQuá trình trích xuất đã hoàn tất.")
    except Exception as e:
        print(f"\nCó lỗi xảy ra: {e}")


if __name__ == '__main__':
    # Hỗ trợ đóng băng tiến trình trên Windows để tránh Crash
    multiprocessing.freeze_support() 
    main()
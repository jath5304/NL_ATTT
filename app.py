import streamlit as st
import joblib
import os
import tempfile
import numpy as np

from feature_extractor import extract_features

# ==========================================
# HÀM XỬ LÝ DỮ LIỆU (LỌC TỪ 2381 -> 2064)
def apply_preprocessing(raw_features_2381, pipeline_path="ember_inference_pipeline.pkl"):
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Không tìm thấy file {pipeline_path}. Bạn đã chạy lại file tiền xử lý chưa?")
        
    pipeline = joblib.load(pipeline_path)
    feature_map = pipeline['feature_map']
    kept_features = pipeline['kept_features']

    #Xử lý giá trị đặc biệt 
    cleaned_features = np.nan_to_num(raw_features_2381, nan=0.0, posinf=0.0, neginf=0.0)

    processed_blocks = []
    
    #Lặp qua từng nhóm để cắt
    for g in feature_map.keys():
        start, end = feature_map[g]
        group_data = cleaned_features[:, start:end]

        #Áp dụng cắt cột
        if g in kept_features:
            mask_or_indices = kept_features[g]
            group_data = group_data[:, mask_or_indices]

        #Áp dụng Binarizer cho nhóm imports 
        if g == 'imports':
            group_data = (group_data > 0.0).astype(float) 

        processed_blocks.append(group_data)

    #Gộp lại thành mảng 1D cuối cùng
    final_features_2064 = np.hstack(processed_blocks)
    return final_features_2064

# ==========================================
# GIAO DIỆN WEB
st.set_page_config(page_title="AI Antivirus", page_icon="🛡️")
st.title("🛡️ AI Antivirus Scanner")
st.write("Tải lên một file thực thi (.exe, .dll) để kiểm tra xem đó có phải là mã độc hay không.")

@st.cache_resource
def load_model():
    model_path = "final_xgb_model.pkl" # Đổi file mô hình
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy file mô hình {model_path}.")
        return None
    return joblib.load(model_path)

model = load_model()

if model:
    uploaded_file = st.file_uploader("Kéo thả file vào đây hoặc chọn file...", type=['exe', 'dll'])

    if uploaded_file is not None:
        if st.button("🚀 Quét File Ngay"):
            with st.spinner("Đang phân tích cấu trúc file..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    #Trích xuất file (Ra 2381 cột)
                    raw_features = extract_features(tmp_file_path)
                    
                    #Chạy qua phễu lọc (lọc xuống 2064 cột)
                    final_features = apply_preprocessing(raw_features, pipeline_path="ember_inference_pipeline.pkl")
                    
                    #Đưa vào mô hình
                    prediction = model.predict(final_features)
                    
                    st.markdown("---")
                    st.subheader("Kết quả phân tích:")
                    
                    # Tùy thuộc vào output của LightGBM (trả về 0/1 hay xác suất)
                    # Nếu model trả về xác suất, ta dùng ngưỡng 0.5
                    is_malware = prediction[0] >= 0.5 if isinstance(prediction[0], float) else prediction[0] == 1
                    
                    if is_malware:
                        st.error("CẢNH BÁO: PHÁT HIỆN MÃ ĐỘC (MALWARE)!")
                    else:
                        st.success("File an toàn (BENIGN).")
                        
                except Exception as e:
                    st.error(f"Có lỗi xảy ra trong quá trình quét: {e}")
                finally:
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
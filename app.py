import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

st.set_page_config(
    page_title="Mask Detection AI",
    page_icon="😷",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = 'mask_project/train_run1/weights/best.pt' 
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Không tìm thấy file model! Hãy kiểm tra lại đường dẫn: {e}")
    st.stop()

st.sidebar.title("⚙️ Cấu hình")
conf_threshold = st.sidebar.slider("Độ tin cậy (Confidence Threshold)", 0.0, 1.0, 0.5, 0.05)
source_type = st.sidebar.radio("Chọn nguồn ảnh:", ["Upload Ảnh", "Chụp từ Webcam"])

st.title("😷 Hệ thống Phát hiện Khẩu trang")
st.markdown("Tải ảnh lên hoặc dùng webcam để kiểm tra mô hình.")

if source_type == "Upload Ảnh":
    uploaded_file = st.file_uploader("Chọn một file ảnh...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Ảnh gốc", use_container_width=True)

        if st.button("🔍 Phân tích ngay", type="primary"):
            with st.spinner('Đang xử lý...'):
                results = model.predict(image, conf=conf_threshold)
                res = results[0]
                
                res_plotted = res.plot()
                
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

                boxes = res.boxes
                num_mask = 0
                num_no_mask = 0
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0: num_mask += 1
                    else: num_no_mask += 1

            with col2:
                st.image(res_rgb, caption="Kết quả phát hiện", use_container_width=True)
            
            st.success("✅ Hoàn thành phân tích!")
            st.metric(label="Đeo khẩu trang đúng", value=f"{num_mask} người")
            st.metric(label="Không đeo khẩu trang", value=f"{num_no_mask} người", delta_color="inverse")

elif source_type == "Chụp từ Webcam":
    picture = st.camera_input("Chụp một bức ảnh để kiểm tra")

    if picture:
        image = Image.open(picture)
        results = model.predict(image, conf=conf_threshold)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption="Kết quả Webcam")

st.divider()
st.caption("Project YOLOv8 - Object Detection")
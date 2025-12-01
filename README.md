# 😷 Face Mask Detection - YOLO11

Hệ thống phát hiện đeo khẩu trang sử dụng YOLO11 với giao diện Streamlit.

---

## 📦 Cài đặt thư viện

```bash
pip install ultralytics opencv-python scikit-learn tqdm streamlit pillow
```

---

## 📁 Cấu trúc thư mục

```
project/
├── MaskFaceDataset/              # Dataset gốc
│   ├── images/                   # Các file ảnh .png
│   └── annotations/              # Các file nhãn .xml (Pascal VOC)
├── preprocessing.py              # Script tiền xử lý dữ liệu
├── train.py                      # Script huấn luyện mô hình
├── app.py                        # Ứng dụng Streamlit demo
├── requirements.txt              # Danh sách thư viện
└── README.md                     # File này
```

---

## 🚀 Hướng dẫn sử dụng

### **Bước 1: Chuẩn bị dataset gốc**

Đảm bảo dataset của bạn có cấu trúc:

```
MaskFaceDataset/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── annotations/
    ├── image1.xml
    ├── image2.xml
    └── ...
```

**Lưu ý:**

- File XML phải ở định dạng Pascal VOC
- Tên file ảnh và XML phải khớp nhau (trừ phần đuôi)

---

### **Bước 2: Tiền xử lý dữ liệu**

Chạy script preprocessing để chuyển đổi dataset sang định dạng YOLO:

```bash
python preprocessing.py
```

---

### **Bước 3: Huấn luyện mô hình**

Chạy script training:

```bash
python train.py
```

**Cấu hình mặc định:**

- Model: YOLO11 Nano (`yolo11n.pt`)
- Epochs: 50
- Batch size: 16
- Image size: 640×640

**Kết quả sẽ được lưu tại:**

```
mask_project/
└── train_runX/               # X tăng dần (1, 2, 3...)
    ├── weights/
    │   ├── best.pt          # Model tốt nhất (theo mAP)
    │   └── last.pt          # Model cuối cùng
    ├── results.png          # Biểu đồ training curves
    ├── confusion_matrix.png # Ma trận nhầm lẫn
    ├── results.csv          # Chi tiết metrics theo epoch
    └── args.yaml            # Các tham số training
```

### **Bước 4: Chạy ứng dụng demo**

Sau khi training xong, chỉnh sửa đường dẫn model trong `app.py`:

```python
# Dòng 14 trong app.py
model_path = 'mask_project/train_run1/weights/best.pt'  # Đổi thành train_runX của bạn
```

Chạy ứng dụng:

```bash
streamlit run app.py
```

**Giao diện sẽ mở trên:** http://localhost:8501

**Tính năng:**

- 📤 Upload ảnh để phát hiện khẩu trang
- 📷 Chụp ảnh từ webcam (real-time)
- 🎚️ Điều chỉnh độ tin cậy (confidence threshold)
- 📊 Hiển thị kết quả với bounding box và nhãn

---

## 🎯 Mục đích sử dụng

- ✅ Bài tập lớn môn học AI/Computer Vision
- ✅ Demo phát hiện đeo khẩu trang real-time
- ✅ Nghiên cứu Object Detection với YOLO
- ✅ Ứng dụng thực tế tại nơi công cộng (mở rộng)

---


## 📚 Tài liệu tham khảo

- Ultralytics YOLO11 Docs: https://docs.ultralytics.com/models/yolo11/

---
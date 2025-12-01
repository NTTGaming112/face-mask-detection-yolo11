import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data='yolo_dataset/data.yaml', 
    epochs=50, 
    imgsz=640,
    batch=16,
    project='mask_project',
    name='train_run'
)

metrics = model.val(split='test')
print(f"mAP50-95: {metrics.box.map}")
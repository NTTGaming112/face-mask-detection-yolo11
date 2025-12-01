import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import cv2

CLASSES = ['with_mask', 'without_mask']
IMG_PATH = 'MaskFaceDataset/images'
XML_PATH = 'MaskFaceDataset/annotations'
OUTPUT_DIR = 'yolo_dataset'

def convert_xml_to_yolo(xml_file, width, height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_data = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASSES:
            continue
        
        class_id = CLASSES.index(name)
        bbox = obj.find('bndbox')
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        yolo_data.append(f"{class_id} {x_center} {y_center} {w} {h}")
    
    return yolo_data

def create_dirs():
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

def process_files(file_list, split_type):
    print(f"Processing {split_type} data...")
    for filename in tqdm(file_list):
        src_img = os.path.join(IMG_PATH, filename)
        dst_img = os.path.join(OUTPUT_DIR, split_type, 'images', filename)
        
        img = cv2.imread(src_img)
        if img is None: continue
        h, w, _ = img.shape
        
        shutil.copy(src_img, dst_img)

        xml_filename = os.path.splitext(filename)[0] + '.xml'
        src_xml = os.path.join(XML_PATH, xml_filename)
        
        if os.path.exists(src_xml):
            yolo_lines = convert_xml_to_yolo(src_xml, w, h)
            
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            dst_txt = os.path.join(OUTPUT_DIR, split_type, 'labels', txt_filename)
            
            with open(dst_txt, 'w') as f:
                f.write('\n'.join(yolo_lines))

if __name__ == "__main__":
    images = [f for f in os.listdir(IMG_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    create_dirs()
    process_files(train_imgs, 'train')
    process_files(val_imgs, 'val')
    process_files(test_imgs, 'test')
    
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: val/images
test: test/images

nc: {len(CLASSES)}
names: {CLASSES}
    """
    with open(f"{OUTPUT_DIR}/data.yaml", 'w', encoding='utf-8') as f:
        f.write(yaml_content)
        
    print(f"\nXong! Du lieu da san sang tai folder: {OUTPUT_DIR}")
    print(f"File cau hinh tai: {OUTPUT_DIR}/data.yaml")
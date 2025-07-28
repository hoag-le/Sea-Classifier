import cv2
import os
import numpy as np

def preprocess_image(img_path, out_path, size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Lỗi file: {img_path}")
        return
    img = cv2.resize(img, size)

    # Cân bằng sáng (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Làm sắc nét
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    # Giảm nhiễu nhẹ
    img = cv2.GaussianBlur(img, (3,3), 0)

    cv2.imwrite(out_path, img)

def batch_preprocess_folder(src_folder, dst_folder, size=(224,224)):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for root, dirs, files in os.walk(src_folder):
        rel_path = os.path.relpath(root, src_folder)
        out_dir = os.path.join(dst_folder, rel_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                in_path = os.path.join(root, f)
                out_path = os.path.join(out_dir, f)
                preprocess_image(in_path, out_path, size)

# Xử lý toàn bộ ảnh trong data/train và lưu ra data_pp/train
batch_preprocess_folder('data/train', 'data_pp/train', size=(224,224))
batch_preprocess_folder('data/test', 'data_pp/test', size=(224,224))
print("Xử lý xong toàn bộ ảnh!")

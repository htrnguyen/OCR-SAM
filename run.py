#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
# from PIL import Image
# from pillow_heif import register_heif_opener
# from PIL import ImageFile

# # Cho phép xử lý tệp bị lỗi hoặc không đầy đủ
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # Đăng ký HEIF/HEIC opener với Pillow
# register_heif_opener()

# def convert_heic_to_png(heic_file, png_file):
#     """Chuyển đổi tệp HEIC sang PNG"""
#     try:
#         image = Image.open(heic_file)  # Mở tệp HEIC
#         image.save(png_file, "PNG")   # Lưu dưới dạng PNG
#         print(f"Converted {heic_file} to {png_file}")
#     except Exception as e:
#         print(f"Failed to convert {heic_file}: {e}")

# # Thư mục chứa tệp HEIC
# file_path = "./DATA"

# # Duyệt qua tất cả các tệp trong thư mục
# for file in os.listdir(file_path):
#     if file.lower().endswith(".heic"):  # Kiểm tra đuôi tệp
#         heic_file = os.path.join(file_path, file)
#         png_file = os.path.splitext(heic_file)[0] + ".png"

#         # Kiểm tra nếu file PNG đã tồn tại
#         if os.path.exists(png_file):
#             print(f"Skipping {file}, {png_file} already exists.")
#             continue

#         # Chuyển đổi nếu file PNG chưa tồn tại
#         convert_heic_to_png(heic_file, png_file)
#     else:
#         print(f"Skipping {file}, not a HEIC file.")

# print("Done")


# In[ ]:


# !jupyter nbconvert --to script run.ipynb


# In[2]:


import os
import shutil
import warnings
import torch
import cv2
from mmocr.apis.inferencers import MMOCRInferencer

# Ẩn cảnh báo không cần thiết
warnings.filterwarnings("ignore")


# In[3]:


# Cấu hình thư mục và mô hình
INPUT_DIR = "./imgs"  # Thư mục chứa ảnh gốc
OUTPUT_DIR = "./OUTPUT"  # Thư mục lưu kết quả
os.makedirs(OUTPUT_DIR, exist_ok=True)


# In[4]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# In[5]:


# Cấu hình MMOCR Inferencer
mmocr_inferencer = MMOCRInferencer(
    det="mmocr_dev/configs/textdet/dbnetpp/dbnetpp_swinv2_base_w16_in21k.py",
    det_weights="checkpoints/mmocr/db_swin_mix_pretrain.pth",
    rec="mmocr_dev/configs/textrecog/abinet/abinet_20e_st-an_mj.py",
    rec_weights="checkpoints/mmocr/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth",
    device=DEVICE,  
)


# In[6]:


# Hàm scale ảnh
def scale_image(image, scale_factor):
    # Tính kích thước mới
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Thực hiện scale ảnh
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return scaled_image


# In[7]:


# Hàm xử lý ảnh OCR
def process_image(image_path, output_dir, mmocr_inferencer, scale_factor):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return

    # Scale ảnh
    scaled_image = scale_image(image, scale_factor)

    # Thực hiện OCR
    results = mmocr_inferencer(scaled_image, save_vis=True, out_dir=output_dir)

    # Lấy kết quả văn bản
    texts = []
    for prediction in results["predictions"]:
        for text in prediction["rec_texts"]:
            texts.append(text)  

    # Ghi kết quả văn bản vào file txt
    if texts:
        output_text_file = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", ".txt").replace(".png", ".txt"))
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        print(f"Text saved to: {output_text_file}")

# Xóa dữ liệu cũ trong thư mục OUTPUT
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lặp qua tất cả các ảnh trong thư mục
SCALE_FACTOR = 1.5  # Scale ảnh lên 1.5 lần
image_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith((".jpg", ".png"))]

# for image_file in image_files:
#     # Xử lý 5 ảnh đầu tiên
#     if len(os.listdir(OUTPUT_DIR)) >= 5:
#         break
#     process_image(image_file, OUTPUT_DIR, mmocr_inferencer, SCALE_FACTOR)


# In[8]:


# Chỉ xử lý 1 ảnh 
process_image("./data/DataFood/IMG_3496.png", OUTPUT_DIR, mmocr_inferencer, SCALE_FACTOR)


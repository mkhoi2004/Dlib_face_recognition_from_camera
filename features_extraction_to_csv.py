# 从人脸图像文件中提取人脸特征存入 "features_all.csv" / Extract features from images and save into "features_all.csv"

import os
import dlib
import csv
import numpy as np
import logging
import cv2
from PIL import Image
from database.connection import get_collection

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Truy xuất dữ liệu từ MongoDB
def get_data_from_db():
    collection = get_collection()
    data = []
    for document in collection.find():
        data.append((document["name"], document["age"], document["location"], document["image_path"]))
    return data

# 返回单张图像的 128D 特征 / Return 128D features for single image
# Input:    path_img           <class 'str'>
# Output:   face_descriptor    <class 'dlib.vector'>
def return_128d_features(path_img):
    img_pil = Image.open(path_img)
    img_np = np.array(img_pil)
    img_rd = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    faces = detector(img_rd, 1)

    logging.info("%-40s %-20s", "检测到人脸的图像 / Image with faces detected:", path_img)

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了, 所以要确保是 检测到人脸的人脸图像拿去算特征
    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("no face")
    return face_descriptor

def main():
    logging.basicConfig(level=logging.INFO)
    # Truy xuất dữ liệu từ MongoDB
    data = get_data_from_db()

    with open("data/features_all.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for name, age, location, img_path in data:
            # Get the 128D features of the face
            logging.info("Processing image for person: %s", name)
            features_128d = return_128d_features(img_path)
            if features_128d != 0:
                # Chuyển đổi đặc trưng 128D thành danh sách để có thể thêm các thông tin khác
                features_128d = list(features_128d)
                # Thêm tên, tuổi, vị trí vào đầu danh sách đặc trưng
                features_128d = [name, age, location] + features_128d
                # Ghi đặc trưng vào tệp CSV
                writer.writerow(features_128d)
                logging.info('\n')
        logging.info("所有录入人脸数据存入 / Save all the features of faces registered into: data/features_all.csv")

if __name__ == '__main__':
    main()
import dlib
import cv2
import time
import numpy as np
import csv
import os
from database.connection1 import get_collection
import base64
from datetime import datetime

# 1. Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# 2. Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# 3. Dlib Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# 读取 features_all.csv
def read_features_from_csv():
    features = []
    with open('data/features_all.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            name = row[0]
            age = row[1]
            location = row[2]
            feature = np.array(row[3:], dtype=float)
            features.append((name, age, location, feature))
    return features

# 计算欧氏距离
def euclidean_distance(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

class Face_Descriptor:
    def __init__(self):
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_cnt = 0
        self.known_faces = read_features_from_csv()
        self.current_face_desc = None
        self.current_face_name = "Unknown"
        self.save_success = False

        # Kết nối tới MongoDB
        self.collection = get_collection()

        # Tạo thư mục current_photo nếu chưa tồn tại
        if not os.path.exists('current_photo'):
            os.makedirs('current_photo')

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 480)
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()

    def process(self, stream):
        while stream.isOpened():
            flag, img_rd = stream.read()
            self.frame_cnt += 1
            k = cv2.waitKey(1)

            print('- Frame ', self.frame_cnt, " starts:")

            timestamp1 = time.time()
            faces = detector(img_rd, 0)
            timestamp2 = time.time()
            print("--- Time used to `detector`:                  %s seconds ---" % (timestamp2 - timestamp1))

            font = cv2.FONT_HERSHEY_SIMPLEX

            # 检测到人脸
            if len(faces) != 0:
                for face in faces:
                    timestamp3 = time.time()
                    face_shape = predictor(img_rd, face)
                    timestamp4 = time.time()
                    print("--- Time used to `predictor`:                 %s seconds ---" % (timestamp4 - timestamp3))

                    timestamp5 = time.time()
                    self.current_face_desc = face_reco_model.compute_face_descriptor(img_rd, face_shape)
                    timestamp6 = time.time()
                    print("--- Time used to `compute_face_descriptor：`   %s seconds ---" % (timestamp6 - timestamp5))

                    # 比较特征向量
                    min_distance = float('inf')
                    self.current_face_name = "Unknown"
                    for known_name, known_age, known_location, known_feature in self.known_faces:
                        distance = euclidean_distance(self.current_face_desc, known_feature)
                        if distance < min_distance:
                            min_distance = distance
                            self.current_face_name = known_name
                            self.current_face_age = known_age
                            self.current_face_location = known_location

                    # 显示结果
                    if min_distance < 0.6:  # 阈值可以根据实际情况调整
                        cv2.putText(img_rd, f"Name: {self.current_face_name}", (face.left(), face.top() - 10), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(img_rd, "Unknown", (face.left(), face.top() - 10), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            # 添加说明
            cv2.putText(img_rd, "Face descriptor", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 140), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "S: Save current face", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            # 显示 "Nhận diện thành công" nếu lưu thành công
            if self.save_success:
                cv2.putText(img_rd, "Nhận diện thành công", (20, 450), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("camera", img_rd)
                cv2.waitKey(2000)  # Hiển thị thông báo trong 2 giây
                break

            # 按下 's' 键保存当前人脸特征
            if k == ord('s') and self.current_face_desc is not None:
                self.save_feature_to_mongodb(img_rd)
                print("Face feature saved to MongoDB.")

            self.update_fps()

            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_rd)
            print('\n')

    def save_feature_to_mongodb(self, img_rd):
        # Chụp ảnh hiện tại
        ret, buffer = cv2.imencode('.jpg', img_rd)
        img_str = base64.b64encode(buffer).decode()

        # Lưu ảnh vào thư mục current_photo
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f'current_photo/{self.current_face_name}_{current_time}.jpg'
        cv2.imwrite(img_path, img_rd)

        # Chuyển đổi đối tượng _dlib_pybind11.vector thành danh sách Python
        face_desc_list = [f for f in self.current_face_desc]

        # Định dạng thời gian hiện tại
        current_time_str = datetime.now().strftime("%H:%M ngày %d/%m/%Y")

        # Lưu dữ liệu vào MongoDB
        document = {
            "name": self.current_face_name,
            "age": self.current_face_age,
            "location": self.current_face_location,
            "feature": face_desc_list,
            "image": img_str,
            "timestamp": current_time_str
        }
        self.collection.insert_one(document)
        self.save_success = True


def main():
    Face_Descriptor_con = Face_Descriptor()
    Face_Descriptor_con.run()


if __name__ == '__main__':
    main()
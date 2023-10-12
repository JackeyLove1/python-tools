import os

from imutils import face_utils
import numpy as np
import dlib
import cv2
from typing import List
import time

def eye_aspect_ratio(eye):
    # (|e1-e5|+|e2-e4|) / (2|e0-e3|)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 嘴长宽比例
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[1] - mouth[7])  # 61, 67
    B = np.linalg.norm(mouth[3] - mouth[5])  # 63, 65
    C = np.linalg.norm(mouth[0] - mouth[4])  # 60, 64
    mar = (A + B) / (2.0 * C)
    return mar

def time_cost(func):
    def wrapper(*arg, **kwargs):
        start = time.time()
        func(*arg, **kwargs)
        print(f"Cost:{time.time() - start}")
    return wrapper

#  进行活体检测（包含眨眼和张嘴）
@time_cost
def liveness_detection(images_path : List[str]):
    # images_path 为一组连续图片的地址
    # 眼长宽比例值
    EAR_THRESH = 0.2 # 可以适当放宽
    EAR_CONSEC_FRAMES_MIN = 1
    EAR_CONSEC_FRAMES_MAX = 3 # 当EAR小于阈值时，接连多少帧一定发生眨眼动作

    # 嘴长宽比例值
    MAR_THRESH = 0.2

    # 初始化眨眼的连续帧数
    blink_counter = 0
    # 初始化眨眼次数总数
    blink_total = 0
    # 初始化张嘴次数
    mouth_total = 0
    # 初始化张嘴状态为闭嘴
    mouth_status_open = 0

    print("[INFO] loading facial landmark predictor...")
    # 人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 特征点检测器
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    # 获取左眼的特征点
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # 获取右眼的特征点
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # 获取嘴巴特征点
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

    print("[INFO] starting video stream thread...")
    counts = 0
    for image_path in images_path:
        frame = cv2.imread(image_path)
        if frame is not None:
            counts += 1
            # 图片转换成灰色
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rects = detector(gray, 0)  # 人脸检测
            if len(rects) == 1:
                shape = predictor(gray, rects[0])  # 保存68个特征点坐标的<class 'dlib.dlib.full_object_detection'>对象
                shape = face_utils.shape_to_np(shape)  # 将shape转换为numpy数组，数组中每个元素为特征点坐标

                left_eye = shape[lStart:lEnd]  # 取出左眼对应的特征点
                right_eye = shape[rStart:rEnd]  # 取出右眼对应的特征点
                left_ear = eye_aspect_ratio(left_eye)  # 计算左眼EAR
                right_ear = eye_aspect_ratio(right_eye)  # 计算右眼EAR
                ear = (left_ear + right_ear) / 2.0   # 求左右眼EAR的均值

                inner_mouth = shape[mStart:mEnd]  # 取出嘴巴对应的特征点
                mar = mouth_aspect_ratio(inner_mouth)  # 求嘴巴mar的均值

                left_eye_hull = cv2.convexHull(left_eye)  # 寻找左眼轮廓
                right_eye_hull = cv2.convexHull(right_eye)  # 寻找右眼轮廓
                mouth_hull = cv2.convexHull(inner_mouth)  # 寻找嘴巴轮廓

                # EAR低于阈值，有可能发生眨眼，眨眼连续帧数加一次
                if ear < EAR_THRESH:
                    blink_counter += 1

                # EAR高于阈值，判断前面连续闭眼帧数，如果在合理范围内，说明发生眨眼
                else:
                    if EAR_CONSEC_FRAMES_MIN <= blink_counter <= EAR_CONSEC_FRAMES_MAX:
                        blink_total += 1
                    blink_counter = 0
                # 通过张、闭来判断一次张嘴动作
                if mar > MAR_THRESH:
                     mouth_status_open = 1
                else:
                    if mouth_status_open:
                        mouth_total += 1
                    mouth_status_open = 0
            # cv2.imshow("Frame", frame)
    print("blink_counter:", blink_counter)
    print("blink_total:", blink_total)
    print("mouth_total:", mouth_total)
    print("mouth_status_open:", mouth_total)
    print("count:", counts)

images_dir = os.path.join(os.getcwd(), "output/test2")
step = 10
# images_path = [os.path.join(images_dir, sub_path) for sub_path in os.listdir(images_dir)][::10]
# images_path.sort()
idx_ranges = list(range(75))[::7]
print("idxs:", idx_ranges)
# 获取一组连续的图片地址
images_path = [os.path.join(images_dir,f"frame_{num}.jpg") for num in idx_ranges]
print("images_path:", images_path)

# 核心函数 liveness_detection
liveness_detection(images_path)

# 计算内存消耗
import psutil
def get_readable_size(size_in_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    if size_in_bytes == 0:
        return "0B"
    i = 0
    while size_in_bytes >= 1024 and i < len(units) - 1:
        size_in_bytes /= 1024
        i += 1
    return f"{size_in_bytes:.2f} {units[i]}"

pid = psutil.Process()
memory_info = pid.memory_info()

memory_usage = get_readable_size(memory_info.rss)
print(f"Memory usage: {memory_usage}")
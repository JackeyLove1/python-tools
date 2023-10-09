# -*- codeing=utf-8 -*-
# @Time :2022/8/3 11:38
# @Author :wczh
# @File :眨眼.张口检测.py
# @Software:PyCharm

import face_recognition
from imutils import face_utils
import numpy as np
import dlib
import cv2
import sys

# 初始化眨眼次数
blink_total = 0
# 初始化张嘴次数
mouth_total = 0
# 设置图片存储路径
pic_path = r'images\viode_face.jpg'
# 图片数量
pic_total = 0
# 初始化眨眼的连续帧数以及总的眨眼次数
blink_counter = 0
# 初始化张嘴状态为闭嘴
mouth_status_open = 0

def getFaceEncoding(src):
    image = face_recognition.load_image_file(src)  # 加载人脸图片
    # 获取图片人脸定位[(top,right,bottom,left )]
    face_locations = face_recognition.face_locations(image)
    img_ = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    # display(img_)
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]  # 对人脸图片进行编码
    return face_encoding


def simcos(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b)  # 二范数
    sim = 1.0 / (1.0 + dist)  #
    return sim

# 提供对外比对的接口 返回比对的相似度
def comparison(face_src1, face_src2):
    xl1 = getFaceEncoding(face_src1)
    xl2 = getFaceEncoding(face_src2)
    value = simcos(xl1, xl2)
    print(value)

# 眼长宽比例
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

#  进行活体检测（包含眨眼和张嘴）
#  filePath 视频路径
def liveness_detection():
    global blink_total  # 使用global声明blink_total，在函数中就可以修改全局变量的值
    global mouth_total
    global pic_total
    global blink_counter
    global mouth_status_open
    # 眼长宽比例值
    EAR_THRESH = 0.15
    EAR_CONSEC_FRAMES_MIN = 1
    EAR_CONSEC_FRAMES_MAX = 5  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作
    # 嘴长宽比例值
    MAR_THRESH = 0.2

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
    vs = cv2.VideoCapture(video_path)
    # 总帧数(frames)
    frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_total = int(frames)
    for i in range(frames_total):
        ok, frame = vs.read(i)  # 读取视频流的一帧
        if not ok:
            break
        if frame is not None and i % 2 == 0:
            # 图片转换成灰色（去除色彩干扰，让图片识别更准确）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)  # 人脸检测
            # 只能处理一张人脸
            if len(rects) == 1:
                if pic_total == 0:
                    cv2.imwrite(pic_path, frame)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                    cv2.waitKey(1)
                    pic_total += 1

                shape = predictor(gray, rects[0])  # 保存68个特征点坐标的<class 'dlib.dlib.full_object_detection'>对象
                shape = face_utils.shape_to_np(shape)  # 将shape转换为numpy数组，数组中每个元素为特征点坐标

                left_eye = shape[lStart:lEnd]  # 取出左眼对应的特征点
                right_eye = shape[rStart:rEnd]  # 取出右眼对应的特征点
                left_ear = eye_aspect_ratio(left_eye)  # 计算左眼EAR
                right_ear = eye_aspect_ratio(right_eye)  # 计算右眼EAR
                ear = (left_ear + right_ear) / 2.0   # 求左右眼EAR的均值

                mouth = shape[mStart:mEnd]   # 取出嘴巴对应的特征点
                mar = mouth_aspect_ratio(mouth)  # 求嘴巴mar的均值

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
            elif len(rects) == 0 and i == 90:
                print("No face!")
                break
            elif len(rects) > 1:
                print("More than one face!")
        # 判断眨眼次数大于2、张嘴次数大于1则为活体,退出循环
        if blink_total >= 1 or mouth_total >= 1:
            break
    cv2.destroyAllWindows()
    vs.release()


# video_path, src = sys.argv[1], sys.argv[2]

video_path = r'video\face13.mp4'      # 输入的video文件夹位置
# src = r'C:\Users\666\Desktop\zz5.jpg'
liveness_detection()
print("眨眼次数》》", blink_total)
print("张嘴次数》》", mouth_total)
# comparison(pic_path, src)








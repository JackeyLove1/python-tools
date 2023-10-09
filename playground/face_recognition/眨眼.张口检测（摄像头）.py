# -*- codeing=utf-8 -*-
# @Time :2022/8/3 11:38
# @Author :wczh
# @File :眨眼.张口检测.py
# @Software:PyCharm

from imutils import face_utils
import numpy as np
import dlib
import cv2

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
def liveness_detection():
    vs = cv2.VideoCapture(0)  # 调用第一个摄像头的信息

    # 眼长宽比例值
    EAR_THRESH = 0.15
    EAR_CONSEC_FRAMES_MIN = 1
    EAR_CONSEC_FRAMES_MAX = 5  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作

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
    while True:

        flag, frame = vs.read()  # 返回一帧的数据
        if not flag:
            print("不支持摄像头", flag)
            break

        if frame is not None:
            # 图片转换成灰色（去除色彩干扰，让图片识别更准确）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rects = detector(gray, 0)  # 人脸检测
            # 只能处理一张人脸
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
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)   # 绘制左眼轮廓
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)  # 绘制右眼轮廓
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)  # 绘制嘴巴轮廓

                # EAR低于阈值，有可能发生眨眼，眨眼连续帧数加一次
                if ear < EAR_THRESH:
                    blink_counter += 1

                # EAR高于阈值，判断前面连续闭眼帧数，如果在合理范围内，说明发生眨眼
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
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

                cv2.putText(frame, "Blinks: {}".format(blink_total), (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Mouth: {}".format(mouth_total),
                            (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "MAR: {:.2f}".format(mar), (450, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(rects) == 0:
                cv2.putText(frame, "No face!", (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "More than one face!", (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame", frame)
            # 按下q键退出循环（鼠标要点击一下图片使图片获得焦点）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    vs.release()


liveness_detection()
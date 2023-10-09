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


#  进行活体检测（包含眨眼和张嘴）
def liveness_detection():
    vs = cv2.VideoCapture(0)  # 调用第一个摄像头的信息

    # 眼长宽比例值
    EAR_THRESH = 0.15
    EAR_CONSEC_FRAMES_MIN = 1
    EAR_CONSEC_FRAMES_MAX = 3  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作

    # 初始化眨眼的连续帧数
    blink_counter = 0
    # 初始化眨眼次数总数
    blink_total = 0

    print("[INFO] loading facial landmark predictor...")
    # 人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 特征点检测器
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    # 获取左眼的特征点
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # 获取右眼的特征点
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("[INFO] starting video stream thread...")
    while True:
        flag, frame = vs.read()  # 返回一帧的数据
        if not flag:
            print("不支持摄像头", flag)
            break

        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转成灰度图像
            rects = detector(gray, 0)  # 人脸检测
            # 只能处理一张人脸
            if len(rects) == 1:
                shape = predictor(gray, rects[0])  # 保存68个特征点坐标的<class 'dlib.dlib.full_object_detection'>对象
                shape = face_utils.shape_to_np(shape)  # 将shape转换为numpy数组，数组中每个元素为特征点坐标

                left_eye = shape[lStart:lEnd]  # 取出左眼对应的特征点
                right_eye = shape[rStart:rEnd]  # 取出右眼对应的特征点
                left_ear = eye_aspect_ratio(left_eye)  # 计算左眼EAR
                right_ear = eye_aspect_ratio(right_eye)  # 计算右眼EAR
                ear = (left_ear + right_ear) / 2.0  # 求左右眼EAR的均值

                left_eye_hull = cv2.convexHull(left_eye)  # 寻找左眼轮廓
                right_eye_hull = cv2.convexHull(right_eye)  # 寻找右眼轮廓
                # mouth_hull = cv2.convexHull(mouth)  # 寻找嘴巴轮廓
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)  # 绘制左眼轮廓
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)  # 绘制右眼轮廓

                # EAR低于阈值，有可能发生眨眼，眨眼连续帧数加一次
                if ear < EAR_THRESH:
                    blink_counter += 1

                # EAR高于阈值，判断前面连续闭眼帧数，如果在合理范围内，说明发生眨眼
                else:
                    if EAR_CONSEC_FRAMES_MIN <= blink_counter <= EAR_CONSEC_FRAMES_MAX:
                        blink_total += 1
                    blink_counter = 0

                cv2.putText(frame, "Blinks: {}".format(blink_total), (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
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
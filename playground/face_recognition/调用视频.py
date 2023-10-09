# -*-coding:GBK -*-
import cv2
from PIL import Image, ImageDraw
import numpy as np

# 1.调用摄像头
# 2.读取摄像头图像信息
# 3.在图像上添加文字信息
# 4.保存图像

cap = cv2.VideoCapture(r'video\face13.mp4')  # 调用第一个摄像头信息

while True:
    flag, frame = cap.read()  # 返回一帧的数据
    if not flag:
        break
    if frame is not None:
        # BGR是cv2 的图像保存格式，RGB是PIL的图像保存格式，在转换时需要做格式上的转换
        img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        draw.text((100, 100), 'press q to exit', fill=(255, 255, 255))
        # # 将frame对象转换成cv2的格式
        frame = cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)
        cv2.imshow('capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('images/out.jpg', frame)
            break

cv2.destroyAllWindows()
cap.release()


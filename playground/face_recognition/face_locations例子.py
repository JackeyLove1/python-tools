# -*- codeing=utf-8 -*-
# @Time :2022/8/17 15:08
# @Author :wczh
# @File :face_locations例子.py
# @Software:PyCharm
# 通过face_locations方法找人脸

import face_recognition
import cv2

# load_image_file 将图像文件（.jpg，.png等）加载到（多维数组）numpy数组中，记录图片的所有像数的特征向量
image = face_recognition.load_image_file('images/img_2.png')
# print(image)
# 通过face_locations方法，得到图像中所有人脸的位置
# 返回值是一个列表形式，列表中每一行是一张人脸的位置信息，包括[top,right,bottom,left],也可以认为每个人脸就是一组元组信息，主要用于标识图像中所有人脸的位置信息
face_locations = face_recognition.face_locations(image)
print(face_locations)
for face_location in face_locations:
    top, right, bottom, left = face_location  # 解包操作，得到图片种每张人脸的四个位置信息
    star = (left, top)
    end = (right, bottom)
    # 在图片上绘制人脸的矩形框，以star开始 end结束，矩形框的颜色(0,0,255) 边框粗细为2
    cv2.rectangle(image, star, end, (0, 0, 255), thickness=2)
cv2.imshow('window', image)
cv2.waitKey()
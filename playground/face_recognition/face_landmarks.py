# -*- codeing=utf-8 -*-
# @Time :2022/7/27 16:30
# @Author :wczh
# @File :face_landmarks.py
# @Software:PyCharm

# face_landmarks识别人脸关键特征点
# 参数仍然是待检测的图像对象，返回值包含面部特征点字典的列表，列表长度就是图像中的人脸数
# 面部特征包含以下几个部分 chin（下巴）,
#         left_eyebrow（左眼眉）,
#         'right_eyebrow'（右眼眉）,
#         'left_eye'（左眼）,
#         'right_eye'（右眼）,
#         nose_bridge（鼻梁）,
#         nose_tip（下眼部）,
#         bottom_lip（下嘴唇）,
#         top_lip（上嘴唇）
# 勾勒脸部大体的轮廓

import face_recognition
from PIL import Image, ImageDraw

# load_image_file 主要用于加载要识别的人脸图像，加载返回的数据是（多维数组）Numpy数组，记录图片的所有像数的特征向量
image = face_recognition.load_image_file('images/img.png')

face_landmarks_list = face_recognition.face_landmarks(image)

# print(face_landmarks_list)
pil_image = Image.fromarray(image)

d = ImageDraw.Draw(pil_image)  # 生成一张pil图像

for face_landmarks in face_landmarks_list:
    facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'left_eye',
        'right_eye',
        'nose_bridge',
        'nose_tip',
        'bottom_lip',
        'top_lip'
    ]
    for facial_feature in facial_features:
        # print("{}每个人的面部特征显示在以下位置：{}".format(facial_feature,face_landmarks[facial_feature]))
        # 调用pil的line方法，绘制所有特征点
        d.line(face_landmarks[facial_feature], width=2)
pil_image.show()


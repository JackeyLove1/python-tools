# -*- codeing=utf-8 -*-
# @Time :2022/7/27 17:50
# @Author :wczh
# @File :compare_faces例子.py
# @Software:PyCharm
# • compare_faces由面部编码信息进行面部识别匹配
# 。主要用于匹配两个面部特征编码,利用这两个特征向量的内积来衡量相似度,根据阔值确认,是否是同一个人。
# 。第一个参数就是一个面部编码列表(很多张脸),第二个参数就是给出单个面部编码(一张脸),compare_faces会将第二个参数中的编码信息与第一个参数中的所有编码信息依次匹配,返回值是一个布尔列表,匹配成功则返回True,匹配失败则返回False,顺序与第一个参数中脸部编码顺序一致。
# 。参数里有一个tolerance=0.6,大家可以根据实际的效果进行调整,一般经验值是0.39,tolerance值越小,匹配越严格。

import face_recognition

# 加载一张合照
image1 = face_recognition.load_image_file('images/yml.png')

# 加载一张当人照
image2 = face_recognition.load_image_file('images/yami.png')

known_face_encodings = face_recognition.face_encodings(image1)

# 要进行识别的单张图片的特征   只需要拿到第一个人脸的编码
compare_face_encoding = face_recognition.face_encodings(image2)[0]
# 注意第二个参数，只能是单个面部特征编码，不能列表
matches = face_recognition.compare_faces(known_face_encodings, compare_face_encoding, tolerance=0.39)

print(matches)

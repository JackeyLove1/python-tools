# -*- codeing=utf-8 -*-
# @Time :2022/7/27 15:15
# @Author :wczh
# @File :face_encodings例子.py
# @Software:PyCharm
# face_encodings 获取图像文件中所有面部编码信息
#     1：返回值是一个编码列表，参数仍然是要识别的图像对象，如果后续访问时需要加上索引或遍历进行访问，每张人脸的编码信息时一个128维向量
#     2:面部编码信息时进行人像识别的重要参数


import face_recognition

# load_image_file 主要用于加载要识别的人脸图像，加载返回的数据是（多维数组）Numpy数组，记录图片的所有像数的特征向量
image = face_recognition.load_image_file('images/yml.png')

face_encodings = face_recognition.face_encodings(image)
for face_encoding in face_encodings:
    print("信息编码长度为:{}\n编码信息为：{}".format(len(face_encoding), face_encoding))







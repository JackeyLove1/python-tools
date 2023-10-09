# -*- codeing=utf-8 -*-
# @Time :2022/7/28 14:50
# @Author :wczh
# @File :显示未知图片中已知人物的脸.py
# @Software:PyCharm
import face_recognition
import cv2


def compareFaces(known_image, name):
    known_face_encoding = face_recognition.face_encodings(known_image)[0]
    for i in range(len(face_locations)):  # face_Locations的长度就代表有多少张脸
        top1, right1, bottom1, left1 = face_locations[i]
        face_image = unknown_image[top1:bottom1, left1:right1]
        face_encoding = face_recognition.face_encodings(face_image)
        if face_encoding:
            result = {}
            matches = face_recognition.compare_faces([unknown_face_encodings[i]], known_face_encoding, tolerance=0.39)
            if True in matches:
                print('在未知图片中找到了已知面孔')
                result['face_encoding'] = face_encoding
                result['is_view'] = True
                result['location'] = face_locations[i]
                result['face_id'] = i + 1
                result['face_name'] = name
                results.append(result)
                if result['is_view']:
                    print('已知面孔匹配照片上的第{}张脸!!'.format(result['face_id']))


unknown_image = face_recognition.load_image_file('images/yml.png')
known_image1 = face_recognition.load_image_file('images/yami.png')
known_image2 = face_recognition.load_image_file('images/lkw.png')
results = []
unknown_face_encodings = face_recognition.face_encodings(unknown_image)
face_locations = face_recognition.face_locations(unknown_image)
compareFaces(known_image1, 'yami')
compareFaces(known_image2, 'lkw')

view_faces = [i for i in results if i['is_view']]

if len(view_faces) > 0:
    for view_face in view_faces:
        top, right, bottom, left = view_face['location']
        start = (left, top)
        end = (right, bottom)

        cv2.rectangle(unknown_image, start, end, (0, 0, 255), thickness=2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(unknown_image, view_face['face_name'], (left+6, bottom+16), font, 1.0, (255, 255, 255), thickness=1)

cv2.imshow('windows', unknown_image)
cv2.waitKey()








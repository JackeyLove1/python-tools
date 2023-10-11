# -*-coding:GBK -*-
import cv2
from PIL import Image, ImageDraw
import numpy as np

# 1.��������ͷ
# 2.��ȡ����ͷͼ����Ϣ
# 3.��ͼ�������������Ϣ
# 4.����ͼ��

cap = cv2.VideoCapture(r'video\face13.mp4')  # ���õ�һ������ͷ��Ϣ

while True:
    flag, frame = cap.read()  # ����һ֡������
    if not flag:
        break
    if frame is not None:
        # BGR��cv2 ��ͼ�񱣴��ʽ��RGB��PIL��ͼ�񱣴��ʽ����ת��ʱ��Ҫ����ʽ�ϵ�ת��
        img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        draw.text((100, 100), 'press q to exit', fill=(255, 255, 255))
        # # ��frame����ת����cv2�ĸ�ʽ
        frame = cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)
        cv2.imshow('capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('images/out.jpg', frame)
            break

cv2.destroyAllWindows()
cap.release()


import cv2
import os
video_path = "video/test2.mp4"
video = cv2.VideoCapture(video_path)
file_name = video_path.split("/")[-1].split(".")[0]
save_path = os.path.join(os.getcwd(), "output", file_name)
os.system(f"mkdir -p {save_path}")
frame_count = 0
frame_count_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print("视频总帧数:", frame_count_total)
while True:
    ret, frame = video.read()
    if not ret:
        print("failed to read video")
        break
    image_name = f'frame_{frame_count}.jpg'
    save_image_name = os.path.join(save_path, image_name)
    cv2.imwrite(save_image_name, frame)
    print(f"Succeed to save to {save_image_name}")
    frame_count += 1

# 释放视频对象
video.release()
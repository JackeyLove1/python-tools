import cv2
video_path = "video/test1.mp4"
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(fps * 3)  # 3秒的帧数
output = cv2.VideoWriter('video/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(video.get(3)), int(video.get(4))))
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    output.write(frame)

    frame_count += 1
    if frame_count == total_frames:
        print("totals frames:", total_frames)
        break

video.release()
output.release()
cv2.destroyAllWindows()
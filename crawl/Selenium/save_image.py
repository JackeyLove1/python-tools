image_link = "https://images.wsj.net/im-818277?width=700&size=1.5005861664712778&pixel_ratio=2"
import requests
def save_image_from_url(url, file_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print("图像保存成功！")
    else:
        print("无法获取图像。")
save_image_from_url(image_link, "test.jpg")
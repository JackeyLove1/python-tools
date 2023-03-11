# pip3 install tesserocr pillow
import requests
def get_picture(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        print("Successful request!")
    else:
        print("Request Failed!")
        exit()
    with open(path, 'wb') as f:
        f.write(response.content)
path = "img.jpg"
img_url = "https://raw.githubusercontent.com/Python3WebSpider/TestTess/master/image.png"
get_picture(img_url, path)

# image to text get more informations https://github.com/tesseract-ocr/tessdata
from tesserocr import image_to_text
from PIL import Image
pic = Image.open(path)
print("load picture success")
print(image_to_text(pic))
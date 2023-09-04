import urllib.request
import requests
url = "https://www.56wen.com/e/DownSys/doaction.php?enews=DownSoft&classid=3&id=10952&pathid=0&pass=80aada55632991c1c81aaca2cf895ae9&p=:::"
resp = requests.get(url)
text_content = resp.content.decode('gbk')
print(text_content)
with open("test.txt", 'w', encoding='utf-8') as txt_file:
    txt_file.write(text_content)
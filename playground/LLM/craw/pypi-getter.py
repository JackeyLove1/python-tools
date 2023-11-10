import requests
from bs4 import BeautifulSoup
url = "https://mirrors.bfsu.edu.cn/pypi/web/"
resp = requests.get(url)
soup = BeautifulSoup(resp.text, 'lxml')
print(soup.prettify())
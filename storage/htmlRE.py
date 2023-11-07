import requests
from bs4 import BeautifulSoup
url = "http://example.com"
resp = requests.get(url)
soup = BeautifulSoup(resp.content, "html.parser")
links = soup.find_all("a")
for link in links:
    href = link.get("href")
    print(href)
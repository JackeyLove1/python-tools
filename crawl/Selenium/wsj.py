import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time, json
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
driver = webdriver.Chrome()
# url = "https://www.wsj.com/"
# driver.get(url)
driver.get("https://sso.accounts.dowjones.com/login-page?")
email = ""
password=""
driver.find_element(By.XPATH, "//div/input[@name = 'username']").send_keys(email)
time.sleep(2)
driver.find_element(By.XPATH, "//button[@type='button' and @class='solid-button continue-submit new-design']").click()
time.sleep(2)
driver.find_element(By.XPATH, "//input[@type='password' and @class='password' and @name='password' and @id='password-login-password']").send_keys(password)
time.sleep(2)
driver.find_element(By.XPATH, "//button[@type='submit' and contains(@class, 'solid-button') and contains(@class, 'new-design') and contains(span/@class, 'text') and span/@data-token='signIn']").click()
time.sleep(2)
driver.refresh()
time.sleep(2)
driver_cookies = driver.get_cookies()
print(driver_cookies)
cookies = {}
for item in driver_cookies:
    cookies[item['name']] = item['value']
with open("wsjcookies.txt", "w") as f:
    f.write(json.dumps(cookies))

from bs4 import BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')
from lxml import etree
page = etree.HTML(driver.page_source)
href = page.xpath("//article[contains(@class, 'WSJTheme--story--') and contains(@class, 'WSJTheme--story-padding--') and contains(@class, 'WSJTheme--border-bottom--')]/div/h3/a/@href")
title = [element.text for element in page.xpath("//article[contains(@class, 'WSJTheme--story--') and contains(@class, 'WSJTheme--story-padding--') and contains(@class, 'WSJTheme--border-bottom--')]/div/h3/a/span")]
summary = [element.text for element in page.xpath("//p[contains(@class, 'WSJTheme--summary--') and contains(@class, 'typography--')]/span")]

articlelink = "https://www.wsj.com/articles/workers-to-employers-were-just-not-that-into-you-71dbeb6e?mod=hp_lead_pos1#comments_sector"
import requests
session = requests.session()
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:88.0) Gecko/20100101 Firefox/88.0',
"content-type": "application/json; charset=UTF-8",
"Connection": "keep-alive"
}
cookies = {}
for item in driver.get_cookies():
    cookies[item['name']] = item['value']
data = session.get(articlelink, headers=headers, cookies=cookies)
page = etree.HTML(data.content)
content =  page.xpath('//section')
import re
# 创建BeautifulSoup对象
# soup = BeautifulSoup(data.content, 'html.parser')
driver.get(articlelink)
soup = BeautifulSoup(driver.page_source, 'html.parser')
pattern = re.compile(r'css-*')
paragraphs = soup.find_all('p', {'data-type': 'paragraph', 'class': pattern})
# 提取每个<p>标签的文本内容
texts = [p.get_text() for p in paragraphs]
# 打印提取到的文本
for text in texts:
    print(text)
all_texts = ''.join(texts)

# find headers and summarys
headers_soup = soup.find_all("div", clas_="article-header")
page = etree.HTML(driver.page_source)
headers_nodes = page.xpath("//div[contains(@class, 'article-header')]/div/div[contains(@class, 'crawler')]/h1")
header_text = ""
if len(headers_nodes) > 0:
    header_text = headers_nodes[0].text
summary_nodes = page.xpath("//div[contains(@class, 'article-header')]/div/h2")
summary_text = ""
if len(summary_nodes) > 0:
    summary_text = summary_nodes[0].text
print("header:", header_text)
print("summary:", summary_text)

# get image link
pattern = r'srcset="(.*?)"'
match = re.search(pattern, driver.page_source)
image_link = ""
import validators
if match:
    srcset = match.group(0)
    url_match = re.search(r'(https?://images.wsj.net/[^?]+)', srcset)
    if url_match and validators.url(url_match.group(0)):
        image_link = url_match.group(0)
        print(image_link)

# extract link
url_pattern = re.compile(r'(https?://www.wsj.com/articles/[^"]+)')
url_match = re.search(url_pattern, driver.page_source)

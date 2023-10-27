from bs4 import BeautifulSoup
import requests

# Fetch the webpage content
url = 'https://burger.bytedance.net/bytenas/operation/upgradeClusterList/5d3feaaa-73c4-11ee-9f72-3436ac1201a0'  # Replace with the URL you're interested in
response = requests.get(url)
web_content = response.text

# Parse the webpage using BeautifulSoup
soup = BeautifulSoup(web_content, 'html.parser')

# Find the button named 'go on'
print(soup)
button = soup.find('button', {'name': 'go on'})

# If the button is found, you can perform further actions (e.g., clicking it if you were using Selenium)
if button:
    print("Button found:", button)
else:
    print("Button not found")
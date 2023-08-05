import requests

url = 'http://localhost:5000/upload'
file_path = "test.txt"

with open(file_path, 'rb') as f:
    response = requests.post(url, files={'file': f})

print(response.json())

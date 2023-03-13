import requests
import json

url = "https://google.serper.dev/search"

payload = json.dumps({
  "q": "山上打老虎"
})

headers = {
  'X-API-KEY': '',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

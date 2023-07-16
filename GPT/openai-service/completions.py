# completion
import requests
import json

url = "https:/your_end_point/openai/deployments/chatgptv1/chat/completions?api-version=2023-03-15-preview"

headers = {
    "Content-Type": "application/json",
    "api-key": "your_key"
}

data = {
    "messages": [
        {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user", "content": "hello"}
    ],
    "max_tokens": 800,
    "temperature": 0.7,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "top_p": 0.95,
    "stop": None
}

response = requests.post(url, headers=headers, data=json.dumps(data))

json_data = response.json()
print(json_data['choices'][0]['message']['content'])
import requests
# use proxies
proxies = {
    "http": "http://127.0.0.1:8787",
    "https": "http://127.0.0.1:8787",
}
url = 'http://httpbin.org/ip'
r = requests.get(url, proxies=proxies)
print(r.text)
# requests from version 2.10.0 support socks proxy
# pip install -U requests[socks]
proxies = {'http': "socks5://myproxy:9191"}
requests.get('http://example.org', proxies=proxies)
# tornado proxy demo
# sudo apt-get install libcurl-dev librtmp-dev
import requests

ip = requests.get('https://api.ipify.org').text
print(f'My IP address is: {ip}')
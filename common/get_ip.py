import socket
import requests

def get_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip

def get_public_ip():
    response = requests.get('https://api.ipify.org?format=json')
    public_ip = response.json()['ip']
    return public_ip
print(f'Local IP: {get_local_ip()}')
print(f'Public IP: {get_public_ip()}')
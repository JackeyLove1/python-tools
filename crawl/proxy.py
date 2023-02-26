import requests
# define proxies
proxies = {
        'http' : 'http://xx.xxx.xxx.xxx:xxxx',
        'http' : 'http://xxx.xx.xx.xxx:xxx',
        ....
    }
# use proxy
response = requests.get(url,proxies=proxies)
import requests

def get_one_page(url:str):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome / 65.0.3325.162 Safari / 537.36'}
    resp = requests.get(url, headers=headers)
    if resp .status_code == 200:
        return resp.text
    return None

def main():
    url = 'http://maoyan.com/board/4'
    html = get_one_page(url)
    print(html)

if __name__ == '__main__':
    main()
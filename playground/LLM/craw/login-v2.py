import requests
from bs4 import BeautifulSoup

# 第一步：发送第一次请求，获取csrftoken
r1 = requests.get(
    url='https://github.com/login'
)
bs1 = BeautifulSoup(r1.text, 'html.parser')
obj_token = bs1.find(
    name='input',
    attrs={'name': 'authenticity_token'}
)
# token = obj_token.attrs.get('value')
token = obj_token.get('value')
r1_cookie = r1.cookies.get_dict()  # 获取第一次cookie值、格式化成字典
print(r1_cookie)

# 第二步：发送post请求，携带用户名密码并伪造请求头
r2 = requests.post(
    url='https://github.com/session',
    data={
        'commit': 'Sign in',
        'utf8': '✓',
        'authenticity_token': token,
        'login': 'mbasfrvqfl@gmail.com',
        'password': 'password'
    },
    cookies=r1_cookie  # 带入第一次的cookie做验证
)

r2_cookie = r2.cookies.get_dict()
print(r2_cookie)
r1_cookie.update(r2_cookie)  # 更新到第一次response的cookie字典里
print(r1_cookie)
# 因为是form data提交所以网页是走的重定向，获取状态码&location
# 1、根据状态码；2、根据错误提示

# 第三步：访问个人页面，携带cookie
url = "https://github.com/bytedance/lightseq/archive/refs/heads/master.zip" # Replace username and repo with your desired repository
response = requests.get(
    url=url,
    cookies=r1_cookie
)

print(response.status_code)
with open("lightseq-master.zip", 'wb') as f:
    f.write(response.content)

# pip download --no-binary :all: -d ./pkg <package-name>
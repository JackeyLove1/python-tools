import requests
from typing import Dict

TELEGRAPH_URL = 'https://api.openai.com'


def handler(event: Dict, context: Dict):
    PRESHARED_AUTH_HEADER_KEY = "X-Custom-PSK"
    PRESHARED_AUTH_HEADER_VALUE = "openai"
    psk = event['headers'].get(PRESHARED_AUTH_HEADER_KEY)

    if psk != PRESHARED_AUTH_HEADER_VALUE:
        return {"body": "Sorry, you have supplied an invalid key.", "statusCode": 403}

    url = event['url']
    headers_Origin = event['headers'].get("Access-Control-Allow-Origin", "*")
    event['headers'].pop(PRESHARED_AUTH_HEADER_KEY, None)

    # proxy
    response = requests.request(
        method=event['httpMethod'],
        url=TELEGRAPH_URL + url,
        headers=event['headers'],
        data=event.get('body'),
        allow_redirects=True
    )

    # Copy headers and add Access-Control-Allow-Origin
    headers = response.headers
    headers['Access-Control-Allow-Origin'] = headers_Origin

    return {"body": response.content, "headers": dict(headers), "statusCode": response.status_code}

'''
{
    "body": "",
    "requestContext": {
        "apiId": "bc1dcffd-aa35-474d-897c-d53425a4c08e",
        "requestId": "11cdcdcf33949dc6d722640a13091c77",
        "stage": "RELEASE"
    },
    "queryStringParameters": {
        "responseType": "html"
    },
    "httpMethod": "GET",
    "pathParameters": {},
    "headers": {
        "accept-language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "accept-encoding": "gzip, deflate, br",
        "x-forwarded-port": "443",
        "x-forwarded-for": "103.218.216.98",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "upgrade-insecure-requests": "1",
        "host": "50eedf92-c9ad-4ac0-827e-d7c11415d4f1.apigw.cn-north-1.huaweicloud.com",
        "x-forwarded-proto": "https",
        "pragma": "no-cache",
        "cache-control": "no-cache",
        "x-real-ip": "103.218.216.98",
        "user-agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0",
        "Authorization": "Bearer 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",
        "X-Custom-PSK": "openai"
    },
    "path": "https://www.openai.com/v1/models",
    "isBase64Encoded": true
}
'''
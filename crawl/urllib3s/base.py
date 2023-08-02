import urllib3
# Create an HTTPHeaderDict and add headers
headers = urllib3.HTTPHeaderDict()
headers.add("Accept", "application/json")
headers.add("Accept", "text/plain")

# Make the request using the headers
resp = urllib3.request(
    "GET",
    "https://httpbin.org/headers",
    headers=headers
)

print(resp.json()["headers"])
# {"Accept": "application/json, text/plain", ...}

# cookies
import urllib3

resp = urllib3.request(
    "GET",
    "https://httpbin.org/cookies",
    headers={
        "Cookie": "session=f3efe9db; id=30"
    }
)

print(resp.json())
# {"cookies": {"id": "30", "session": "f3efe9db"}}

# GET Methods
import urllib3

resp = urllib3.request(
    "GET",
    "https://httpbin.org/get",
    fields={"arg": "value"}
)

print(resp.json()["args"])
# {"arg": "value"}

# POST/PUT Methods
from urllib.parse import urlencode
import urllib3

# Encode the args into url grammar.
encoded_args = urlencode({"arg": "value"})

# Create a URL with args encoded.
url = "https://httpbin.org/post?" + encoded_args
resp = urllib3.request("POST", url)

print(resp.json()["args"])
# {"arg": "value"}

# From Data
import urllib3

resp = urllib3.request(
    "POST",
    "https://httpbin.org/post",
    fields={"field": "value"}
)

print(resp.json()["form"])
# {"field": "value"}

# JSON
import urllib3

resp = urllib3.request(
    "POST",
    "https://httpbin.org/post",
    json={"attribute": "value"},
    headers={"Content-Type": "application/json"}
)

print(resp.json())
# {'headers': {'Content-Type': 'application/json', ...},
#  'data': '{"attribute":"value"}', 'json': {'attribute': 'value'}, ...}

# Files & Binary Data
import urllib3

# Reading the text file from local storage.
with open("example.txt") as fp:
    file_data = fp.read()

# Sending the request.
resp = urllib3.request(
    "POST",
    "https://httpbin.org/post",
    fields={
       "filefield": ("example.txt", file_data),
    }
)

print(resp.json()["files"])
# {"filefield": "..."}

resp = urllib3.request(
    "POST",
    "https://httpbin.org/post",
    fields={
        "filefield": ("example.txt", file_data, "text/plain"),
    }
)
import urllib3

with open("/home/samad/example.jpg", "rb") as fp:
    binary_data = fp.read()

resp = urllib3.request(
    "POST",
    "https://httpbin.org/post",
    body=binary_data,
    headers={"Content-Type": "image/jpeg"}
)

print(resp.json()["data"])
# data:application/octet-stream;base64,...

# using timeout
import urllib3

resp = urllib3.request(
    "GET",
    "https://httpbin.org/delay/3",
    timeout=4.0
)

print(type(resp))
# <class "urllib3.response.HTTPResponse">

# This request will take more time to process than timeout.
urllib3.request(
    "GET",
    "https://httpbin.org/delay/3",
    timeout=2.5
)
# MaxRetryError caused by ReadTimeoutError

import urllib3

resp = urllib3.request(
    "GET",
    "https://httpbin.org/delay/3",
    timeout=4.0
)

print(type(resp))
# <class "urllib3.response.HTTPResponse">

# This request will take more time to process than timeout.
urllib3.request(
    "GET",
    "https://httpbin.org/delay/3",
    timeout=2.5
)
# MaxRetryError caused by ReadTimeoutError

# retries
import urllib3

urllib3.request("GET", "https://httpbin.org/ip", retries=10)
import urllib3

urllib3.request(
    "GET",
    "https://nxdomain.example.com",
    retries=False
)
# NewConnectionError

resp = urllib3.request(
    "GET",
    "https://httpbin.org/redirect/1",
    retries=False
)

print(resp.status)
# 302

import urllib3
resp = urllib3.request("GET", "https://www.openbmb.org/community/course")
print(resp.data.decode('utf-8'))
from urllib.parse import  urlparse
import validators
url = "https://aaronhertzmann.com/2020/04/19/lines-as-edges.html"
print(validators.url(url))
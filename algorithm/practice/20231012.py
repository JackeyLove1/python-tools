import re

input_string = "CurrentLeader : (1,10.152.180.197:15004,11.153.199.167:15005,voter)"

# Extract IP address using regular expression
ip_address = re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', input_string).group(0)

import re

def extract_ipv4_addresses(text):
    ipv4_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
    ipv4_addresses = re.findall(ipv4_pattern, text)
    return ipv4_addresses

print(extract_ipv4_addresses(input_string))
print(ip_address)
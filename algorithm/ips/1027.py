import os
import pandas as pd
df = pd.read_excel("hosts.xlsx")
print(df['ip'])
print(df['ipv6'])
v4s = df['ip'].to_list()
v6s = df['ipv6'].to_list()

assert len(v4s) == len(v6s), "v4 and v6 are not equal"
port = "8600"
command = "curl -X POST http://{}:{} -d @./commands"
with open("commands", "w") as file:
    for (v4, v6) in zip(v4s, v6s):
        file.write(command.format(v4, v6) + "\n")
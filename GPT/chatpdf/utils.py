import openai
import random
import time

random.seed(int(time.time()))
from common import *


def chat(message: str, system_prompt=Common_Prompt):
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"  # may be change
    kv = random.choice(kvs)
    openai.api_base = kv[0]
    openai.api_key = kv[1]
    response = openai.ChatCompletion.create(
        engine="chatgptv1",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": message}],
        temperature=0.7,
        max_tokens=Default_Max_Tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return response['choices'][0]['message']['content']




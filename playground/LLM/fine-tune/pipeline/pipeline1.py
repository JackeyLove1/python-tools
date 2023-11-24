# pip install -q transformers
from transformers import pipeline
checkpoint = "MBZUAI/LaMini-GPT-1.5B"
model = pipeline('text-generation', model = checkpoint)
instruction = 'Please let me know your thoughts on the given place and why you think it deserves to be visited: \n"Barcelona, Spain"'
input_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
generated_text = model(input_prompt, max_length=512, do_sample=True)[0]['generated_text']
print("Response", generated_text)
'''
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
text = generator("import pandas as pd \n import numpy as np\n " , do_sample=True, min_length=200)
print(text)
print(text[0]['generated_text'])
'''
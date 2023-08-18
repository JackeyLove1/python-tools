from transformers import AutoTokenizer

model = "bert-base-cased"
example = "Hello, World!"
tokenizer = AutoTokenizer.from_pretrained(model)
encoding = tokenizer(example)
print(type(encoding))
print("encoding:", encoding)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
# 可以直接通过 tokens() 函数来获取切分出的 token
print(encoding.tokens())
# 可以通过 word_ids() 函数来获取每一个 token 对应的词语索引
print(encoding.word_ids())

# 词语/token => 文本：通过 word_to_chars()、token_to_chars()
token_index = 5
print('the 5th token is:', encoding.tokens()[token_index])
start, end = encoding.token_to_chars(token_index)
print('corresponding text span is:', example[start:end])
word_index = encoding.word_ids()[token_index] # 3
start, end = encoding.word_to_chars(word_index)
print('corresponding word span is:', example[start:end])

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
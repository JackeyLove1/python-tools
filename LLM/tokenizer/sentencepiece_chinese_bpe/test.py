import os

from sentencepiece import sentencepiece_model_pb2  as model
import sentencepiece as spm
from transformers import LlamaTokenizer

# parse pretrain model
m = model.ModelProto()
llama_tokenizer_dir = "transformers_tokenizer/llama/tokenizer.model"
chinese_sp_model_file = "tokenizer.model"
tokens = m.ParseFromString(open(chinese_sp_model_file, "rb").read())
print(f"Chinese use tokens:{tokens}")
print(f"vocab size:{len(m.pieces)}")
chinese_model = spm.SentencePieceProcessor()
chinese_model.Load(chinese_sp_model_file)
print(chinese_model.encode("萧炎你好大的胆子！", out_type=str))
chinese_model_spm = model.ModelProto()
chinese_model_spm.ParseFromString(chinese_model.serialized_model_proto())
print("pieces:",len(chinese_model_spm.pieces))

import sentencepiece as spm
llama_model = model.ModelProto()
llama_model_path = "llama/tokenizer.model"
tokens = llama_model.ParseFromString(open(llama_model_path, "rb").read())
print(f"llama use tokens:{tokens}")

# merge tokens
llama_token_set = set(p.piece for p in llama_model.pieces)
print("llama model size before:", len(llama_token_set))
for p in chinese_model_spm.pieces:
    token = p.piece
    if token not in llama_token_set:
        new_token = model.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0
        llama_model.pieces.append(new_token)
print("New model pieces:", len(llama_model.pieces))

# save
output_sp_dir = 'transformers_tokenizer/llama_chinese'
output_hf_dir = 'transformers_tokenizer/llama_chinese'
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + "/chinese_llama.model", "wb") as f:
    f.write(llama_model.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+"/chinese_llama.model")

tokenizer.save_pretrained(output_hf_dir)
print(f"Chinese-LLAMA tokenizer has been saved to {output_hf_dir}")
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)

# Test
print("all special tokens:", tokenizer.all_special_tokens)
print(tokenizer.special_tokens_map)
text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
The primary use of LLaMA is research on large language models, including'''
print("Text:", text)
print("llama:", llama_tokenizer.tokenize(text))
print("chinese:", tokenizer.tokenize(text))

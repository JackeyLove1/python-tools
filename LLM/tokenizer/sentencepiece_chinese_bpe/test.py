from sentencepiece import sentencepiece_model_pb2  as model
import sentencepiece as spm
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

import sentencepiece as spm
llama_model = model.ModelProto()
llama_model_path = "llama/tokenizer.model"
tokens = llama_model.ParseFromString(open(llama_model_path, "rb").read())
print(f"llama use tokens:{tokens}")

# merge tokens
llama_token_set = set()
# add new tokens to sentencepiece model

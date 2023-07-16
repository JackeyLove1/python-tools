#!pip install -U text2vec
'''
embedding_name = ["shibing624/text2vec-base-chinese" ,
                "GanymedeNil/text2vec-large-chinese" , # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # 支持多语言的句向量模型（Sentence-BERT），英文语义匹配任务推荐，支持fine-tune继续训练
                "w2v-light-tencent-chinese",  # 中文词向量模型(word2vec)，中文字面匹配任务和冷启动适用
                "nghuyong/ernie-3.0-base-zh",
                "nghuyong/ernie-3.0-nano-zh",
                'moka-ai/m3e-base',
                ]
'''
from text2vec import SentenceModel,Word2Vec
# create embedding models
t2v_model_name = "shibing624/text2vec-base-chinese"
t2v_model = SentenceModel(t2v_model_name)

# calculate sentence's embedding
'''
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("shibing624/text2vec-base-chinese")
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
sentence_embeddings = m.encode(sentences)
print("Sentence embeddings:")
print(sentence_embeddings)
'''
sentences="你好！"
sentence_embeddings = t2v_model.encode(sentences)
print(type(sentence_embeddings), sentence_embeddings.shape)
# print("sentence_embeddings:", sentence_embeddings)

# calculate similarity
# pip install -U similarities
from similarities import Similarity
model = Similarity(model_name_or_path="shibing624/text2vec-base-chinese")
sentences = ['如何更换花呗绑定银行卡',
             '花呗更改绑定银行卡']
similarity_score = model.similarity(sentences[0], sentences[1])
print(f"{sentences[0]} vs {sentences[1]}, score: {float(similarity_score):.4f}")
# 基于字面的文本相似度计算和匹配搜索
from similarities.literalsim import SimHashSimilarity, TfidfSimilarity, BM25Similarity, \
    WordEmbeddingSimilarity, CilinSimilarity, HownetSimilarity
text1 = "如何更换花呗绑定银行卡"
text2 = "花呗更改绑定银行卡"

corpus = [
    '花呗更改绑定银行卡',
    '我什么时候开通了花呗',
    '俄罗斯警告乌克兰反对欧盟协议',
    '暴风雨掩埋了东北部；新泽西16英寸的降雪',
    '中央情报局局长访问以色列叙利亚会谈',
    '人在巴基斯坦基地的炸弹袭击中丧生',
]
m = TfidfSimilarity()
print(text1, text2, ' sim score: ', m.similarity(text1, text2))

m.add_corpus(corpus)
queries = [
    '我的花呗开通了？',
    '乌克兰被俄罗斯警告'
]
res = m.most_similar(queries, topn=3)

# M3E: https://huggingface.co/moka-ai/m3e-base
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('moka-ai/m3e-base')

#Our sentences we like to encode
sentences = [
    '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
    '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
    '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
]

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", type(embedding))
    print("Embedding:", embedding.shape) # (768, )


# SGPT: https://github.com/Muennighoff/sgpt
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine

# Get our models - The package will take care of downloading the models automatically
# For best performance: Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit
tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
model = AutoModel.from_pretrained("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
# Deactivate Dropout (There is no dropout in the above models so it makes no difference here but other SGPT models may have dropout)
model.eval()

# Tokenize input texts
texts = [
    "deep learning",
    "artificial intelligence",
    "deep diving",
    "artificial snow",
]
batch_tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    # Get hidden state of shape [bs, seq_len, hid_dim]
    last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

# Get weights of shape [bs, seq_len, hid_dim]
weights = (
    torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
    .unsqueeze(0)
    .unsqueeze(-1)
    .expand(last_hidden_state.size())
    .float().to(last_hidden_state.device)
)

# Get attn mask of shape [bs, seq_len, hid_dim]
input_mask_expanded = (
    batch_tokens["attention_mask"]
    .unsqueeze(-1)
    .expand(last_hidden_state.size())
    .float()
)

# Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

embeddings = sum_embeddings / sum_mask

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
cosine_sim_0_3 = 1 - cosine(embeddings[0], embeddings[3])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[3], cosine_sim_0_3))
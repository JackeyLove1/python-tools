#!pip install -U text2vec
'''
embedding_name = ["shibing624/text2vec-base-chinese" ,
                "GanymedeNil/text2vec-large-chinese" , # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # 支持多语言的句向量模型（Sentence-BERT），英文语义匹配任务推荐，支持fine-tune继续训练
                "w2v-light-tencent-chinese",  # 中文词向量模型(word2vec)，中文字面匹配任务和冷启动适用
                "nghuyong/ernie-3.0-base-zh",
                "nghuyong/ernie-3.0-nano-zh",
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
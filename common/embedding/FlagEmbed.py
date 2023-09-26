from sentence_transformers import SentenceTransformer
#
sentences_1 = ["你好！"]
sentences_2 = ["如何更换花呗绑定银行卡"]
sentences_3 = ["花呗更改绑定银行卡？"]
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
embeddings_3 = model.encode(sentences_3, normalize_embeddings=True)
# from scipy.spatial.distance import cosine
import numpy as np

# 将列表转换为NumPy数组
array1 = np.array(embeddings_1).reshape(-1)
array2 = np.array(embeddings_2).reshape(-1)
array3 = np.array(embeddings_3).reshape(-1)


# 计算余弦相似度
def simility(vec1, vec2):
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim


print(simility(array1, array2))
print(simility(array1, array3))
print(simility(array2, array3))

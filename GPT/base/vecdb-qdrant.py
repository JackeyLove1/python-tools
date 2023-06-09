'''
!pip install qdrant-client>=1.1.1
!pip install -U sentence-transformers
'''
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
# encoder = SentenceTransformer("shibing624/text2vec-base-chinese")
encoder = SentenceTransformer("GanymedeNil/text2vec-large-chinese")
# Let's make a semantic search for Sci-Fi books!
documents =  [
{ "name": "时间机器", "description": "一名男子穿越时空，目睹人类的演化。", "author": "H.G.威尔斯", "year": 1895 },
{ "name": "安德的游戏", "description": "一个年轻男孩被训练成为一名军事领袖，参与对抗外星种族的战争。", "author": "奥森·斯科特·卡德", "year": 1985 },
{ "name": "美丽新世界", "description": "一个反乌托邦社会，人们被基因工程和行为训练来符合严格的社会等级制度。", "author": "阿道斯·赫胥黎", "year": 1932 },
{ "name": "银河系漫游指南", "description": "一部喜剧科幻系列，讲述了一个无意中卷入一系列冒险的人类和他的外星朋友的故事。", "author": "道格拉斯·亚当斯", "year": 1979 },
{ "name": "沙丘", "description": "一个沙漠星球上的政治阴谋和权力斗争。", "author": "弗兰克·赫伯特", "year": 1965 },
{ "name": "基地", "description": "一名数学家开发了一种预测人类未来的科学，并致力于拯救文明免于崩溃。", "author": "艾萨克·阿西莫夫", "year": 1951 },
{ "name": "雪崩", "description": "一个未来世界，互联网已经发展成为一个虚拟现实的元宇宙。", "author": "尼尔·斯蒂芬森", "year": 1992 },
{ "name": "神经漫游者", "description": "一名黑客被雇佣进行一次几乎不可能的黑客攻击，并卷入了一系列的阴谋。", "author": "威廉·吉布森", "year": 1984 },
{ "name": "世界大战", "description": "火星人入侵地球，使人类陷入混乱。", "author": "H.G.威尔斯", "year": 1898 },
{ "name": "饥饿游戏", "description": "一个反乌托邦社会，青少年被迫在电视上打斗至死。", "author": "苏珊·柯林斯", "year": 2008 },
{ "name": "安德洛梅达奇兵", "description": "来自外太空的致命病毒威胁要消灭人类。", "author": "迈克尔·克莱顿", "year": 1969 },
{ "name": "黑暗之左手", "description": "一名人类大使被派往一个居民没有性别、可以随意改变性别的星球。", "author": "乌苏拉·勒·吉恩", "year": 1969 },
{ "name": "时间旅行者的妻子", "description": "一段男子无意中穿越时空和他所爱的女人之间的爱情故事。", "author": "奥黛丽·尼芬格", "year": 2003 }
]
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance

# Create collection to store books
qdrant.recreate_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

# Let's vectorize descriptions and upload to qdrant
qdrant.upload_records(
    collection_name="my_books",
    records=[
        models.Record(
            id=idx,
            vector=encoder.encode(doc["description"]).tolist(),
            payload=doc
        ) for idx, doc in enumerate(documents)
    ]
)

# Let's now search for something
hits = qdrant.search(
    collection_name="my_books",
    query_vector=encoder.encode("外星人袭击了我们星球").tolist(),
    limit=3
)
for hit in hits:
  print(hit.payload, "score:", hit.score)

# Let's now search only for books from 21st century
hits = qdrant.search(
    collection_name="my_books",
    query_vector=encoder.encode("暴政社会").tolist(),
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="year",
                range=models.Range(
                    gte=2000
                )
            )
        ]
    ),
    limit=3
)
for hit in hits:
  print(hit.payload, "score:", hit.score)
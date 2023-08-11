# python -m spacy download en_core_web_sm
# python -m spacy download zh_core_web_sm/md
# pip install -U spacy

# 词性标注
import spacy
# 读取小版本的中文流程
nlp = spacy.load("zh_core_web_sm")
# 处理文本
doc = nlp("我吃了个肉夹馍")
# 遍历词符
for token in doc:
    # Print the text and the predicted part-of-speech tag
    print(token.text, token.pos_)
'''
我 PRON
吃 VERB
了 PART
个 NUM
肉夹馍 NOUN
'''
# 依存关系解析
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)
'''
我 PRON nsubj 吃
吃 VERB ROOT 吃
了 PART aux:asp 吃
个 NUM nummod 肉夹馍
肉夹馍 NOUN dobj 吃
'''

# 命名实体识别
# 处理文本
nlp = spacy.load("zh_core_web_sm")
doc = nlp("微软准备用十亿美金买下这家英国的创业公司。")

# 遍历识别出的实体
for ent in doc.ents:
    # 打印实体文本及其标注
    print(ent.text, ent.label_)
'''
微软 ORG
十亿美金 MONEY
英国 GPE
'''

# 基于规则的匹配
# TODO

# 共享词汇表和字符串库
nlp.vocab.strings.add("咖啡")
coffee_hash = nlp.vocab.strings["咖啡"]
coffee_string = nlp.vocab.strings[coffee_hash]

# Doc、Span和Token
import spacy
nlp = spacy.blank("en")

# 导入Doc类
from spacy.tokens import Doc

# 用来创建doc的词汇和空格
words = ["Hello", "world", "!"]
spaces = [True, False, False]

# 手动创建一个doc
doc = Doc(nlp.vocab, words=words, spaces=spaces)

# 词向量和语义相似度
'''
对比语义相似度
spaCy可以对比两个实例来判断它们之间的相似度
Doc.similarity()、Span.similarity()和Token.similarity()
使用另一个实例作为参数返回一个相似度分数(在0和1之间)
注意：我们需要一个含有词向量的流程，比如：
✅ en_core_web_md (中等)
✅ en_core_web_lg (大)
🚫 而不是 en_core_web_sm (小)
'''
# 读取一个有词向量的较大流程
nlp = spacy.load("en_core_web_sm")

# 比较两个文档
doc1 = nlp("I like fast food")
doc2 = nlp("I like pizza")
print(doc1.similarity(doc2))
doc = nlp("I like pizza and pasta")
token1 = doc[2]
token2 = doc[4]
print(token1.similarity(token2))
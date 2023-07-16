kvs = []

Default_Max_Tokens = 4096
Chinese_Embedding_Base = "shibing624/text2vec-base-chinese"
Chinese_Embedding_Large = "GanymedeNil/text2vec-large-chinese"
# prompt
Common_Prompt = "You are an AI assistant that helps people find information."

Chinese_QA_Prompt = """\
基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

已知内容:
{context_str}

问题:
{query_str}
"""

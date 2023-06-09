'''
!pip install -U chatllm
'''
from chatllm.applications.chatpdf import ChatPDF

qa = ChatPDF(encode_model='nghuyong/ernie-3.0-nano-zh')
qa.load_llm(model_name_or_path="THUDM/chatglm-6b")
qa.create_index('财报.pdf')  # 构建知识库

for i in qa(query='东北证券主营业务'):
    print(i, end='')
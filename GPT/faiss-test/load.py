import os
import time

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFium2Loader, UnstructuredPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from typing import Union
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

BASE_URL = "your_endpoint"
API_KEY = "your_key"
DEPLOYMENT_NAME = "chatgptv1"
SPLIT_CHUNK_SIZE = 1000
SPLIT_CHUNK_OVERLAP = 200

start = time.time()
loader = PyPDFium2Loader("demo.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-03-15-preview",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure")
'''
from langchain.schema import Document
list_of_documents = [
    Document(page_content="foo", metadata=dict(page=1)),
    Document(page_content="bar", metadata=dict(page=1)),
    Document(page_content="foo", metadata=dict(page=2)),
    Document(page_content="barbar", metadata=dict(page=2)),
    Document(page_content="foo", metadata=dict(page=3)),
    Document(page_content="bar burr", metadata=dict(page=3)),
    Document(page_content="foo", metadata=dict(page=4)),
    Document(page_content="bar bruh", metadata=dict(page=4)),
]
'''

db_name = "faiss_index_test_demo"
'''
db = FAISS.from_documents(docs, embeddings)
results_with_scores = db.similarity_search_with_score("what is faiss ?", k=1)
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
db.save_local(db_name)
'''
query = "What is faiss"
new_db = FAISS.load_local(db_name, embeddings)
load_db_time = time.time()
print(f"Load Cost:{load_db_time - start}")
docs_find = new_db.similarity_search(query, k=3)
query_db_time = time.time()
print(f"Query Cost:{query_db_time - load_db_time}")
chain = load_qa_chain(llm, chain_type="stuff")
response = chain.run(input_documents=docs_find, question=query)
print(response)
print(f"Answer Cost:{time.time() - query_db_time}")
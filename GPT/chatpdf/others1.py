import os
import time

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFium2Loader, UnstructuredPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from typing import Union
BASE_URL = ""
API_KEY = ""
DEPLOYMENT_NAME = "chatgptv1"
SPLIT_CHUNK_SIZE = 1000
SPLIT_CHUNK_OVERLAP = 200


class PDFQuery:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=SPLIT_CHUNK_SIZE, chunk_overlap=SPLIT_CHUNK_SIZE)
        self.llm = AzureChatOpenAI(
            openai_api_base=BASE_URL,
            openai_api_version="2023-03-15-preview",
            deployment_name=DEPLOYMENT_NAME,
            openai_api_key=API_KEY,
            openai_api_type="azure")
        self.loader = None
        self.chain = None
        self.db = None


    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            docs = self.db.get_relevant_documents(question)
            response = self.chain.run(input_documents=docs, question=question)
        return response

    def ingest(self, file : Union[bytes, str]) -> None:
        if isinstance(file, str):
            self.loader = UnstructuredPDFLoader(file)
        else:
            self.loader = UnstructuredPDFLoader(file_path="test", file=file)
        documents = self.loader.load()
        docs = self.text_splitter.split_documents(documents)
        self.db = Qdrant.from_documents(docs, self.embeddings,
                                        location=":memory:", collection_name="my_documents").as_retriever()
        self.chain = load_qa_chain(self.llm, chain_type="stuff")

    def forget(self) -> None:
        self.db = None
        self.chain = None

start = time.time()
file_path = "demo.pdf"
chat_pdf = PDFQuery()
chat_pdf.ingest(file_path)
embedding_time = time.time()
print(f"Embedding Cost:{embedding_time - start}")
answer = chat_pdf.ask("What the paper talk about?")
print("answer:", answer)
print(f"Answer Cost:{time.time() - embedding_time}")
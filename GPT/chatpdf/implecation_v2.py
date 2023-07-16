import time

from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import VectorDBQA
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from loader import PyPDFLoader
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

start = time.time()
file_path = "demo.pdf"
loader = PDFMinerLoader(file_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
qdrant = Qdrant.from_documents(
    docs, embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
)
query = "what is ANNS?"
found_docs = qdrant.similarity_search_with_score(query, k=4)
for d in found_docs:
    print(d[0].page_content)
print("cost:", time.time() - start)
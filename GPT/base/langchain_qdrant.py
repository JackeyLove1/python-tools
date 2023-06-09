'''
!pip install qdrant-client>=1.1.1
!pip install -U sentence-transformers
!pip install langchain
!pip install openai
!pip install pdfminer
!pip install google-search-results
!pip install unstructured
!pip install chromadb
!pip install pinecone-client
!pip install youtube-transcript-api
!pip install pytube
'''
import os
os.environ["OPENAI_API_KEY"] = "your openai key"

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader

loader = TextLoader('your txt path')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# in memory mode
qdrant = Qdrant.from_documents(
    docs, embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
)

del qdrant
# in disk mode
qdrant = Qdrant.from_documents(
    docs, embeddings,
    path="/tmp/local_qdrant",
    collection_name="my_documents",
)

del qdrant

# load from disk
import qdrant_client

client = qdrant_client.QdrantClient(
    path="/tmp/local_qdrant", prefer_grpc=True
)
qdrant = Qdrant(
    client=client, collection_name="my_documents",
    embeddings=embeddings
)

# similarity search
query = "What did the president say about Ketanji Brown Jackson"
found_docs = qdrant.similarity_search(query)
print(found_docs[0].page_content)

# Similarity search with score
query = "What did the president say about Ketanji Brown Jackson"
found_docs = qdrant.similarity_search_with_score(query)
document, score = found_docs[0]
print(document.page_content)
print(f"\nScore: {score}")


# pip install faiss-cpu
import numpy as np
import faiss

# Dimension of our vector space
d = 64
# Number of vectors
n_data = 10000
# Randomly generate data
np.random.seed(1234)
data = np.random.random((n_data, d)).astype('float32')
# Build the index
index = faiss.IndexFlatL2(d)
print(index.is_trained)
# Add vectors to the index
index.add(data)
print(index.ntotal)
# Search the index
n_query = 10
query = np.random.random((n_query, d)).astype('float32')
k = 4  # we want to see 4 nearest neighbors
D, I = index.search(query, k)
print(I)
print(D)

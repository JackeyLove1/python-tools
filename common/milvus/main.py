# wget https://github.com/milvus-io/milvus/releases/download/v2.2.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
# sudo docker-compose up -d
# sudo mkdir -p  /var/lib/docker
# sudo chmod 777 /var/lib/docker
import time
import json

from pymilvus import connections, db
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
num_entities, dim = 100, 10
connections.connect("default", host="localhost", port="19530")
cname = "qa"
has = utility.has_collection(cname)
print(f"Does collection {cname} exist in Milvus: {has}")
fields = [
    FieldSchema(name="pk",dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="qa", dtype=DataType.JSON),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
schema = CollectionSchema(fields, f"{cname} is the simplest demo to introduce the APIs")
hello_milvus = Collection(f"{cname}", schema, consistency_level="Strong")
import numpy as np
from faker import Faker
fake = Faker()
Faker.seed(int(time.time()))
rng = np.random.default_rng(seed=19530)
entities = [
    [{"q":fake.name(),"a":fake.text()} for _ in range(num_entities)],
    rng.random((num_entities, dim)),    # field embeddings, supports numpy.ndarray and list
]
insert_result = hello_milvus.insert(entities)
hello_milvus.flush()
print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entites
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

hello_milvus.create_index("embeddings", index)
hello_milvus.load()
vectors_to_search = entities[-1][-2:]
print(f"vec2search:{str(vectors_to_search)}")
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
start_time = time.time()
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["qa"])
end_time = time.time()

for hits in result:
    for hit in hits:
        qa = hit.entity.get('qa')
        q, a = qa.get('q'), qa.get('a')
        # print(f"hit: {hit}, qa field: {hit.entity.get('qa')}")
        print(f"hit: {hit}, q:{q}, a:{q}")
print(f"Cost{end_time - start_time}")

# utility.drop_collection(f"{cname}")
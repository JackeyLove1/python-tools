from pymilvus import connections
import numpy as np
# 导入的 connections 是一个单例对象，维护用户所有的连接

# 连接数据库
def connect_db():

    # 连接数据库, 如果 alias 没有指定的话，默认名字为 default
    connections.connect(alias='main1', host='localhost', port=19530)

    # 判断连接对象状态
    print(connections.has_connection('main1'))

    # 获得 alias 对应的连接地址
    print(connections.get_connection_addr('main1'))

    # 该函数可以添加连接信息，当创建数据库时可以直接使用 alias 标记在那个连接中进行操作
    # connections.add_connection(main2={"host": "localhost", "port": '19530'})
    '''
         connections.add_connection(
                default={"host": "localhost", "port": "19530"},
                dev1={"host": "localhost", "port": "19531"},
                dev2={"uri": "http://random.com/random"},
                dev3={"uri": "http://localhost:19530"},
                dev4={"uri": "tcp://localhost:19530"},
                dev5={"address": "localhost:19530"},
                prod={"uri": "http://random.random.random.com:19530"},
            )
    '''

    # 获得所有连接对象
    print(connections.list_connections())

    # 断开名字为 alias 连接
    connections.disconnect('main')

# 创建集合
from pymilvus import connections
from pymilvus import CollectionSchema
from pymilvus import Collection
from pymilvus import FieldSchema
from pymilvus import DataType
from pymilvus import list_collections
from pymilvus import has_collection
from pymilvus import drop_collection

def create_collection():

    # 连接数据库
    connections.connect(alias='main', host='localhost', port=19530)


    # 1. 定义 collection 字段信息
    field1 = FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, description='主键')
    field2 = FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=100, description='名字')
    field3 = FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=328, description='向量')

    # 2. 定义 collection 配置信息
    collection_schema = CollectionSchema(fields=[field1, field2, field3], description='数据库')

    # 3. 创建数据库和表
    # using 如果不指定，则使用 default 连接
    collection = Collection(name='my_collection',        # collection 的名字
                            schema=collection_schema,    # collection 的配置信息
                            using='main')                # 向哪个连接中创建数据库和表

    # 4. 其他关于 collection 的操作
    # 获得指定连接中所有的 collection 名字
    print('获得所有的集合:', list_collections(using='main'))

    # 判断是否存在指定名字的 collection 集合
    print('是否存在某集合:', has_collection(collection_name='my_collection', using='main'))

    # 删除指定名字的 collection 集合
    drop_collection(collection_name='my_collection', using='main')
    print('是否存在某集合:', has_collection(collection_name='my_collection', using='main'))


    # 断开数据库
    connections.disconnect('main')

# 插入数据
def insert_data():

    # 连接数据库
    connections.connect(alias='main', host='localhost', port=19530)


    # 1. 定义 collection 字段信息
    field1 = FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, description='主键')
    field2 = FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=100, description='名字')
    field3 = FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=328, description='向量')

    # 2. 定义 collection 配置信息
    collection_schema = CollectionSchema(fields=[field1, field2, field3], description='数据库')

    # 3. 创建数据库和表
    # using 如果不指定，则使用 default 连接
    collection = Collection(name='my_collection',        # collection 的名字
                            schema=collection_schema,    # collection 的配置信息
                            using='main')                # 向哪个连接中创建数据库和表

    # 4. 其他关于 collection 的操作
    # 获得指定连接中所有的 collection 名字
    print('获得所有的集合:', list_collections(using='main'))

    # 判断是否存在指定名字的 collection 集合
    print('是否存在某集合:', has_collection(collection_name='my_collection', using='main'))

    # 删除指定名字的 collection 集合
    drop_collection(collection_name='my_collection', using='main')
    print('是否存在某集合:', has_collection(collection_name='my_collection', using='main'))


    # 断开数据库
    connections.disconnect('main')

# 查询数据库
def query_db():

    # 连接数据库
    connections.connect(alias='main', host='localhost', port=19530)

    # 获得 collection 对象
    collection = Collection(name='my_collection',using='main')
    collection.release()

    # 我们要查询 ID 字段，给其构建索引
    # 构建索引需要在 collection 未加载到内存时进行
    collection.create_index(field_name='id', index_name='PK_index')

    # 给向量字段构建索引，并指定索引类型，以及相似度度量方式
    # nlist 表示簇的个数，该参数可以将向量划分成多个区域，有利于加快搜索
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name='vector', index_name='vector_index', index_params=index_params)

    # 将整个 collection 加载到内存中，也可以只加载某个 Partition
    collection.load()

    # 根据某个字段查询
    # query 函数的 output_fields 字段可以返回任意字段
    res = collection.query(expr='id > 0', output_fields=['id', 'name'])
    print(len(res), res)


    # 查询相似的向量
    data = np.random.randn(1, 328)
    res = collection.search(data=data,   # 要搜索的向量
                            limit=3,     # 返回记录数量
                            anns_field='vector',  # 要搜索的字段
                            param={'nprobe': 10, 'metric_type': 'L2'},   # nprobe 在最近的10个簇中搜索
                            output_fields=['id'])  # 输出字段名字，只能是标量字段(数字或者小数字段，字符串无法返回)

    print(res)


    # 断开数据库
    connections.disconnect('main')
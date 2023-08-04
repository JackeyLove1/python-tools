连接方式
redis-py提供两个类Redis和StrictRedis用于实现Redis的命令，StrictRedis用于实现大部分官方的命令，并使用官方的语法和命令，Redis是StrictRedis的子类。

import redis
r = redis.Redis(host='192.168.0.110', port=6379,db=0)
r.set('name', 'zhangsan')   #添加
print (r.get('name'))   #获取
连接池
redis-py使用connection pool来管理对一个redis server的所有连接，避免每次建立、释放连接的开销。默认，每个Redis实例都会维护一个自己的连接池。可以直接建立一个连接池，然后作为参数Redis，这样就可以实现多个Redis实例共享一个连接池。

import redis
pool = redis.ConnectionPool(host='192.168.0.110', port=6379)
r = redis.Redis(connection_pool=pool)
r.set('name', 'zhangsan')   #添加
print (r.get('name'))   #获取
管道
redis-py默认在执行每次请求都会创建（连接池申请连接）和断开（归还连接池）一次连接操作，如果想要在一次请求中指定多个命令，则可以使用pipline实现一次请求指定多个命令，并且默认情况下一次pipline 是原子性操作。

import redis

pool = redis.ConnectionPool(host='192.168.0.110', port=6379)
r = redis.Redis(connection_pool=pool)
pipe = r.pipeline(transaction=True)
r.set('name', 'zhangsan')
r.set('name', 'lisi')
pipe.execute()
发布和订阅
首先定义一个RedisHelper类，连接Redis，定义频道为monitor，定义发布(publish)及订阅(subscribe)方法。

import redis
class RedisHelper(object):
    def __init__(self):
        self.__conn = redis.Redis(host='192.168.0.110',port=6379)#连接Redis
        self.channel = 'monitor' #定义名称
    def publish(self,msg):#定义发布方法
        self.__conn.publish(self.channel,msg)
        return True
    def subscribe(self):#定义订阅方法
        pub = self.__conn.pubsub()
        pub.subscribe(self.channel)
        pub.parse_response()
        return pub

set()
#在Redis中设置值，默认不存在则创建，存在则修改
r.set('name', 'zhangsan')
'''参数：
     set(name, value, ex=None, px=None, nx=False, xx=False)
     ex，过期时间（秒）
     px，过期时间（毫秒）
     nx，如果设置为True，则只有name不存在时，当前set操作才执行,同setnx(name, value)
     xx，如果设置为True，则只有name存在时，当前set操作才执行'''
setex(name, value, time)
#设置过期时间（秒）
psetex(name, time_ms, value)
#设置过期时间（豪秒）

mset()
#批量设置值
r.mset(name1='zhangsan', name2='lisi')
#或
r.mset({"name1":'zhangsan', "name2":'lisi'})

get(name) 和 mget(keys, *args)
#批量获取
print(r.mget("name1","name2"))
#或
li=["name1","name2"]
print(r.mget(li))

getset(name, value)
#设置新值，打印原值
print(r.getset("name1","wangwu")) #输出:zhangsan
print(r.get("name1")) #输出:wangwu


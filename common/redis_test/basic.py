import redis
pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
r = redis.Redis(connection_pool=pool)
r.set("k1", "v1")
print(r.get('k1'))
print(r.ping())
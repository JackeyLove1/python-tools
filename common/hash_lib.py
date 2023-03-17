import hashlib
md5 = hashlib.md5()
md5.update(b"how to use md5 in python hashlib?")
print (md5.hexdigest())
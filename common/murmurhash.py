# pip install murmurhash3
import mmh3
mmh3.hash('foo')  # 32-bit signed int

mmh3.hash64('foo')  # two 64-bit signed ints (the 128-bit hash sliced in half)

mmh3.hash128('foo')  # 128-bit signed int

mmh3.hash_bytes('foo')  # 128-bit value as bytes

mmh3.hash('foo', 42)  # uses 42 for its seed
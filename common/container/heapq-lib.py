import heapq

list1 = [34, 25, 12, 99, 87, 63, 58, 78, 88, 92]
list2 = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}]

print(heapq.nlargest(3, list1))
print(heapq.nsmallest(3, list1))

# use lambda key
print(heapq.nlargest(3, list2, key=lambda x : x['name']))
print(heapq.nlargest(3, list2, key=lambda x : x['shares']))

# insert and pop
h = []
heapq.heappush(h, (5, 'apple'))
heapq.heappush(h, (7, 'banana'))
heapq.heappush(h, (1, 'cat'))
print(h)

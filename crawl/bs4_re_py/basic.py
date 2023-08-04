import re
from bs4 import BeautifulSoup
html = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title" name="dromouse"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1"><!-- Elsie --></a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
"""
#创建 Beautiful Soup 对象
soup = BeautifulSoup(html,'html.parser')
#打开本地 HTML 文件的方式来创建对象
#soup = BeautifulSoup(open('index.html'))
#格式化输出 soup 对象的内容
print(soup.prettify())

print(soup.title)

print(soup.a)

print(soup.name)
# [document] #soup 对象本身比较特殊，它的 name 即为 [document]
print(soup.head.name)
# head #对于其他内部标签，输出的值便为标签本身的名称
print(soup.p.attrs)
# {'class': ['title'], 'name': 'dromouse'}
# 在这里，我们把 p 标签的所有属性打印输出了出来，得到的类型是一个字典。
print(soup.p['class']) # soup.p.get('class')
# ['title'] #还可以利用get方法，传入属性的名称，二者是等价的
soup.p['class'] = "newClass"
print(soup.p) # 可以对这些属性和内容等等进行修改
# <p class="newClass" name="dromouse"><b>The Dormouse's story</b></p>
del soup.p['class'] # 还可以对这个属性进行删除
print(soup.p)
# <p name="dromouse"><b>The Dormouse's story</b></p>

# 遍历文档树
'''
1. 直接子节点 ：.contents .children 属性
.content
tag 的 .content 属性可以将tag的子节点以列表的方式输出
'''

print(soup.head.contents)
#[<title>The Dormouse's story</title>]
# 输出方式为列表，我们可以用列表索引来获取它的某一个元素

print(soup.head.contents[0])
#<title>The Dormouse's story</title>

# 2. .children
print(soup.head.children)
#<listiterator object at 0x7f71457f5710>
for child in  soup.body.children:
    print(child)
# 所有子孙节点: .descendants 属性
for child in soup.descendants:
    print(child)

# 搜索文档树
# A.传字符串
soup.find_all('b')
# [<b>The Dormouse's story</b>]
print(soup.find_all('a'))
#[<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>, <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
# B.传正则表达式
# 如果传入正则表达式作为参数,Beautiful Soup会通过正则表达式的 match() 来匹配内容.下面例子中找出所有以b开头的标签,这表示< body >和< b >标签都应该被找到
import re
for tag in soup.find_all(re.compile("^b")):
    print(tag.name)
# body
# 如果传入列表参数,Beautiful Soup会将与列表中任一元素匹配的内容返回.下面代码找到文档中所有< a >标签和< b >标签:
soup.find_all(["a", "b"])
# [<b>The Dormouse's story</b>,
#  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
# keyword 参数
soup.find_all(class_ = "sister")
#[<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>, <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
soup.find_all(id='link2')
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]
'''
3. CSS选择器
这就是另一种与 find_all 方法有异曲同工之妙的查找方法，也是返回所有匹配结果的列表。

写 CSS 时，标签名不加任何修饰，类名前加.，id名前加#
在这里我们也可以利用类似的方法来筛选元素，用到的方法是 soup.select()，返回类型是 list
'''
print(soup.select('title'))
#[<title>The Dormouse's story</title>]
print(soup.select('a'))
#[<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>, <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
print(soup.select('b'))
#[<b>The Dormouse's story</b>]
#（2）通过类名查找
print(soup.select('.sister'))
#[<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>, <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
#（3）通过 id 名查找
print(soup.select('#link1'))
#[<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]
#（4）组合查找
# 组合查找即和写 class 文件时，标签名与类名、id名进行的组合原理是一样的，例如查找 p 标签中，id 等于 link1的内容，二者需要用空格分开

print(soup.select('p #link1'))
#[<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]
# 直接子标签查找，则使用 > 分隔

print(soup.select("head > title"))
#[<title>The Dormouse's story</title>]
#（5）属性查找
# 查找时还可以加入属性元素，属性需要用中括号括起来，注意属性和标签属于同一节点，所以中间不能加空格，否则会无法匹配到。

print(soup.select('a[class="sister"]'))
#[<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>, <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
print(soup.select('a[href="http://example.com/elsie"]'))
#[<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]
# 同样，属性仍然可以与上述查找方式组合，不在同一节点的空格隔开，同一节点的不加空格

print(soup.select('p a[href="http://example.com/elsie"]'))
#[<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]
# (6) 获取内容
# 以上的 select 方法返回的结果都是列表形式，可以遍历形式输出，然后用 get_text() 方法来获取它的内容。

soup = BeautifulSoup(html, 'lxml')
print(type(soup.select('title')))
print(soup.select('title')[0].get_text())
for title in soup.select('title'):
    print(title.get_text())
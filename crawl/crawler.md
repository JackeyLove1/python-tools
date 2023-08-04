# Title
## Basic
### RE
.  matches any character except a newline("\n")
^  Matches the start of the string
$  Matches the end of the string or just before the newline at the end of the string
match 方法：从起始位置开始查找，一次匹配
search 方法：从任何位置开始查找，一次匹配
findall 方法：全部匹配，返回列表
finditer 方法：全部匹配，返回迭代器
split 方法：分割字符串，返回列表
sub 方法：替换
当匹配成功时返回一个 Match 对象，其中：
group([group1, …]) 方法用于获得一个或多个分组匹配的字符串，当要获得整个匹配的子串时，可直接使用 group() 或 group(0)；
start([group]) 方法用于获取分组匹配的子串在整个字符串中的起始位置（子串第一个字符的索引），参数默认值为 0；
end([group]) 方法用于获取分组匹配的子串在整个字符串中的结束位置（子串最后一个字符的索引+1），参数默认值为 0；
span([group]) 方法返回 (start(group), end(group))。

贪婪模式：在整个表达式匹配成功的前提下，尽可能多的匹配 ( * )；
非贪婪模式：在整个表达式匹配成功的前提下，尽可能少的匹配 ( ? )；
Python里数量词默认是贪婪的。

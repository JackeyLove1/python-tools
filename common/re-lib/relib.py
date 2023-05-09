'''
compile(pattern, flags=0)	编译正则表达式返回正则表达式对象
match(pattern, string, flags=0)	用正则表达式匹配字符串 成功返回匹配对象 否则返回None
search(pattern, string, flags=0)	搜索字符串中第一次出现正则表达式的模式 成功返回匹配对象 否则返回None
split(pattern, string, maxsplit=0, flags=0)	用正则表达式指定的模式分隔符拆分字符串 返回列表
sub(pattern, repl, string, count=0, flags=0)	用指定的字符串替换原字符串中与正则表达式匹配的模式 可以用count指定替换的次数
fullmatch(pattern, string, flags=0)	match函数的完全匹配（从字符串开头到结尾）版本
findall(pattern, string, flags=0)	查找字符串所有与正则表达式匹配的模式 返回字符串的列表
finditer(pattern, string, flags=0)	查找字符串所有与正则表达式匹配的模式 返回一个迭代器
purge()	清除隐式编译的正则表达式的缓存
re.I / re.IGNORECASE	忽略大小写匹配标记
re.M / re.MULTILINE	多行匹配标记
'''
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

import re


def main():
    username = "usernameJacky001"
    qq = "123456789"
    m1 = re.match(r'^[0-9a-zA-Z]{6,20}', username)
    if not m1:
        print("请输入有效的用户名")
    m2 = re.match(r'^[1-9]\d{4,11}$', qq)
    if not m2:
        print('请输入QQ号')
    if m1 and m2:
        print('right information: %s:%s' % (username, qq))

    sentence ="apple banana cat dog"
    purified = re.sub('apple | dog', '*', sentence, flags=re.I)
    print(purified)

    poem = '窗前明月光，疑是地上霜。举头望明月，低头思故乡。'
    sentence2 = re.split(r'[，。,.]', poem)
    print(sentence2)

if __name__ == '__main__':
    main()

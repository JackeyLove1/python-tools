'''
'r'	读取 （默认）
'w'	写入（会先截断之前的内容）
'x'	写入，如果文件已经存在会产生异常
'a'	追加，将内容写入到已有文件的末尾
'b'	二进制模式
't'	文本模式（默认）
'+'	更新（既可以读又可以写）
'''

import time


def main():
    with open('test.txt', 'r', encoding='utf-8') as f:
        print(f.read())
    print()

    with open('test.txt', mode='r') as f:
        for line in f:
            print(line, end='')
    print()

    with open('test.txt') as f:
        lines = f.readline()
        print(lines)


if __name__ == '__main__':
    main()

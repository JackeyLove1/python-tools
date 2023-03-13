import jieba
import jieba.posseg as pseg
import paddle
def word_print(word):
    for word, flag in words:
        print("{} : {}".format(word, flag))

words = pseg.cut("我爱北京天安门") #jieba默认模式
word_print(words)

paddle.enable_static()
jieba.enable_paddle()
words = pseg.cut("我爱北京天安门",use_paddle=True)
word_print(words)
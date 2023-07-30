import urllib
import chardet
# encoding url query
city=u'北京'.encode('utf-8')
district=u'朝阳区'.encode('utf-8')
bizArea=u'望京'.encode('utf-8')
query={
    'city':city,
    'district':district,
    'bizArea':bizArea
}
print(chardet.detect(query['city']))
from urllib.parse import urlencode
print (urllib.parse.urlencode(query))
city="%E5%8C%97%E4%BA%AC&bizArea=%E6%9C%9B%E4%BA%AC&district=%E6%9C%9B%E4%BA%AC"
print('http://www.lagou.com/jobs/list_Python?px=default&'+urllib.parse.urlencode(query)+'#filterBox')

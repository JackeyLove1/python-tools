import gzip, requests, json, time, urllib


bianma = 'utf-8'

he = {
    "Host": 'api.bilibili.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:67.0) Gecko/20100101 Firefox/67.0',
    'Accept': '*/*',
    'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
    'Accept-Encoding': 'gzip,deflate',
    'Connection': 'keep-alive',
    }

def downAll(bv):
    '''
    传入BV号，下载该BV号下的全部字幕
    '''
    videoList = getVideoList(bv)
    p = 1
    for i in videoList:
        cid = i['cid']
        downSolo(cid, bv, p, i['part'])                         # 下载该P视频的字幕，part是单P视频名。
        print ('【任务总进度：%s/%sP】\n'%(p,len(videoList)))
        p += 1
    print ('\n\n*** 任务完成 ***\n')


def getVideoList(bv):
    '''
    传入BV号，返回该BV号的视频列表
    '''
    url = 'https://api.bilibili.com/x/player/pagelist?bvid=%s' % bv # 创建URL
    response  = requests.get(url,headers=he)                # 获取Json
    videoList = json.loads(response.text)['data']           # Json转换
    print ('请求URL:', url)
    print ('视频目录获取成功！共%sP。\n'%len(videoList))      #汇报
    return videoList


def downSolo(cid, bv, p, part=''):
    '''
    根据cid，下载单P里的全部语言字幕
    '''
    url = 'https://api.bilibili.com/x/player/v2?bvid=%s&cid=%s'%(bv,cid)
    response = requests.get(url,headers=he)
    data = json.loads(response.text)

    subList = data['data']['subtitle']['subtitles']      # 字幕信息列表

    if len(subList) == 0:print('【警告】P%s无字幕！' % p)

    i = 1
    for d in subList:
        lan = d['lan']                          # 字幕的语言编号（ZH JP EN之类）
        name = bv + ' - P' + str(p) + '：' + rep(part) + ' - ' + lan   # 根据BV号、P数、语言，生成字幕文件名
        subUrl = 'http:' + d['subtitle_url']    # 字幕的URL

        # urllib.request.urlretrieve(subUrl,'%s.json' % name)   # 下载json字幕文件
        response = urllib.request.urlopen(subUrl)               # 不下载了，直接获取内容
        if response.info().get('Content-Encoding') == 'gzip':   # 在响应头中获取编码格式
            j = gzip.decompress(response.read())
        else:
            j = response.read()
        jsonToSrt (name, j)

        print ('P%s 第%s种语言下载完成，进度：%s/%s'%(p,i,i,len(subList)))    #报告任务进度（以该P视频的字幕语言数算）

        i += 1
        time.sleep(0.2)


def jsonToSrt(fileName,j):
    '''
    传入文件名和json字幕内容，将json输出为Srt字幕文件
    '''
    data = json.loads(j)['body']
    file = open('%s.srt'%fileName,'w',encoding=bianma)     # 创建srt字幕文件
    i = 1   # Srt字幕的序号计数器
    for d in data:
        f = round(d['from'],3)      # 开始时间 （round(n，3)四舍五入为三位小数）
        t = round(d['to'],3)        # 结束时间
        c = d['content']            # 字幕内容
        ff = time.strftime("%H:%M:%S",time.gmtime(f)) + ',' + miao(f)   # 开始时间，秒数转 时:分:秒 格式，加逗号、毫秒修正为三位
        tt = time.strftime("%H:%M:%S",time.gmtime(t)) + ',' + miao(t)   # 结束时间，处理方式同上

        srt = str(i) + '\n' + ff + ' ' + '-->' + ' ' + tt + '\n' + c + '\n\n'     # 格式化为Srt字幕
        file.write(srt)             # 写入文件
        i += 1                      # 计数器+1

    file.close()
    print ('%s OK.' % fileName)



li = [
    ['/','、'],
    ['\\','、'],
    ['|','、'],
    ['*','X'],
    [':','：'],
    ['?','？'],
    ['<','《'],
    ['>','》'],
    ['\"','“'],
    ['\"','”'],
]

def rep(s=''):
    '''
    根据列表li，去除特殊符号（不能用于文件名的）
    '''
    for i in li:
        s.replace(i[0],i[1])
    return s

def miao(m):
    '''
    修正毫秒为三位
    '''
    m = str(m).partition('.')[2]   #取小数部分
    if len(m)==0:
        m = '000'         #补齐三位小数
    if len(m)==1:
        m = m + '00'
    if len(m)==2:
        m = m + '0'
    return m                           #返回标准三位的毫秒数

# BV1m3411P7TG gzip
# BV13f4y1G7sA 无压缩
# BV18a4y1H73s 很多P


test_bv = "BV18a4y1H73s"
downAll(test_bv)

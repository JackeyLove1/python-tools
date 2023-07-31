# -*- coding:utf-8 -*-
import urllib3
import urllib
from lxml import etree
import chardet
import json
import codecs
def GetTimeByArticle(url):
    request = urllib3.Request(url)
    response = urllib3.urlopen(request)
    resHtml = response.read()
    html = etree.HTML(resHtml)
    time = html.xpath('//span[@class="tail-info"]')[1].text
    print(time)
    return time
def main():
    output = codecs.open('tieba0812.json', 'w', encoding='utf-8')
    for pn in range(0, 250, 50):
        kw = u'网络爬虫'.encode('utf-8')
        url = 'http://tieba.baidu.com/f?kw=' + urllib.quote(kw) + '&ie=utf-8&pn=' + str(pn)
        print(url)
        request = urllib3.Request(url)
        response = urllib3.urlopen(request)
        resHtml = response.read()
        print(resHtml)
        html_dom = etree.HTML(resHtml)
        # print etree.tostring(html_dom)
        html = html_dom
        # site = html.xpath('//li[@data-field]')[0]
        for site in html.xpath('//li[@data-field]'):
            # print etree.tostring(site.xpath('.//a')[0])
            title = site.xpath('.//a')[0].text
            Article_url = site.xpath('.//a')[0].attrib['href']
            reply_date = GetTimeByArticle('http://tieba.baidu.com' + Article_url)
            jieshao = site.xpath('.//*[@class="threadlist_abs threadlist_abs_onlyline "]')[0].text.strip()
            author = site.xpath('.//*[@class="frs-author-name j_user_card "]')[0].text.strip()
            lastName = site.xpath('.//*[@class="frs-author-name j_user_card "]')[1].text.strip()
            print(title, jieshao, Article_url, author, lastName)
            item = {}
            item['title'] = title
            item['author'] = author
            item['lastName'] = lastName
            item['reply_date'] = reply_date
            print(item)
            line = json.dumps(item, ensure_ascii=False)
            print(line)
            print(type(line))
            output.write(line + "\n")
        output.close()
    print('end')
if __name__ == '__main__':
    main()
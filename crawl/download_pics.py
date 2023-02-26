import os
import requests
import concurrent.futures
def download_Pic(title, image_list):
    # 新建文件夹
    os.mkdir(title)
    j = 1    # 下载图片
    for item in image_list:
        filename = '%s/%s.jpg' % (title,str(j))
        print('downloading....%s : NO.%s' % (title,str(j)))
        with open(filename, 'wb') as f:
            img = requests.get(item,headers=header(item)).content
            f.write(img)
        j+=1

def download_all_images(list_page_urls):
    # 获取每一个详情妹纸
    works = len(list_page_urls)
    with concurrent.futures.ThreadPoolExecutor(works) as exector:
        for url in list_page_urls:
            exector.submit(download,url)

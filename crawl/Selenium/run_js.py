# 隐藏百度图片
from selenium import webdriver
driver = webdriver.Chrome()
driver.get("https://www.baidu.com/")
# 给搜索输入框标红的javascript脚本
js = "var q=document.getElementById(\"kw\");q.style.border=\"2px solid red\";"
# 调用给搜索输入框标红js脚本
driver.execute_script(js)
#查看页面快照
driver.save_screenshot("redbaidu.png")
#js隐藏元素，将获取的图片元素隐藏
img = driver.find_element_by_xpath("//*[@id='lg']/img")
driver.execute_script('$(arguments[0]).fadeOut()',img)
# 向下滚动到页面底部
driver.execute_script("$('.scroll_top').click(function(){$('html,body').animate({scrollTop: '0px'}, 800);});")
#查看页面快照
driver.save_screenshot("nullbaidu.png")
driver.quit()

# 模拟滚动条滚动到底部
from selenium import webdriver
import time
driver = webdriver.PhantomJS()
driver.get("https://movie.douban.com/typerank?type_name=剧情&type=11&interval_id=100:90&action=")
# 向下滚动10000像素
js = "document.body.scrollTop=10000"
#js="var q=document.documentElement.scrollTop=10000"
time.sleep(3)
#查看页面快照
driver.save_screenshot("douban.png")
# 执行JS语句
driver.execute_script(js)
time.sleep(10)
#查看页面快照
driver.save_screenshot("newdouban.png")
driver.quit()
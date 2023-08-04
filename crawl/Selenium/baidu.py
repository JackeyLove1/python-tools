# https://selenium-python.readthedocs.io/getting-started.html#selenium-remote-webdriver
# pip install selenium
# install drivers: https://sites.google.com/chromium.org/driver/
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
driver = webdriver.Chrome()
# get方法会一直等到页面被完全加载，然后才会继续程序，通常测试会在这里选择 time.sleep(2)
driver.get("http://www.baidu.com/")
# 获取页面名为 wrapper的id标签的文本内容
data = driver.find_element(By.ID,"wrapper").text
driver.save_screenshot('baidu1.png')
print(driver.page_source)
# id="kw"是百度搜索输入框，输入字符串"长城"
driver.find_element(By.ID, 'kw').send_keys(u"长城")
# id="su"是百度搜索按钮，click() 是模拟点击
driver.find_element(By.ID, 'su').click()
# 获取新的页面快照
driver.save_screenshot("长城.png")
# 打印网页渲染后的源代码
print(driver.page_source)
# 获取当前页面Cookie
print(driver.get_cookies())
# 调用键盘按键操作时需要引入的Keys包
from selenium.webdriver.common.keys import Keys
# ctrl+a 全选输入框内容
driver.find_element(By.ID, "kw").send_keys(Keys.CONTROL,'a')
# ctrl+x 剪切输入框内容
driver.find_element(By.ID, "kw").send_keys(Keys.CONTROL,'x')
# 输入框重新输入内容
driver.find_element(By.ID, "kw").send_keys("byd")
# 模拟Enter回车键
driver.find_element(By.ID, "su").send_keys(Keys.RETURN)
# 清除输入框内容
driver.find_element(By.ID, "kw").clear()
# 生成新的页面快照
driver.save_screenshot("itcast.png")
# 获取当前url
print(driver.current_url)
# 获取id标签值
element = driver.find_element(By.ID, "passwd-id")
# 获取name标签值
element = driver.find_element(By.NAME, "user-name")
# 获取标签名值
element = driver.find_elements(By.TAG_NAME, "input")
# 也可以通过XPath来匹配
element = driver.find_element(By.XPATH, "//input[@id='passwd-id']")

# 关闭当前页面，如果只有一个页面，会关闭浏览器
# driver.close()
# 关闭浏览器
driver.quit()

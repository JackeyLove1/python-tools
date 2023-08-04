# 导入 ActionChains 类
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver

driver = webdriver.Chrome()
# get方法会一直等到页面被完全加载
# 鼠标移动到 ac 位置
ac = driver.find_element(By.XPATH, 'element')
ActionChains(driver).move_to_element(ac).perform()
# 在 ac 位置单击
ac = driver.find_element(By.XPATH, "elementA")
ActionChains(driver).move_to_element(ac).click(ac).perform()
# 在 ac 位置双击
ac = driver.find_element(By.XPATH, "elementB")
ActionChains(driver).move_to_element(ac).double_click(ac).perform()
# 在 ac 位置右击
ac = driver.find_element(By.XPATH, "elementC")
ActionChains(driver).move_to_element(ac).context_click(ac).perform()
# 在 ac 位置左键单击hold住
ac = driver.find_element(By.XPATH, 'elementF')
ActionChains(driver).move_to_element(ac).click_and_hold(ac).perform()
# 将 ac1 拖拽到 ac2 位置
ac1 = driver.find_element(By.XPATH, 'elementD')
ac2 = driver.find_element(By.XPATH, 'elementE')
ActionChains(driver).drag_and_drop(ac1, ac2).perform()

# fullfil the form
# 导入 Select 类
from selenium.webdriver.support.ui import Select
# 找到 name 的选项卡
select = Select(driver.find_element(By.NAME, 'status'))
#
select.select_by_index(1)
select.select_by_value("0")
select.select_by_visible_text(u"未审核")

# alert
alert = driver.switch_to_alert()

# 操作页面的前进和后退功能：
driver.forward()     #前进
driver.back()        # 后退

# 获取页面每个Cookies值，用法如下
for cookie in driver.get_cookies():
    print("%s=%s;" % (cookie['name'], cookie['value']))
# 删除Cookies，用法如下
# By name
driver.delete_cookie("BAIDUID")
# all
driver.delete_all_cookies()

# wait
# explicit
from selenium import webdriver
from selenium.webdriver.common.by import By
# WebDriverWait 库，负责循环等待
from selenium.webdriver.support.ui import WebDriverWait
# expected_conditions 类，负责条件出发
from selenium.webdriver.support import expected_conditions as EC
driver = webdriver.Chrome()
driver.get("http://www.xxxxx.com/loading")
try:
    # 每隔10秒查找页面元素 id="myDynamicElement"，直到出现则返回
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "myDynamicElement"))
    )
finally:
    driver.quit()
# implicit
from selenium import webdriver
driver = webdriver.Chrome()
driver.implicitly_wait(10) # seconds
driver.get("http://www.xxxxx.com/loading")
myDynamicElement = driver.find_element(By.ID, "myDynamicElement")

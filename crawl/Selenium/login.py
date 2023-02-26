username = WAIT.until(EC.presence_of_element_located((By.CSS_SELECTOR, "帐号的selector")))
password = WAIT.until(EC.presence_of_element_located((By.CSS_SELECTOR, "密码的selector")))
submit = WAIT.until(EC.element_to_be_clickable((By.XPATH, '按钮的xpath')))

username.send_keys('你的帐号')
password.send_keys('你的密码')submit.click()

cookies = webdriver.get_cookies()
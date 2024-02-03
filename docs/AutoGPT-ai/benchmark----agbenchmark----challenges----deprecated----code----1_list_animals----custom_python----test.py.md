# `.\AutoGPT\benchmark\agbenchmark\challenges\deprecated\code\1_list_animals\custom_python\test.py`

```py
# 导入必要的库
import os
import time

# 导入 Selenium 相关模块
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 获取当前文件所在目录的路径
current_directory = os.path.dirname(current_path)
# 构建 HTML 文件的路径
file_path = f"file://{current_directory}/animal_list.html"

# 创建 Chrome 浏览器实例
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1024x768")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

# 打开 HTML 文件
driver.get(file_path)

# 等待最多 10 秒，直到 "dog" 元素可用
wait = WebDriverWait(driver, 10)
dog_li = wait.until(EC.presence_of_element_located((By.ID, "dog")))

# 点击 "dog" 列表项
dog_li.click()

# 查找 "info" div 元素并获取其文本内容
info_div = driver.find_element(By.ID, "info")
info_text = info_div.text

# 断言文本内容是否符合预期
assert info_text == "Dogs are known as man's best friend!"

# 输出测试通过信息
print(" passed!")

# 等待 5 秒
time.sleep(5)

# 关闭浏览器窗口
driver.quit()
```
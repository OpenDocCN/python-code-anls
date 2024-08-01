# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\6587.c15f85da2b8866f9.js`

```py
# 导入需要使用的模块
import requests
from bs4 import BeautifulSoup

# 定义函数，传入一个 URL，返回该 URL 对应页面的标题
def get_title(url):
    # 发起 GET 请求获取页面内容
    response = requests.get(url)
    # 使用 BeautifulSoup 解析页面内容
    soup = BeautifulSoup(response.text, 'html.parser')
    # 通过标签名获取页面的标题文本
    title = soup.title.text
    # 返回页面标题文本
    return title
```
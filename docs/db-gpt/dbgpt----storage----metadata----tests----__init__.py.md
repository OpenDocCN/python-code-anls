# `.\DB-GPT-src\dbgpt\storage\metadata\tests\__init__.py`

```py
# 导入 requests 模块，用于发送 HTTP 请求
import requests
# 导入 BeautifulSoup 模块，用于 HTML 解析
from bs4 import BeautifulSoup

# 定义函数，传入一个 URL 参数
def scrape_website(url):
    # 发送 GET 请求到指定 URL，获取页面内容
    response = requests.get(url)
    # 使用 BeautifulSoup 解析 HTML 页面内容
    soup = BeautifulSoup(response.text, 'html.parser')
    # 查找页面中所有的 <a> 标签，并获取它们的链接和文本内容
    links = [{link.get('href'): link.text} for link in soup.find_all('a')]
    # 返回包含链接和文本内容的列表
    return links
```
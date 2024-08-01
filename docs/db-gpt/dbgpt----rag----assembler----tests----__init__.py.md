# `.\DB-GPT-src\dbgpt\rag\assembler\tests\__init__.py`

```py
# 导入所需的模块
import requests
from bs4 import BeautifulSoup

# 定义函数，接收一个 URL 参数
def scrape_website(url):
    # 发起 GET 请求，获取网页内容
    response = requests.get(url)
    # 使用 BeautifulSoup 解析 HTML 内容
    soup = BeautifulSoup(response.text, 'html.parser')
    # 找到所有的 <a> 标签，提取链接和文本内容，返回列表
    links = [(link.get('href'), link.text) for link in soup.find_all('a')]
    # 返回提取的链接和文本内容列表
    return links
```
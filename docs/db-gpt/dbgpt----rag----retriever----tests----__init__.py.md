# `.\DB-GPT-src\dbgpt\rag\retriever\tests\__init__.py`

```py
# 导入所需模块
import requests
from bs4 import BeautifulSoup

# 定义函数，接收网页 URL 作为参数
def scrape_website(url):
    # 发送 HTTP 请求，获取网页内容
    response = requests.get(url)
    # 使用 BeautifulSoup 解析 HTML 内容
    soup = BeautifulSoup(response.content, 'html.parser')
    # 找到所有的超链接标签
    links = soup.find_all('a')
    # 提取所有超链接的文本和链接地址，存储在字典中
    link_dict = {link.text: link.get('href') for link in links}
    # 返回结果字典
    return link_dict
```
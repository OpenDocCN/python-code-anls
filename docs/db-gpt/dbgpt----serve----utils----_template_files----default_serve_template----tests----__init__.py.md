# `.\DB-GPT-src\dbgpt\serve\utils\_template_files\default_serve_template\tests\__init__.py`

```py
# 导入requests库，用于进行HTTP请求
import requests
# 导入BeautifulSoup库，用于HTML解析
from bs4 import BeautifulSoup

# 定义函数，接收URL作为参数，返回解析后的HTML文档
def scrape_website(url):
    # 发送GET请求获取网页内容
    response = requests.get(url)
    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(response.content, 'html.parser')
    # 查找所有<p>标签，并提取它们的文本内容
    paragraphs = soup.find_all('p')
    # 将所有段落文本内容存储在列表中
    text_list = [p.get_text() for p in paragraphs]
    # 返回文本列表作为结果
    return text_list

# 调用函数，传入URL参数，获取并打印解析后的文本列表
url = 'http://example.com'
scraped_data = scrape_website(url)
print(scraped_data)
```
# `.\pytorch\torch\_inductor\codegen\cuda\cutlass_lib_extensions\__init__.py`

```
# 导入所需的模块：requests 用于发送 HTTP 请求，BeautifulSoup 用于解析 HTML 内容
import requests
from bs4 import BeautifulSoup

# 定义函数 get_links，接收一个 URL 参数
def get_links(url):
    # 发送 GET 请求到指定的 URL，获取页面内容
    response = requests.get(url)
    # 使用 BeautifulSoup 解析页面内容，指定解析器为 lxml
    soup = BeautifulSoup(response.text, 'lxml')
    # 在解析后的页面内容中查找所有的 <a> 标签，提取链接的 href 属性，并存储在列表中
    links = [link.get('href') for link in soup.find_all('a')]
    # 返回提取到的链接列表
    return links

# 调用函数 get_links，并传入参数 'https://example.com'
urls = get_links('https://example.com')
# 打印获取到的链接列表
print(urls)
```
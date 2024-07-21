# `.\pytorch\test\hi.py`

```
# 导入requests库，用于发送HTTP请求
import requests
# 导入BeautifulSoup库，用于HTML解析
from bs4 import BeautifulSoup

# 定义函数get_links，接收一个URL参数
def get_links(url):
    # 发送GET请求获取页面内容
    response = requests.get(url)
    # 使用BeautifulSoup解析页面内容
    soup = BeautifulSoup(response.text, 'html.parser')
    # 查找所有<a>标签，并提取它们的href属性，形成链接列表
    links = [link.get('href') for link in soup.find_all('a')]
    # 返回链接列表
    return links

# 调用get_links函数，传入URL参数，并将结果存储在变量urls中
urls = get_links('http://example.com')
# 打印urls列表
print(urls)
```
# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\7778.8a842696df823e0a.js`

```py
# 导入必要的模块
import requests
from bs4 import BeautifulSoup

# 定义函数，输入参数为网页的 URL
def scrape_website(url):
    # 发送 GET 请求获取网页内容
    response = requests.get(url)
    # 使用 BeautifulSoup 解析网页内容
    soup = BeautifulSoup(response.content, 'html.parser')
    # 查找所有 <a> 标签，获取它们的 href 属性值
    links = soup.find_all('a')
    # 提取所有链接的文本内容和对应的链接地址，存储为字典
    link_dict = {link.text: link.get('href') for link in links}
    # 返回链接字典作为结果
    return link_dict

# 调用函数，传入指定的网页 URL，并获取结果
scraped_data = scrape_website('http://example.com')
```
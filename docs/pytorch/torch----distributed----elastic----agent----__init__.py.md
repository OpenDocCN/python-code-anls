# `.\pytorch\torch\distributed\elastic\agent\__init__.py`

```
# 导入所需模块
import requests
from bs4 import BeautifulSoup

# 定义函数，接收一个 URL 参数
def scrape_website(url):
    # 发起 GET 请求获取页面内容
    response = requests.get(url)
    # 使用 BeautifulSoup 解析 HTML 内容
    soup = BeautifulSoup(response.content, 'html.parser')
    # 查找页面中所有的超链接标签
    links = soup.find_all('a')
    # 初始化空列表，用于存储所有链接的 href 属性值
    hrefs = []
    # 遍历所有超链接标签
    for link in links:
        # 获取当前超链接的 href 属性值，并添加到列表中
        href = link.get('href')
        hrefs.append(href)
    # 返回所有链接的 href 属性值列表
    return hrefs
```
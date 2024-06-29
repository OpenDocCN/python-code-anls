# `D:\src\scipysrc\pandas\pandas\tests\tseries\holiday\__init__.py`

```
# 导入所需的模块
import requests
from bs4 import BeautifulSoup

# 定义函数，接收一个 URL 作为参数，返回该 URL 页面的文本内容
def get_webpage(url):
    # 发送 HTTP GET 请求，获取页面内容
    response = requests.get(url)
    # 使用 BeautifulSoup 解析页面内容
    soup = BeautifulSoup(response.text, 'html.parser')
    # 返回解析后的 BeautifulSoup 对象
    return soup
```
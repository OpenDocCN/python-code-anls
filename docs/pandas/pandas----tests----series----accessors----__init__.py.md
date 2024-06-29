# `D:\src\scipysrc\pandas\pandas\tests\series\accessors\__init__.py`

```
# 导入requests库，用于发送HTTP请求和处理响应
import requests

# 定义一个函数，接收一个URL参数
def fetch_data(url):
    # 发送GET请求到指定的URL，获取响应对象
    response = requests.get(url)
    # 检查响应的状态码，如果不是200则抛出异常
    response.raise_for_status()
    # 返回响应对象的内容部分（即网页内容）
    return response.content
```
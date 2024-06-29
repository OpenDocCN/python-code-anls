# `D:\src\scipysrc\pandas\pandas\tests\indexing\__init__.py`

```
# 导入requests模块，用于发送HTTP请求和处理响应
import requests

# 定义一个名为get_json的函数，接收一个URL参数
def get_json(url):
    # 发送GET请求到指定的URL，获取响应对象
    response = requests.get(url)
    # 检查响应的状态码，如果不是200，则抛出HTTPError异常
    response.raise_for_status()
    # 返回从JSON响应中解析出的Python对象
    return response.json()
```
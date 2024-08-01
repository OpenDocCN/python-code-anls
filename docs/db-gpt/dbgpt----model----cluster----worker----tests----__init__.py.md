# `.\DB-GPT-src\dbgpt\model\cluster\worker\tests\__init__.py`

```py
# 导入所需的模块：requests 用于发送 HTTP 请求
import requests

# 定义一个函数，名为 fetch_url，接受一个参数 url
def fetch_url(url):
    # 使用 requests 模块发送一个 GET 请求到指定的 URL，并获取响应对象
    response = requests.get(url)
    # 检查响应的状态码，如果不是 200 OK，则抛出异常
    response.raise_for_status()
    # 返回响应对象的内容作为结果
    return response.content
```
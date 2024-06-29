# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\__init__.py`

```
# 导入 requests 模块，用于发送 HTTP 请求
import requests

# 定义函数 fetch_url_data，接收一个 URL 参数
def fetch_url_data(url):
    # 发送 GET 请求获取 URL 对应的响应对象
    response = requests.get(url)
    # 检查响应状态码是否为 200 OK
    if response.status_code == 200:
        # 返回响应内容的文本形式
        return response.text
    else:
        # 如果响应状态码不是 200，则返回空字符串
        return ''

# 调用 fetch_url_data 函数，传入参数 'https://example.com'
fetch_url_data('https://example.com')
```
# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\__init__.py`

```
# 导入所需模块：requests 用于发送 HTTP 请求，json 用于处理 JSON 格式数据
import requests
import json

# 定义一个函数 send_request，接收一个 URL 参数，发送 GET 请求，并返回响应内容的 JSON 解析结果
def send_request(url):
    # 发送 GET 请求到指定的 URL，获取响应对象
    response = requests.get(url)
    # 使用 json() 方法解析响应内容，返回 JSON 格式的数据
    return response.json()

# 调用 send_request 函数，传入指定的 URL，并将返回的 JSON 数据赋给变量 data
data = send_request('https://api.example.com/data')

# 打印输出变量 data 的内容
print(data)
```
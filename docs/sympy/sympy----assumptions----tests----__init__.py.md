# `D:\src\scipysrc\sympy\sympy\assumptions\tests\__init__.py`

```
# 导入 requests 库，用于发送 HTTP 请求
import requests

# 设置目标 URL
url = 'https://api.example.com/data'

# 发送 GET 请求到目标 URL，并将响应对象存储在变量 response 中
response = requests.get(url)

# 检查响应的状态码，如果状态码为 200 表示请求成功
if response.status_code == 200:
    # 提取 JSON 格式的响应数据
    data = response.json()
    # 打印获取的数据内容
    print(data)
else:
    # 如果请求失败，打印错误信息和状态码
    print(f'Request failed with status code {response.status_code}')
```
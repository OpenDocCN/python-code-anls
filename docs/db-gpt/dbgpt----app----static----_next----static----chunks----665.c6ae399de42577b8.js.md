# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\665.c6ae399de42577b8.js`

```py
# 导入名为 requests 的库，用于发起 HTTP 请求
import requests

# 设置 URL 变量，指定要请求的网页地址
url = 'https://api.github.com/user'

# 设置 headers 变量，包含请求时发送的 HTTP 头信息，指定 User-Agent 为 Python-urllib/3.9
headers = {'User-Agent': 'Python-urllib/3.9'}

# 使用 requests 库发送 GET 请求到指定 URL，传递 headers 变量作为请求头信息
response = requests.get(url, headers=headers)

# 打印响应对象的状态码
print(response.status_code)

# 打印响应对象的 JSON 数据
print(response.json())
```
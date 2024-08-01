# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\249.8c791ea49f6fb4e4.js`

```py
# 导入所需的模块
import requests

# 定义函数 `get_random_quote`，用于获取随机名言，并返回该名言的文本内容
def get_random_quote():
    # API 的 URL 地址，用于获取随机名言
    url = 'https://api.quotable.io/random'
    # 发送 GET 请求，获取响应对象
    response = requests.get(url)
    # 从响应对象中解析出 JSON 格式的数据，并将其转换为 Python 字典格式
    data = response.json()
    # 从返回的数据中获取名言的文本内容
    quote = data['content']
    # 返回获取到的名言文本内容
    return quote
```
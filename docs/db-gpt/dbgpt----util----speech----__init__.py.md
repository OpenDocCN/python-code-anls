# `.\DB-GPT-src\dbgpt\util\speech\__init__.py`

```py
# 导入所需的模块：requests 用于 HTTP 请求，json 用于处理 JSON 数据
import requests
import json

# 定义函数 fetch_data，接收一个 URL 参数
def fetch_data(url):
    # 发起 GET 请求，获取数据
    response = requests.get(url)
    # 将响应的 JSON 数据解析为 Python 字典
    data = response.json()
    # 返回解析后的数据字典
    return data
```
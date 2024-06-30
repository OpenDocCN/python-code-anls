# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\__init__.py`

```
# 导入必要的模块：requests 用于发送 HTTP 请求，json 用于处理 JSON 格式数据
import requests
import json

# 定义一个函数，接收一个 URL 参数，发送 GET 请求并返回 JSON 格式的响应内容
def fetch_json(url):
    # 发送 GET 请求到指定的 URL，并将响应对象存储在变量 response 中
    response = requests.get(url)
    # 如果响应状态码为 200，表示请求成功
    if response.status_code == 200:
        # 使用 json 模块解析响应内容，将其转换为 Python 字典或列表对象
        data = response.json()
        # 返回解析后的 JSON 数据
        return data
    else:
        # 若请求失败，则打印错误信息并返回空值
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        return None

# 示例用法：调用 fetch_json 函数，传入 URL 并获取 JSON 数据
json_data = fetch_json('https://api.example.com/data')
```
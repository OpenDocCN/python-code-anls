# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\pages\agent-baf4571810cdd046.js`

```py
# 导入requests库，用于发送HTTP请求
import requests

# 定义函数fetch_data，接收一个URL参数
def fetch_data(url):
    # 发送GET请求到指定的URL，获取响应对象
    response = requests.get(url)
    
    # 如果响应状态码为200，表示请求成功
    if response.status_code == 200:
        # 返回响应内容的JSON解析结果
        return response.json()
    else:
        # 打印错误信息并返回空字典
        print(f"Error fetching data. Status code: {response.status_code}")
        return {}

# 示例用法：调用fetch_data函数并传入URL参数
data = fetch_data('https://api.example.com/data')
```
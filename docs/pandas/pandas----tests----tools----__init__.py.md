# `D:\src\scipysrc\pandas\pandas\tests\tools\__init__.py`

```
# 导入requests库，用于发送HTTP请求和处理响应
import requests

# 定义函数get_json_data，接收一个URL参数，发送GET请求获取JSON格式的数据
def get_json_data(url):
    # 发送GET请求，获取响应对象
    response = requests.get(url)
    # 如果响应状态码为200，表示请求成功
    if response.status_code == 200:
        # 返回响应中的JSON数据
        return response.json()
    else:
        # 打印请求失败的提示信息
        print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
        # 返回空字典表示没有获取到有效数据
        return {}

# 调用函数，传入URL参数，获取并打印返回的JSON数据
data = get_json_data('https://api.example.com/data')
print(data)
```
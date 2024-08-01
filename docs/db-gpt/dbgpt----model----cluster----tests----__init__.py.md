# `.\DB-GPT-src\dbgpt\model\cluster\tests\__init__.py`

```py
# 导入模块requests用于发送HTTP请求
import requests

# 定义函数fetch_data，接收一个url参数
def fetch_data(url):
    # 发送GET请求到指定的URL，并获取响应对象
    response = requests.get(url)
    # 如果响应状态码为200，表示请求成功
    if response.status_code == 200:
        # 返回响应的文本内容
        return response.text
    # 如果响应状态码不是200，打印出错信息并返回空字符串
    else:
        print(f"Error fetching data from {url}. Status code: {response.status_code}")
        return ''

# 示例用法
data = fetch_data('https://api.example.com/data')
print(data)
```
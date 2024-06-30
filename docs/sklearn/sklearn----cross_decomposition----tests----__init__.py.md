# `D:\src\scipysrc\scikit-learn\sklearn\cross_decomposition\tests\__init__.py`

```
# 导入requests库，用于发送HTTP请求
import requests

# 定义函数fetch_data，接收一个url参数，用来获取指定URL的数据
def fetch_data(url):
    # 发送GET请求到指定的URL，并将响应对象存储在response变量中
    response = requests.get(url)
    # 检查响应状态码是否为200，表示请求成功
    if response.status_code == 200:
        # 如果请求成功，则返回响应的文本内容
        return response.text
    else:
        # 如果请求失败，则打印出错误信息并返回空字符串
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        return ''

# 调用fetch_data函数，传入一个URL参数，并将返回的数据打印出来
print(fetch_data('https://jsonplaceholder.typicode.com/posts/1'))
```
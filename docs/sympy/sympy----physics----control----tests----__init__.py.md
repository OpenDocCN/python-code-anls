# `D:\src\scipysrc\sympy\sympy\physics\control\tests\__init__.py`

```
# 导入模块requests，用于发送HTTP请求
import requests

# 定义函数fetch_data，接收一个url参数，用于从指定URL获取数据
def fetch_data(url):
    # 发送GET请求到指定URL，获取响应对象
    response = requests.get(url)
    # 如果响应状态码不等于200，即请求不成功
    if response.status_code != 200:
        # 抛出异常并显示错误信息
        raise Exception(f"Failed to fetch data from {url}. Status code: {response.status_code}")
    # 返回响应对象的文本内容
    return response.text

# 调用fetch_data函数，传入URL参数，打印获取到的数据文本内容
print(fetch_data('https://example.com/data'))
```
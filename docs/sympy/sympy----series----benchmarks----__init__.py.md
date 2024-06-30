# `D:\src\scipysrc\sympy\sympy\series\benchmarks\__init__.py`

```
# 导入requests库，用于发起HTTP请求
import requests

# 定义函数fetch_data，接收一个URL参数
def fetch_data(url):
    # 使用requests库发起GET请求，获取URL对应的响应对象
    response = requests.get(url)
    # 如果响应状态码为200，表示请求成功
    if response.status_code == 200:
        # 返回响应对象的文本内容
        return response.text
    else:
        # 若请求失败，则打印错误信息
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        # 返回空字符串表示失败
        return ""
```
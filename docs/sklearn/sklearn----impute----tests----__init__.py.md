# `D:\src\scipysrc\scikit-learn\sklearn\impute\tests\__init__.py`

```
# 导入 requests 模块，用于发送 HTTP 请求
import requests

# 定义一个函数，名为 fetch_data，接受一个参数 url
def fetch_data(url):
    # 使用 requests 模块发送 GET 请求，获取 URL 返回的响应对象
    response = requests.get(url)
    # 如果响应状态码为 200，表示请求成功
    if response.status_code == 200:
        # 返回响应对象的 JSON 数据
        return response.json()
    else:
        # 若请求失败，则打印错误信息并返回空值
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        return None
```
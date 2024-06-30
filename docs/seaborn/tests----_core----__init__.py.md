# `D:\src\scipysrc\seaborn\tests\_core\__init__.py`

```
# 导入 requests 库，用于发送 HTTP 请求
import requests

# 定义一个名为 fetch_data 的函数，接受一个 URL 参数
def fetch_data(url):
    # 发送 GET 请求到指定的 URL，获取响应对象
    response = requests.get(url)
    # 检查响应状态码是否为 200，表示请求成功
    if response.status_code == 200:
        # 返回响应内容的文本表示
        return response.text
    else:
        # 如果请求失败，打印错误信息并返回空字符串
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        return ''
```
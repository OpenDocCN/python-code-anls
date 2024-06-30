# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\__init__.py`

```
# 导入requests库，用于发送HTTP请求
import requests

# 定义一个名为get_json的函数，接收一个url参数
def get_json(url):
    # 使用requests库发送GET请求，获取URL对应的JSON数据
    response = requests.get(url)
    # 如果响应状态码不是200，则抛出HTTPError异常
    response.raise_for_status()
    # 返回从JSON响应中解析出的Python对象
    return response.json()
```
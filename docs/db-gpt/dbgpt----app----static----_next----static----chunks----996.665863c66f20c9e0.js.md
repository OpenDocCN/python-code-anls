# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\996.665863c66f20c9e0.js`

```py
# 导入所需的模块：os（操作系统接口）、re（正则表达式操作）、json（JSON编码和解码）、requests（发送HTTP请求）
import os
import re
import json
import requests

# 定义一个函数，接收一个URL作为参数，返回该URL的域名
def get_domain(url):
    # 使用正则表达式从URL中提取域名部分
    domain = re.search(r"https?://([^/?]+)", url).group(1)
    # 返回提取出的域名
    return domain

# 使用requests模块发送一个GET请求到指定的URL，并返回响应对象
response = requests.get('https://api.github.com')

# 如果响应状态码为200（请求成功），则打印响应内容
if response.status_code == 200:
    # 将响应内容解析为JSON格式的数据，并打印出来
    data = response.json()
    print(json.dumps(data, indent=2))
```
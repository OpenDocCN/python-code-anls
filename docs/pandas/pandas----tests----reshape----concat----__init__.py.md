# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\__init__.py`

```
# 导入必要的模块：requests 用于发送 HTTP 请求，os 用于操作文件路径
import requests
import os

# 定义函数 download_file，接收文件 URL 和保存路径作为参数
def download_file(url, save_path):
    # 发送 GET 请求获取文件内容，并保存到 response 变量中
    response = requests.get(url)
    # 打开文件保存路径，以二进制写入模式创建文件对象
    with open(save_path, 'wb') as f:
        # 将获取的文件内容写入到本地文件中
        f.write(response.content)

# 调用 download_file 函数，下载指定 URL 的文件到本地路径
download_file('http://example.com/sample.zip', '/path/to/save/sample.zip')
```
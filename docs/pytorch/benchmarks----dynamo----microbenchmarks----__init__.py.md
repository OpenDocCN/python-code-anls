# `.\pytorch\benchmarks\dynamo\microbenchmarks\__init__.py`

```py
# 导入所需的模块：requests 用于 HTTP 请求操作，os 用于系统相关操作
import requests
import os

# 定义一个函数 download_file，接受两个参数：url（文件的远程地址）、save_path（本地保存路径）
def download_file(url, save_path):
    # 发起 GET 请求，获取远程文件的内容
    response = requests.get(url, stream=True)
    # 打开本地文件，以二进制写入模式打开，准备写入远程文件内容
    with open(save_path, 'wb') as f:
        # 遍历响应内容的每个数据块，每次 1024 字节
        for chunk in response.iter_content(chunk_size=1024):
            # 将当前数据块写入本地文件
            if chunk:
                f.write(chunk)
    # 返回文件保存路径，表示文件下载完成
    return save_path
```
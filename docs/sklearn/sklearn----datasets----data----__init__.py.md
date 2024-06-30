# `D:\src\scipysrc\scikit-learn\sklearn\datasets\data\__init__.py`

```
# 导入必要的模块：requests 用于发送 HTTP 请求
import requests

# 定义一个函数 download_file，接收两个参数：url（下载文件的地址）、local_filename（本地保存的文件名）
def download_file(url, local_filename):
    # 发送 GET 请求到指定的 URL，将响应保存在变量 r 中
    r = requests.get(url, stream=True)
    # 打开本地文件，以二进制写入模式创建文件对象，保存为变量 f
    with open(local_filename, 'wb') as f:
        # 遍历通过流方式读取的响应内容块
        for chunk in r.iter_content(chunk_size=1024):
            # 将读取的每个数据块写入到本地文件中
            if chunk:
                f.write(chunk)
    # 返回下载后的本地文件名
    return local_filename
```
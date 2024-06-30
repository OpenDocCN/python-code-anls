# `D:\src\scipysrc\seaborn\seaborn\_marks\__init__.py`

```
# 导入 requests 库，用于发送 HTTP 请求
import requests

# 定义函数 download_file，接收两个参数：url（下载地址）和 dest_filename（目标文件名）
def download_file(url, dest_filename):
    # 发送 GET 请求到指定的 URL，获取文件内容
    response = requests.get(url)
    
    # 将获取到的文件内容写入目标文件中，以二进制形式写入
    with open(dest_filename, 'wb') as f:
        f.write(response.content)
```
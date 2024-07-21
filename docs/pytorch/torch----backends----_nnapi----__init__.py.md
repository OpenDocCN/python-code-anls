# `.\pytorch\torch\backends\_nnapi\__init__.py`

```
# 导入 requests 模块，用于发送 HTTP 请求
import requests

# 定义函数 download_file，接受两个参数：url（下载文件的链接）和 dest_filename（目标文件名）
def download_file(url, dest_filename):
    # 发送 GET 请求到指定的 url，获取文件内容
    response = requests.get(url, stream=True)
    # 打开目标文件，以二进制写入模式
    with open(dest_filename, 'wb') as f:
        # 遍历请求返回的数据流，每次写入到目标文件
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # 如果数据块不为空
                f.write(chunk)  # 将数据块写入目标文件
```
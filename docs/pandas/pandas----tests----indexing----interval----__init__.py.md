# `D:\src\scipysrc\pandas\pandas\tests\indexing\interval\__init__.py`

```
# 导入需要使用的 requests 库
import requests

# 定义函数 download_file，用于从指定 URL 下载文件到本地
def download_file(url, dest_filename):
    # 发起 HTTP GET 请求，获取文件内容
    response = requests.get(url, stream=True)
    # 打开本地文件，以二进制写入模式创建或覆盖，准备写入下载的数据
    with open(dest_filename, 'wb') as f:
        # 遍历下载的数据流的每个块，写入到本地文件中
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    # 返回下载完成后的本地文件名
    return dest_filename
```
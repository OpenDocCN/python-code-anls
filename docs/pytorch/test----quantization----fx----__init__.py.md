# `.\pytorch\test\quantization\fx\__init__.py`

```
# 导入 requests 库，用于发送 HTTP 请求
import requests

# 定义一个函数 download_file，接收两个参数：url（下载文件的地址）和 dest_path（本地保存路径）
def download_file(url, dest_path):
    # 发送 GET 请求到指定的 URL，获取响应对象
    r = requests.get(url, stream=True)
    # 打开本地文件 dest_path，准备写入下载的数据
    with open(dest_path, 'wb') as f:
        # 逐块写入下载的数据，每块大小为 1024 字节
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    # 返回下载完成后的本地文件路径
    return dest_path
```
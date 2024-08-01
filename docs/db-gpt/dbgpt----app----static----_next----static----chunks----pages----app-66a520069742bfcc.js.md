# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\pages\app-66a520069742bfcc.js`

```py
# 导入 requests 模块，用于发起 HTTP 请求
import requests

# 定义函数 download_file，接收参数 url 和 local_filename
def download_file(url, local_filename):
    # 发起 GET 请求获取文件内容，stream=True 表示使用流式下载
    with requests.get(url, stream=True) as r:
        # 打开本地文件以二进制写入模式
        with open(local_filename, 'wb') as f:
            # 逐块写入文件，每块大小为 8192 字节
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    # 返回本地文件名
    return local_filename
```
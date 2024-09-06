# `.\HippoRAG\src\baselines\__init__.py`

```py
# 导入 requests 模块，用于发送 HTTP 请求
import requests

# 定义函数 download_file，接收两个参数：url（下载地址）和 save_path（保存路径）
def download_file(url, save_path):
    # 发送 GET 请求到指定的 URL，获取响应对象
    response = requests.get(url, stream=True)
    # 打开二进制文件保存路径，以写入（'wb'）方式打开
    with open(save_path, 'wb') as f:
        # 遍历响应内容的每个数据块
        for chunk in response.iter_content(chunk_size=1024):
            # 将每个数据块写入文件
            if chunk:
                f.write(chunk)
    # 返回文件保存路径，表示下载完成
    return save_path
```
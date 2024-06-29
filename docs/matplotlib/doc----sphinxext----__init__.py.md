# `D:\src\scipysrc\matplotlib\doc\sphinxext\__init__.py`

```py
# 导入requests库，用于发送HTTP请求和处理响应
import requests

# 定义函数download_file，接受两个参数：url（下载地址）和save_as（保存路径）
def download_file(url, save_as):
    # 发送GET请求到指定的URL，获取响应对象
    response = requests.get(url, stream=True)
    
    # 打开二进制文件，准备写入下载的数据
    with open(save_as, 'wb') as f:
        # 遍历响应内容的每个二进制块，写入到本地文件
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # 确保块不为空
                f.write(chunk)
    
    # 返回成功下载的文件路径
    return save_as
```
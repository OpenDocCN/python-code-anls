# `.\pytorch\test\jit\_imported_class_test\very\very\__init__.py`

```py
# 导入requests库，用于发送HTTP请求
import requests

# 定义函数download_file，接收url和保存路径作为参数
def download_file(url, save_path):
    # 发送GET请求获取url对应的文件内容
    response = requests.get(url)
    
    # 打开保存路径的文件以二进制写模式
    with open(save_path, 'wb') as f:
        # 将获取的文件内容写入到文件中
        f.write(response.content)

# 调用download_file函数，下载文件并保存到本地
download_file('http://example.com/example.zip', '/path/to/save/example.zip')
```
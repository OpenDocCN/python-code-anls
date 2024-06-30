# `D:\src\scipysrc\sympy\sympy\solvers\tests\__init__.py`

```
# 导入 requests 库，用于发送 HTTP 请求
import requests

# 定义一个函数 download_file，接受两个参数：url（下载文件的地址）和 save_path（保存文件的路径）
def download_file(url, save_path):
    # 发送 GET 请求获取文件内容，并将响应对象赋值给变量 response
    response = requests.get(url)
    
    # 打开保存文件的二进制写入模式，并使用 with 语句确保文件操作安全关闭
    with open(save_path, 'wb') as f:
        # 将响应内容的二进制写入到文件
        f.write(response.content)
```
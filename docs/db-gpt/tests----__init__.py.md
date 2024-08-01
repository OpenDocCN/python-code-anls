# `.\DB-GPT-src\tests\__init__.py`

```py
# 导入所需的模块：requests 用于发送 HTTP 请求，os 用于操作文件路径
import requests
import os

# 定义一个函数 download_file，接收两个参数：url 表示文件的远程地址，save_path 表示本地保存路径
def download_file(url, save_path):
    # 发送 GET 请求获取文件内容
    response = requests.get(url)
    # 打开本地文件，以二进制写入模式（'wb'）保存获取的文件内容
    with open(save_path, 'wb') as f:
        # 将获取的文件内容写入到本地文件中
        f.write(response.content)

# 主程序入口，如果当前脚本被直接执行（而非被导入），执行以下代码
if __name__ == "__main__":
    # 设置远程文件的 URL
    url = 'https://example.com/example.zip'
    # 设置本地文件保存路径
    save_path = os.path.join(os.getcwd(), 'example.zip')
    # 调用 download_file 函数，传入 URL 和保存路径作为参数，下载并保存文件
    download_file(url, save_path)
```
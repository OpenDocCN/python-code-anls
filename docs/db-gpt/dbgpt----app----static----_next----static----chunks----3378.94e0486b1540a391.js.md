# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\3378.94e0486b1540a391.js`

```py
# 导入所需的模块：os（操作系统接口）、json（处理 JSON 数据）、requests（发送 HTTP 请求）
import os
import json
import requests

# 定义函数 download_file，接受两个参数：url（下载文件的 URL）、save_path（保存文件的路径）
def download_file(url, save_path):
    # 发送 GET 请求到指定的 URL，获取响应对象
    response = requests.get(url)
    
    # 确保响应状态码为 200，表示请求成功
    if response.status_code == 200:
        # 以二进制写模式打开指定路径的文件，准备写入下载的内容
        with open(save_path, 'wb') as f:
            # 将响应内容（文件的二进制数据）写入到文件中
            f.write(response.content)
    else:
        # 如果请求失败，打印错误信息
        print(f"Failed to download file from {url}")

# 定义主函数 main
def main():
    # 设置文件的下载 URL 和保存路径
    url = 'http://example.com/data.json'
    save_path = os.path.join(os.getcwd(), 'data.json')
    
    # 调用 download_file 函数，传入 URL 和保存路径
    download_file(url, save_path)

# 如果该脚本被直接运行（而不是作为模块导入），执行主函数 main
if __name__ == "__main__":
    main()
```
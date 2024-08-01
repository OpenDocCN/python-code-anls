# `.\DB-GPT-src\dbgpt\serve\agent\__init__.py`

```py
# 导入必要的模块
import os
import sys
import hashlib

# 定义一个函数，用于计算给定文件的 MD5 哈希值
def calculate_md5(filename):
    # 打开文件并以二进制模式读取其内容
    with open(filename, 'rb') as f:
        # 创建一个 MD5 哈希对象
        md5 = hashlib.md5()
        # 逐步读取文件内容并更新 MD5 哈希对象
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    # 返回计算得到的 MD5 哈希值的十六进制表示
    return md5.hexdigest()

# 如果命令行参数的数量小于2，则打印使用方法并退出
if len(sys.argv) < 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

# 获取命令行参数中的文件名
filename = sys.argv[1]

# 如果文件不存在，打印错误信息并退出
if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found!")
    sys.exit(1)

# 计算文件的 MD5 哈希值
md5_hash = calculate_md5(filename)

# 打印文件的 MD5 哈希值
print(f"MD5 Hash of file '{filename}': {md5_hash}")
```
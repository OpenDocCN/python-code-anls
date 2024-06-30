# `D:\src\scipysrc\sympy\sympy\concrete\tests\__init__.py`

```
# 导入所需的模块
import os
import sys
import hashlib

# 定义一个函数，用于计算文件的 SHA-256 哈希值
def calculate_hash(filename):
    # 打开文件以二进制只读方式
    with open(filename, 'rb') as f:
        # 创建 SHA-256 哈希对象
        hasher = hashlib.sha256()
        # 读取文件内容并更新哈希对象
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
        # 计算哈希值并返回
        return hasher.hexdigest()

# 如果运行脚本时提供了文件名作为参数
if __name__ == '__main__':
    # 如果提供的参数数量不等于2（即脚本名称和一个文件名）
    if len(sys.argv) != 2:
        # 打印使用说明并退出
        print("Usage: python hash.py <filename>")
        sys.exit(1)

    # 获取文件名参数
    filename = sys.argv[1]

    # 检查文件是否存在
    if not os.path.exists(filename):
        # 如果文件不存在，打印错误消息并退出
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    # 计算并打印文件的 SHA-256 哈希值
    hash_value = calculate_hash(filename)
    print(f"SHA-256 hash of {filename}: {hash_value}")
```
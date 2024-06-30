# `D:\src\scipysrc\sympy\sympy\plotting\tests\__init__.py`

```
# 导入所需模块
import os
import sys
import hashlib

# 定义一个函数，用于计算指定文件的 MD5 哈希值
def compute_md5(fname):
    # 初始化 MD5 哈希对象
    md5 = hashlib.md5()
    # 打开文件进行读取，以二进制方式
    with open(fname, "rb") as f:
        # 循环读取文件内容
        for chunk in iter(lambda: f.read(4096), b""):
            # 更新 MD5 哈希对象的值
            md5.update(chunk)
    # 返回计算得到的 MD5 哈希值的十六进制表示
    return md5.hexdigest()

# 主程序入口点
if __name__ == "__main__":
    # 检查参数个数，确保只有一个参数（文件路径）
    if len(sys.argv) != 2:
        # 如果参数不正确，打印使用说明并退出
        print(f"Usage: {sys.argv[0]} <filename>")
        sys.exit(1)
    
    # 获取文件路径参数
    filename = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        # 如果文件不存在，提示错误并退出程序
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    
    # 调用 compute_md5 函数计算文件的 MD5 哈希值
    md5_hash = compute_md5(filename)
    
    # 输出计算得到的 MD5 哈希值
    print(f"MD5 hash of file '{filename}': {md5_hash}")
```
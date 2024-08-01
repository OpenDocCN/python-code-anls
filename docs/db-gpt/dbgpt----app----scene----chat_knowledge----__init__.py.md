# `.\DB-GPT-src\dbgpt\app\scene\chat_knowledge\__init__.py`

```py
# 导入必要的模块
import os
import sys
import hashlib

# 定义一个函数，计算指定文件的 MD5 值
def compute_md5(file_path):
    # 打开文件进行读取，以二进制模式
    with open(file_path, 'rb') as f:
        # 初始化 MD5 计算对象
        md5 = hashlib.md5()
        # 逐步读取文件内容并更新 MD5 值
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
        # 返回计算得到的 MD5 值的十六进制表示
        return md5.hexdigest()

# 如果脚本作为主程序执行，而非被导入，则执行以下代码
if __name__ == "__main__":
    # 如果命令行参数不足，输出提示信息并退出
    if len(sys.argv) < 2:
        print("Usage: python md5sum.py <file_path>")
        sys.exit(1)
    
    # 获取文件路径参数
    file_path = sys.argv[1]
    
    # 如果文件不存在，输出错误信息并退出
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    
    # 计算文件的 MD5 值
    md5_hash = compute_md5(file_path)
    
    # 输出计算得到的 MD5 值
    print(f"MD5 hash of file '{file_path}': {md5_hash}")
```
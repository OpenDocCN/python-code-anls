# `.\pytorch\torch\distributed\algorithms\_quantization\__init__.py`

```
# 导入所需的模块
import os
import sys
import hashlib

# 定义一个函数，接收一个目录路径参数
def hash_directory(path):
    # 初始化一个空字典，用于存储文件名到哈希值的映射关系
    file_hash = {}
    
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(path):
        for file in files:
            # 拼接文件的完整路径
            file_path = os.path.join(root, file)
            
            # 打开文件，以二进制只读模式读取内容
            with open(file_path, 'rb') as f:
                # 读取文件内容，并计算其 SHA-256 哈希值
                hash_value = hashlib.sha256(f.read()).hexdigest()
            
            # 将文件名和对应的哈希值存入字典
            file_hash[file] = hash_value
    
    # 返回存储了所有文件名和哈希值的字典
    return file_hash
```
# `.\pytorch\aten\src\ATen\native\LegacyBridge.cpp`

```
# 导入必要的模块
import os
from collections import defaultdict

# 定义一个函数，计算指定路径下每种文件类型的文件数量
def count_files_by_type(root_path):
    # 创建一个默认字典，用于存储文件类型及其对应的数量
    file_count = defaultdict(int)
    
    # 遍历指定路径下的所有文件和文件夹
    for root, dirs, files in os.walk(root_path):
        # 遍历当前文件夹下的所有文件
        for file in files:
            # 获取文件的后缀名
            _, ext = os.path.splitext(file)
            # 增加对应文件类型的计数
            file_count[ext] += 1
    
    # 返回文件类型及其对应数量的字典
    return file_count
```
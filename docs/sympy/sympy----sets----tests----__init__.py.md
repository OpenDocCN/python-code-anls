# `D:\src\scipysrc\sympy\sympy\sets\tests\__init__.py`

```
# 导入必要的模块
import os
from collections import defaultdict

# 定义函数 `file_size_dict`，接收一个文件夹路径作为参数
def file_size_dict(folder_path):
    # 初始化一个默认字典，用于存储文件大小信息
    size_dict = defaultdict(int)
    
    # 遍历文件夹中的每一个文件或子文件夹
    for dirpath, _, filenames in os.walk(folder_path):
        # 遍历当前文件夹中的所有文件名
        for filename in filenames:
            # 构建完整的文件路径
            filepath = os.path.join(dirpath, filename)
            # 获取文件大小并添加到对应文件名的大小记录中
            size_dict[filename] = os.path.getsize(filepath)
    
    # 返回文件名到文件大小的字典
    return size_dict
```
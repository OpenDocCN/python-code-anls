# `D:\src\scipysrc\seaborn\tests\_marks\__init__.py`

```
# 导入所需的模块
import os
from collections import defaultdict

# 定义函数 `file_stats`，接收一个参数 `dirname` 表示目录路径
def file_stats(dirname):
    # 初始化一个默认字典，用于存储文件大小信息
    stats = defaultdict(int)
    
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(dirname):
        # 遍历当前目录下的所有文件
        for file in files:
            # 构造文件的完整路径
            filepath = os.path.join(root, file)
            # 获取文件大小
            size = os.path.getsize(filepath)
            # 将文件大小添加到字典中，以文件路径为键
            stats[filepath] = size
    
    # 返回包含文件大小信息的字典
    return stats
```
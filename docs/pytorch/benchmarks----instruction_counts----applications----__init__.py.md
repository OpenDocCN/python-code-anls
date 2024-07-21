# `.\pytorch\benchmarks\instruction_counts\applications\__init__.py`

```
# 导入所需的模块：os（用于文件操作）和 json（用于 JSON 数据处理）
import os
import json

# 定义一个函数，接收一个目录路径作为参数
def process_directory(dir_path):
    # 初始化一个空列表，用于存储符合条件的文件名
    file_list = []
    
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(dir_path):
        # 遍历当前目录中的所有文件名
        for file in files:
            # 如果文件名以".json"结尾，则将其加入到列表中
            if file.endswith(".json"):
                # 构造完整的文件路径
                file_path = os.path.join(root, file)
                # 将文件路径添加到列表中
                file_list.append(file_path)
    
    # 返回包含符合条件的文件路径列表
    return file_list
```
# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\__init__.py`

```
# 导入所需模块：os（操作系统接口）、re（正则表达式操作）、json（处理 JSON 数据）
import os
import re
import json

# 定义一个函数，用于统计给定目录下每种文件类型的文件数目
def count_files_by_type(directory):
    # 初始化一个空字典，用于存储不同类型文件的计数结果
    type_counts = {}
    
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        # 遍历当前目录中的每个文件
        for file in files:
            # 使用正则表达式匹配文件名，提取文件扩展名作为文件类型
            match = re.search(r'\.(\w+)$', file)
            if match:
                # 获取文件类型（扩展名）
                file_type = match.group(1)
                # 如果该文件类型已存在于字典中，则递增其计数值
                if file_type in type_counts:
                    type_counts[file_type] += 1
                else:
                    # 如果该文件类型尚不存在于字典中，则初始化其计数值为1
                    type_counts[file_type] = 1
    
    # 将统计结果字典转换为 JSON 格式的字符串，并返回
    return json.dumps(type_counts)
```
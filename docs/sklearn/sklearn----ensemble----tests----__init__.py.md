# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\tests\__init__.py`

```
# 导入所需模块：os 模块用于操作文件系统，re 模块用于正则表达式匹配
import os
import re

# 定义函数 search_files，接收文件夹路径和文件名模式作为参数
def search_files(folder, pattern):
    # 初始化空列表，用于存储匹配到的文件路径
    matches = []
    # 遍历指定文件夹下的所有文件和文件夹
    for root, _, files in os.walk(folder):
        # 遍历当前文件夹中的所有文件
        for file in files:
            # 使用正则表达式匹配文件名是否符合指定的模式
            if re.match(pattern, file):
                # 如果匹配成功，将文件的完整路径添加到 matches 列表中
                matches.append(os.path.join(root, file))
    # 返回匹配到的文件路径列表
    return matches
```
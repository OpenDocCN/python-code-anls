# `D:\src\scipysrc\pandas\pandas\tests\frame\constructors\__init__.py`

```
# 导入必要的模块
import os
from typing import List

# 定义一个名为`File`的类，表示文件
class File:
    # 初始化方法，接受文件名和内容
    def __init__(self, name: str, content: str):
        self.name = name  # 文件名属性
        self.content = content  # 文件内容属性

# 定义一个名为`get_files`的函数，接受目录路径作为参数
def get_files(directory: str) -> List[File]:
    # 创建一个空列表，用于存储文件对象
    files = []
    # 遍历目录中的每个文件
    for filename in os.listdir(directory):
        # 拼接文件的完整路径
        filepath = os.path.join(directory, filename)
        # 如果该路径是文件而不是目录
        if os.path.isfile(filepath):
            # 打开文件，读取文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            # 创建一个文件对象，并将其添加到文件列表中
            files.append(File(filename, content))
    # 返回所有文件对象的列表
    return files
```
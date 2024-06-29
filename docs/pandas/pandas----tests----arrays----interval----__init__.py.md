# `D:\src\scipysrc\pandas\pandas\tests\arrays\interval\__init__.py`

```
# 导入所需的模块：os 和 re
import os
import re

# 定义函数 extract_numbers
def extract_numbers(text):
    # 使用正则表达式查找文本中的数字序列，并返回找到的第一个匹配结果
    match = re.search(r'\d+', text)
    return match.group(0) if match else None

# 定义函数 find_files
def find_files(directory, extension):
    # 初始化一个空列表，用于存储找到的文件名
    files = []
    # 遍历指定目录下的所有文件和子目录
    for root, _, filenames in os.walk(directory):
        # 遍历当前目录下的所有文件名
        for filename in filenames:
            # 检查文件名是否以指定的扩展名结尾
            if filename.endswith(extension):
                # 如果是，则将文件的绝对路径添加到列表中
                files.append(os.path.join(root, filename))
    # 返回找到的所有符合条件的文件名列表
    return files
```
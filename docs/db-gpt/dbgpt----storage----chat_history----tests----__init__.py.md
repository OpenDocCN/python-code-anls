# `.\DB-GPT-src\dbgpt\storage\chat_history\tests\__init__.py`

```py
# 导入需要的模块：os 和 re
import os
import re

# 定义一个函数，用于列出指定目录下所有的文件名
def list_files(directory):
    # 使用 os 模块的 listdir 函数列出目录中的所有文件和文件夹
    files = os.listdir(directory)
    # 使用列表推导式过滤出所有的文件名，去除文件夹名
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    # 返回过滤后的文件名列表
    return files

# 定义一个函数，用于在指定文本中查找所有符合给定模式的字符串
def find_patterns(text, pattern):
    # 使用 re 模块的 findall 函数查找所有匹配给定模式的字符串
    matches = re.findall(pattern, text)
    # 返回找到的所有匹配字符串列表
    return matches
```
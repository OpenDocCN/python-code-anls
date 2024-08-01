# `.\DB-GPT-src\dbgpt\app\openapi\api_v1\editor\__init__.py`

```py
# 导入必要的模块：os 用于操作文件系统，re 用于正则表达式匹配
import os
import re

# 定义函数 `find_files`，接收两个参数：`pattern`（用于匹配文件名的正则表达式）和 `path`（搜索的根目录）
def find_files(pattern, path):
    # 遍历指定路径下的所有文件和子目录，返回一个生成器对象
    for root, dirs, files in os.walk(path):
        # 遍历当前目录下的所有文件
        for file in files:
            # 使用正则表达式 `pattern` 匹配文件名
            if re.match(pattern, file):
                # 拼接文件的完整路径
                filepath = os.path.join(root, file)
                # 使用 `yield` 关键字返回匹配到的文件路径，使函数变为一个生成器
                yield filepath

# 测试代码
pattern = r'.*\.txt'  # 匹配所有以 `.txt` 结尾的文件
path = '/path/to/search'  # 搜索的根目录
for file in find_files(pattern, path):
    print(file)
```
# `.\numpy\numpy\core\__init__.pyi`

```py
# 导入所需的模块：re 用于正则表达式操作，os 用于文件路径操作
import re
import os

# 定义函数 find_files，接收文件路径和正则表达式作为参数
def find_files(dir, regex):
    # 初始化一个空列表，用于存储符合条件的文件路径
    files = []
    # 遍历指定目录及其子目录下的所有文件和文件夹
    for root, dirs, filenames in os.walk(dir):
        # 对每个文件名进行正则匹配
        for filename in filenames:
            # 如果文件名符合正则表达式条件
            if re.match(regex, filename):
                # 构建文件的完整路径并添加到列表中
                files.append(os.path.join(root, filename))
    # 返回符合条件的文件路径列表
    return files
```
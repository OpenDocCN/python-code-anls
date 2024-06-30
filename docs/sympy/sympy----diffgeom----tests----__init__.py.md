# `D:\src\scipysrc\sympy\sympy\diffgeom\tests\__init__.py`

```
# 导入所需模块
import os
import sys

# 定义一个名为 `process_file` 的函数，接受一个参数 `filename`
def process_file(filename):
    # 打开文件，使用 `with` 语句确保文件在使用后自动关闭
    with open(filename, 'r') as f:
        # 逐行读取文件内容，将每行内容存入 `lines` 列表中
        lines = f.readlines()
        
    # 对读取的每一行进行处理，使用 `enumerate` 函数获取行号和内容
    for idx, line in enumerate(lines):
        # 打印行号（从1开始）和行内容（去掉末尾的换行符）
        print(f"Line {idx + 1}: {line.rstrip()}")

# 调用 `process_file` 函数，处理名为 `test.txt` 的文件
process_file('test.txt')
```
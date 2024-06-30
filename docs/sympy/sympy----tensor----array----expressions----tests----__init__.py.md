# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\tests\__init__.py`

```
# 导入所需的模块：os 模块用于操作文件路径，re 模块用于正则表达式匹配
import os
import re

# 定义函数 extract_data，接收文件名作为参数
def extract_data(filename):
    # 使用 with 语句打开文件，确保文件操作后会自动关闭
    with open(filename, 'r') as file:
        # 读取文件的所有内容并存储在变量 data 中
        data = file.read()

    # 使用正则表达式模式匹配文件中的数字，返回匹配到的第一个数字作为结果
    result = re.search(r'\d+', data)

    # 如果找到匹配的数字，则将其作为整数返回；否则返回 None
    return int(result.group()) if result else None
```
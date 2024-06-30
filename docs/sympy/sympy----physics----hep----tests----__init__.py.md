# `D:\src\scipysrc\sympy\sympy\physics\hep\tests\__init__.py`

```
# 导入所需模块：导入了标准库中的 pathlib 模块，并为其定义了别名 Path
from pathlib import Path

# 定义一个名为 count_lines 的函数，接收一个参数 filename
def count_lines(filename):
    # 打开指定文件，'r' 表示以只读模式打开
    with open(filename, 'r') as file:
        # 使用列表推导式遍历文件的每一行，并计算总行数
        num_lines = sum(1 for line in file)
    # 返回计算得到的总行数
    return num_lines
```
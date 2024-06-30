# `D:\src\scipysrc\seaborn\doc\tools\extract_examples.py`

```
"""Turn the examples section of a function docstring into a notebook."""
# 导入所需的库
import re
import sys
import pydoc
import seaborn
from seaborn.external.docscrape import NumpyDocString
import nbformat

# 定义函数，判断给定行的类型（代码还是 Markdown）
def line_type(line):
    if line.startswith("    "):
        return "code"
    else:
        return "markdown"

# 定义函数，向 Jupyter 笔记本添加新的单元格
def add_cell(nb, lines, cell_type):
    # 定义不同类型单元格的对象
    cell_objs = {
        "code": nbformat.v4.new_code_cell,
        "markdown": nbformat.v4.new_markdown_cell,
    }
    # 将行列表连接成单个文本
    text = "\n".join(lines)
    # 创建新的单元格对象，并添加到笔记本的单元格列表中
    cell = cell_objs[cell_type](text)
    nb["cells"].append(cell)

# 主程序入口
if __name__ == "__main__":
    # 从命令行参数中获取函数名
    _, name = sys.argv

    # 获取函数或方法对象
    obj = getattr(seaborn, name)
    # 如果不是函数，则尝试获取其 __init__ 方法作为对象
    if obj.__class__.__name__ != "function":
        obj = obj.__init__
    # 解析对象的文档字符串，提取其中的示例部分
    lines = NumpyDocString(pydoc.getdoc(obj))["Examples"]

    # 编译用于匹配代码段缩进和 matplotlib 绘图指令的正则表达式
    pat = re.compile(r"\s{4}[>\.]{3} (ax = ){0,1}(g = ){0,1}")

    # 创建新的笔记本对象
    nb = nbformat.v4.new_notebook()

    # 默认开始第一个单元格是 Markdown 类型
    cell_type = "markdown"
    # 初始化单元格列表
    cell = []

    # 遍历示例中的每一行
    for line in lines:

        # 忽略 matplotlib 绘图指令和特定上下文信息
        if ".. plot" in line or ":context:" in line:
            continue

        # 忽略空白行
        if not line:
            continue

        # 判断当前行的类型（代码还是 Markdown）
        if line_type(line) != cell_type:
            # 如果当前行类型与前一个单元格类型不同，则封装并添加上一个单元格
            add_cell(nb, cell, cell_type)
            # 更新当前单元格类型为当前行的类型
            cell_type = line_type(line)
            # 清空当前单元格列表，为新类型单元格做准备
            cell = []

        # 如果当前行是代码类型，则去除代码缩进和 matplotlib 绘图指令
        if line_type(line) == "code":
            line = re.sub(pat, "", line)

        # 将当前行添加到当前单元格列表中
        cell.append(line)

    # 封装和添加最后一个单元格
    add_cell(nb, cell, cell_type)

    # 将生成的笔记本保存到文件
    nbformat.write(nb, f"docstrings/{name}.ipynb")
```
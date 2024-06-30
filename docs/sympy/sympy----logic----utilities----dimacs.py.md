# `D:\src\scipysrc\sympy\sympy\logic\utilities\dimacs.py`

```
"""For reading in DIMACS file format

www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/satformat.ps

"""

# 从 sympy.core 中导入 Symbol 符号类
from sympy.core import Symbol
# 从 sympy.logic.boolalg 中导入 And 和 Or 逻辑操作类
from sympy.logic.boolalg import And, Or
# 导入正则表达式模块
import re


def load(s):
    """Loads a boolean expression from a string.

    Examples
    ========

    >>> from sympy.logic.utilities.dimacs import load
    >>> load('1')
    cnf_1
    >>> load('1 2')
    cnf_1 | cnf_2
    >>> load('1 \\n 2')
    cnf_1 & cnf_2
    >>> load('1 2 \\n 3')
    cnf_3 & (cnf_1 | cnf_2)
    """
    # 初始化空的子句列表
    clauses = []

    # 将字符串 s 按行分割为列表 lines
    lines = s.split('\n')

    # 编译正则表达式，用于匹配注释行和状态行
    pComment = re.compile(r'c.*')
    pStats = re.compile(r'p\s*cnf\s*(\d*)\s*(\d*)')

    # 处理每一行直到 lines 为空
    while len(lines) > 0:
        # 弹出 lines 中的第一行
        line = lines.pop(0)

        # 只处理非注释的行
        if not pComment.match(line):
            # 尝试匹配状态行
            m = pStats.match(line)

            if not m:
                # 如果不是状态行，则分割出数字，并转换为相应的符号变量或其否定
                nums = line.rstrip('\n').split(' ')
                list = []
                for lit in nums:
                    if lit != '':
                        if int(lit) == 0:
                            continue
                        num = abs(int(lit))
                        sign = True
                        if int(lit) < 0:
                            sign = False

                        if sign:
                            list.append(Symbol("cnf_%s" % num))
                        else:
                            list.append(~Symbol("cnf_%s" % num))

                # 如果列表不为空，则将其转换为 Or 逻辑操作，并添加到子句列表中
                if len(list) > 0:
                    clauses.append(Or(*list))

    # 将所有子句用 And 逻辑操作连接起来，并返回结果
    return And(*clauses)


def load_file(location):
    """Loads a boolean expression from a file."""
    # 打开文件并读取其内容到字符串 s
    with open(location) as f:
        s = f.read()

    # 调用 load 函数处理字符串 s，并返回结果
    return load(s)
```
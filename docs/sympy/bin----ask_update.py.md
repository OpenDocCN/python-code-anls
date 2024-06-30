# `D:\src\scipysrc\sympy\bin\ask_update.py`

```
#!/usr/bin/env python

""" Update the ``ask_generated.py`` file.

This must be run each time ``known_facts()`` in ``assumptions.facts`` module
is changed.

This must be run each time ``_generate_assumption_rules()`` in
``sympy.core.assumptions`` module is changed.

Should be run from sympy root directory.

$ python bin/ask_update.py
"""

# hook in-tree SymPy into Python path, if possible
import os          # 导入操作系统相关的功能模块
import sys         # 导入系统相关的功能模块
import pprint      # 导入格式化打印模块

isympy_path = os.path.abspath(__file__)     # 获取当前脚本的绝对路径
isympy_dir = os.path.dirname(isympy_path)   # 获取当前脚本所在目录的路径
sympy_top = os.path.split(isympy_dir)[0]    # 获取 sympy 根目录的路径
sympy_dir = os.path.join(sympy_top, 'sympy')  # 组合出 sympy 目录的路径

if os.path.isdir(sympy_dir):                # 检查 sympy 目录是否存在
    sys.path.insert(0, sympy_top)           # 将 sympy 根目录加入到系统路径中

from sympy.core.assumptions import _generate_assumption_rules  # 导入 _generate_assumption_rules 函数
from sympy.assumptions.cnf import CNF, Literal  # 导入 CNF 类和 Literal 类
from sympy.assumptions.facts import (get_known_facts,  # 导入多个函数
    generate_known_facts_dict, get_known_facts_keys, get_matrix_facts, get_number_facts)
from sympy.core import Symbol  # 导入 Symbol 类
from textwrap import dedent, wrap  # 导入文本处理相关函数

def generate_code():

    LINE = ",\n        "   # 定义字符串常量 LINE，用于格式化代码字符串
    HANG = ' '*8            # 定义字符串常量 HANG，用于缩进
    code_string = dedent('''\
    """
    Do NOT manually edit this file.
    Instead, run ./bin/ask_update.py.
    """

    from sympy.assumptions.ask import Q  # 导入 Q 对象
    from sympy.assumptions.cnf import Literal  # 导入 Literal 类
    from sympy.core.cache import cacheit  # 导入 cacheit 函数

    @cacheit
    def get_all_known_facts():
        """
        Known facts between unary predicates as CNF clauses.
        """
        return {
            %s
        }

    @cacheit
    def get_all_known_matrix_facts():
        """
        Known facts between unary predicates for matrices as CNF clauses.
        """
        return {
            %s
        }

    @cacheit
    def get_all_known_number_facts():
        """
        Known facts between unary predicates for numbers as CNF clauses.
        """
        return {
            %s
        }

    @cacheit
    def get_known_facts_dict():
        """
        Logical relations between unary predicates as dictionary.

        Each key is a predicate, and item is two groups of predicates.
        First group contains the predicates which are implied by the key, and
        second group contains the predicates which are rejected by the key.

        """
        return {
            %s
        }
    ''')

    x = Symbol('x')  # 创建一个符号 x
    fact = get_known_facts(x)  # 获取 x 的已知事实
    matrix_fact = get_matrix_facts(x)  # 获取 x 在矩阵中的已知事实
    number_fact = get_number_facts(x)  # 获取 x 在数字中的已知事实

    # Generate CNF of facts between known unary predicates
    cnf = CNF.to_CNF(fact)  # 将已知事实转换为 CNF
    all_clauses = LINE.join(sorted([
        'frozenset(('
         + ', '.join(str(Literal(lit.arg.function, lit.is_Not))
                     for lit in sorted(clause, key=str))
        + '))' for clause in cnf.clauses]))  # 生成 CNF 子句的字符串表示，并按字母顺序排序

    # Generate CNF of matrix facts
    cnf = CNF.to_CNF(matrix_fact)  # 将矩阵中的已知事实转换为 CNF
    matrix_clauses = LINE.join(sorted([
        'frozenset(('
         + ', '.join(str(Literal(lit.arg.function, lit.is_Not))
                     for lit in sorted(clause, key=str))
        + '))' for clause in cnf.clauses]))  # 生成 CNF 子句的字符串表示，并按字母顺序排序
    # 生成数字事实的合取范式（CNF）
    cnf = CNF.to_CNF(number_fact)
    
    # 对CNF中的每个子句进行排序和字符串化，以生成用于表示数字事实的Clausal Normal Form字符串
    number_clauses = LINE.join(sorted([
        'frozenset(('
         + ', '.join(str(Literal(lit.arg.function, lit.is_Not))
                     for lit in sorted(clause, key=str))
        + '))' for clause in cnf.clauses]))
    
    # 生成已知一元谓词之间事实的字典
    keys = [pred(x) for pred in get_known_facts_keys()]
    mapping = generate_known_facts_dict(keys, fact)
    
    # 对字典项按键进行排序
    items = sorted(mapping.items(), key=str)
    
    # 获取已排序键的字符串表示
    keys = [str(i[0]) for i in items]
    
    # 获取已排序值的字符串表示，格式化为元组形式的字符串
    values = ['(set(%s), set(%s))' % (sorted(i[1][0], key=str),
                                      sorted(i[1][1], key=str))
              for i in items]
    
    # 将键值对格式化为字符串列表，每行包含一个键值对，并根据特定格式进行换行和缩进
    m = LINE.join(['\n'.join(
        wrap("{}: {}".format(k, v),
            subsequent_indent=HANG,
            break_long_words=False))
        for k, v in zip(keys, values)]) + ','
    
    # 返回格式化后的代码字符串，使用提供的参数替换占位符%s
    return code_string % (all_clauses, matrix_clauses, number_clauses, m)
# 打开文件 'sympy/assumptions/ask_generated.py' 以便写入内容，文件句柄为 f
with open('sympy/assumptions/ask_generated.py', 'w') as f:
    # 生成代码
    code = generate_code()
    # 将生成的代码写入文件
    f.write(code)

# 打开文件 'sympy/core/assumptions_generated.py' 以便写入内容，文件句柄为 f
with open('sympy/core/assumptions_generated.py', 'w') as f:
    # 生成假设规则的表示形式，并转换为 Python 代码
    representation = _generate_assumption_rules()._to_python()

    # 构建带有注释的代码字符串模板
    code_string = dedent('''\
    """
    Do NOT manually edit this file.
    Instead, run ./bin/ask_update.py.
    """

    %s
    ''')

    # 将假设规则的 Python 表示形式插入到代码模板中，并生成最终的代码
    code = code_string % (representation,)
    # 将生成的代码写入文件
    f.write(code)
```
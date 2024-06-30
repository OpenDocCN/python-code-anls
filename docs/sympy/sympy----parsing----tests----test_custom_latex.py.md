# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_custom_latex.py`

```
import os  # 导入操作系统相关模块
import tempfile  # 导入临时文件相关模块

import sympy  # 导入sympy符号计算库
from sympy.testing.pytest import raises  # 导入用于测试的raises函数
from sympy.parsing.latex.lark import LarkLaTeXParser, TransformToSymPyExpr, parse_latex_lark  # 导入latex解析相关函数
from sympy.external import import_module  # 导入外部模块导入函数

lark = import_module("lark")  # 使用import_module函数导入lark模块

# 如果lark模块不可用，则禁用测试
disabled = lark is None

# 设置latex语法文件的路径
grammar_file = os.path.join(os.path.dirname(__file__), "../latex/lark/grammar/latex.lark")

# 第一个修改，修改latex语法，重定义DIV_SYMBOL和MUL_SYMBOL
modification1 = """
%override DIV_SYMBOL: DIV
%override MUL_SYMBOL: MUL | CMD_TIMES
"""

# 第二个修改，重定义number规则，允许逗号作为小数点
modification2 = r"""
%override number: /\d+(,\d*)?/
"""

def init_custom_parser(modification, transformer=None):
    # 打开latex语法文件并读取内容
    with open(grammar_file, encoding="utf-8") as f:
        latex_grammar = f.read()

    # 将修改内容添加到latex语法中
    latex_grammar += modification

    # 使用临时文件创建临时文件对象
    with tempfile.NamedTemporaryFile() as f:
        f.write(bytes(latex_grammar, encoding="utf8"))

        # 初始化自定义的Latex解析器对象
        parser = LarkLaTeXParser(grammar_file=f.name, transformer=transformer)

    return parser

def test_custom1():
    # 测试函数1：移除解析器对 \cdot 和 \div 的理解能力

    # 初始化自定义解析器，应用第一个修改
    parser = init_custom_parser(modification1)

    # 使用raises函数断言抛出异常：解析器无法理解 \cdot 和 \div
    with raises(lark.exceptions.UnexpectedCharacters):
        parser.doparse(r"a \cdot b")
        parser.doparse(r"x \div y")

class CustomTransformer(TransformToSymPyExpr):
    def number(self, tokens):
        if "," in tokens[0]:
            # 如果token中包含逗号，则将其视为浮点数，替换逗号为点作为小数点
            return sympy.core.numbers.Float(tokens[0].replace(",", "."))
        else:
            # 否则视为整数
            return sympy.core.numbers.Integer(tokens[0])

def test_custom2():
    # 测试函数2：使解析器将逗号视为小数点的分隔符

    # 初始化自定义解析器，应用第二个修改和自定义的转换器
    parser = init_custom_parser(modification2, CustomTransformer)

    # 使用raises函数断言抛出异常：默认解析器无法解析使用逗号作为小数点的数值
    with raises(lark.exceptions.UnexpectedCharacters):
        parse_latex_lark("100,1")
        parse_latex_lark("0,009")

    # 使用自定义解析器解析使用逗号作为小数点的数值
    parser.doparse("100,1")
    parser.doparse("0,009")
    parser.doparse("2,71828")
    parser.doparse("3,14159")
```
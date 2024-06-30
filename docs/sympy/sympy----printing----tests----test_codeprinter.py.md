# `D:\src\scipysrc\sympy\sympy\printing\tests\test_codeprinter.py`

```
# 导入必要的模块和函数
from sympy.printing.codeprinter import CodePrinter, PrintMethodNotImplementedError
from sympy.core import symbols
from sympy.core.symbol import Dummy
from sympy.testing.pytest import raises

# 设置测试用的打印机实例，使用给定的参数设置
def setup_test_printer(**kwargs):
    p = CodePrinter(settings=kwargs)
    # 初始化不支持的功能集合和数字符号集合
    p._not_supported = set()
    p._number_symbols = set()
    return p

# 测试打印 Dummy 符号
def test_print_Dummy():
    # 创建一个 Dummy 符号 'd'
    d = Dummy('d')
    # 设置打印机实例
    p = setup_test_printer()
    # 断言打印 Dummy 符号后的输出与预期相符
    assert p._print_Dummy(d) == "d_%i" % d.dummy_index

# 测试打印 Symbol 符号
def test_print_Symbol():
    # 创建符号 'x' 和 'if'
    x, y = symbols('x, if')

    # 设置打印机实例
    p = setup_test_printer()
    # 断言打印 'x' 符号后的输出为 'x'
    assert p._print(x) == 'x'
    # 断言打印 'if' 符号后的输出为 'if'
    assert p._print(y) == 'if'

    # 更新保留字列表，添加 'if' 作为保留字
    p.reserved_words.update(['if'])
    # 断言再次打印 'if' 符号后的输出为 'if_'
    assert p._print(y) == 'if_'

    # 使用设置了错误处理的打印机实例
    p = setup_test_printer(error_on_reserved=True)
    p.reserved_words.update(['if'])
    # 断言尝试打印 'if' 符号时抛出 ValueError 异常
    with raises(ValueError):
        p._print(y)

    # 使用指定了保留字后缀的打印机实例
    p = setup_test_printer(reserved_word_suffix='_He_Man')
    p.reserved_words.update(['if'])
    # 断言打印 'if' 符号后的输出为 'if_He_Man'
    assert p._print(y) == 'if_He_Man'

# 测试问题 #15791 的情况
def test_issue_15791():
    # 定义一个特定的打印机类 CrashingCodePrinter，用于测试异常情况
    class CrashingCodePrinter(CodePrinter):
        # 定义一个未实现的打印函数
        def emptyPrinter(self, obj):
            raise NotImplementedError

    # 导入需要测试的矩阵类
    from sympy.matrices import (
        MutableSparseMatrix,
        ImmutableSparseMatrix,
    )

    # 创建一个 CrashingCodePrinter 实例
    c = CrashingCodePrinter()

    # 断言尝试打印 ImmutableSparseMatrix 时抛出 PrintMethodNotImplementedError 异常
    with raises(PrintMethodNotImplementedError):
        c.doprint(ImmutableSparseMatrix(2, 2, {}))
    # 断言尝试打印 MutableSparseMatrix 时抛出 PrintMethodNotImplementedError 异常
    with raises(PrintMethodNotImplementedError):
        c.doprint(MutableSparseMatrix(2, 2, {}))
```
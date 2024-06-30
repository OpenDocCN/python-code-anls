# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_sym_expr.py`

```
# 从 sympy.parsing.sym_expr 模块导入 SymPyExpression 类
from sympy.parsing.sym_expr import SymPyExpression
# 从 sympy.testing.pytest 模块导入 raises 函数
from sympy.testing.pytest import raises
# 从 sympy.external 模块导入 import_module 函数
from sympy.external import import_module

# 使用 import_module 函数导入 lfortran 模块
lfortran = import_module('lfortran')
# 使用 import_module 函数导入 clang.cindex 模块，并通过 import_kwargs 参数指定导入 cindex 子模块
cin = import_module('clang.cindex', import_kwargs={'fromlist': ['cindex']})

# 如果 lfortran 和 cin 都成功导入，则执行以下代码
if lfortran and cin:
    # 从 sympy.codegen.ast 模块导入多个类和函数
    from sympy.codegen.ast import (Variable, IntBaseType, FloatBaseType, String,
                                   Declaration, FloatType)
    # 从 sympy.core 模块导入 Integer 和 Float 类
    from sympy.core import Integer, Float
    # 从 sympy.core.symbol 模块导入 Symbol 类
    from sympy.core.symbol import Symbol

    # 创建 SymPyExpression 类的实例对象 expr1
    expr1 = SymPyExpression()
    
    # 定义一个包含 Fortran 类型声明的源代码字符串
    src = """\
    integer :: a, b, c, d
    real :: p, q, r, s
    """

    # 定义测试函数 test_c_parse
    def test_c_parse():
        # 定义一个包含 C 语言类型声明的源代码字符串 src1
        src1 = """\
        int a, b = 4;
        float c, d = 2.4;
        """
        # 将 src1 转换为表达式，并设置语言类型为 'c'
        expr1.convert_to_expr(src1, 'c')
        # 获取转换后的表达式列表 ls
        ls = expr1.return_expr()

        # 断言表达式列表中的元素与预期的 Declaration 对象相等
        assert ls[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc'))
            )
        )
        assert ls[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(4)
            )
        )
        assert ls[2] == Declaration(
            Variable(
                Symbol('c'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    )
            )
        )
        assert ls[3] == Declaration(
            Variable(
                Symbol('d'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('2.3999999999999999', precision=53)
            )
        )
    def test_fortran_parse():
        # 创建 SymPyExpression 对象，使用 'f' 标识符和给定的源代码字符串 src
        expr = SymPyExpression(src, 'f')
        # 调用 return_expr 方法，获取返回的表达式列表 ls
        ls = expr.return_expr()

        # 断言表达式 ls 中的第一个元素是声明语句，声明变量 'a' 为整型，初始值为 0
        assert ls[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        # 断言表达式 ls 中的第二个元素是声明语句，声明变量 'b' 为整型，初始值为 0
        assert ls[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        # 断言表达式 ls 中的第三个元素是声明语句，声明变量 'c' 为整型，初始值为 0
        assert ls[2] == Declaration(
            Variable(
                Symbol('c'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        # 断言表达式 ls 中的第四个元素是声明语句，声明变量 'd' 为整型，初始值为 0
        assert ls[3] == Declaration(
            Variable(
                Symbol('d'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        # 断言表达式 ls 中的第五个元素是声明语句，声明变量 'p' 为浮点型，初始值为 0.0
        assert ls[4] == Declaration(
            Variable(
                Symbol('p'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )
        # 断言表达式 ls 中的第六个元素是声明语句，声明变量 'q' 为浮点型，初始值为 0.0
        assert ls[5] == Declaration(
            Variable(
                Symbol('q'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )
        # 断言表达式 ls 中的第七个元素是声明语句，声明变量 'r' 为浮点型，初始值为 0.0
        assert ls[6] == Declaration(
            Variable(
                Symbol('r'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )
        # 断言表达式 ls 中的第八个元素是声明语句，声明变量 's' 为浮点型，初始值为 0.0
        assert ls[7] == Declaration(
            Variable(
                Symbol('s'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )


    def test_convert_py():
        # 将源代码字符串 src 和附加的字符串连接，构成 src1
        src1 = (
            src +
            """\
            a = b + c
            s = p * q / r
            """
        )
        # 将 src1 转换为表达式对象，类型为 'f'
        expr1.convert_to_expr(src1, 'f')
        # 将表达式对象转换为 Python 表达式，并获取转换结果
        exp_py = expr1.convert_to_python()
        # 断言转换结果与预期的 Python 代码列表相等
        assert exp_py == [
            'a = 0',
            'b = 0',
            'c = 0',
            'd = 0',
            'p = 0.0',
            'q = 0.0',
            'r = 0.0',
            's = 0.0',
            'a = b + c',
            's = p*q/r'
        ]


    def test_convert_fort():
        # 将源代码字符串 src 和附加的字符串连接，构成 src1
        src1 = (
            src +
            """\
            a = b + c
            s = p * q / r
            """
        )
        # 将 src1 转换为表达式对象，类型为 'f'
        expr1.convert_to_expr(src1, 'f')
        # 将表达式对象转换为 Fortran 代码，并获取转换结果
        exp_fort = expr1.convert_to_fortran()
        # 断言转换结果与预期的 Fortran 代码列表相等
        assert exp_fort == [
            '      integer*4 a',
            '      integer*4 b',
            '      integer*4 c',
            '      integer*4 d',
            '      real*8 p',
            '      real*8 q',
            '      real*8 r',
            '      real*8 s',
            '      a = b + c',
            '      s = p*q/r'
        ]
    def test_convert_c():
        src1 = (
            src +
            """\
            a = b + c
            s = p * q / r
            """
        )
        # 将源码字符串 src1 转换为表达式对象，使用 'f' 模式
        expr1.convert_to_expr(src1, 'f')
        # 将表达式对象转换为 C 语言代码并保存在 exp_c 变量中
        exp_c = expr1.convert_to_c()
        # 断言 exp_c 是否等于预期的列表
        assert exp_c == [
            'int a = 0',
            'int b = 0',
            'int c = 0',
            'int d = 0',
            'double p = 0.0',
            'double q = 0.0',
            'double r = 0.0',
            'double s = 0.0',
            'a = b + c;',   # 将 a 赋值为 b 加 c
            's = p*q/r;'    # 将 s 赋值为 p 乘以 q 除以 r
        ]


    def test_exceptions():
        src = 'int a;'
        # 测试 SymPyExpression 构造函数是否能正确处理值错误
        raises(ValueError, lambda: SymPyExpression(src))
        # 测试 SymPyExpression 构造函数是否能正确处理模式为 'c' 的情况
        raises(ValueError, lambda: SymPyExpression(mode='c'))
        # 测试 SymPyExpression 构造函数是否能正确处理模式为 'd' 的情况
        raises(NotImplementedError, lambda: SymPyExpression(src, mode='d'))
# 如果既不是LFortran也不是CIN，则定义一个名为test_raise的函数
elif not lfortran and not cin:
    # 定义一个用于测试异常抛出的函数test_raise
    def test_raise():
        # 断言引发ImportError异常，用lambda函数调用SymPyExpression('int a;', 'c')来检查是否会抛出异常
        raises(ImportError, lambda: SymPyExpression('int a;', 'c'))
        # 断言引发ImportError异常，用lambda函数调用SymPyExpression('integer :: a', 'f')来检查是否会抛出异常
        raises(ImportError, lambda: SymPyExpression('integer :: a', 'f'))
```
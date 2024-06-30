# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_fortran_parser.py`

```
# 引入 pytest 中的 raises 函数，用于处理测试中的异常情况
from sympy.testing.pytest import raises
# 从 sympy.parsing.sym_expr 导入 SymPyExpression 类
from sympy.parsing.sym_expr import SymPyExpression
# 从 sympy.external 导入 import_module 函数
from sympy.external import import_module

# 使用 import_module 函数导入 lfortran 模块
lfortran = import_module('lfortran')

# 如果成功导入了 lfortran 模块，则执行以下代码块
if lfortran:
    # 从 sympy.codegen.ast 中导入多个类和函数
    from sympy.codegen.ast import (Variable, IntBaseType, FloatBaseType, String,
                                   Return, FunctionDefinition, Assignment,
                                   Declaration, CodeBlock)
    # 从 sympy.core 中导入 Integer 和 Float 类
    from sympy.core import Integer, Float
    # 从 sympy.core.symbol 中导入 Symbol 类
    from sympy.core.symbol import Symbol

    # 创建 SymPyExpression 的实例对象 expr1 和 expr2
    expr1 = SymPyExpression()
    expr2 = SymPyExpression()

    # 定义一个包含变量声明的字符串 src
    src = """\
    integer :: a, b, c, d
    real :: p, q, r, s
    """

    # 定义一个测试函数 test_sym_expr
    def test_sym_expr():
        # 将 src 字符串与额外的变量声明合并为 src1 字符串
        src1 = (
            src +
            """\
            d = a + b -c
            """
        )
        # 使用 SymPyExpression 类创建 SymPyExpression 对象 expr3 和 expr4
        # 分别使用 src 和 src1 字符串初始化
        expr3 = SymPyExpression(src, 'f')
        expr4 = SymPyExpression(src1, 'f')

        # 调用 return_expr 方法获取 SymPyExpression 对象的表达式列表 ls1 和 ls2
        ls1 = expr3.return_expr()
        ls2 = expr4.return_expr()

        # 使用循环验证 ls1 和 ls2 中的前7个元素均为 Declaration 对象
        for i in range(0, 7):
            assert isinstance(ls1[i], Declaration)
            assert isinstance(ls2[i], Declaration)

        # 验证 ls2 中第9个元素为 Assignment 对象
        assert isinstance(ls2[8], Assignment)

        # 使用 assert 语句逐一验证 ls1 中的前8个 Declaration 对象的内容和类型
        assert ls1[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls1[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls1[2] == Declaration(
            Variable(
                Symbol('c'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls1[3] == Declaration(
            Variable(
                Symbol('d'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls1[4] == Declaration(
            Variable(
                Symbol('p'),
                type=FloatBaseType(String('real')),
                value=Float(0.0)
            )
        )
        assert ls1[5] == Declaration(
            Variable(
                Symbol('q'),
                type=FloatBaseType(String('real')),
                value=Float(0.0)
            )
        )
        assert ls1[6] == Declaration(
            Variable(
                Symbol('r'),
                type=FloatBaseType(String('real')),
                value=Float(0.0)
            )
        )
        assert ls1[7] == Declaration(
            Variable(
                Symbol('s'),
                type=FloatBaseType(String('real')),
                value=Float(0.0)
            )
        )

        # 验证 ls2 中的第9个 Assignment 对象的内容
        assert ls2[8] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') + Symbol('b') - Symbol('c')
        )
    # 定义一个测试函数，用于测试赋值语句的解析和转换
    def test_assignment():
        # 构建包含赋值语句的源码字符串
        src1 = (
            src +
            """\
            a = b
            c = d
            p = q
            r = s
            """
        )
        # 将源码字符串转换为表达式对象
        expr1.convert_to_expr(src1, 'f')
        # 获取转换后的表达式列表
        ls1 = expr1.return_expr()
        # 遍历表达式列表的索引范围
        for iter in range(0, 12):
            # 前8个表达式应为声明语句
            if iter < 8:
                assert isinstance(ls1[iter], Declaration)
            else:
                # 后4个表达式应为赋值语句
                assert isinstance(ls1[iter], Assignment)
        # 验证第9至12个表达式是否正确表示赋值
        assert ls1[8] == Assignment(
            Variable(Symbol('a')),
            Variable(Symbol('b'))
        )
        assert ls1[9] == Assignment(
            Variable(Symbol('c')),
            Variable(Symbol('d'))
        )
        assert ls1[10] == Assignment(
            Variable(Symbol('p')),
            Variable(Symbol('q'))
        )
        assert ls1[11] == Assignment(
            Variable(Symbol('r')),
            Variable(Symbol('s'))
        )


    # 定义一个测试函数，用于测试加法表达式的解析和转换
    def test_binop_add():
        # 构建包含加法表达式的源码字符串
        src1 = (
            src +
            """\
            c = a + b
            d = a + c
            s = p + q + r
            """
        )
        # 将源码字符串转换为表达式对象
        expr1.convert_to_expr(src1, 'f')
        # 获取转换后的表达式列表
        ls1 = expr1.return_expr()
        # 遍历表达式列表的索引范围
        for iter in range(8, 11):
            # 第8至10个表达式应为赋值语句
            assert isinstance(ls1[iter], Assignment)
        # 验证第9至11个表达式是否正确表示加法表达式
        assert ls1[8] == Assignment(
            Variable(Symbol('c')),
            Symbol('a') + Symbol('b')
        )
        assert ls1[9] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') + Symbol('c')
        )
        assert ls1[10] == Assignment(
            Variable(Symbol('s')),
            Symbol('p') + Symbol('q') + Symbol('r')
        )


    # 定义一个测试函数，用于测试减法表达式的解析和转换
    def test_binop_sub():
        # 构建包含减法表达式的源码字符串
        src1 = (
            src +
            """\
            c = a - b
            d = a - c
            s = p - q - r
            """
        )
        # 将源码字符串转换为表达式对象
        expr1.convert_to_expr(src1, 'f')
        # 获取转换后的表达式列表
        ls1 = expr1.return_expr()
        # 遍历表达式列表的索引范围
        for iter in range(8, 11):
            # 第8至10个表达式应为赋值语句
            assert isinstance(ls1[iter], Assignment)
        # 验证第9至11个表达式是否正确表示减法表达式
        assert ls1[8] == Assignment(
            Variable(Symbol('c')),
            Symbol('a') - Symbol('b')
        )
        assert ls1[9] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') - Symbol('c')
        )
        assert ls1[10] == Assignment(
            Variable(Symbol('s')),
            Symbol('p') - Symbol('q') - Symbol('r')
        )
    # 定义测试函数 test_binop_mul，测试乘法运算符在表达式转换中的行为
    def test_binop_mul():
        # 构造测试用源码，将 src 和多行字符串合并成一个字符串 src1
        src1 = (
            src +
            """\
            c = a * b
            d = a * c
            s = p * q * r
            """
        )
        # 调用表达式转换函数，将 src1 转换为表达式树，标记为 'f'
        expr1.convert_to_expr(src1, 'f')
        # 获取转换后的表达式列表 ls1
        ls1 = expr1.return_expr()
        # 验证 ls1 中第 8 到 10 项是否为赋值语句对象
        for iter in range(8, 11):
            assert isinstance(ls1[iter], Assignment)
        # 验证第 9 项，应为 c = a * c 的赋值语句
        assert ls1[8] == Assignment(
            Variable(Symbol('c')),
            Symbol('a') * Symbol('b')
        )
        # 验证第 10 项，应为 d = a * c 的赋值语句
        assert ls1[9] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') * Symbol('c')
        )
        # 验证第 11 项，应为 s = p * q * r 的赋值语句
        assert ls1[10] == Assignment(
            Variable(Symbol('s')),
            Symbol('p') * Symbol('q') * Symbol('r')
        )


    # 定义测试函数 test_binop_div，测试除法运算符在表达式转换中的行为
    def test_binop_div():
        # 构造测试用源码，将 src 和多行字符串合并成一个字符串 src1
        src1 = (
            src +
            """\
            c = a / b
            d = a / c
            s = p / q
            r = q / p
            """
        )
        # 调用表达式转换函数，将 src1 转换为表达式树，标记为 'f'
        expr1.convert_to_expr(src1, 'f')
        # 获取转换后的表达式列表 ls1
        ls1 = expr1.return_expr()
        # 验证 ls1 中第 8 到 11 项是否为赋值语句对象
        for iter in range(8, 12):
            assert isinstance(ls1[iter], Assignment)
        # 验证第 9 项，应为 c = a / b 的赋值语句
        assert ls1[8] == Assignment(
            Variable(Symbol('c')),
            Symbol('a') / Symbol('b')
        )
        # 验证第 10 项，应为 d = a / c 的赋值语句
        assert ls1[9] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') / Symbol('c')
        )
        # 验证第 11 项，应为 s = p / q 的赋值语句
        assert ls1[10] == Assignment(
            Variable(Symbol('s')),
            Symbol('p') / Symbol('q')
        )
        # 验证第 12 项，应为 r = q / p 的赋值语句
        assert ls1[11] == Assignment(
            Variable(Symbol('r')),
            Symbol('q') / Symbol('p')
        )


    # 定义测试函数 test_mul_binop，测试混合乘法和除法运算符在表达式转换中的行为
    def test_mul_binop():
        # 构造测试用源码，将 src 和多行字符串合并成一个字符串 src1
        src1 = (
            src +
            """\
            d = a + b - c
            c = a * b + d
            s = p * q / r
            r = p * s + q / p
            """
        )
        # 调用表达式转换函数，将 src1 转换为表达式树，标记为 'f'
        expr1.convert_to_expr(src1, 'f')
        # 获取转换后的表达式列表 ls1
        ls1 = expr1.return_expr()
        # 验证 ls1 中第 8 到 11 项是否为赋值语句对象
        for iter in range(8, 12):
            assert isinstance(ls1[iter], Assignment)
        # 验证第 9 项，应为 d = a + b - c 的赋值语句
        assert ls1[8] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') + Symbol('b') - Symbol('c')
        )
        # 验证第 10 项，应为 c = a * b + d 的赋值语句
        assert ls1[9] == Assignment(
            Variable(Symbol('c')),
            Symbol('a') * Symbol('b') + Symbol('d')
        )
        # 验证第 11 项，应为 s = p * q / r 的赋值语句
        assert ls1[10] == Assignment(
            Variable(Symbol('s')),
            Symbol('p') * Symbol('q') / Symbol('r')
        )
        # 验证第 12 项，应为 r = p * s + q / p 的赋值语句
        assert ls1[11] == Assignment(
            Variable(Symbol('r')),
            Symbol('p') * Symbol('s') + Symbol('q') / Symbol('p')
        )
    # 定义一个测试函数
    def test_function():
        # 定义一个包含 Fortran 代码的字符串
        src1 = """\
        integer function f(a,b)
        integer :: x, y
        f = x + y
        end function
        """
        # 调用 expr1 对象的方法，将 src1 转换为表达式树，函数名为 'f'
        expr1.convert_to_expr(src1, 'f')
        # 遍历返回的表达式树中的每个元素
        for iter in expr1.return_expr():
            # 断言每个元素是 FunctionDefinition 类型的实例
            assert isinstance(iter, FunctionDefinition)
            # 断言 iter 对象等于预期的 FunctionDefinition 对象
            assert iter == FunctionDefinition(
                IntBaseType(String('integer')),  # 函数返回类型为整型
                name=String('f'),  # 函数名为 'f'
                parameters=(
                    Variable(Symbol('a')),  # 函数参数 'a'
                    Variable(Symbol('b'))   # 函数参数 'b'
                ),
                body=CodeBlock(  # 函数体的代码块
                    Declaration(  # 声明语句：声明变量 a
                        Variable(
                            Symbol('a'),
                            type=IntBaseType(String('integer')),  # 变量 a 的类型为整型
                            value=Integer(0)  # 变量 a 的初始值为 0
                        )
                    ),
                    Declaration(  # 声明语句：声明变量 b
                        Variable(
                            Symbol('b'),
                            type=IntBaseType(String('integer')),  # 变量 b 的类型为整型
                            value=Integer(0)  # 变量 b 的初始值为 0
                        )
                    ),
                    Declaration(  # 声明语句：声明变量 f
                        Variable(
                            Symbol('f'),
                            type=IntBaseType(String('integer')),  # 变量 f 的类型为整型
                            value=Integer(0)  # 变量 f 的初始值为 0
                        )
                    ),
                    Declaration(  # 声明语句：声明变量 x
                        Variable(
                            Symbol('x'),
                            type=IntBaseType(String('integer')),  # 变量 x 的类型为整型
                            value=Integer(0)  # 变量 x 的初始值为 0
                        )
                    ),
                    Declaration(  # 声明语句：声明变量 y
                        Variable(
                            Symbol('y'),
                            type=IntBaseType(String('integer')),  # 变量 y 的类型为整型
                            value=Integer(0)  # 变量 y 的初始值为 0
                        )
                    ),
                    Assignment(  # 赋值语句：f = x + y
                        Variable(Symbol('f')),  # 将表达式的结果赋给变量 f
                        Add(Symbol('x'), Symbol('y'))  # 表达式为 x + y
                    ),
                    Return(Variable(Symbol('f')))  # 返回语句：返回变量 f
                )
            )
    # 定义一个测试函数，用于测试某个表达式对象的变量生成和返回
    def test_var():
        # 将源代码转换为表达式对象，并指定函数类型为 'f'
        expr1.convert_to_expr(src, 'f')
        # 获取表达式对象返回的变量声明列表
        ls = expr1.return_expr()
        # 遍历返回的变量声明列表，确保每个元素都是 Declaration 类型的对象
        for iter in expr1.return_expr():
            assert isinstance(iter, Declaration)
        # 断言第一个变量声明是否符合预期，即变量名为 'a'，类型为整数，初始值为 0
        assert ls[0] == Declaration(
            Variable(
                Symbol('a'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        # 断言第二个变量声明是否符合预期，即变量名为 'b'，类型为整数，初始值为 0
        assert ls[1] == Declaration(
            Variable(
                Symbol('b'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        # 断言第三个变量声明是否符合预期，即变量名为 'c'，类型为整数，初始值为 0
        assert ls[2] == Declaration(
            Variable(
                Symbol('c'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        # 断言第四个变量声明是否符合预期，即变量名为 'd'，类型为整数，初始值为 0
        assert ls[3] == Declaration(
            Variable(
                Symbol('d'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        # 断言第五个变量声明是否符合预期，即变量名为 'p'，类型为实数，初始值为 0.0
        assert ls[4] == Declaration(
            Variable(
                Symbol('p'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
        # 断言第六个变量声明是否符合预期，即变量名为 'q'，类型为实数，初始值为 0.0
        assert ls[5] == Declaration(
            Variable(
                Symbol('q'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
        # 断言第七个变量声明是否符合预期，即变量名为 'r'，类型为实数，初始值为 0.0
        assert ls[6] == Declaration(
            Variable(
                Symbol('r'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
        # 断言第八个变量声明是否符合预期，即变量名为 's'，类型为实数，初始值为 0.0
        assert ls[7] == Declaration(
            Variable(
                Symbol('s'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
else:
    # 定义一个测试函数 test_raise
    def test_raise():
        # 导入 ASR2PyVisitor 类，用于处理 Fortran 解析
        from sympy.parsing.fortran.fortran_parser import ASR2PyVisitor
        # 断言引发 ImportError 异常，因为 ASR2PyVisitor 应该无法导入
        raises(ImportError, lambda: ASR2PyVisitor())
        # 断言引发 ImportError 异常，因为 SymPyExpression 应该无法导入
        raises(ImportError, lambda: SymPyExpression(' ', mode='f'))
```
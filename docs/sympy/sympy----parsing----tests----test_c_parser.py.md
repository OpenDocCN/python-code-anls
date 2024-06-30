# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_c_parser.py`

```
# 导入 SymPyExpression 类从 sympy.parsing.sym_expr 模块
from sympy.parsing.sym_expr import SymPyExpression
# 导入 raises 和 XFAIL 函数从 sympy.testing.pytest 模块
from sympy.testing.pytest import raises, XFAIL
# 导入 import_module 函数从 sympy.external 模块
from sympy.external import import_module

# 使用 import_module 函数导入 'clang.cindex' 模块，设置 import_kwargs 参数为包含 cindex 的字典
cin = import_module('clang.cindex', import_kwargs = {'fromlist': ['cindex']})

# 如果成功导入 'clang.cindex' 模块
if cin:
    # 从 sympy.codegen.ast 模块导入以下类和函数
    from sympy.codegen.ast import (Variable, String, Return,
        FunctionDefinition, Integer, Float, Declaration, CodeBlock,
        FunctionPrototype, FunctionCall, NoneToken, Assignment, Type,
        IntBaseType, SignedIntType, UnsignedIntType, FloatType,
        AddAugmentedAssignment, SubAugmentedAssignment,
        MulAugmentedAssignment, DivAugmentedAssignment,
        ModAugmentedAssignment, While)
    # 从 sympy.codegen.cnodes 模块导入以下类
    from sympy.codegen.cnodes import (PreDecrement, PostDecrement,
        PreIncrement, PostIncrement)
    # 从 sympy.core 模块导入以下类和函数
    from sympy.core import (Add, Mul, Mod, Pow, Rational,
        StrictLessThan, LessThan, StrictGreaterThan, GreaterThan,
        Equality, Unequality)
    # 从 sympy.logic.boolalg 模块导入 And, Not, Or 类和 true, false 布尔值
    from sympy.logic.boolalg import And, Not, Or
    # 从 sympy.core.symbol 模块导入 Symbol 类
    from sympy.core.symbol import Symbol
    # 导入 os 模块
    import os
    # 定义一个名为 test_bool 的测试函数
    def test_bool():
        # 构造C语言代码片段，声明两个布尔变量并赋初值为 true 和 false
        c_src1 = (
            'bool a = true, b = false;'
        )

        # 构造C语言代码片段，声明两个布尔变量并分别用整数 1 和 0 初始化
        c_src2 = (
            'bool a = 1, b = 0;'
        )

        # 构造C语言代码片段，声明两个布尔变量并分别用整数 10 初始化
        c_src3 = (
            'bool a = 10, b = 20;'
        )

        # 构造C语言代码片段，声明三个布尔变量并分别用浮点数 19.1, 9.0, 0.0 初始化
        c_src4 = (
            'bool a = 19.1, b = 9.0, c = 0.0;'
        )

        # 使用 SymPyExpression 类解析代码片段 c_src1，查找返回的表达式
        res1 = SymPyExpression(c_src1, 'c').return_expr()
        # 使用 SymPyExpression 类解析代码片段 c_src2，查找返回的表达式
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        # 使用 SymPyExpression 类解析代码片段 c_src3，查找返回的表达式
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        # 使用 SymPyExpression 类解析代码片段 c_src4，查找返回的表达式
        res4 = SymPyExpression(c_src4, 'c').return_expr()

        # 断言第一个返回表达式中的第一个声明是否符合预期的布尔变量声明
        assert res1[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=true
                )
            )

        # 断言第一个返回表达式中的第二个声明是否符合预期的布尔变量声明
        assert res1[1] == Declaration(
            Variable(Symbol('b'),
                type=Type(String('bool')),
                value=false
                )
            )

        # 断言第二个返回表达式中的第一个声明是否符合预期的布尔变量声明
        assert res2[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=true)
            )

        # 断言第二个返回表达式中的第二个声明是否符合预期的布尔变量声明
        assert res2[1] == Declaration(
            Variable(Symbol('b'),
                type=Type(String('bool')),
                value=false
                )
            )

        # 断言第三个返回表达式中的第一个声明是否符合预期的布尔变量声明
        assert res3[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=true
                )
            )

        # 断言第三个返回表达式中的第二个声明是否符合预期的布尔变量声明
        assert res3[1] == Declaration(
            Variable(Symbol('b'),
                type=Type(String('bool')),
                value=true
                )
            )

        # 断言第四个返回表达式中的第一个声明是否符合预期的布尔变量声明
        assert res4[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=true)
            )

        # 断言第四个返回表达式中的第二个声明是否符合预期的布尔变量声明
        assert res4[1] == Declaration(
            Variable(Symbol('b'),
                type=Type(String('bool')),
                value=true
                )
            )

        # 断言第四个返回表达式中的第三个声明是否符合预期的布尔变量声明
        assert res4[2] == Declaration(
            Variable(Symbol('c'),
                type=Type(String('bool')),
                value=false
                )
            )

    @XFAIL # 这里预期会因为C语法解析器的一个bug而失败。
    # 定义一个测试函数，用于测试 SymPyExpression 类的功能
    def test_function():
        # 创建第一个 C 语言源码字符串
        c_src1 = (
            'void fun1()' + '\n' +
            '{' + '\n' +
            'int a;' + '\n' +
            '}'
        )
        # 创建第二个 C 语言源码字符串
        c_src2 = (
            'int fun2()' + '\n' +
            '{'+ '\n' +
            'int a;' + '\n' +
            'return a;' + '\n' +
            '}'
        )
        # 创建第三个 C 语言源码字符串
        c_src3 = (
            'float fun3()' + '\n' +
            '{' + '\n' +
            'float b;' + '\n' +
            'return b;' + '\n' +
            '}'
        )
        # 创建第四个 C 语言源码字符串
        c_src4 = (
            'float fun4()' + '\n' +
            '{}'
        )

        # 使用 SymPyExpression 类分析第一个 C 源码字符串，返回表达式结果
        res1 = SymPyExpression(c_src1, 'c').return_expr()
        # 使用 SymPyExpression 类分析第二个 C 源码字符串，返回表达式结果
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        # 使用 SymPyExpression 类分析第三个 C 源码字符串，返回表达式结果
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        # 使用 SymPyExpression 类分析第四个 C 源码字符串，返回表达式结果
        res4 = SymPyExpression(c_src4, 'c').return_expr()

        # 断言检查第一个 C 源码字符串的分析结果是否符合预期
        assert res1[0] == FunctionDefinition(
            NoneToken(),
            name=String('fun1'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('a'),
                        type=IntBaseType(String('intc'))
                    )
                )
            )
        )

        # 断言检查第二个 C 源码字符串的分析结果是否符合预期
        assert res2[0] == FunctionDefinition(
            IntBaseType(String('intc')),
            name=String('fun2'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('a'),
                        type=IntBaseType(String('intc'))
                    )
                ),
                Return('a')
            )
        )

        # 断言检查第三个 C 源码字符串的分析结果是否符合预期
        assert res3[0] == FunctionDefinition(
            FloatType(
                String('float32'),
                nbits=Integer(32),
                nmant=Integer(23),
                nexp=Integer(8)
                ),
            name=String('fun3'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('b'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            )
                    )
                ),
                Return('b')
            )
        )

        # 断言检查第四个 C 源码字符串的分析结果是否符合预期
        assert res4[0] == FunctionPrototype(
            FloatType(
                String('float32'),
                nbits=Integer(32),
                nmant=Integer(23),
                nexp=Integer(8)
                ),
            name=String('fun4'),
            parameters=()
        )

    # 使用 XFAIL 装饰器标记的两个注释，表示预期会因 C 解析器中的 bug 而失败
    @XFAIL # this is expected to fail because of a bug in the C parser.
    @XFAIL # this is expected to fail because of a bug in the C parser.
    # 定义一个测试函数 `test_parse`
    def test_parse():
        # 创建 C 源码字符串 `c_src1`，包含两个整型变量声明
        c_src1 = (
            'int a;' + '\n' +
            'int b;' + '\n'
        )
        # 创建 C 源码字符串 `c_src2`，包含一个函数定义和一个整型变量声明
        c_src2 = (
            'void fun1()' + '\n' +
            '{' + '\n' +
            'int a;' + '\n' +
            '}'
        )

        # 打开文件 '..a.h' 用于写入
        f1 = open('..a.h', 'w')
        # 打开文件 '..b.h' 用于写入
        f2 = open('..b.h', 'w')

        # 将字符串 `c_src1` 写入文件 '..a.h'
        f1.write(c_src1)
        # 将字符串 `c_src2` 写入文件 '..b.h'
        f2. write(c_src2)

        # 关闭文件 '..a.h'
        f1.close()
        # 关闭文件 '..b.h'
        f2.close()

        # 使用 SymPyExpression 解析 '..a.h' 文件中的表达式，返回结果
        res1 = SymPyExpression('..a.h', 'c').return_expr()
        # 使用 SymPyExpression 解析 '..b.h' 文件中的表达式，返回结果
        res2 = SymPyExpression('..b.h', 'c').return_expr()

        # 删除文件 '..a.h'
        os.remove('..a.h')
        # 删除文件 '..b.h'
        os.remove('..b.h')

        # 断言第一个解析结果 `res1[0]` 是声明 `a` 的变量
        assert res1[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc'))
            )
        )
        # 断言第二个解析结果 `res1[1]` 是声明 `b` 的变量
        assert res1[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('intc'))
            )
        )
        # 断言第一个解析结果 `res2[0]` 是函数 `fun1` 的定义
        assert res2[0] == FunctionDefinition(
            NoneToken(),
            name=String('fun1'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('a'),
                        type=IntBaseType(String('intc'))
                    )
                )
            )
        )
    # 定义测试函数 test_paren_expr()
    def test_paren_expr():
        # 设置第一个 C 语言源码字符串，包含两个赋值语句
        c_src1 = (
            'int a = (1);'
            'int b = (1 + 2 * 3);'
        )

        # 设置第二个 C 语言源码字符串，包含多个赋值语句
        c_src2 = (
            'int a = 1, b = 2, c = 3;'
            'int d = (a);'
            'int e = (a + 1);'
            'int f = (a + b * c - d / e);'
        )

        # 使用 SymPyExpression 类处理第一个 C 语言源码字符串，返回处理结果
        res1 = SymPyExpression(c_src1, 'c').return_expr()
        # 使用 SymPyExpression 类处理第二个 C 语言源码字符串，返回处理结果
        res2 = SymPyExpression(c_src2, 'c').return_expr()

        # 断言检查第一个表达式处理结果是否符合预期
        assert res1[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        # 断言检查第一个表达式处理结果是否符合预期
        assert res1[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(7)
                )
            )

        # 断言检查第二个表达式处理结果是否符合预期
        assert res2[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        # 断言检查第二个表达式处理结果是否符合预期
        assert res2[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(2)
                )
            )

        # 断言检查第二个表达式处理结果是否符合预期
        assert res2[2] == Declaration(
            Variable(Symbol('c'),
                type=IntBaseType(String('intc')),
                value=Integer(3)
                )
            )

        # 断言检查第二个表达式处理结果是否符合预期
        assert res2[3] == Declaration(
            Variable(Symbol('d'),
                type=IntBaseType(String('intc')),
                value=Symbol('a')
                )
            )

        # 断言检查第二个表达式处理结果是否符合预期
        assert res2[4] == Declaration(
            Variable(Symbol('e'),
                type=IntBaseType(String('intc')),
                value=Add(
                    Symbol('a'),
                    Integer(1)
                    )
                )
            )

        # 断言检查第二个表达式处理结果是否符合预期
        assert res2[5] == Declaration(
            Variable(Symbol('f'),
                type=IntBaseType(String('intc')),
                value=Add(
                    Symbol('a'),
                    Mul(
                        Symbol('b'),
                        Symbol('c')
                        ),
                    Mul(
                        Integer(-1),
                        Symbol('d'),
                        Pow(
                            Symbol('e'),
                            Integer(-1)
                            )
                        )
                    )
                )
            )
    # 定义一个名为 test_compound_assignment_operator 的测试函数
    def test_compound_assignment_operator():
        # 设置一个包含 C 代码的字符串
        c_src = (
            'void func()' +
            '{' + '\n' +
                'int a = 100;' + '\n' +  # 在函数内部声明一个整型变量 a 并初始化为 100
                'a += 10;' + '\n' +      # 使用复合赋值运算符 +=，将 a 增加 10
                'a -= 10;' + '\n' +      # 使用复合赋值运算符 -=，将 a 减少 10
                'a *= 10;' + '\n' +      # 使用复合赋值运算符 *=，将 a 乘以 10
                'a /= 10;' + '\n' +      # 使用复合赋值运算符 /=，将 a 除以 10
                'a %= 10;' + '\n' +      # 使用复合赋值运算符 %=，将 a 取余数 10
            '}'
        )

        # 调用 SymPyExpression 类处理 C 代码，返回处理结果
        res = SymPyExpression(c_src, 'c').return_expr()

        # 使用断言验证处理结果的第一个元素
        assert res[0] == FunctionDefinition(
            NoneToken(),  # 使用 NoneToken 表示没有特定的函数名
            name=String('func'),  # 函数名为 'func'
            parameters=(),  # 没有函数参数
            body=CodeBlock(  # 函数体开始
                Declaration(  # 声明部分
                    Variable(  # 变量声明
                        Symbol('a'),  # 变量名为 'a'
                        type=IntBaseType(String('intc')),  # 类型为整型 'int'
                        value=Integer(100)  # 初始值为整数 100
                        )
                    ),
                AddAugmentedAssignment(  # 使用复合赋值运算符 += 的操作
                    Variable(Symbol('a')),  # 对变量 'a' 执行 +=
                    Integer(10)  # 将变量 'a' 增加 10
                    ),
                SubAugmentedAssignment(  # 使用复合赋值运算符 -= 的操作
                    Variable(Symbol('a')),  # 对变量 'a' 执行 -=
                    Integer(10)  # 将变量 'a' 减少 10
                    ),
                MulAugmentedAssignment(  # 使用复合赋值运算符 *= 的操作
                    Variable(Symbol('a')),  # 对变量 'a' 执行 *=
                    Integer(10)  # 将变量 'a' 乘以 10
                    ),
                DivAugmentedAssignment(  # 使用复合赋值运算符 /= 的操作
                    Variable(Symbol('a')),  # 对变量 'a' 执行 /=
                    Integer(10)  # 将变量 'a' 除以 10
                    ),
                ModAugmentedAssignment(  # 使用复合赋值运算符 %= 的操作
                    Variable(Symbol('a')),  # 对变量 'a' 执行 %=
                    Integer(10)  # 将变量 'a' 取余数 10
                    )
                )  # 函数体结束
            )

    @XFAIL # 这里预期会因为 C 解析器中的一个 bug 而失败。
else:
    # 定义一个名为 test_raise 的函数
    def test_raise():
        # 从 sympy.parsing.c.c_parser 模块中导入 CCodeConverter 类
        from sympy.parsing.c.c_parser import CCodeConverter
        # 使用 raises 断言来检查 ImportError 是否被正确地引发
        raises(ImportError, lambda: CCodeConverter())
        # 使用 raises 断言来检查 ImportError 是否被正确地引发
        raises(ImportError, lambda: SymPyExpression(' ', mode = 'c'))
```
# `D:\src\scipysrc\sympy\sympy\printing\tests\test_lambdarepr.py`

```
from sympy.concrete.summations import Sum  # 导入 Sum 类，用于表示求和表达式
from sympy.core.expr import Expr  # 导入 Expr 类，表示 Sympy 表达式的基类
from sympy.core.symbol import symbols  # 导入 symbols 函数，用于创建符号变量
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数，表示平方根
from sympy.functions.elementary.piecewise import Piecewise  # 导入 Piecewise 类，表示分段函数
from sympy.functions.elementary.trigonometric import sin  # 导入 sin 函数，表示正弦函数
from sympy.matrices.dense import MutableDenseMatrix as Matrix  # 导入 MutableDenseMatrix 类并重命名为 Matrix，用于表示可变密集矩阵
from sympy.sets.sets import Interval  # 导入 Interval 类，表示数学上的区间
from sympy.utilities.lambdify import lambdify  # 导入 lambdify 函数，用于将 Sympy 表达式转换为可调用的 Python 函数
from sympy.testing.pytest import raises  # 导入 raises 函数，用于测试异常情况

from sympy.printing.tensorflow import TensorflowPrinter  # 导入 TensorflowPrinter 类，用于将 Sympy 表达式转换为 TensorFlow 代码
from sympy.printing.lambdarepr import lambdarepr, LambdaPrinter, NumExprPrinter  # 导入 lambdarepr 函数和相关的打印器类


x, y, z = symbols("x,y,z")  # 创建符号变量 x, y, z
i, a, b = symbols("i,a,b")  # 创建符号变量 i, a, b
j, c, d = symbols("j,c,d")  # 创建符号变量 j, c, d


def test_basic():
    assert lambdarepr(x*y) == "x*y"  # 检查 lambdarepr 函数对乘法表达式的打印是否正确
    assert lambdarepr(x + y) in ["y + x", "x + y"]  # 检查 lambdarepr 函数对加法表达式的打印是否在预期的列表中
    assert lambdarepr(x**y) == "x**y"  # 检查 lambdarepr 函数对幂运算表达式的打印是否正确


def test_matrix():
    # 测试打印具有 LambdaPrinter 打印器不同的元素的矩阵
    e = x % 2  # 创建取模运算的表达式 e
    assert lambdarepr(e) != str(e)  # 断言 lambdarepr 函数对 e 的打印结果与 str(e) 不相等
    assert lambdarepr(Matrix([e])) == 'ImmutableDenseMatrix([[x % 2]])'  # 断言 lambdarepr 函数对包含 e 的矩阵的打印结果是否正确


def test_piecewise():
    # 在每种情况下，测试 eval() lambdarepr()，确保括号数量正确。如果不正确，将引发 SyntaxError。

    h = "lambda x: "

    p = Piecewise((x, x < 0))  # 创建 Piecewise 对象 p，包含一个条件 (x, x < 0)
    l = lambdarepr(p)  # 使用 lambdarepr 函数打印 Piecewise 对象 p
    eval(h + l)  # 使用 eval() 函数验证打印结果的可执行性
    assert l == "((x) if (x < 0) else None)"  # 断言打印结果是否符合预期格式

    p = Piecewise(
        (1, x < 1),
        (2, x < 2),
        (0, True)
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((1) if (x < 1) else (2) if (x < 2) else (0))"

    p = Piecewise(
        (1, x < 1),
        (2, x < 2),
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((1) if (x < 1) else (2) if (x < 2) else None)"

    p = Piecewise(
        (x, x < 1),
        (x**2, Interval(3, 4, True, False).contains(x)),
        (0, True),
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((x) if (x < 1) else (x**2) if (((x <= 4)) and ((x > 3))) else (0))"

    p = Piecewise(
        (x**2, x < 0),
        (x, x < 1),
        (2 - x, x >= 1),
        (0, True), evaluate=False
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((x**2) if (x < 0) else (x) if (x < 1)"\
                                " else (2 - x) if (x >= 1) else (0))"

    p = Piecewise(
        (x**2, x < 0),
        (x, x < 1),
        (2 - x, x >= 1), evaluate=False
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((x**2) if (x < 0) else (x) if (x < 1)"\
                    " else (2 - x) if (x >= 1) else None)"

    p = Piecewise(
        (1, x >= 1),
        (2, x >= 2),
        (3, x >= 3),
        (4, x >= 4),
        (5, x >= 5),
        (6, True)
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((1) if (x >= 1) else (2) if (x >= 2) else (3) if (x >= 3)"\
                        " else (4) if (x >= 4) else (5) if (x >= 5) else (6))"
    # 创建一个分段函数对象，根据不同的条件返回不同的值
    p = Piecewise(
        (1, x <= 1),       # 如果 x 小于等于 1，则返回 1
        (2, x <= 2),       # 如果 x 大于 1 且小于等于 2，则返回 2
        (3, x <= 3),       # 如果 x 大于 2 且小于等于 3，则返回 3
        (4, x <= 4),       # 如果 x 大于 3 且小于等于 4，则返回 4
        (5, x <= 5),       # 如果 x 大于 4 且小于等于 5，则返回 5
        (6, True)          # 对于所有其他情况，返回 6
    )
    l = lambdarepr(p)     # 将分段函数对象转换为 lambda 表达式的字符串表示形式
    eval(h + l)            # 执行字符串 h 和 l 的表达式
    # 断言 lambda 表达式字符串 l 是否等于预期的字符串
    assert l == "((1) if (x <= 1) else (2) if (x <= 2) else (3) if (x <= 3)"\
                " else (4) if (x <= 4) else (5) if (x <= 5) else (6))"
    
    # 创建另一个分段函数对象，根据不同的条件返回不同的值
    p = Piecewise(
        (1, x > 1),        # 如果 x 大于 1，则返回 1
        (2, x > 2),        # 如果 x 大于 2，则返回 2
        (3, x > 3),        # 如果 x 大于 3，则返回 3
        (4, x > 4),        # 如果 x 大于 4，则返回 4
        (5, x > 5),        # 如果 x 大于 5，则返回 5
        (6, True)          # 对于所有其他情况，返回 6
    )
    l = lambdarepr(p)     # 将分段函数对象转换为 lambda 表达式的字符串表示形式
    eval(h + l)            # 执行字符串 h 和 l 的表达式
    # 断言 lambda 表达式字符串 l 是否等于预期的字符串
    assert l == "((1) if (x > 1) else (2) if (x > 2) else (3) if (x > 3)"\
                " else (4) if (x > 4) else (5) if (x > 5) else (6))"
    
    # 创建另一个分段函数对象，根据不同的条件返回不同的值
    p = Piecewise(
        (1, x < 1),        # 如果 x 小于 1，则返回 1
        (2, x < 2),        # 如果 x 小于 2，则返回 2
        (3, x < 3),        # 如果 x 小于 3，则返回 3
        (4, x < 4),        # 如果 x 小于 4，则返回 4
        (5, x < 5),        # 如果 x 小于 5，则返回 5
        (6, True)          # 对于所有其他情况，返回 6
    )
    l = lambdarepr(p)     # 将分段函数对象转换为 lambda 表达式的字符串表示形式
    eval(h + l)            # 执行字符串 h 和 l 的表达式
    # 断言 lambda 表达式字符串 l 是否等于预期的字符串
    assert l == "((1) if (x < 1) else (2) if (x < 2) else (3) if (x < 3)"\
                " else (4) if (x < 4) else (5) if (x < 5) else (6))"
    
    # 创建一个嵌套的分段函数对象，根据复杂的条件返回不同的值
    p = Piecewise(
        (Piecewise(
            (1, x > 0),     # 如果 x 大于 0，则返回 1
            (2, True)       # 否则返回 2
        ), y > 0),          # 如果 y 大于 0，则返回内部分段函数的结果，否则返回 3
        (3, True)           # 对于所有其他情况，返回 3
    )
    l = lambdarepr(p)     # 将分段函数对象转换为 lambda 表达式的字符串表示形式
    eval(h + l)            # 执行字符串 h 和 l 的表达式
    # 断言 lambda 表达式字符串 l 是否等于预期的字符串
    assert l == "((((1) if (x > 0) else (2))) if (y > 0) else (3))"
def test_sum__1():
    # 对于每种情况，测试 eval() 的 lambdarepr()，确保其计算结果与符号表达式相同
    s = Sum(x ** i, (i, a, b))
    # 使用 lambdarepr() 获取表达式 s 的字符串表示，并与预期字符串进行比较
    l = lambdarepr(s)
    assert l == "(builtins.sum(x**i for i in range(a, b+1)))"

    args = x, a, b
    # 使用 lambdify() 将符号表达式 s 转换为可调用的函数 f
    f = lambdify(args, s)
    v = 2, 3, 8
    # 测试函数 f 的计算结果是否与通过替换参数后计算符号表达式的结果一致
    assert f(*v) == s.subs(zip(args, v)).doit()

def test_sum__2():
    s = Sum(i * x, (i, a, b))
    l = lambdarepr(s)
    assert l == "(builtins.sum(i*x for i in range(a, b+1)))"

    args = x, a, b
    f = lambdify(args, s)
    v = 2, 3, 8
    assert f(*v) == s.subs(zip(args, v)).doit()


def test_multiple_sums():
    s = Sum(i * x + j, (i, a, b), (j, c, d))
    # 使用 lambdarepr() 获取多重求和表达式 s 的字符串表示，并与预期字符串进行比较
    l = lambdarepr(s)
    assert l == "(builtins.sum(i*x + j for i in range(a, b+1) for j in range(c, d+1)))"

    args = x, a, b, c, d
    f = lambdify(args, s)
    vals = 2, 3, 4, 5, 6
    # 检查函数 f 在给定参数 vals 下的计算结果是否与符号表达式通过参数替换后的结果一致
    f_ref = s.subs(zip(args, vals)).doit()
    f_res = f(*vals)
    assert f_res == f_ref


def test_sqrt():
    prntr = LambdaPrinter({'standard' : 'python3'})
    # 测试 LambdaPrinter 对象 prntr 的 _print_Pow 方法，验证对 sqrt(x) 的打印输出是否正确
    assert prntr._print_Pow(sqrt(x), rational=False) == 'sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'


def test_settings():
    # 测试是否会引发 TypeError 异常，lambda: lambdarepr(sin(x), method="garbage") 应该引发异常
    raises(TypeError, lambda: lambdarepr(sin(x), method="garbage"))


def test_numexpr():
    # 测试 ITE 是否被重写为 Piecewise
    from sympy.logic.boolalg import ITE
    expr = ITE(x > 0, True, False, evaluate=False)
    # 检查 NumExprPrinter 对象的 doprint 方法是否正确打印 ITE 表达式
    assert NumExprPrinter().doprint(expr) == \
           "numexpr.evaluate('where((x > 0), True, False)', truediv=True)"

    from sympy.codegen.ast import Return, FunctionDefinition, Variable, Assignment
    func_def = FunctionDefinition(None, 'foo', [Variable(x)], [Assignment(y,x), Return(y**2)])
    expected = "def foo(x):\n"\
               "    y = numexpr.evaluate('x', truediv=True)\n"\
               "    return numexpr.evaluate('y**2', truediv=True)"
    # 检查 NumExprPrinter 对象的 doprint 方法是否正确打印函数定义 func_def
    assert NumExprPrinter().doprint(func_def) == expected


class CustomPrintedObject(Expr):
    def _lambdacode(self, printer):
        return 'lambda'

    def _tensorflowcode(self, printer):
        return 'tensorflow'

    def _numpycode(self, printer):
        return 'numpy'

    def _numexprcode(self, printer):
        return 'numexpr'

    def _mpmathcode(self, printer):
        return 'mpmath'


def test_printmethod():
    # 测试 LambdaPrinter、TensorflowPrinter 和 NumExprPrinter 对象的 doprint 方法
    obj = CustomPrintedObject()
    assert LambdaPrinter().doprint(obj) == 'lambda'
    assert TensorflowPrinter().doprint(obj) == 'tensorflow'
    assert NumExprPrinter().doprint(obj) == "numexpr.evaluate('numexpr', truediv=True)"

    assert NumExprPrinter().doprint(Piecewise((y, x >= 0), (z, x < 0))) == \
            "numexpr.evaluate('where((x >= 0), y, z)', truediv=True)"
```
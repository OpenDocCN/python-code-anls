# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_applyfunc.py`

```
# 导入需要的符号、虚拟变量和函数类别
from sympy.core.symbol import symbols, Dummy
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.core.function import Lambda
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matmul import MatMul
from sympy.simplify.simplify import simplify

# 定义矩阵符号 X 和 Y
X = MatrixSymbol("X", 3, 3)
Y = MatrixSymbol("Y", 3, 3)

# 定义一个符号 k
k = symbols("k")

# 定义具有符号 k 的矩阵符号 Xk
Xk = MatrixSymbol("X", k, k)

# 将 X 转换为明确的矩阵对象
Xd = X.as_explicit()

# 定义符号变量 x, y, z, t
x, y, z, t = symbols("x y z t")

# 定义测试函数 test_applyfunc_matrix
def test_applyfunc_matrix():
    # 定义一个虚拟变量 x
    x = Dummy('x')
    # 定义一个双参数 Lambda 函数，表示 x 的平方
    double = Lambda(x, x**2)

    # 对明确的矩阵 Xd 应用 ElementwiseApplyFunction 函数
    expr = ElementwiseApplyFunction(double, Xd)
    # 断言 expr 是 ElementwiseApplyFunction 类的实例
    assert isinstance(expr, ElementwiseApplyFunction)
    # 断言应用表达式的结果等于 Xd 每个元素平方的结果
    assert expr.doit() == Xd.applyfunc(lambda x: x**2)
    # 断言表达式的形状为 (3, 3)
    assert expr.shape == (3, 3)
    # 断言表达式的构造函数等于其自身
    assert expr.func(*expr.args) == expr
    # 断言简化后的表达式等于其自身
    assert simplify(expr) == expr
    # 断言表达式的第一个元素为 double 函数应用于 Xd 第一个元素的结果
    assert expr[0, 0] == double(Xd[0, 0])

    # 对矩阵 X 应用 ElementwiseApplyFunction 函数
    expr = ElementwiseApplyFunction(double, X)
    # 断言 expr 是 ElementwiseApplyFunction 类的实例
    assert isinstance(expr, ElementwiseApplyFunction)
    # 断言 expr 的应用结果是 ElementwiseApplyFunction 类的实例
    assert isinstance(expr.doit(), ElementwiseApplyFunction)
    # 断言 expr 等于 X 应用 double 函数的结果
    assert expr == X.applyfunc(double)
    # 断言表达式的构造函数等于其自身
    assert expr.func(*expr.args) == expr

    # 对表达式 exp(X*Y) 应用 ElementwiseApplyFunction 函数
    expr = ElementwiseApplyFunction(exp, X*Y)
    # 断言 expr 的表达式部分为 X*Y
    assert expr.expr == X*Y
    # 断言 expr 的函数部分 dummy_eq(Lambda(x, exp(x)))
    assert expr.function.dummy_eq(Lambda(x, exp(x)))
    # 断言 expr 的 dummy_eq((X*Y).applyfunc(exp))
    assert expr.dummy_eq((X*Y).applyfunc(exp))
    # 断言表达式的构造函数等于其自身
    assert expr.func(*expr.args) == expr

    # 断言 X*expr 是 MatMul 类的实例
    assert isinstance(X*expr, MatMul)
    # 断言 (X*expr) 的形状为 (3, 3)
    assert (X*expr).shape == (3, 3)

    # 定义矩阵符号 Z，形状为 (2, 3)
    Z = MatrixSymbol("Z", 2, 3)
    # 断言 (Z*expr) 的形状为 (2, 3)
    assert (Z*expr).shape == (2, 3)

    # 定义表达式为 exp(Z.T)*exp(Z)，形状为 (3, 3)
    expr = ElementwiseApplyFunction(exp, Z.T)*ElementwiseApplyFunction(exp, Z)
    assert expr.shape == (3, 3)
    # 交换顺序后的表达式形状为 (2, 2)
    expr = ElementwiseApplyFunction(exp, Z)*ElementwiseApplyFunction(exp, Z.T)
    assert expr.shape == (2, 2)

    # 定义矩阵 M
    M = Matrix([[x, y], [z, t]])
    # 对矩阵 M 应用 ElementwiseApplyFunction 函数
    expr = ElementwiseApplyFunction(sin, M)
    # 断言 expr 是 ElementwiseApplyFunction 类的实例
    assert isinstance(expr, ElementwiseApplyFunction)
    # 断言 expr 的函数部分 dummy_eq(Lambda(x, sin(x)))
    assert expr.function.dummy_eq(Lambda(x, sin(x)))
    # 断言 expr 的表达式部分为 M
    assert expr.expr == M
    # 断言 expr 的应用结果等于 M.applyfunc(sin)
    assert expr.doit() == M.applyfunc(sin)
    # 断言 expr 的应用结果等于 Matrix([[sin(x), sin(y)], [sin(z), sin(t)]])
    assert expr.doit() == Matrix([[sin(x), sin(y)], [sin(z), sin(t)]])
    # 断言表达式的构造函数等于其自身
    assert expr.func(*expr.args) == expr

    # 对矩阵符号 Xk 应用 ElementwiseApplyFunction 函数
    expr = ElementwiseApplyFunction(double, Xk)
    # 断言 expr 的应用结果等于其自身
    assert expr.doit() == expr
    # 断言对 k 赋值为 2 后的 expr 的形状为 (2, 2)
    assert expr.subs(k, 2).shape == (2, 2)
    # 断言 expr*expr 的形状为 (k, k)
    assert (expr*expr).shape == (k, k)
    # 定义矩阵符号 M，形状为 (k, t)
    M = MatrixSymbol("M", k, t)
    # 定义表达式 expr2 为 M.T*expr*M
    expr2 = M.T*expr*M
    # 断言 expr2 是 MatMul 类的实例
    assert isinstance(expr2, MatMul)
    # 断言 expr2 的第二个参数为 expr
    assert expr2.args[1] == expr
    # 断言 expr2 的形状为 (t, t)
    assert expr2.shape == (t, t)
    # 定义表达式 expr3 为 expr*M
    expr3 = expr*M
    # 断言 expr3 的形状为 (k, t)

# 定义测试函数 test_applyfunc_entry
def test_applyfunc_entry():
    # 对矩阵符号 X 应用 ElementwiseApplyFunction 函数 sin
    af = X.applyfunc(sin)
    # 断言 af 的第一个元素等于 sin(X[0, 0])
    assert af[0, 0] == sin(X[0, 0])

    # 对明确的矩阵 Xd 应用 ElementwiseApplyFunction 函数 sin
    af = Xd.applyfunc(sin)
    # 断言 af 的第一个元素等于 sin(X[0, 0])

# 定义测试函数 test_applyfunc_as_explicit
def test_applyfunc_as_explicit():
    # 对矩阵符号 X 应用 ElementwiseApplyFunction 函数 sin
    af = X.applyfunc(sin)
    # 断言：验证 af 对象调用 as_explicit() 方法的返回值是否等于给定的矩阵
    assert af.as_explicit() == Matrix([
        # 第一行矩阵元素为 X 矩阵中各元素的正弦值
        [sin(X[0, 0]), sin(X[0, 1]), sin(X[0, 2])],
        # 第二行矩阵元素为 X 矩阵中各元素的正弦值
        [sin(X[1, 0]), sin(X[1, 1]), sin(X[1, 2])],
        # 第三行矩阵元素为 X 矩阵中各元素的正弦值
        [sin(X[2, 0]), sin(X[2, 1]), sin(X[2, 2])],
    ])
# 定义测试函数，用于测试矩阵的元素级函数应用和转置操作
def test_applyfunc_transpose():
    # 对矩阵 Xk 中的每个元素应用 sin 函数，返回新的矩阵 af
    af = Xk.applyfunc(sin)
    # 断言转置后的矩阵 af.T 与原始矩阵 Xk.T 分别经过 dummy_eq 函数后相等
    assert af.T.dummy_eq(Xk.T.applyfunc(sin))


# 定义测试函数，用于测试形状为 1x1 的矩阵 M 的元素级函数应用和函数对象应用
def test_applyfunc_shape_11_matrices():
    # 创建一个形状为 1x1 的矩阵符号 M
    M = MatrixSymbol("M", 1, 1)
    
    # 定义一个 lambda 函数 double，用于将输入参数乘以 2
    double = Lambda(x, x*2)

    # 对矩阵 M 中的每个元素应用 sin 函数，返回新的表达式 expr
    expr = M.applyfunc(sin)
    # 断言 expr 是 ElementwiseApplyFunction 类的实例
    assert isinstance(expr, ElementwiseApplyFunction)

    # 对矩阵 M 中的每个元素应用 double 函数，返回新的表达式 expr
    expr = M.applyfunc(double)
    # 断言 expr 是 MatMul 类的实例，且 expr 等于 2*M
    assert isinstance(expr, MatMul)
    assert expr == 2*M
```
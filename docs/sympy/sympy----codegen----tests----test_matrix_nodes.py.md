# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_matrix_nodes.py`

```
# 导入需要的符号、函数、矩阵和简化函数
from sympy.core.symbol import symbols
from sympy.core.function import Function
from sympy.matrices.dense import Matrix
from sympy.matrices.dense import zeros
from sympy.simplify.simplify import simplify
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.utilities.lambdify import lambdify
from sympy.printing.numpy import NumPyPrinter
from sympy.testing.pytest import skip
from sympy.external import import_module


# 定义测试函数，验证在解决矩阵问题时是否存在问题
def test_matrix_solve_issue_24862():
    # 创建一个3x3的符号矩阵A
    A = Matrix(3, 3, symbols('a:9'))
    # 创建一个3x1的符号矩阵b
    b = Matrix(3, 1, symbols('b:3'))
    # 对MatrixSolve对象进行哈希处理
    hash(MatrixSolve(A, b))


# 定义测试函数，验证在求解导数时的精确性
def test_matrix_solve_derivative_exact():
    # 创建一个符号变量q
    q = symbols('q')
    # 使用符号函数创建符号函数a11, a12, a21, a22, b1, b2
    a11, a12, a21, a22, b1, b2 = (
        f(q) for f in symbols('a11 a12 a21 a22 b1 b2', cls=Function))
    # 创建一个2x2的符号矩阵A
    A = Matrix([[a11, a12], [a21, a22]])
    # 创建一个2x1的符号矩阵b
    b = Matrix([b1, b2])
    # 使用LU分解求解线性方程组，并计算其对q的导数
    x_lu = A.LUsolve(b)
    # 使用LU分解和矩阵求导数，计算dxdq_lu
    dxdq_lu = A.LUsolve(b.diff(q) - A.diff(q) * A.LUsolve(b))
    # 断言简化后的差值为零矩阵
    assert simplify(x_lu.diff(q) - dxdq_lu) == zeros(2, 1)
    # 使用MatrixSolve对象计算导数dxdq_ms，并声明其等效于dxdq_lu
    dxdq_ms = MatrixSolve(A, b.diff(q) - A.diff(q) * MatrixSolve(A, b))
    assert MatrixSolve(A, b).diff(q) == dxdq_ms


# 定义测试函数，验证在求解导数时使用NumPy的准确性
def test_matrix_solve_derivative_numpy():
    # 导入numpy模块
    np = import_module('numpy')
    # 如果numpy未安装，则跳过测试
    if not np:
        skip("numpy not installed.")
    # 创建一个符号变量q
    q = symbols('q')
    # 使用符号函数创建符号函数a11, a12, a21, a22, b1, b2
    a11, a12, a21, a22, b1, b2 = (
        f(q) for f in symbols('a11 a12 a21 a22 b1 b2', cls=Function))
    # 创建一个2x2的符号矩阵A
    A = Matrix([[a11, a12], [a21, a22]])
    # 创建一个2x1的符号矩阵b
    b = Matrix([b1, b2])
    # 使用LU分解求解线性方程组，并计算其对q的导数
    dx_lu = A.LUsolve(b).diff(q)
    # 创建一个符号替换字典subs，用于数值替换
    subs = {a11.diff(q): 0.2, a12.diff(q): 0.3, a21.diff(q): 0.1,
            a22.diff(q): 0.5, b1.diff(q): 0.4, b2.diff(q): 0.9,
            a11: 1.3, a12: 0.5, a21: 1.2, a22: 4, b1: 6.2, b2: 3.5}
    # 分离字典subs的键和值
    p, p_vals = zip(*subs.items())
    # 使用MatrixSolve对象计算导数dx_sm
    dx_sm = MatrixSolve(A, b).diff(q)
    # 使用numpy.testing.assert_allclose函数验证数值计算的准确性
    np.testing.assert_allclose(
        lambdify(p, dx_sm, printer=NumPyPrinter)(*p_vals),
        lambdify(p, dx_lu, printer=NumPyPrinter)(*p_vals))
```
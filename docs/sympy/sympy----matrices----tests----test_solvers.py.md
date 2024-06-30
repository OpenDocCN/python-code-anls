# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_solvers.py`

```
# 导入 pytest 库，用于运行测试
import pytest
# 从 sympy.core.function 模块导入 expand_mul 函数
from sympy.core.function import expand_mul
# 从 sympy.core.numbers 模块导入 I 和 Rational 类
from sympy.core.numbers import (I, Rational)
# 从 sympy.core.singleton 模块导入 S 单例对象
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Symbol 和 symbols 符号变量
from sympy.core.symbol import (Symbol, symbols)
# 从 sympy.core.sympify 模块导入 sympify 函数
from sympy.core.sympify import sympify
# 从 sympy.simplify.simplify 模块导入 simplify 函数
from sympy.simplify.simplify import simplify
# 从 sympy.matrices.exceptions 模块导入 ShapeError 和 NonSquareMatrixError 异常
from sympy.matrices.exceptions import (ShapeError, NonSquareMatrixError)
# 从 sympy.matrices 模块导入各种矩阵类和函数：ImmutableMatrix, Matrix, eye, ones, ImmutableDenseMatrix, dotprodsimp
from sympy.matrices import (
    ImmutableMatrix, Matrix, eye, ones, ImmutableDenseMatrix, dotprodsimp)
# 从 sympy.matrices.determinant 模块导入 _det_laplace 函数
from sympy.matrices.determinant import _det_laplace
# 从 sympy.testing.pytest 模块导入 raises 函数
from sympy.testing.pytest import raises
# 从 sympy.matrices.exceptions 模块导入 NonInvertibleMatrixError 异常
from sympy.matrices.exceptions import NonInvertibleMatrixError
# 从 sympy.polys.matrices.exceptions 模块导入 DMShapeError 异常
from sympy.polys.matrices.exceptions import DMShapeError
# 从 sympy.solvers.solveset 模块导入 linsolve 函数
from sympy.solvers.solveset import linsolve
# 从 sympy.abc 模块导入 x, y 符号变量
from sympy.abc import x, y

# 定义一个测试函数，用于测试 issue 17247 中的表达式膨胀问题
def test_issue_17247_expression_blowup_29():
    # 定义一个 Matrix 对象 M，包含指定的数学表达式
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    # 启用 dotprodsimp 优化上下文管理器
    with dotprodsimp(True):
        # 使用 Gauss-Jordan 方法求解线性方程组 M * ones(4, 1) = 0，并断言其结果
        assert M.gauss_jordan_solve(ones(4, 1)) == (Matrix(S('''[
            [                          -32549314808672/3306971225785 - 17397006745216*I/3306971225785],
            [                               67439348256/3306971225785 - 9167503335872*I/3306971225785],
            [-15091965363354518272/21217636514687010905 + 16890163109293858304*I/21217636514687010905],
            [                                                          -11328/952745 + 87616*I/952745]]''')), Matrix(0, 1, []))

# 定义另一个测试函数，用于测试 issue 17247 中的表达式膨胀问题
def test_issue_17247_expression_blowup_30():
    # 定义一个 Matrix 对象 M，包含指定的数学表达式
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    # 启用 dotprodsimp 优化上下文管理器
    with dotprodsimp(True):
        # 使用 Cholesky 分解方法求解线性方程组 M * ones(4, 1) = 0，并断言其结果
        assert M.cholesky_solve(ones(4, 1)) == Matrix(S('''[
            [                          -32549314808672/3306971225785 - 17397006745216*I/3306971225785],
            [                               67439348256/3306971225785 - 9167503335872*I/3306971225785],
            [-15091965363354518272/21217636514687010905 + 16890163109293858304*I/21217636514687010905],
            [                                                          -11328/952745 + 87616*I/952745]]'''))

# @XFAIL # This calculation hangs with dotprodsimp.
# 定义一个被标记为预期失败的测试函数，用于测试 issue 17247 中的表达式膨胀问题
# def test_issue_17247_expression_blowup_31():
#     M = Matrix([
#         [x + 1, 1 - x,     0,     0],
#         [1 - x, x + 1,     0, x + 1],
#         [    0, 1 - x, x + 1,     0],
#         [    0,     0,     0, x + 1]])
#     with dotprodsimp(True):
# 测试 LUsolve 方法在 iszerofunc 参数为 lambda 表达式时的功能
def test_LUsolve_iszerofunc():
    # 创建一个 2x2 的矩阵 M，元素为符号表达式
    M = Matrix([[(x + 1)**2 - (x**2 + 2*x + 1), x], [x, 0]])
    # 创建一个 2x1 的矩阵 b，元素为符号表达式
    b = Matrix([1, 1])
    # 定义一个 lambda 函数 is_zero_func，用于确定参数是否为零
    is_zero_func = lambda e: False if e._random() else True
    # 期望的解 x_exp，元素为符号表达式
    x_exp = Matrix([1/x, (1-(-x**2 - 2*x + (x+1)**2 - 1)/x)/x])
    # 断言 LUsolve 方法返回的解与期望解 x_exp 的差为零矩阵
    assert (x_exp - M.LUsolve(b, iszerofunc=is_zero_func)) == Matrix([0, 0])


# 测试 LUsolve 方法的特定问题
def test_issue_17247_expression_blowup_32():
    # 创建一个 4x4 的矩阵 M，元素为符号表达式
    M = Matrix([
        [x + 1, 1 - x,     0,     0],
        [1 - x, x + 1,     0, x + 1],
        [    0, 1 - x, x + 1,     0],
        [    0,     0,     0, x + 1]])
    # 使用 dotprodsimp 上下文环境
    with dotprodsimp(True):
        # 断言 LUsolve 方法返回的结果与期望解矩阵相等
        assert M.LUsolve(ones(4, 1)) == Matrix([
            [(x + 1)/(4*x)],
            [(x - 1)/(4*x)],
            [(x + 1)/(4*x)],
            [    1/(x + 1)]])


# 测试 LUsolve 方法的多种情况
def test_LUsolve():
    # 情况 1
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.LUsolve(b)
    # 断言解与期望解 x 相等
    assert soln == x
    
    # 情况 2
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.LUsolve(b)
    # 断言解与期望解 x 相等
    assert soln == x
    
    # 情况 3
    A = Matrix([[2, 1], [1, 0], [1, 0]])   # issue 14548
    b = Matrix([3, 1, 1])
    # 断言解与期望解相等
    assert A.LUsolve(b) == Matrix([1, 1])
    b = Matrix([3, 1, 2])                  # inconsistent
    # 断言解抛出 ValueError 异常
    raises(ValueError, lambda: A.LUsolve(b))
    
    # 情况 4
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4],
                [2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix([2, 1, -4])
    b = A*x
    soln = A.LUsolve(b)
    # 断言解与期望解 x 相等
    assert soln == x
    
    # 情况 5
    A = Matrix([[0, -1, 2], [5, 10, 7]])  # underdetermined
    x = Matrix([-1, 2, 0])
    b = A*x
    # 断言解抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: A.LUsolve(b))
    
    # 情况 6
    A = Matrix(4, 4, lambda i, j: 1/(i+j+1) if i != 3 else 0)
    b = Matrix.zeros(4, 1)
    # 断言解抛出 NonInvertibleMatrixError 异常
    raises(NonInvertibleMatrixError, lambda: A.LUsolve(b))


# 测试 QRsolve 方法的多种情况
def test_QRsolve():
    # 情况 1
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.QRsolve(b)
    # 断言解与期望解 x 相等
    assert soln == x
    
    # 情况 2
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.QRsolve(b)
    # 断言解与期望解 x 相等
    assert soln == x


# 测试错误情况
def test_errors():
    # 断言解抛出 ShapeError 异常
    raises(ShapeError, lambda: Matrix([1]).LUsolve(Matrix([[1, 2], [3, 4]])))


# 测试 Cholesky_solve 方法的一种情况
def test_cholesky_solve():
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.cholesky_solve(b)
    # 断言解与期望解 x 相等
    assert soln == x
    # 创建一个 3x3 的矩阵 A，其元素由列表提供
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    
    # 创建一个 3x1 的矩阵 x，其元素由列表提供
    x = Matrix(3, 1, [-1, 2, 5])
    
    # 计算矩阵乘法 A*x，得到结果向量 b
    b = A*x
    
    # 使用 Cholesky 分解解决线性方程组 A * soln = b，soln 是未知向量
    soln = A.cholesky_solve(b)
    
    # 断言求解出的 soln 向量与预期的 x 向量相等
    assert soln == x
    
    # 创建一个 2x2 的矩阵 A，其元素由元组提供
    A = Matrix(((1, 5), (5, 1)))
    
    # 创建一个 2x1 的矩阵 x，其元素由元组提供
    x = Matrix((4, -3))
    
    # 计算矩阵乘法 A*x，得到结果向量 b
    b = A*x
    
    # 使用 Cholesky 分解解决线性方程组 A * soln = b
    soln = A.cholesky_solve(b)
    
    # 断言求解出的 soln 向量与预期的 x 向量相等
    assert soln == x
    
    # 创建一个 2x2 的矩阵 A，其中包含复数元素，由元组提供
    A = Matrix(((9, 3*I), (-3*I, 5)))
    
    # 创建一个 2x1 的矩阵 x，其元素由元组提供
    x = Matrix((-2, 1))
    
    # 计算矩阵乘法 A*x，得到结果向量 b
    b = A*x
    
    # 使用 Cholesky 分解解决线性方程组 A * soln = b
    soln = A.cholesky_solve(b)
    
    # 断言求解出的 soln 向量经过展开乘法后与预期的 x 向量相等
    assert expand_mul(soln) == x
    
    # 创建一个 2x2 的矩阵 A，其中包含复数元素，由元组提供
    A = Matrix(((9*I, 3), (-3 + I, 5)))
    
    # 创建一个 2x1 的矩阵 x，其元素由元组提供
    x = Matrix((2 + 3*I, -1))
    
    # 计算矩阵乘法 A*x，得到结果向量 b
    b = A*x
    
    # 使用 Cholesky 分解解决线性方程组 A * soln = b
    soln = A.cholesky_solve(b)
    
    # 断言求解出的 soln 向量经过展开乘法后与预期的 x 向量相等
    assert expand_mul(soln) == x
    
    # 创建符号变量 a00, a01, a11, b0, b1
    a00, a01, a11, b0, b1 = symbols('a00, a01, a11, b0, b1')
    
    # 创建一个 2x2 的矩阵 A，其中元素由符号变量提供
    A = Matrix(((a00, a01), (a01, a11)))
    
    # 创建一个 2x1 的矩阵 b，其元素由符号变量提供
    b = Matrix((b0, b1))
    
    # 使用 Cholesky 分解解决线性方程组 A * soln = b
    x = A.cholesky_solve(b)
    
    # 断言简化后的 A*x 结果与预期的 b 向量相等
    assert simplify(A*x) == b
# 定义测试函数 test_LDLsolve，用于测试矩阵的 LDL 分解求解线性方程组的功能
def test_LDLsolve():
    # 创建一个 3x3 的矩阵 A
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    # 创建一个 3x1 的列向量 x
    x = Matrix(3, 1, [3, 7, 5])
    # 计算右端向量 b = A*x
    b = A*x
    # 使用 LDL 分解求解线性方程组 A*soln = b，并断言得到的解 soln 等于预期的 x
    soln = A.LDLsolve(b)
    assert soln == x

    # 创建另一个 3x3 的矩阵 A
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    # 创建另一个 3x1 的列向量 x
    x = Matrix(3, 1, [-1, 2, 5])
    # 计算右端向量 b = A*x
    b = A*x
    # 使用 LDL 分解求解线性方程组 A*soln = b，并断言得到的解 soln 等于预期的 x
    soln = A.LDLsolve(b)
    assert soln == x

    # 创建一个复数域上的 2x2 矩阵 A
    A = Matrix(((9, 3*I), (-3*I, 5)))
    # 创建一个 2x1 的复数列向量 x
    x = Matrix((-2, 1))
    # 计算右端向量 b = A*x
    b = A*x
    # 使用 LDL 分解求解复数域上的线性方程组 A*soln = b，并断言得到的解 soln 扩展乘积后等于预期的 x
    soln = A.LDLsolve(b)
    assert expand_mul(soln) == x

    # 创建另一个复数域上的 2x2 矩阵 A
    A = Matrix(((9*I, 3), (-3 + I, 5)))
    # 创建另一个 2x1 的复数列向量 x
    x = Matrix((2 + 3*I, -1))
    # 计算右端向量 b = A*x
    b = A*x
    # 使用 LDL 分解求解复数域上的线性方程组 A*soln = b，并断言得到的解 soln 扩展乘积后等于预期的 x
    soln = A.LDLsolve(b)
    assert expand_mul(soln) == x

    # 创建一个 2x2 的矩阵 A
    A = Matrix(((9, 3), (3, 9)))
    # 创建一个 2x1 的列向量 x
    x = Matrix((1, 1))
    # 计算右端向量 b = A*x
    b = A * x
    # 使用 LDL 分解求解线性方程组 A*soln = b，并断言得到的解 soln 扩展乘积后等于预期的 x
    soln = A.LDLsolve(b)
    assert expand_mul(soln) == x

    # 创建一个 2x3 的矩阵 A
    A = Matrix([[-5, -3, -4], [-3, -7, 7]])
    # 创建一个 3x1 的列向量 x
    x = Matrix([[8], [7], [-2]])
    # 计算右端向量 b = A*x
    b = A * x
    # 调用 LDLsolve 方法，预期会抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: A.LDLsolve(b))


```  
# 定义测试函数 test_lower_triangular_solve，用于测试下三角矩阵求解线性方程组的功能
def test_lower_triangular_solve():

    # 断言下三角矩阵和非方阵求解会抛出 NonSquareMatrixError 异常
    raises(NonSquareMatrixError,
           lambda: Matrix([1, 0]).lower_triangular_solve(Matrix([0, 1])))
    raises(ShapeError,
           lambda: Matrix([[1, 0], [0, 1]]).lower_triangular_solve(Matrix([1])))
    raises(ValueError,
           lambda: Matrix([[2, 1], [1, 2]]).lower_triangular_solve(
               Matrix([[1, 0], [0, 1]])))

    # 创建一个 2x2 的单位下三角矩阵 A
    A = Matrix([[1, 0], [0, 1]])
    # 创建一个 2x2 的矩阵 B
    B = Matrix([[x, y], [y, x]])
    # 断言 A 下三角矩阵求解线性方程组 A*soln = B 后得到的解 soln 等于预期的 B
    assert A.lower_triangular_solve(B) == B
    # 创建一个 2x2 的矩阵 C
    C = Matrix([[4, 8], [2, 9]])
    # 断言 A 下三角矩阵求解线性方程组 A*soln = C 后得到的解 soln 等于预期的 C
    assert A.lower_triangular_solve(C) == C


# 定义测试函数 test_upper_triangular_solve，用于测试上三角矩阵求解线性方程组的功能
def test_upper_triangular_solve():

    # 断言上三角矩阵和非方阵求解会抛出 NonSquareMatrixError 异常
    raises(NonSquareMatrixError,
           lambda: Matrix([1, 0]).upper_triangular_solve(Matrix([0, 1])))
    raises(ShapeError,
           lambda: Matrix([[1, 0], [0, 1]]).upper_triangular_solve(Matrix([1])))
    raises(TypeError,
           lambda: Matrix([[2, 1], [1, 2]]).upper_triangular_solve(
               Matrix([[1, 0], [0, 1]])))

    # 创建一个 2x2 的单位上三角矩阵 A
    A = Matrix([[1, 0], [0, 1]])
    # 创建一个 2x2 的矩阵 B
    B = Matrix([[x, y], [y, x]])
    # 断言 A 上三角矩阵求解线性方程组 A*soln = B 后得到的解 soln 等于预期的 B
    assert A.upper_triangular_solve(B) == B
    # 创建一个 2x2 的矩阵 C
    C = Matrix([[2, 4], [3, 8]])
    # 断言 A 上三角矩阵求解线性方程组 A*soln = C 后得到的解 soln 等于预期的 C
    assert A.upper_triangular_solve(C) == C


# 定义测试函数 test_diagonal_solve，用于测试对角矩阵求解线性方程组的功能
def test_diagonal_solve():
    # 断言对角矩阵和非方阵求解会抛出 TypeError 异常
    raises(TypeError, lambda: Matrix([1, 1]).diagonal_solve(Matrix([1])))
    # 创建一个 2x2 的对角矩阵 A
    A = Matrix([[1, 0], [0, 1]]) * 2
    # 创建一个 2x2 的矩阵 B
    B = Matrix([[x, y], [y, x]])
    # 断言对角矩阵求解线性方程组 A*soln = B 后得到的解 soln 等于预期的 B/2
    assert A.diagonal_solve(B) == B / 2

    # 创建一个非对角矩阵 A
    A = Matrix([[1, 0], [1, 2]])
    # 断言对角矩阵求解非对角矩阵的线性方程组会抛出 TypeError 异常
    raises(TypeError, lambda: A.diagonal_solve(B))


# 定义测试函数 test_pinv_solve，用于测试广义逆求解线性方程组的功能
def test_pinv_solve():
    # 全定系统（唯一解，与其他求
    # 创建一个 2x3 的矩阵 A，表示一个欠定系统（有无限解）。
    A = Matrix([[1, 0, 1], [0, 1, 1]])
    # 创建一个长度为 2 的列向量 B。
    B = Matrix([5, 7])
    # 使用广义逆解决线性方程组，并得到解 solution。
    solution = A.pinv_solve(B)
    # 创建一个空字典 w 用于存储解 solution 中的符号变量。
    w = {}
    # 遍历 solution 中的所有符号变量，并存储其名称及对应的符号对象。
    for s in solution.atoms(Symbol):
        # 提取解中使用的虚拟符号。
        w[s.name] = s
    # 断言解 solution 是否满足特定的线性方程组关系。
    assert solution == Matrix([[w['w0_0']/3 + w['w1_0']/3 - w['w2_0']/3 + 1],
                               [w['w0_0']/3 + w['w1_0']/3 - w['w2_0']/3 + 3],
                               [-w['w0_0']/3 - w['w1_0']/3 + w['w2_0']/3 + 4]])
    # 断言 A * A.pinv() * B 是否等于 B，用于验证解的准确性。
    assert A * A.pinv() * B == B
    
    # 创建一个 3x2 的矩阵 A，表示一个超定系统（最小二乘解）。
    A = Matrix([[1, 0], [0, 0], [0, 1]])
    # 创建一个长度为 3 的列向量 B。
    B = Matrix([3, 2, 1])
    # 断言使用广义逆解得到的解是否等于预期的向量。
    assert A.pinv_solve(B) == Matrix([3, 1])
    
    # 断言 A * A.pinv() * B 是否不等于 B，用于证明解不是精确解。
    assert A * A.pinv() * B != B
def test_pinv_rank_deficient():
    # Test the four properties of the pseudoinverse for various matrices.
    # 定义三个矩阵，用于测试伪逆的四个性质
    As = [Matrix([[1, 1, 1], [2, 2, 2]]),
          Matrix([[1, 0], [0, 0]]),
          Matrix([[1, 2], [2, 4], [3, 6]])]

    # 遍历每个矩阵进行测试
    for A in As:
        # 计算伪逆矩阵，使用方法"RD"
        A_pinv = A.pinv(method="RD")
        # 计算 A * A_pinv 和 A_pinv * A
        AAp = A * A_pinv
        ApA = A_pinv * A
        # 断言验证性质：AAp * A == A
        assert simplify(AAp * A) == A
        # 断言验证性质：ApA * A_pinv == A_pinv
        assert simplify(ApA * A_pinv) == A_pinv
        # 断言验证共轭转置等于自身
        assert AAp.H == AAp
        assert ApA.H == ApA

    # 再次遍历每个矩阵进行测试
    for A in As:
        # 计算伪逆矩阵，使用方法"ED"
        A_pinv = A.pinv(method="ED")
        # 计算 A * A_pinv 和 A_pinv * A
        AAp = A * A_pinv
        ApA = A_pinv * A
        # 断言验证性质：AAp * A == A
        assert simplify(AAp * A) == A
        # 断言验证性质：ApA * A_pinv == A_pinv
        assert simplify(ApA * A_pinv) == A_pinv
        # 断言验证共轭转置等于自身
        assert AAp.H == AAp
        assert ApA.H == ApA

    # 测试处理秩亏矩阵的求解
    A = Matrix([[1, 0], [0, 0]])
    # 精确非唯一解
    B = Matrix([3, 0])
    solution = A.pinv_solve(B)
    # 提取解中的符号
    w1 = solution.atoms(Symbol).pop()
    # 断言验证符号名为'w1_0'
    assert w1.name == 'w1_0'
    # 断言验证解为 Matrix([3, w1])
    assert solution == Matrix([3, w1])
    # 断言验证 A * A.pinv() * B == B
    assert A * A.pinv() * B == B
    # 最小二乘，非唯一解
    B = Matrix([3, 1])
    solution = A.pinv_solve(B)
    # 提取解中的符号
    w1 = solution.atoms(Symbol).pop()
    # 断言验证符号名为'w1_0'
    assert w1.name == 'w1_0'
    # 断言验证解为 Matrix([3, w1])
    assert solution == Matrix([3, w1])
    # 断言验证 A * A.pinv() * B != B

def test_gauss_jordan_solve():

    # 正方形，满秩，唯一解
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    b = Matrix([3, 6, 9])
    # 高斯-约当消元求解
    sol, params = A.gauss_jordan_solve(b)
    # 断言验证解
    assert sol == Matrix([[-1], [2], [0]])
    # 断言验证参数
    assert params == Matrix(0, 1, [])

    # 正方形，满秩，唯一解，B列数大于行数
    A = eye(3)
    B = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    # 高斯-约当消元求解
    sol, params = A.gauss_jordan_solve(B)
    # 断言验证解
    assert sol == B
    # 断言验证参数
    assert params == Matrix(0, 4, [])

    # 正方形，降秩，参数化解
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = Matrix([3, 6, 9])
    # 高斯-约当消元求解，返回自由变量
    sol, params, freevar = A.gauss_jordan_solve(b, freevar=True)
    w = {}
    for s in sol.atoms(Symbol):
        # 提取解中使用的虚拟符号
        w[s.name] = s
    # 断言验证解
    assert sol == Matrix([[w['tau0'] - 1], [-2*w['tau0'] + 2], [w['tau0']]])
    # 断言验证参数
    assert params == Matrix([[w['tau0']]])
    # 断言验证自由变量
    assert freevar == [2]

    # 正方形，降秩，参数化解，B有两列
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = Matrix([[3, 4], [6, 8], [9, 12]])
    # 高斯-约当消元求解，返回自由变量
    sol, params, freevar = A.gauss_jordan_solve(B, freevar=True)
    w = {}
    for s in sol.atoms(Symbol):
        # 提取解中使用的虚拟符号
        w[s.name] = s
    # 断言验证解
    assert sol == Matrix([[w['tau0'] - 1, w['tau1'] - Rational(4, 3)],
                          [-2*w['tau0'] + 2, -2*w['tau1'] + Rational(8, 3)],
                          [w['tau0'], w['tau1']],])
    # 断言验证参数
    assert params == Matrix([[w['tau0'], w['tau1']]])
    # 断言验证自由变量
    assert freevar == [2]

    # 正方形，降秩，参数化解
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    b = Matrix([0, 0, 0])
    # 使用 Gauss-Jordan 方法解线性方程组 A * sol = b
    sol, params = A.gauss_jordan_solve(b)
    
    # 初始化空字典 w，用于存储解 sol 中的符号变量和它们的值
    w = {}
    
    # 遍历解 sol 中的每个符号变量，将其名称作为键，符号本身作为值存入字典 w
    for s in sol.atoms(Symbol):
        w[s.name] = s
    
    # 断言解 sol 符合特定的矩阵形式
    assert sol == Matrix([[-2*w['tau0'] - 3*w['tau1']],
                         [w['tau0']], [w['tau1']]])
    
    # 断言参数 params 符合特定的矩阵形式
    assert params == Matrix([[w['tau0']], [w['tau1']]])

    # Square, reduced rank, parametrized solution
    # 创建一个全零方阵 A 和一个全零向量 b
    A = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    b = Matrix([0, 0, 0])
    
    # 使用 Gauss-Jordan 方法解线性方程组 A * sol = b
    sol, params = A.gauss_jordan_solve(b)
    
    # 初始化空字典 w，用于存储解 sol 中的符号变量和它们的值
    w = {}
    
    # 遍历解 sol 中的每个符号变量，将其名称作为键，符号本身作为值存入字典 w
    for s in sol.atoms(Symbol):
        w[s.name] = s
    
    # 断言解 sol 符合特定的矩阵形式
    assert sol == Matrix([[w['tau0']], [w['tau1']], [w['tau2']]])
    
    # 断言参数 params 符合特定的矩阵形式
    assert params == Matrix([[w['tau0']], [w['tau1']], [w['tau2']]])

    # Square, reduced rank, no solution
    # 创建一个 A 矩阵使得线性方程组无解，抛出 ValueError 异常
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    b = Matrix([0, 0, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Rectangular, tall, full rank, unique solution
    # 创建一个长方形 A 矩阵和一个相应的 b 向量，确保存在唯一解
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    b = Matrix([0, 0, 1, 0])
    
    # 使用 Gauss-Jordan 方法解线性方程组 A * sol = b
    sol, params = A.gauss_jordan_solve(b)
    
    # 断言解 sol 符合特定的矩阵形式
    assert sol == Matrix([[Rational(-1, 2)], [0], [Rational(1, 6)]])
    
    # 断言参数 params 符合特定的矩阵形式
    assert params == Matrix(0, 1, [])

    # Rectangular, tall, full rank, unique solution, B has less columns than rows
    # 创建一个长方形 A 矩阵和一个 B 矩阵，确保存在唯一解并且 B 的列数少于行数
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0,0], [0, 0], [1, 2], [0, 0]])
    
    # 使用 Gauss-Jordan 方法解线性方程组 A * sol = B
    sol, params = A.gauss_jordan_solve(B)
    
    # 断言解 sol 符合特定的矩阵形式
    assert sol == Matrix([[Rational(-1, 2), Rational(-2, 2)], [0, 0], [Rational(1, 6), Rational(2, 6)]])
    
    # 断言参数 params 符合特定的矩阵形式
    assert params == Matrix(0, 2, [])

    # Rectangular, tall, full rank, no solution
    # 创建一个长方形 A 矩阵使得线性方程组无解，抛出 ValueError 异常
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    b = Matrix([0, 0, 0, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Rectangular, tall, full rank, no solution, B has two columns (2nd has no solution)
    # 创建一个长方形 A 矩阵和一个 B 矩阵，使得线性方程组无解，并确保 B 的第二列也无解
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0,0], [0, 0], [1, 0], [0, 1]])
    raises(ValueError, lambda: A.gauss_jordan_solve(B))

    # Rectangular, tall, full rank, no solution, B has two columns (1st has no solution)
    # 创建一个长方形 A 矩阵和一个 B 矩阵，使得线性方程组无解，并确保 B 的第一列也无解
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0,0], [0, 0], [0, 1], [1, 0]])
    raises(ValueError, lambda: A.gauss_jordan_solve(B))

    # Rectangular, tall, reduced rank, parametrized solution
    # 创建一个长方形 A 矩阵和相应的 b 向量，确保存在参数化的解
    A = Matrix([[1, 5, 3], [2, 10, 6], [3, 15, 9], [1, 4, 3]])
    b = Matrix([0, 0, 0, 1])
    
    # 使用 Gauss-Jordan 方法解线性方程组 A * sol = b
    sol, params = A.gauss_jordan_solve(b)
    
    # 初始化空字典 w，用于存储解 sol 中的符号变量和它们的值
    w = {}
    
    # 遍历解 sol 中的每个符号变量，将其名称作为键，符号本身作为值存入字典 w
    for s in sol.atoms(Symbol):
        w[s.name] = s
    
    # 断言解 sol 符合特定的矩阵形式
    assert sol == Matrix([[-3*w['tau0'] + 5], [-1], [w['tau0']]])
    
    # 断言参数 params 符合特定的矩阵形式
    assert params == Matrix([[w['tau0']]])

    # Rectangular, tall, reduced rank, no solution
    # 创建一个长方形 A 矩阵使得线性方程组无解，抛出 ValueError 异常
    A = Matrix([[1, 5, 3], [2, 10, 6], [3, 15, 9], [1, 4, 3]])
    b = Matrix([0, 0, 1, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Rectangular, wide, full rank, parametrized solution
    # 创建一个宽方形 A 矩阵和相应的 b 向量，确保存在参数化的解
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 1, 12]])
    b = Matrix([1, 1, 1])
    
    # 使用 Gauss-Jordan 方法解线性方程组 A * sol = b
    sol, params = A.gauss_jordan_solve(b)
    for s in sol.atoms(Symbol):
        # 将解向量中的符号变量加入字典 w 中
        w[s.name] = s
    # 断言：验证解向量 sol 是否等于特定的矩阵形式
    assert sol == Matrix([[2*w['tau0'] - 1], [-3*w['tau0'] + 1], [0], [w['tau0']]])
    # 断言：验证参数向量 params 是否等于特定的矩阵形式
    assert params == Matrix([[w['tau0']]])

    # Rectangular, wide, reduced rank, parametrized solution
    # 定义矩阵 A 和向量 b
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]])
    b = Matrix([0, 1, 0])
    # 求解线性方程组 A*x = b
    sol, params = A.gauss_jordan_solve(b)
    # 初始化符号变量字典 w
    w = {}
    for s in sol.atoms(Symbol):
        # 将解向量中的符号变量加入字典 w 中
        w[s.name] = s
    # 断言：验证解向量 sol 是否等于特定的矩阵形式
    assert sol == Matrix([[w['tau0'] + 2*w['tau1'] + S.Half], [-2*w['tau0'] - 3*w['tau1'] - Rational(1, 4)], [w['tau0']], [w['tau1']]])
    # 断言：验证参数向量 params 是否等于特定的矩阵形式
    assert params == Matrix([[w['tau0']], [w['tau1']]])
    # 提示：需要注意符号变量的冲突

    # watch out for clashing symbols
    # 定义一些符号变量，注意可能的冲突
    x0, x1, x2, _x0 = symbols('_tau0 _tau1 _tau2 tau1')
    # 定义矩阵 M，分割出 A 和 b
    M = Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
    A = M[:, :-1]
    b = M[:, -1:]
    # 求解线性方程组 A*x = b
    sol, params = A.gauss_jordan_solve(b)
    # 断言：验证参数向量 params 是否等于特定的矩阵形式
    assert params == Matrix(3, 1, [x0, x1, x2])
    # 断言：验证解向量 sol 是否等于特定的矩阵形式
    assert sol == Matrix(5, 1, [x0, 0, x1, _x0, x2])

    # Rectangular, wide, reduced rank, no solution
    # 定义矩阵 A 和向量 b，此处期望无解
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]])
    b = Matrix([1, 1, 1])
    # 断言：验证求解线性方程组 A*x = b 会抛出 ValueError 异常
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Test for immutable matrix
    # 使用不可变矩阵测试
    A = ImmutableMatrix([[1, 0], [0, 1]])
    B = ImmutableMatrix([1, 2])
    # 求解不可变矩阵 A*x = B
    sol, params = A.gauss_jordan_solve(B)
    # 断言：验证解向量 sol 是否等于特定的不可变矩阵形式
    assert sol == ImmutableMatrix([1, 2])
    # 断言：验证参数向量 params 是否等于特定的不可变矩阵形式
    assert params == ImmutableMatrix(0, 1, [])
    # 断言：验证 sol 和 params 的类型是否为 ImmutableDenseMatrix
    assert sol.__class__ == ImmutableDenseMatrix
    assert params.__class__ == ImmutableDenseMatrix

    # Test placement of free variables
    # 测试自由变量的位置
    A = Matrix([[1, 0, 0, 0], [0, 0, 0, 1]])
    b = Matrix([1, 1])
    # 求解线性方程组 A*x = b
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        # 将解向量中的符号变量加入字典 w 中
        w[s.name] = s
    # 断言：验证解向量 sol 是否等于特定的矩阵形式
    assert sol == Matrix([[1], [w['tau0']], [w['tau1']], [1]])
    # 断言：验证参数向量 params 是否等于特定的矩阵形式
    assert params == Matrix([[w['tau0']], [w['tau1']]])
# 定义测试函数，用于测试线性方程组求解和高斯-约当消元求解
def test_linsolve_underdetermined_AND_gauss_jordan_solve():
    # 创建系数矩阵 A
    A = Matrix([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    # 创建常数向量 B
    B = Matrix([1, 2, 1, 1, 1, 1, 1, 2])
    # 调用 gauss_jordan_solve 方法解线性方程组，返回解 sol 和参数 params
    sol, params = A.gauss_jordan_solve(B)
    
    # 创建空字典 w，用于存储解中的符号变量
    w = {}
    # 遍历解 sol 中的符号变量，将其存入字典 w 中
    for s in sol.atoms(Symbol):
        w[s.name] = s
    
    # 断言：params 应该与符号变量字典 w 的值构成的矩阵相等
    assert params == Matrix([[w['tau0']], [w['tau1']], [w['tau2']],
                             [w['tau3']], [w['tau4']], [w['tau5']]])
    
    # 断言：sol 应该与给定的解析解矩阵相等
    assert sol == Matrix([[1 - 1*w['tau2']],
                          [w['tau2']],
                          [1 - 1*w['tau0'] + w['tau1']],
                          [w['tau0']],
                          [w['tau3'] + w['tau4']],
                          [-1*w['tau3'] - 1*w['tau4'] - 1*w['tau1']],
                          [1 - 1*w['tau2']],
                          [w['tau1']],
                          [w['tau2']],
                          [w['tau3']],
                          [w['tau4']],
                          [1 - 1*w['tau5']],
                          [w['tau5']],
                          [1]])
    
    # 导入 sympy.abc 中的符号 j 和 f
    from sympy.abc import j, f
    
    # 创建新的系数矩阵 A，用于另一个测试案例
    A = Matrix([
        [1,  1, 1,  1, 1,  1, 1,  1,  1],
        [0, -1, 0, -1, 0, -1, 0, -1, -j],
        [0,  0, 0,  0, 1,  1, 1,  1,  f]
    ])
    
    # 使用 linsolve 方法求解线性方程组，并取第一个解
    sol_1 = Matrix(list(linsolve(A))[0])
    
    # 创建符号变量 tau0 到 tau4
    tau0, tau1, tau2, tau3, tau4 = symbols('tau:5')
    
    # 断言：sol_1 应该与给定的解析解矩阵相等
    assert sol_1 == Matrix([[-f - j - tau0 + tau2 + tau4 + 1],
                            [j - tau1 - tau2 - tau4],
                            [tau0],
                            [tau1],
                            [f - tau2 - tau3 - tau4],
                            [tau2],
                            [tau3],
                            [tau4]])
    
    # 计算线性变换 A[:, :-1] * sol_1 - A[:, -1] 的结果
    sol_2 = A[:, :-1] * sol_1 - A[:, -1]
    
    # 断言：sol_2 应该是一个全零矩阵
    assert sol_2 == Matrix([[0], [0], [0]])


# 使用 pytest 的参数化测试功能，测试 Cramer 解法
@pytest.mark.parametrize("det_method", ["bird", "laplace"])
@pytest.mark.parametrize("M, rhs", [
    (Matrix([[2, 3, 5], [3, 6, 2], [8, 3, 6]]), Matrix(3, 1, [3, 7, 5])),
    (Matrix([[2, 3, 5], [3, 6, 2], [8, 3, 6]]),
     Matrix([[1, 2], [3, 4], [5, 6]])),
    (Matrix(2, 2, symbols("a:4")), Matrix(2, 1, symbols("b:2"))),
])
def test_cramer_solve(det_method, M, rhs):
    # 断言：使用 Cramer 解法和 LU 分解解法求解结果的简化应该是一个全零矩阵
    assert simplify(M.cramer_solve(rhs, det_method=det_method) - M.LUsolve(rhs)) == Matrix.zeros(M.rows, rhs.cols)


# 测试 Cramer 解法的错误情况
@pytest.mark.parametrize("det_method, error", [
    ("bird", DMShapeError), (_det_laplace, NonSquareMatrixError)])
def test_cramer_solve_errors(det_method, error):
    # 非方阵的情况下应该引发特定的错误
    # 创建一个2x3的矩阵A，包含整数元素，用于线性方程组求解
    A = Matrix([[0, -1, 2], [5, 10, 7]])
    # 创建一个长度为2的列向量b，包含整数元素，作为线性方程组的右侧常数项
    b = Matrix([-2, 15])
    # 调用矩阵A的克莱姆法则求解方法，传入右侧常数项b和特定的行列式计算方法det_method，
    # 并期望此处抛出一个名为error的异常
    raises(error, lambda: A.cramer_solve(b, det_method=det_method))
# 定义一个名为 test_solve 的测试函数
def test_solve():
    # 创建一个 2x2 的矩阵 A，其值为 [[1,2], [2,4]]
    A = Matrix([[1,2], [2,4]])
    # 创建一个 2x1 的矩阵 b，其值为 [[3], [4]]
    b = Matrix([[3], [4]])
    # 断言调用 A.solve(b) 会引发 ValueError 异常，表示没有解
    raises(ValueError, lambda: A.solve(b)) #no solution
    
    # 将 b 修改为 [[4], [8]]
    b = Matrix([[ 4], [8]])
    # 再次断言调用 A.solve(b) 会引发 ValueError 异常，表示有无穷解
    raises(ValueError, lambda: A.solve(b)) #infinite solution
```
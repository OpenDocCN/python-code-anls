# `D:\src\scipysrc\sympy\sympy\stats\tests\test_symbolic_multivariate.py`

```
# 导入需要的模块和函数
from sympy.stats import Expectation, Normal, Variance, Covariance
from sympy.testing.pytest import raises
from sympy.core.symbol import symbols
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.stats.rv import RandomMatrixSymbol
from sympy.stats.symbolic_multivariate_probability import (ExpectationMatrix,
                            VarianceMatrix, CrossCovarianceMatrix)

# 定义符号变量 j, k
j, k = symbols("j,k")

# 创建符号矩阵对象 A, B, C, D, a, b, A2, B2, X, Y, Z, W, R, X2
A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)

A2 = MatrixSymbol("A2", 2, 2)
B2 = MatrixSymbol("B2", 2, 2)

X = RandomMatrixSymbol("X", k, 1)
Y = RandomMatrixSymbol("Y", k, 1)
Z = RandomMatrixSymbol("Z", k, 1)
W = RandomMatrixSymbol("W", k, 1)

R = RandomMatrixSymbol("R", k, k)

X2 = RandomMatrixSymbol("X2", 2, 1)

normal = Normal("normal", 0, 1)

# 创建具有随机变量的矩阵 m1
m1 = Matrix([
    [1, j * Normal("normal2", 2, 1)],
    [normal, 0]
])

# 定义测试函数 test_multivariate_expectation()
def test_multivariate_expectation():
    # 测试对矩阵向量的期望运算
    expr = Expectation(a)
    assert expr == Expectation(a) == ExpectationMatrix(a)
    assert expr.expand() == a

    # 测试对随机矩阵符号的期望运算
    expr = Expectation(X)
    assert expr == Expectation(X) == ExpectationMatrix(X)
    assert expr.shape == (k, 1)
    assert expr.rows == k
    assert expr.cols == 1
    assert isinstance(expr, ExpectationMatrix)

    # 测试对表达式 A*X + b 的期望运算
    expr = Expectation(A*X + b)
    assert expr == ExpectationMatrix(A*X + b)
    assert expr.expand() == A*ExpectationMatrix(X) + b
    assert isinstance(expr, ExpectationMatrix)
    assert expr.shape == (k, 1)

    # 测试对表达式 m1*X2 的期望运算
    expr = Expectation(m1*X2)
    assert expr.expand() == expr

    # 测试对表达式 A2*m1*B2*X2 的期望运算
    expr = Expectation(A2*m1*B2*X2)
    assert expr.args[0].args == (A2, m1, B2, X2)
    assert expr.expand() == A2*ExpectationMatrix(m1*B2*X2)

    # 测试对表达式 (X + Y)*(X - Y).T 的期望运算
    expr = Expectation((X + Y)*(X - Y).T)
    assert expr.expand() == ExpectationMatrix(X*X.T) - ExpectationMatrix(X*Y.T) +\
                ExpectationMatrix(Y*X.T) - ExpectationMatrix(Y*Y.T)

    # 测试对表达式 A*X + B*Y 的期望运算
    expr = Expectation(A*X + B*Y)
    assert expr.expand() == A*ExpectationMatrix(X) + B*ExpectationMatrix(Y)

    # 测试对矩阵 m1 的期望运算结果
    assert Expectation(m1).doit() == Matrix([[1, 2*j], [0, 0]])

    # 创建两个矩阵 x1 和 x2，包含具有不同正态分布的随机变量
    x1 = Matrix([
    [Normal('N11', 11, 1), Normal('N12', 12, 1)],
    [Normal('N21', 21, 1), Normal('N22', 22, 1)]
    ])
    x2 = Matrix([
    [Normal('M11', 1, 1), Normal('M12', 2, 1)],
    [Normal('M21', 3, 1), Normal('M22', 4, 1)]
    ])

    # 测试嵌套期望的结果是否与期望矩阵匹配
    assert Expectation(Expectation(x1 + x2)).doit(deep=False) == ExpectationMatrix(x1 + x2)
    assert Expectation(Expectation(x1 + x2)).doit() == Matrix([[12, 14], [24, 26]])


# 定义测试函数 test_multivariate_variance()
def test_multivariate_variance():
    # 测试不支持的形状错误
    raises(ShapeError, lambda: Variance(A))

    # 测试对向量 a 的方差运算
    expr = Variance(a)
    assert expr == Variance(a) == VarianceMatrix(a)
    assert expr.expand() == ZeroMatrix(k, k)

    # 测试对向量 a 转置的方差运算
    expr = Variance(a.T)
    assert expr == Variance(a.T) == VarianceMatrix(a.T)
    # 断言表达式 expr.expand() 等于 ZeroMatrix(k, k)
    assert expr.expand() == ZeroMatrix(k, k)

    # 将 expr 定义为随机变量 X 的方差，断言多个表达式相等
    expr = Variance(X)
    assert expr == Variance(X) == VarianceMatrix(X)
    # 断言 expr 的形状为 (k, k)
    assert expr.shape == (k, k)
    # 断言 expr 的行数为 k
    assert expr.rows == k
    # 断言 expr 的列数为 k
    assert expr.cols == k
    # 断言 expr 是 VarianceMatrix 类型的实例
    assert isinstance(expr, VarianceMatrix)

    # 将 expr 定义为矩阵 A 乘以随机变量 X 的方差，断言多个表达式相等
    expr = Variance(A*X)
    assert expr == VarianceMatrix(A*X)
    # 断言 expr 扩展后等于 A * VarianceMatrix(X) * A 的转置
    assert expr.expand() == A*VarianceMatrix(X)*A.T
    # 断言 expr 是 VarianceMatrix 类型的实例
    assert isinstance(expr, VarianceMatrix)
    # 断言 expr 的形状为 (k, k)
    assert expr.shape == (k, k)

    # 将 expr 定义为矩阵 A 乘以矩阵 B 乘以随机变量 X 的方差，断言表达式相等
    expr = Variance(A*B*X)
    assert expr.expand() == A*B*VarianceMatrix(X)*B.T*A.T

    # 将 expr 定义为矩阵 m1 乘以随机变量 X2 的方差，断言表达式相等
    expr = Variance(m1*X2)
    assert expr.expand() == expr

    # 将 expr 定义为矩阵 A2 乘以矩阵 m1 乘以矩阵 B2 乘以随机变量 X2 的方差，断言表达式的第一个参数的参数等于 (A2, m1, B2, X2)，并且表达式扩展后等于自身
    expr = Variance(A2*m1*B2*X2)
    assert expr.args[0].args == (A2, m1, B2, X2)
    assert expr.expand() == expr

    # 将 expr 定义为矩阵 A 乘以随机变量 X 加上矩阵 B 乘以随机变量 Y 的方差，断言表达式扩展后等于 2*A*CrossCovarianceMatrix(X, Y)*B.T + A*VarianceMatrix(X)*A.T + B*VarianceMatrix(Y)*B.T
    assert expr.expand() == 2*A*CrossCovarianceMatrix(X, Y)*B.T +\
                    A*VarianceMatrix(X)*A.T + B*VarianceMatrix(Y)*B.T
# 定义一个测试函数，用于测试多变量交叉协方差计算的函数
def test_multivariate_crosscovariance():
    # 测试当输入的矩阵维度不匹配时，是否会引发 ShapeError 异常
    raises(ShapeError, lambda: Covariance(X, Y.T))
    raises(ShapeError, lambda: Covariance(X, A))

    # 创建一个协方差表达式对象，计算 a.T 和 b.T 的协方差
    expr = Covariance(a.T, b.T)
    # 断言该表达式的形状为 (1, 1)
    assert expr.shape == (1, 1)
    # 断言该表达式展开后为零矩阵
    assert expr.expand() == ZeroMatrix(1, 1)

    # 创建一个协方差表达式对象，计算 a 和 b 的协方差
    expr = Covariance(a, b)
    # 断言该表达式与 Covariance(a, b) 和 CrossCovarianceMatrix(a, b) 相等
    assert expr == Covariance(a, b) == CrossCovarianceMatrix(a, b)
    # 断言该表达式展开后为 k*k 的零矩阵
    assert expr.expand() == ZeroMatrix(k, k)
    # 断言该表达式的形状为 (k, k)
    assert expr.shape == (k, k)
    # 断言该表达式的行数为 k
    assert expr.rows == k
    # 断言该表达式的列数为 k
    assert expr.cols == k
    # 断言该表达式是 CrossCovarianceMatrix 的实例
    assert isinstance(expr, CrossCovarianceMatrix)

    # 创建一个协方差表达式对象，计算 A*X + a 和 b 的协方差
    expr = Covariance(A*X + a, b)
    # 断言该表达式展开后为 k*k 的零矩阵
    assert expr.expand() == ZeroMatrix(k, k)

    # 创建一个协方差表达式对象，计算 X 和 Y 的协方差
    expr = Covariance(X, Y)
    # 断言该表达式是 CrossCovarianceMatrix 的实例
    assert isinstance(expr, CrossCovarianceMatrix)
    # 断言该表达式展开后与其本身相等
    assert expr.expand() == expr

    # 创建一个协方差表达式对象，计算 X 和 X 的协方差
    expr = Covariance(X, X)
    # 断言该表达式是 CrossCovarianceMatrix 的实例
    assert isinstance(expr, CrossCovarianceMatrix)
    # 断言该表达式展开后与 VarianceMatrix(X) 相等
    assert expr.expand() == VarianceMatrix(X)

    # 创建一个协方差表达式对象，计算 X + Y 和 Z 的协方差
    expr = Covariance(X + Y, Z)
    # 断言该表达式是 CrossCovarianceMatrix 的实例
    assert isinstance(expr, CrossCovarianceMatrix)
    # 断言该表达式展开后与 CrossCovarianceMatrix(X, Z) + CrossCovarianceMatrix(Y, Z) 相等
    assert expr.expand() == CrossCovarianceMatrix(X, Z) + CrossCovarianceMatrix(Y, Z)

    # 创建一个协方差表达式对象，计算 A*X 和 Y 的协方差
    expr = Covariance(A*X, Y)
    # 断言该表达式是 CrossCovarianceMatrix 的实例
    assert isinstance(expr, CrossCovarianceMatrix)
    # 断言该表达式展开后与 A*CrossCovarianceMatrix(X, Y) 相等
    assert expr.expand() == A*CrossCovarianceMatrix(X, Y)

    # 创建一个协方差表达式对象，计算 X 和 B*Y 的协方差
    expr = Covariance(X, B*Y)
    # 断言该表达式是 CrossCovarianceMatrix 的实例
    assert isinstance(expr, CrossCovarianceMatrix)
    # 断言该表达式展开后与 CrossCovarianceMatrix(X, Y)*B.T 相等
    assert expr.expand() == CrossCovarianceMatrix(X, Y)*B.T

    # 创建一个协方差表达式对象，计算 A*X + B*Y + a 和 C.T*Z + D.T*W + b 的协方差
    expr = Covariance(A*X + B*Y + a, C.T*Z + D.T*W + b)
    # 断言该表达式是 CrossCovarianceMatrix 的实例
    assert isinstance(expr, CrossCovarianceMatrix)
    # 断言该表达式展开后与 A*CrossCovarianceMatrix(X, W)*D + A*CrossCovarianceMatrix(X, Z)*C
    #               + B*CrossCovarianceMatrix(Y, W)*D + B*CrossCovarianceMatrix(Y, Z)*C 相等
    assert expr.expand() == A*CrossCovarianceMatrix(X, W)*D + A*CrossCovarianceMatrix(X, Z)*C \
        + B*CrossCovarianceMatrix(Y, W)*D + B*CrossCovarianceMatrix(Y, Z)*C
```
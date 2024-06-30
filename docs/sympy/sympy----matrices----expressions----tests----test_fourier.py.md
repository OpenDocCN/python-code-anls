# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_fourier.py`

```
# 从 sympy.assumptions.ask 模块导入 Q 和 ask 函数
# 从 sympy.core.numbers 模块导入 I 和 Rational
# 从 sympy.core.singleton 模块导入 S
# 从 sympy.functions.elementary.complexes 模块导入 Abs
# 从 sympy.functions.elementary.exponential 模块导入 exp
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt
# 从 sympy.simplify.simplify 模块导入 simplify
# 从 sympy.core.symbol 模块导入 symbols
# 从 sympy.matrices.expressions.fourier 模块导入 DFT 和 IDFT
# 从 sympy.matrices 模块导入 det, Matrix, Identity
# 从 sympy.testing.pytest 模块导入 raises

# 定义一个测试函数，用于测试 DFT 的创建
def test_dft_creation():
    # 检查 DFT(2) 的真值
    assert DFT(2)
    # 检查 DFT(0) 的真值
    assert DFT(0)
    # 检查 DFT(-1) 抛出 ValueError 异常
    raises(ValueError, lambda: DFT(-1))
    # 检查 DFT(2.0) 抛出 ValueError 异常
    raises(ValueError, lambda: DFT(2.0))
    # 检查 DFT(2+1j) 抛出 ValueError 异常
    raises(ValueError, lambda: DFT(2 + 1j))

    # 创建一个符号 n
    n = symbols('n')
    # 检查 DFT(n) 的真值
    assert DFT(n)
    # 创建一个非整数符号 n
    n = symbols('n', integer=False)
    # 检查 DFT(n) 抛出 ValueError 异常
    raises(ValueError, lambda: DFT(n))
    # 创建一个负数符号 n
    n = symbols('n', negative=True)
    # 检查 DFT(n) 抛出 ValueError 异常
    raises(ValueError, lambda: DFT(n))

# 定义一个测试函数，用于测试 DFT 相关性质
def test_dft():
    # 创建符号 n, i, j
    n, i, j = symbols('n i j')
    # 检查 DFT(4) 的形状是否为 (4, 4)
    assert DFT(4).shape == (4, 4)
    # 检查 DFT(4) 是否满足 unitary 属性
    assert ask(Q.unitary(DFT(4)))
    # 检查矩阵 DFT(4) 行列式的绝对值是否等于 1
    assert Abs(simplify(det(Matrix(DFT(4))))) == 1
    # 检查 DFT(n) * IDFT(n) 是否等于单位矩阵 Identity(n)
    assert DFT(n)*IDFT(n) == Identity(n)
    # 检查 DFT(n)[i, j] 的值是否符合给定表达式
    assert DFT(n)[i, j] == exp(-2*S.Pi*I/n)**(i*j) / sqrt(n)

# 定义一个测试函数，用于测试 DFT 的具体表达式
def test_dft2():
    # 检查 DFT(1) 转换为显式矩阵是否为 [[1]]
    assert DFT(1).as_explicit() == Matrix([[1]])
    # 检查 DFT(2) 转换为显式矩阵是否为 1/sqrt(2) * [[1, 1], [1, -1]]
    assert DFT(2).as_explicit() == 1/sqrt(2)*Matrix([[1, 1], [1, -1]])
    # 检查 DFT(4) 转换为显式矩阵是否为给定的复杂矩阵表达式
    assert DFT(4).as_explicit() == Matrix([[S.Half, S.Half, S.Half, S.Half],
                                           [S.Half, -I/2, Rational(-1, 2), I/2],
                                           [S.Half, Rational(-1, 2), S.Half, Rational(-1, 2)],
                                           [S.Half, I/2, Rational(-1, 2), -I/2]])
```
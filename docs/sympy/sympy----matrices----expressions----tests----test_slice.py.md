# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_slice.py`

```
from sympy.matrices.expressions.slice import MatrixSlice  # 导入 MatrixSlice 类
from sympy.matrices.expressions import MatrixSymbol  # 导入 MatrixSymbol 类
from sympy.abc import a, b, c, d, k, l, m, n  # 导入符号变量 a, b, c, d, k, l, m, n
from sympy.testing.pytest import raises, XFAIL  # 导入测试相关的 raises 和 XFAIL
from sympy.functions.elementary.integers import floor  # 导入 floor 函数
from sympy.assumptions import assuming, Q  # 导入 assuming 和 Q

X = MatrixSymbol('X', n, m)  # 定义一个 n x m 的 MatrixSymbol 对象 X
Y = MatrixSymbol('Y', m, k)  # 定义一个 m x k 的 MatrixSymbol 对象 Y

def test_shape():
    B = MatrixSlice(X, (a, b), (c, d))  # 创建一个 MatrixSlice 对象 B
    assert B.shape == (b - a, d - c)  # 检查 B 的形状是否为 (b - a, d - c)

def test_entry():
    B = MatrixSlice(X, (a, b), (c, d))  # 创建一个 MatrixSlice 对象 B
    assert B[0,0] == X[a, c]  # 检查 B 的元素 (0, 0) 是否等于 X[a, c]
    assert B[k,l] == X[a+k, c+l]  # 检查 B 的元素 (k, l) 是否等于 X[a+k, c+l]
    raises(IndexError, lambda : MatrixSlice(X, 1, (2, 5))[1, 0])  # 检查是否引发 IndexError 异常

    assert X[1::2, :][1, 3] == X[1+2, 3]  # 检查切片操作是否正确
    assert X[:, 1::2][3, 1] == X[3, 1+2]  # 检查切片操作是否正确

def test_on_diag():
    assert not MatrixSlice(X, (a, b), (c, d)).on_diag  # 检查非对角线上的情况
    assert MatrixSlice(X, (a, b), (a, b)).on_diag  # 检查对角线上的情况

def test_inputs():
    assert MatrixSlice(X, 1, (2, 5)) == MatrixSlice(X, (1, 2), (2, 5))  # 检查输入是否一致
    assert MatrixSlice(X, 1, (2, 5)).shape == (1, 3)  # 检查形状是否正确

def test_slicing():
    assert X[1:5, 2:4] == MatrixSlice(X, (1, 5), (2, 4))  # 检查切片操作是否正确
    assert X[1, 2:4] == MatrixSlice(X, 1, (2, 4))  # 检查切片操作是否正确
    assert X[1:5, :].shape == (4, X.shape[1])  # 检查形状是否正确
    assert X[:, 1:5].shape == (X.shape[0], 4)  # 检查形状是否正确

    assert X[::2, ::2].shape == (floor(n/2), floor(m/2))  # 检查切片操作是否正确
    assert X[2, :] == MatrixSlice(X, 2, (0, m))  # 检查切片操作是否正确
    assert X[k, :] == MatrixSlice(X, k, (0, m))  # 检查切片操作是否正确

def test_exceptions():
    X = MatrixSymbol('x', 10, 20)  # 创建一个新的 MatrixSymbol 对象 X
    raises(IndexError, lambda: X[0:12, 2])  # 检查是否引发 IndexError 异常
    raises(IndexError, lambda: X[0:9, 22])  # 检查是否引发 IndexError 异常
    raises(IndexError, lambda: X[-1:5, 2])  # 检查是否引发 IndexError 异常

@XFAIL
def test_symmetry():
    X = MatrixSymbol('x', 10, 10)  # 创建一个新的 MatrixSymbol 对象 X
    Y = X[:5, 5:]  # 对 X 进行切片操作得到 Y
    with assuming(Q.symmetric(X)):  # 假设 X 是对称的
        assert Y.T == X[5:, :5]  # 检查 Y 的转置是否等于 X 的对称部分

def test_slice_of_slice():
    X = MatrixSymbol('x', 10, 10)  # 创建一个新的 MatrixSymbol 对象 X
    assert X[2, :][:, 3][0, 0] == X[2, 3]  # 检查切片操作是否正确
    assert X[:5, :5][:4, :4] == X[:4, :4]  # 检查切片操作是否正确
    assert X[1:5, 2:6][1:3, 2] == X[2:4, 4]  # 检查切片操作是否正确
    assert X[1:9:2, 2:6][1:3, 2] == X[3:7:2, 4]  # 检查切片操作是否正确

def test_negative_index():
    X = MatrixSymbol('x', 10, 10)  # 创建一个新的 MatrixSymbol 对象 X
    assert X[-1, :] == X[9, :]  # 检查负索引操作是否正确
```
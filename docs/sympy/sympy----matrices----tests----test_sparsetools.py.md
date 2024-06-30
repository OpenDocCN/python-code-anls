# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_sparsetools.py`

```
# 导入所需模块和函数
from sympy.matrices.sparsetools import _doktocsr, _csrtodok, banded
# 导入矩阵相关的类和函数
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices import SparseMatrix
# 导入测试相关的异常处理函数
from sympy.testing.pytest import raises


# 测试函数：将 DOK 格式稀疏矩阵转换为 CSR 格式
def test_doktocsr():
    # 创建多个稀疏矩阵实例
    a = SparseMatrix([[1, 2, 0, 0], [0, 3, 9, 0], [0, 1, 4, 0]])
    b = SparseMatrix(4, 6, [10, 20, 0, 0, 0, 0, 0, 30, 0, 40, 0, 0, 0, 0, 50,
        60, 70, 0, 0, 0, 0, 0, 0, 80])
    c = SparseMatrix(4, 4, [0, 0, 0, 0, 0, 12, 0, 2, 15, 0, 12, 0, 0, 0, 0, 4])
    d = SparseMatrix(10, 10, {(1, 1): 12, (3, 5): 7, (7, 8): 12})
    e = SparseMatrix([[0, 0, 0], [1, 0, 2], [3, 0, 0]])
    f = SparseMatrix(7, 8, {(2, 3): 5, (4, 5):12})
    
    # 断言转换后的 CSR 格式是否符合预期
    assert _doktocsr(a) == [[1, 2, 3, 9, 1, 4], [0, 1, 1, 2, 1, 2],
        [0, 2, 4, 6], [3, 4]]
    assert _doktocsr(b) == [[10, 20, 30, 40, 50, 60, 70, 80],
        [0, 1, 1, 3, 2, 3, 4, 5], [0, 2, 4, 7, 8], [4, 6]]
    assert _doktocsr(c) == [[12, 2, 15, 12, 4], [1, 3, 0, 2, 3],
        [0, 0, 2, 4, 5], [4, 4]]
    assert _doktocsr(d) == [[12, 7, 12], [1, 5, 8],
        [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3], [10, 10]]
    assert _doktocsr(e) == [[1, 2, 3], [0, 2, 0], [0, 0, 2, 3], [3, 3]]
    assert _doktocsr(f) == [[5, 12], [3, 5], [0, 0, 0, 1, 1, 2, 2, 2], [7, 8]]


# 测试函数：将 CSR 格式稀疏矩阵转换为 DOK 格式
def test_csrtodok():
    # 创建多个 CSR 格式矩阵的表示
    h = [[5, 7, 5], [2, 1, 3], [0, 1, 1, 3], [3, 4]]
    g = [[12, 5, 4], [2, 4, 2], [0, 1, 2, 3], [3, 7]]
    i = [[1, 3, 12], [0, 2, 4], [0, 2, 3], [2, 5]]
    j = [[11, 15, 12, 15], [2, 4, 1, 2], [0, 1, 1, 2, 3, 4], [5, 8]]
    k = [[1, 3], [2, 1], [0, 1, 1, 2], [3, 3]]
    
    # 将 CSR 转换为 DOK 格式
    m = _csrtodok(h)
    # 断言转换后的结果是否为 SparseMatrix 类型且与预期结果相符
    assert isinstance(m, SparseMatrix)
    assert m == SparseMatrix(3, 4,
        {(0, 2): 5, (2, 1): 7, (2, 3): 5})
    assert _csrtodok(g) == SparseMatrix(3, 7,
        {(0, 2): 12, (1, 4): 5, (2, 2): 4})
    assert _csrtodok(i) == SparseMatrix([[1, 0, 3, 0, 0], [0, 0, 0, 0, 12]])
    assert _csrtodok(j) == SparseMatrix(5, 8,
        {(0, 2): 11, (2, 4): 15, (3, 1): 12, (4, 2): 15})
    assert _csrtodok(k) == SparseMatrix(3, 3, {(0, 2): 1, (2, 1): 3})


# 测试函数：测试 banded 函数
def test_banded():
    # 断言对于不同参数的输入，是否能正确引发预期的异常
    raises(TypeError, lambda: banded())
    raises(TypeError, lambda: banded(1))
    raises(TypeError, lambda: banded(1, 2))
    raises(TypeError, lambda: banded(1, 2, 3))
    raises(TypeError, lambda: banded(1, 2, 3, 4))
    raises(ValueError, lambda: banded({0: (1, 2)}, rows=1))
    raises(ValueError, lambda: banded({0: (1, 2)}, cols=1))
    raises(ValueError, lambda: banded(1, {0: (1, 2)}))
    raises(ValueError, lambda: banded(2, 1, {0: (1, 2)}))
    raises(ValueError, lambda: banded(1, 2, {0: (1, 2)}))
    
    # 断言对于不同参数的输入，banded 函数是否返回符合预期的 SparseMatrix 类型结果
    assert isinstance(banded(2, 4, {}), SparseMatrix)
    assert banded(2, 4, {}) == zeros(2, 4)
    assert banded({0: 0, 1: 0}) == zeros(0)
    assert banded({0: Matrix([1, 2])}) == Matrix([1, 2])
    assert banded({1: [1, 2, 3, 0], -1: [4, 5, 6]}) == \
        banded({1: (1, 2, 3), -1: (4, 5, 6)}) == \
        Matrix([
        [0, 1, 0, 0],
        [4, 0, 2, 0],
        [0, 5, 0, 3],
        [0, 0, 6, 0]])
    # 验证函数`banded`对于特定输入的返回值是否符合预期
    assert banded(3, 4, {-1: 1, 0: 2, 1: 3}) == \
        Matrix([
        [2, 3, 0, 0],
        [1, 2, 3, 0],
        [0, 1, 2, 3]])
    
    # 定义一个 lambda 函数 `s`，用来计算 `(1 + d)**2` 的值
    s = lambda d: (1 + d)**2
    
    # 验证函数`banded`对于特定输入的返回值是否符合预期，其中第二个参数是一个字典 `{0: s, 2: s}`
    assert banded(5, {0: s, 2: s}) == \
        Matrix([
        [1, 0, 1,  0,  0],
        [0, 4, 0,  4,  0],
        [0, 0, 9,  0,  9],
        [0, 0, 0, 16,  0],
        [0, 0, 0,  0, 25]])
    
    # 验证函数`banded`对于特定输入的返回值是否符合预期，其中第二个参数是 `{0: 1}`
    assert banded(2, {0: 1}) == \
        Matrix([
        [1, 0],
        [0, 1]])
    
    # 验证函数`banded`对于特定输入的返回值是否符合预期，其中第二个参数是 `{0: 1}`
    assert banded(2, 3, {0: 1}) == \
        Matrix([
        [1, 0, 0],
        [0, 1, 0]])
    
    # 创建一个垂直的矩阵 `vert`
    vert = Matrix([1, 2, 3])
    
    # 验证函数`banded`对于特定输入的返回值是否符合预期，其中第一个参数是 `{0: vert}`，并指定 `cols=3`
    assert banded({0: vert}, cols=3) == \
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [3, 2, 1],
        [0, 3, 2],
        [0, 0, 3]])
    
    # 验证函数`banded`对于特定输入的返回值是否符合预期，其中第二个参数是 `{0: ones(2)}`
    assert banded(4, {0: ones(2)}) == \
        Matrix([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]])
    
    # 验证函数`banded`是否会引发预期的异常，第一个参数为 `{0: 2, 1: ones(2)}`，并指定 `rows=5`
    raises(ValueError, lambda: banded({0: 2, 1: ones(2)}, rows=5))
    
    # 验证函数`banded`对于特定输入的返回值是否符合预期，其中第一个参数是 `{0: 2, 2: (ones(2),)*3}`
    assert banded({0: 2, 2: (ones(2),)*3}) == \
        Matrix([
        [2, 0, 1, 1, 0, 0, 0, 0],
        [0, 2, 1, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 1, 1, 0, 0],
        [0, 0, 0, 2, 1, 1, 0, 0],
        [0, 0, 0, 0, 2, 0, 1, 1],
        [0, 0, 0, 0, 0, 2, 1, 1]])
    
    # 验证函数`banded`是否会引发预期的异常，第一个参数为 `{0: (2,)*5, 1: (ones(2),)*3}`
    raises(ValueError, lambda: banded({0: (2,)*5, 1: (ones(2),)*3}))
    
    # 定义矩阵 `u2`
    u2 = Matrix([[1, 1], [0, 1]])
    
    # 验证函数`banded`对于特定输入的返回值是否符合预期，其中第一个参数是 `{0: (2,)*5, 1: (u2,)*3}`
    assert banded({0: (2,)*5, 1: (u2,)*3}) == \
        Matrix([
        [2, 1, 1, 0, 0, 0, 0],
        [0, 2, 1, 0, 0, 0, 0],
        [0, 0, 2, 1, 1, 0, 0],
        [0, 0, 0, 2, 1, 0, 0],
        [0, 0, 0, 0, 2, 1, 1],
        [0, 0, 0, 0, 0, 0, 1]])
    
    # 验证函数`banded`对于特定输入的返回值是否符合预期，其中第一个参数是 `{0:(0, ones(2)), 2: 2}`
    assert banded({0:(0, ones(2)), 2: 2}) == \
        Matrix([
        [0, 0, 2],
        [0, 1, 1],
        [0, 1, 1]])
    
    # 验证函数`banded`是否会引发预期的异常，第一个参数为 `{0: (0, ones(2)), 1: 2}`
    raises(ValueError, lambda: banded({0: (0, ones(2)), 1: 2}))
    
    # 验证函数`banded`对于特定输入的返回值是否符合预期，其中第一个参数是 `{0: 1}`，并指定 `cols=3` 和 `rows=3`
    assert banded({0: 1}, cols=3) == banded({0: 1}, rows=3) == eye(3)
    
    # 验证函数`banded`对于特定输入的返回值是否符合预期，其中第一个参数是 `{1: 1}`，并指定 `rows=3`
    assert banded({1: 1}, rows=3) == \
        Matrix([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]])
```
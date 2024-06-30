# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_immutable.py`

```
# 导入 itertools 模块中的 product 函数，用于生成迭代器的笛卡尔积
from itertools import product

# 导入 sympy 库中的各个模块和函数
from sympy.core.relational import (Equality, Unequality)
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import (Matrix, eye, zeros)
from sympy.matrices.immutable import ImmutableMatrix
from sympy.matrices import SparseMatrix
from sympy.matrices.immutable import \
    ImmutableDenseMatrix, ImmutableSparseMatrix

# 导入 sympy 库中的符号变量 x, y
from sympy.abc import x, y

# 导入 pytest 模块中的 raises 函数，用于测试异常抛出
from sympy.testing.pytest import raises

# 创建不可变的稠密矩阵 IM，存储数据为 3x3 的矩阵
IM = ImmutableDenseMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 创建不可变的稀疏矩阵 ISM，存储数据为 3x3 的矩阵
ISM = ImmutableSparseMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 创建单位矩阵的不可变形式，存储数据为 3x3 的单位矩阵
ieye = ImmutableDenseMatrix(eye(3))


def test_creation():
    # 检查 IM 和 ISM 的形状是否都为 (3, 3)
    assert IM.shape == ISM.shape == (3, 3)
    # 检查 IM 和 ISM 中指定位置的元素是否都为 6
    assert IM[1, 2] == ISM[1, 2] == 6
    # 检查 IM 和 ISM 中另一个指定位置的元素是否都为 9
    assert IM[2, 2] == ISM[2, 2] == 9


def test_immutability():
    # 测试尝试修改 IM 和 ISM 中元素的不可变性
    with raises(TypeError):
        IM[2, 2] = 5
    with raises(TypeError):
        ISM[2, 2] = 5


def test_slicing():
    # 检查 IM 和 ISM 在切片操作下的行为
    assert IM[1, :] == ImmutableDenseMatrix([[4, 5, 6]])
    assert IM[:2, :2] == ImmutableDenseMatrix([[1, 2], [4, 5]])
    assert ISM[1, :] == ImmutableSparseMatrix([[4, 5, 6]])
    assert ISM[:2, :2] == ImmutableSparseMatrix([[1, 2], [4, 5]])


def test_subs():
    # 测试矩阵替换操作的结果
    A = ImmutableMatrix([[1, 2], [3, 4]])
    B = ImmutableMatrix([[1, 2], [x, 4]])
    C = ImmutableMatrix([[-x, x*y], [-(x + y), y**2]])
    assert B.subs(x, 3) == A
    assert (x*B).subs(x, 3) == 3*A
    assert (x*eye(2) + B).subs(x, 3) == 3*eye(2) + A
    assert C.subs([[x, -1], [y, -2]]) == A
    assert C.subs([(x, -1), (y, -2)]) == A
    assert C.subs({x: -1, y: -2}) == A
    assert C.subs({x: y - 1, y: x - 1}, simultaneous=True) == \
        ImmutableMatrix([[1 - y, (x - 1)*(y - 1)], [2 - x - y, (x - 1)**2]])


def test_as_immutable():
    # 测试将 Matrix 和 SparseMatrix 对象转换为不可变矩阵 ImmutableMatrix 的行为
    data = [[1, 2], [3, 4]]
    X = Matrix(data)
    assert sympify(X) == X.as_immutable() == ImmutableMatrix(data)

    data = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
    X = SparseMatrix(2, 2, data)
    assert sympify(X) == X.as_immutable() == ImmutableSparseMatrix(2, 2, data)


def test_function_return_types():
    # 测试不可变矩阵在函数调用后返回类型的一致性
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[1], [0]])
    q, r = X.QRdecomposition()
    assert (type(q), type(r)) == (ImmutableMatrix, ImmutableMatrix)

    assert type(X.LUsolve(Y)) == ImmutableMatrix
    assert type(X.QRsolve(Y)) == ImmutableMatrix

    X = ImmutableMatrix([[5, 2], [2, 7]])
    assert X.T == X
    assert X.is_symmetric
    assert type(X.cholesky()) == ImmutableMatrix
    L, D = X.LDLdecomposition()
    assert (type(L), type(D)) == (ImmutableMatrix, ImmutableMatrix)

    X = ImmutableMatrix([[1, 2], [2, 1]])
    assert X.is_diagonalizable()
    assert X.det() == -3
    assert X.norm(2) == 3

    assert type(X.eigenvects()[0][2][0]) == ImmutableMatrix
    # 断言：检查 zeros(3, 3) 生成的不可变矩阵的零空间的第一个向量类型是否为 ImmutableMatrix
    assert type(zeros(3, 3).as_immutable().nullspace()[0]) == ImmutableMatrix
    
    # 创建一个不可变矩阵 X
    X = ImmutableMatrix([[1, 0], [2, 1]])
    # 断言：检查使用 X 解方程 X * X_lower = Y 后得到的解的类型是否为 ImmutableMatrix
    assert type(X.lower_triangular_solve(Y)) == ImmutableMatrix
    # 断言：检查使用 X 转置解方程 X_transpose * X_upper = Y 后得到的解的类型是否为 ImmutableMatrix
    assert type(X.T.upper_triangular_solve(Y)) == ImmutableMatrix
    
    # 断言：检查 X 的主子矩阵（去掉第 0 行和第 0 列后的矩阵）的类型是否为 ImmutableMatrix
    assert type(X.minor_submatrix(0, 0)) == ImmutableMatrix
# issue 6279
# https://github.com/sympy/sympy/issues/6279
# Test that Immutable _op_ Immutable => Immutable and not MatExpr

# 测试不可变矩阵的运算结果仍为不可变矩阵而非 MatExpr

def test_immutable_evaluation():
    # 创建一个 3x3 单位矩阵，并封装为不可变矩阵 X
    X = ImmutableMatrix(eye(3))
    # 创建一个 3x3 矩阵 A，内容为 0 到 8 的顺序排列，并封装为不可变矩阵 A
    A = ImmutableMatrix(3, 3, range(9))
    # 断言 X + A 的结果仍为不可变矩阵类型
    assert isinstance(X + A, ImmutableMatrix)
    # 断言 X * A 的结果仍为不可变矩阵类型
    assert isinstance(X * A, ImmutableMatrix)
    # 断言 X * 2 的结果仍为不可变矩阵类型
    assert isinstance(X * 2, ImmutableMatrix)
    # 断言 2 * X 的结果仍为不可变矩阵类型
    assert isinstance(2 * X, ImmutableMatrix)
    # 断言 A 的平方（A^2）的结果仍为不可变矩阵类型
    assert isinstance(A**2, ImmutableMatrix)


def test_deterimant():
    # 断言一个 4x4 矩阵，其元素为 i+j 的 lambda 表达式结果的行列式为 0
    assert ImmutableMatrix(4, 4, lambda i, j: i + j).det() == 0


def test_Equality():
    # 断言两个相同的不可变矩阵 IM 之间的相等关系为 S.true
    assert Equality(IM, IM) is S.true
    # 断言两个相同的不可变矩阵 IM 之间的不等关系为 S.false
    assert Unequality(IM, IM) is S.false
    # 断言一个不可变矩阵 IM 与其元素中某个元素替换后的结果不相等为 S.false
    assert Equality(IM, IM.subs(1, 2)) is S.false
    # 断言一个不可变矩阵 IM 与其元素中某个元素替换后的结果不相等为 S.true
    assert Unequality(IM, IM.subs(1, 2)) is S.true
    # 断言一个不可变矩阵 IM 与整数 2 的比较结果不相等为 S.false
    assert Equality(IM, 2) is S.false
    # 断言一个不可变矩阵 IM 与整数 2 的比较结果不相等为 S.true
    assert Unequality(IM, 2) is S.true
    # 创建一个包含符号 x, y 的不可变矩阵 M
    M = ImmutableMatrix([x, y])
    # 断言不可变矩阵 M 与不可变矩阵 IM 的比较结果不相等为 S.false
    assert Equality(M, IM) is S.false
    # 断言不可变矩阵 M 与不可变矩阵 IM 的比较结果不相等为 S.true
    assert Unequality(M, IM) is S.true
    # 断言不可变矩阵 M 与将其元素中的符号 x 替换为 2 后的结果相等，且在 x 被替换为 2 后为 S.true
    assert Equality(M, M.subs(x, 2)).subs(x, 2) is S.true
    # 断言不可变矩阵 M 与将其元素中的符号 x 替换为 2 后的结果不相等，且在 x 被替换为 2 后为 S.false
    assert Unequality(M, M.subs(x, 2)).subs(x, 2) is S.false
    # 断言不可变矩阵 M 与将其元素中的符号 x 替换为 2 后的结果相等，但在 x 被替换为 3 后为 S.false
    assert Equality(M, M.subs(x, 2)).subs(x, 3) is S.false
    # 断言不可变矩阵 M 与将其元素中的符号 x 替换为 2 后的结果不相等，且在 x 被替换为 3 后为 S.true
    assert Unequality(M, M.subs(x, 2)).subs(x, 3) is S.true


def test_integrate():
    # 对不可变矩阵 IM 进行 x 变量的积分
    intIM = integrate(IM, x)
    # 断言积分结果的形状与不可变矩阵 IM 的形状相同
    assert intIM.shape == IM.shape
    # 断言积分结果的每个元素都满足特定的条件
    assert all(intIM[i, j] == (1 + j + 3*i)*x for i, j in
                product(range(3), range(3)))
```
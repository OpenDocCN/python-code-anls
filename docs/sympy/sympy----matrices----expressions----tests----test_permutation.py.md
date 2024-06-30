# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_permutation.py`

```
from sympy.combinatorics import Permutation  # 导入排列的类
from sympy.core.expr import unchanged  # 导入未改变的表达式类
from sympy.matrices import Matrix  # 导入矩阵类
from sympy.matrices.expressions import \
    MatMul, BlockDiagMatrix, Determinant, Inverse  # 导入矩阵乘积、块对角矩阵、行列式、逆的表达式类
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵符号类
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix, Identity  # 导入特殊矩阵类
from sympy.matrices.expressions.permutation import \
    MatrixPermute, PermutationMatrix  # 导入矩阵排列和排列矩阵类
from sympy.testing.pytest import raises  # 导入测试模块中的异常抛出函数
from sympy.core.symbol import Symbol  # 导入符号类


def test_PermutationMatrix_basic():
    p = Permutation([1, 0])  # 创建排列 [1, 0]
    assert unchanged(PermutationMatrix, p)  # 检查排列矩阵不变性
    raises(ValueError, lambda: PermutationMatrix((0, 1, 2)))  # 检查创建排列矩阵时的值错误异常
    assert PermutationMatrix(p).as_explicit() == Matrix([[0, 1], [1, 0]])  # 检查排列矩阵的显式表示
    assert isinstance(PermutationMatrix(p)*MatrixSymbol('A', 2, 2), MatMul)  # 检查排列矩阵与符号矩阵乘积的类型


def test_PermutationMatrix_matmul():
    p = Permutation([1, 2, 0])  # 创建排列 [1, 2, 0]
    P = PermutationMatrix(p)  # 创建排列矩阵
    M = Matrix([[0, 1, 2], [3, 4, 5], [6, 7, 8]])  # 创建矩阵
    assert (P*M).as_explicit() == P.as_explicit()*M  # 检查排列矩阵乘矩阵的显式表示
    assert (M*P).as_explicit() == M*P.as_explicit()  # 检查矩阵乘排列矩阵的显式表示

    P1 = PermutationMatrix(Permutation([1, 2, 0]))  # 创建排列矩阵 P1
    P2 = PermutationMatrix(Permutation([2, 1, 0]))  # 创建排列矩阵 P2
    P3 = PermutationMatrix(Permutation([1, 0, 2]))  # 创建排列矩阵 P3
    assert P1*P2 == P3  # 检查排列矩阵乘积的结果


def test_PermutationMatrix_matpow():
    p1 = Permutation([1, 2, 0])  # 创建排列 [1, 2, 0]
    P1 = PermutationMatrix(p1)  # 创建排列矩阵 P1
    p2 = Permutation([2, 0, 1])  # 创建排列 [2, 0, 1]
    P2 = PermutationMatrix(p2)  # 创建排列矩阵 P2
    assert P1**2 == P2  # 检查排列矩阵的乘幂运算
    assert P1**3 == Identity(3)  # 检查排列矩阵的乘幂运算


def test_PermutationMatrix_identity():
    p = Permutation([0, 1])  # 创建排列 [0, 1]
    assert PermutationMatrix(p).is_Identity  # 检查是否为单位排列矩阵

    p = Permutation([1, 0])  # 创建排列 [1, 0]
    assert not PermutationMatrix(p).is_Identity  # 检查是否为单位排列矩阵


def test_PermutationMatrix_determinant():
    P = PermutationMatrix(Permutation([0, 1, 2]))  # 创建排列矩阵 P
    assert Determinant(P).doit() == 1  # 检查排列矩阵的行列式值
    P = PermutationMatrix(Permutation([0, 2, 1]))  # 创建排列矩阵 P
    assert Determinant(P).doit() == -1  # 检查排列矩阵的行列式值
    P = PermutationMatrix(Permutation([2, 0, 1]))  # 创建排列矩阵 P
    assert Determinant(P).doit() == 1  # 检查排列矩阵的行列式值


def test_PermutationMatrix_inverse():
    P = PermutationMatrix(Permutation(0, 1, 2))  # 创建排列矩阵 P
    assert Inverse(P).doit() == PermutationMatrix(Permutation(0, 2, 1))  # 检查排列矩阵的逆矩阵


def test_PermutationMatrix_rewrite_BlockDiagMatrix():
    P = PermutationMatrix(Permutation([0, 1, 2, 3, 4, 5]))  # 创建排列矩阵 P
    P0 = PermutationMatrix(Permutation([0]))  # 创建单位排列矩阵 P0
    assert P.rewrite(BlockDiagMatrix) == \
        BlockDiagMatrix(P0, P0, P0, P0, P0, P0)  # 检查排列矩阵转换为块对角矩阵表示

    P = PermutationMatrix(Permutation([0, 1, 3, 2, 4, 5]))  # 创建排列矩阵 P
    P10 = PermutationMatrix(Permutation(0, 1))  # 创建排列矩阵 P10
    assert P.rewrite(BlockDiagMatrix) == \
        BlockDiagMatrix(P0, P0, P10, P0, P0)  # 检查排列矩阵转换为块对角矩阵表示

    P = PermutationMatrix(Permutation([1, 0, 3, 2, 5, 4]))  # 创建排列矩阵 P
    assert P.rewrite(BlockDiagMatrix) == \
        BlockDiagMatrix(P10, P10, P10)  # 检查排列矩阵转换为块对角矩阵表示

    P = PermutationMatrix(Permutation([0, 4, 3, 2, 1, 5]))  # 创建排列矩阵 P
    P3210 = PermutationMatrix(Permutation([3, 2, 1, 0]))  # 创建排列矩阵 P3210
    assert P.rewrite(BlockDiagMatrix) == \
        BlockDiagMatrix(P0, P3210, P0)  # 检查排列矩阵转换为块对角矩阵表示

    P = PermutationMatrix(Permutation([0, 4, 2, 3, 1, 5]))  # 创建排列矩阵 P
    # 创建一个置换矩阵 P3120，其中置换顺序为 [3, 1, 2, 0]
    P3120 = PermutationMatrix(Permutation([3, 1, 2, 0]))
    
    # 断言：将置换矩阵 P 重新写成块对角矩阵 BlockDiagMatrix，应与给定的块对角矩阵相等
    assert P.rewrite(BlockDiagMatrix) == \
        BlockDiagMatrix(P0, P3120, P0)
    
    # 创建一个置换矩阵 P，其置换操作包括 0 ↔ 3，1 ↔ 4，2 ↔ 5
    P = PermutationMatrix(Permutation(0, 3)(1, 4)(2, 5))
    
    # 断言：将置换矩阵 P 重新写成块对角矩阵 BlockDiagMatrix，应与给定的块对角矩阵相等
    assert P.rewrite(BlockDiagMatrix) == BlockDiagMatrix(P)
# 定义一个测试函数，用于测试 MatrixPermute 类的基本功能
def test_MartrixPermute_basic():
    # 创建一个置换对象 p，包含元素 0 和 1
    p = Permutation(0, 1)
    # 根据置换对象 p 创建置换矩阵 P
    P = PermutationMatrix(p)
    # 创建一个符号矩阵 A，大小为 2x2
    A = MatrixSymbol('A', 2, 2)

    # 断言在给定 Symbol('x') 和 p 的情况下，调用 MatrixPermute 抛出 ValueError 异常
    raises(ValueError, lambda: MatrixPermute(Symbol('x'), p))
    # 断言在给定 A 和 Symbol('x') 的情况下，调用 MatrixPermute 抛出 ValueError 异常
    raises(ValueError, lambda: MatrixPermute(A, Symbol('x')))

    # 断言 MatrixPermute(A, P) 等于 MatrixPermute(A, p)
    assert MatrixPermute(A, P) == MatrixPermute(A, p)
    # 断言在给定 A、p 和额外参数 2 的情况下，调用 MatrixPermute 抛出 ValueError 异常
    raises(ValueError, lambda: MatrixPermute(A, p, 2))

    # 创建一个包含 3 个元素的置换对象 pp，元素为 0、1，以及默认大小为 3
    pp = Permutation(0, 1, size=3)
    # 断言 MatrixPermute(A, pp) 等于 MatrixPermute(A, p)
    assert MatrixPermute(A, pp) == MatrixPermute(A, p)
    # 创建一个包含 3 个元素的置换对象 pp，元素为 0、1、2
    pp = Permutation(0, 1, 2)
    # 断言在给定 A 和 pp 的情况下，调用 MatrixPermute 抛出 ValueError 异常
    raises(ValueError, lambda: MatrixPermute(A, pp))


# 定义一个测试函数，用于测试 MatrixPermute 对象的形状
def test_MatrixPermute_shape():
    # 创建一个置换对象 p，包含元素 0 和 1
    p = Permutation(0, 1)
    # 创建一个符号矩阵 A，大小为 2x3
    A = MatrixSymbol('A', 2, 3)
    # 断言 MatrixPermute(A, p) 的形状为 (2, 3)
    assert MatrixPermute(A, p).shape == (2, 3)


# 定义一个测试函数，测试 MatrixPermute 对象的显式表达
def test_MatrixPermute_explicit():
    # 创建一个置换对象 p，包含元素 0、1、2
    p = Permutation(0, 1, 2)
    # 创建一个符号矩阵 A，大小为 3x3
    A = MatrixSymbol('A', 3, 3)
    # 将符号矩阵 A 转换为显式矩阵 AA
    AA = A.as_explicit()
    # 断言 MatrixPermute(A, p, 0) 的显式表示等于 AA 按照置换 p 进行行重排的结果
    assert MatrixPermute(A, p, 0).as_explicit() == \
        AA.permute(p, orientation='rows')
    # 断言 MatrixPermute(A, p, 1) 的显式表示等于 AA 按照置换 p 进行列重排的结果
    assert MatrixPermute(A, p, 1).as_explicit() == \
        AA.permute(p, orientation='cols')


# 定义一个测试函数，测试 MatrixPermute 对象在重写 MatMul 操作时的行为
def test_MatrixPermute_rewrite_MatMul():
    # 创建一个置换对象 p，包含元素 0、1、2
    p = Permutation(0, 1, 2)
    # 创建一个符号矩阵 A，大小为 3x3
    A = MatrixSymbol('A', 3, 3)

    # 断言 MatrixPermute(A, p, 0) 重写为 MatMul 操作后的显式表示等于未重写时的显式表示
    assert MatrixPermute(A, p, 0).rewrite(MatMul).as_explicit() == \
        MatrixPermute(A, p, 0).as_explicit()
    # 断言 MatrixPermute(A, p, 1) 重写为 MatMul 操作后的显式表示等于未重写时的显式表示
    assert MatrixPermute(A, p, 1).rewrite(MatMul).as_explicit() == \
        MatrixPermute(A, p, 1).as_explicit()


# 定义一个测试函数，测试 MatrixPermute 对象的 doit() 方法
def test_MatrixPermute_doit():
    # 创建一个置换对象 p，包含元素 0、1、2
    p = Permutation(0, 1, 2)
    # 创建一个符号矩阵 A，大小为 3x3
    A = MatrixSymbol('A', 3, 3)
    # 断言 MatrixPermute(A, p).doit() 等于 MatrixPermute(A, p)
    assert MatrixPermute(A, p).doit() == MatrixPermute(A, p)

    # 创建一个置换对象 p，包含元素 0，大小为 3
    p = Permutation(0, size=3)
    # 断言 MatrixPermute(A, p).doit() 的显式表示等于 MatrixPermute(A, p) 的显式表示
    assert MatrixPermute(A, p).doit().as_explicit() == \
        MatrixPermute(A, p).as_explicit()

    # 创建一个置换对象 p，包含元素 0、1、2
    p = Permutation(0, 1, 2)
    # 创建一个 3x3 的单位矩阵 A
    A = Identity(3)
    # 断言 MatrixPermute(A, p, 0).doit() 的显式表示等于 MatrixPermute(A, p, 0) 的显式表示
    assert MatrixPermute(A, p, 0).doit().as_explicit() == \
        MatrixPermute(A, p, 0).as_explicit()
    # 断言 MatrixPermute(A, p, 1).doit() 的显式表示等于 MatrixPermute(A, p, 1) 的显式表示
    assert MatrixPermute(A, p, 1).doit().as_explicit() == \
        MatrixPermute(A, p, 1).as_explicit()

    # 创建一个 3x3 的零矩阵 A
    A = ZeroMatrix(3, 3)
    # 断言 MatrixPermute(A, p).doit() 等于 A
    assert MatrixPermute(A, p).doit() == A
    # 创建一个 3x3 的单位矩阵 A
    A = OneMatrix(3, 3)
    # 断言 MatrixPermute(A, p).doit() 等于 A
    assert MatrixPermute(A, p).doit() == A

    # 创建一个符号矩阵 A，大小为 4x4
    A = MatrixSymbol('A', 4, 4)
    # 创建两个置换对象 p1 和 p2
    p1 = Permutation(0, 1, 2, 3)
    p2 = Permutation(0, 2, 3, 1)
    # 创建一个复合表达式 expr，两次置换操作 MatrixPermute(A, p1, 0) 和 MatrixPermute(A, p2, 0)
    expr = MatrixPermute(MatrixPermute(A, p1, 0), p2, 0)
    # 断言 expr 的显式表示等于对其执行 doit() 后的显式表示
    assert expr.as_explicit() == expr.doit().as_explicit()
    # 创建一个复合表达式 expr，两次置换操作 MatrixPermute(A, p1, 1) 和 MatrixPermute(A, p2, 1)
    expr = MatrixPermute(MatrixPermute(A, p1, 1), p2, 1)
    # 断言 expr 的显式表示等于对其执行 doit() 后的显式表示
    assert expr.as_explicit() == expr.doit().as_explicit()
```
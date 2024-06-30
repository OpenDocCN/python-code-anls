# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_decompositions.py`

```
# 从 sympy 库中导入需要的模块和函数

from sympy.core.function import expand_mul
from sympy.core.numbers import I, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import Abs
from sympy.simplify.simplify import simplify
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices import Matrix, zeros, eye, SparseMatrix
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, slow
from sympy.testing.matrices import allclose

# 定义名为 test_LUdecomp 的测试函数
def test_LUdecomp():
    # 创建一个 4x4 的 Matrix 对象 testmat
    testmat = Matrix([[0, 2, 5, 3],
                      [3, 3, 7, 4],
                      [8, 4, 0, 2],
                      [-2, 6, 3, 4]])
    # 对 testmat 进行 LU 分解，并获取返回的 L, U, p
    L, U, p = testmat.LUdecomposition()
    # 断言 L 是下三角矩阵
    assert L.is_lower
    # 断言 U 是上三角矩阵
    assert U.is_upper
    # 断言 (L*U).permute_rows(p, 'backward') - testmat 是一个 4x4 的零矩阵
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(4)

    # 创建另一个 4x4 的 Matrix 对象 testmat
    testmat = Matrix([[6, -2, 7, 4],
                      [0, 3, 6, 7],
                      [1, -2, 7, 4],
                      [-9, 2, 6, 3]])
    # 对 testmat 进行 LU 分解，并获取返回的 L, U, p
    L, U, p = testmat.LUdecomposition()
    # 断言 L 是下三角矩阵
    assert L.is_lower
    # 断言 U 是上三角矩阵
    assert U.is_upper
    # 断言 (L*U).permute_rows(p, 'backward') - testmat 是一个 4x4 的零矩阵
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(4)

    # 创建一个非方阵 Matrix 对象 testmat
    testmat = Matrix([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]])
    # 对 testmat 进行 LU 分解，关闭秩检查(rankcheck=False)，并获取返回的 L, U, p
    L, U, p = testmat.LUdecomposition(rankcheck=False)
    # 断言 L 是下三角矩阵
    assert L.is_lower
    # 断言 U 是上三角矩阵
    assert U.is_upper
    # 断言 (L*U).permute_rows(p, 'backward') - testmat 是一个 4x3 的零矩阵
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(4, 3)

    # 创建一个方阵但奇异的 Matrix 对象 testmat
    testmat = Matrix([[1, 2, 3],
                      [2, 4, 6],
                      [4, 5, 6]])
    # 对 testmat 进行 LU 分解，关闭秩检查(rankcheck=False)，并获取返回的 L, U, p
    L, U, p = testmat.LUdecomposition(rankcheck=False)
    # 断言 L 是下三角矩阵
    assert L.is_lower
    # 断言 U 是上三角矩阵
    assert U.is_upper
    # 断言 (L*U).permute_rows(p, 'backward') - testmat 是一个 3x3 的零矩阵
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(3)

    # 创建一个具有符号的 Matrix 对象 M
    M = Matrix(((1, x, 1), (2, y, 0), (y, 0, z)))
    # 对 M 进行 LU 分解，并获取返回的 L, U, p
    L, U, p = M.LUdecomposition()
    # 断言 L 是下三角矩阵
    assert L.is_lower
    # 断言 U 是上三角矩阵
    assert U.is_upper
    # 断言 (L*U).permute_rows(p, 'backward') - M 是一个 3x3 的零矩阵
    assert (L*U).permute_rows(p, 'backward') - M == zeros(3)

    # 创建一个下三角 Matrix 对象 mL
    mL = Matrix((
        (1, 0, 0),
        (2, 3, 0),
    ))
    # 断言 mL 是下三角矩阵
    assert mL.is_lower is True
    # 断言 mL 不是上三角矩阵
    assert mL.is_upper is False
    # 创建一个上三角 Matrix 对象 mU
    mU = Matrix((
        (1, 2, 3),
        (0, 4, 5),
    ))
    # 断言 mU 不是下三角矩阵
    assert mU.is_lower is False
    # 断言 mU 是上三角矩阵
    assert mU.is_upper is True

    # 测试 FF LUdecomp
    M = Matrix([[1, 3, 3],
                [3, 2, 6],
                [3, 2, 2]])
    # 对 M 进行 FF LU 分解，并获取返回的 P, L, Dee, U
    P, L, Dee, U = M.LUdecompositionFF()
    # 断言 P*M == L*Dee.inv()*U
    assert P*M == L*Dee.inv()*U

    M = Matrix([[1,  2, 3,  4],
                [3, -1, 2,  3],
                [3,  1, 3, -2],
                [6, -1, 0,  2]])
    # 对 M 进行 FF LU 分解，并获取返回的 P, L, Dee, U
    P, L, Dee, U = M.LUdecompositionFF()
    # 断言 P*M == L*Dee.inv()*U
    assert P*M == L*Dee.inv()*U

    M = Matrix([[0, 0, 1],
                [2, 3, 0],
                [3, 1, 4]])
    # 对 M 进行 FF LU 分解，并获取返回的 P, L, Dee, U
    P, L, Dee, U = M.LUdecompositionFF()
    # 断言 P*M == L*Dee.inv()*U
    assert P*M == L*Dee.inv()*U

    # 处理 issue 15794
    M = Matrix(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    )
    # 断言调用 M.LUdecomposition_Simple(rankcheck=True) 会抛出 ValueError 异常
    raises(ValueError, lambda : M.LUdecomposition_Simple(rankcheck=True))

# 定义名为 test_singular_value_decompositionD 的测试函数，用于测试奇异值分解
def test_singular_value_decompositionD():
    # 这个函数尚未完整添加，后续需要根据测试需求继续添加测试代码
    # 创建一个 2x2 的矩阵 A
    A = Matrix([[1, 2], [2, 1]])
    # 对矩阵 A 进行奇异值分解，返回分解后的三个矩阵 U, S, V
    U, S, V = A.singular_value_decomposition()
    # 断言奇异值分解的恢复性质：U * S * V^T 应该等于原始矩阵 A
    assert U * S * V.T == A
    # 断言 U 的转置乘以自身应该等于单位矩阵
    assert U.T * U == eye(U.cols)
    # 断言 V 的转置乘以自身应该等于单位矩阵
    assert V.T * V == eye(V.cols)

    # 创建一个 1x2 的矩阵 B
    B = Matrix([[1, 2]])
    # 对矩阵 B 进行奇异值分解，返回分解后的三个矩阵 U, S, V
    U, S, V = B.singular_value_decomposition()
    # 断言奇异值分解的恢复性质：U * S * V^T 应该等于原始矩阵 B
    assert U * S * V.T == B
    # 断言 U 的转置乘以自身应该等于单位矩阵
    assert U.T * U == eye(U.cols)
    # 断言 V 的转置乘以自身应该等于单位矩阵
    assert V.T * V == eye(V.cols)

    # 创建一个 4x5 的矩阵 C
    C = Matrix([
        [1, 0, 0, 0, 2],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
    ])
    # 对矩阵 C 进行奇异值分解，返回分解后的三个矩阵 U, S, V
    U, S, V = C.singular_value_decomposition()
    # 断言奇异值分解的恢复性质：U * S * V^T 应该等于原始矩阵 C
    assert U * S * V.T == C
    # 断言 U 的转置乘以自身应该等于单位矩阵
    assert U.T * U == eye(U.cols)
    # 断言 V 的转置乘以自身应该等于单位矩阵
    assert V.T * V == eye(V.cols)

    # 创建一个有理数元素的 2x2 矩阵 D
    D = Matrix([[Rational(1, 3), sqrt(2)], [0, Rational(1, 4)]])
    # 对矩阵 D 进行奇异值分解，返回分解后的三个矩阵 U, S, V
    U, S, V = D.singular_value_decomposition()
    # 断言简化后的 U 的转置乘以自身应该等于单位矩阵
    assert simplify(U.T * U) == eye(U.cols)
    # 断言简化后的 V 的转置乘以自身应该等于单位矩阵
    assert simplify(V.T * V) == eye(V.cols)
    # 断言简化后的 U * S * V^T 应该等于原始矩阵 D
    assert simplify(U * S * V.T) == D
def test_QR():
    # 创建一个2x2的矩阵A
    A = Matrix([[1, 2], [2, 3]])
    # 对矩阵A进行QR分解，得到正交矩阵Q和上三角矩阵S
    Q, S = A.QRdecomposition()
    # 定义一个有理数类Rational
    R = Rational
    # 断言语句，验证Q是否等于指定的正交矩阵
    assert Q == Matrix([
        [  5**R(-1, 2),  (R(2)/5)*(R(1)/5)**R(-1, 2)],
        [2*5**R(-1, 2), (-R(1)/5)*(R(1)/5)**R(-1, 2)]])
    # 断言语句，验证S是否等于指定的上三角矩阵
    assert S == Matrix([[5**R(1, 2), 8*5**R(-1, 2)], [0, (R(1)/5)**R(1, 2)]])
    # 断言语句，验证Q * S是否等于原始矩阵A
    assert Q*S == A
    # 断言语句，验证Q的转置与Q相乘是否等于单位矩阵
    assert Q.T * Q == eye(2)

    # 创建一个3x3的矩阵A
    A = Matrix([[1, 1, 1], [1, 1, 3], [2, 3, 4]])
    # 对矩阵A进行QR分解，得到正交矩阵Q和上三角矩阵R
    Q, R = A.QRdecomposition()
    # 断言语句，验证Q的转置与Q相乘是否等于单位矩阵
    assert Q.T * Q == eye(Q.cols)
    # 断言语句，验证R是否为上三角矩阵
    assert R.is_upper
    # 断言语句，验证A是否等于Q * R
    assert A == Q*R

    # 创建一个3x3的矩阵A
    A = Matrix([[12, 0, -51], [6, 0, 167], [-4, 0, 24]])
    # 对矩阵A进行QR分解，得到正交矩阵Q和上三角矩阵R
    Q, R = A.QRdecomposition()
    # 断言语句，验证Q的转置与Q相乘是否等于单位矩阵
    assert Q.T * Q == eye(Q.cols)
    # 断言语句，验证R是否为上三角矩阵
    assert R.is_upper
    # 断言语句，验证A是否等于Q * R
    assert A == Q*R

    # 创建一个包含符号x的1x1矩阵A
    x = Symbol('x')
    A = Matrix([x])
    # 对矩阵A进行QR分解，得到正交矩阵Q和上三角矩阵R
    Q, R = A.QRdecomposition()
    # 断言语句，验证Q是否等于指定的正交矩阵
    assert Q == Matrix([x / Abs(x)])
    # 断言语句，验证R是否等于指定的上三角矩阵
    assert R == Matrix([Abs(x)])

    # 创建一个包含符号x的2x2矩阵A
    A = Matrix([[x, 0], [0, x]])
    # 对矩阵A进行QR分解，得到正交矩阵Q和上三角矩阵R
    Q, R = A.QRdecomposition()
    # 断言语句，验证Q是否等于指定的正交矩阵
    assert Q == x / Abs(x) * Matrix([[1, 0], [0, 1]])
    # 断言语句，验证R是否等于指定的上三角矩阵
    assert R == Abs(x) * Matrix([[1, 0], [0, 1]])


def test_QR_non_square():
    # 窄矩阵（列数小于行数）
    A = Matrix([[9, 0, 26], [12, 0, -7], [0, 4, 4], [0, -3, -3]])
    # 对矩阵A进行QR分解，得到正交矩阵Q和上三角矩阵R
    Q, R = A.QRdecomposition()
    # 断言语句，验证Q的转置与Q相乘是否等于单位矩阵
    assert Q.T * Q == eye(Q.cols)
    # 断言语句，验证R是否为上三角矩阵
    assert R.is_upper
    # 断言语句，验证A是否等于Q * R
    assert A == Q*R

    A = Matrix([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix(2, 1, [1, 2])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    # 宽矩阵（列数大于行数）
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, 2, 3, 4], [1, 4, 9, 16], [1, 8, 27, 64]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix(1, 2, [1, 2])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R


def test_QR_trivial():
    # 秩亏矩阵
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    # 零秩矩阵
    A = Matrix([[0, 0, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0], [0, 0, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    # 断言R是一个上三角矩阵
    assert R.is_upper

    # 断言A等于Q乘以R，即QR分解的正确性
    assert A == Q*R

    # 创建一个3行2列的零矩阵，并转置
    A = Matrix([[0, 0, 0], [0, 0, 0]]).T
    # 对A进行QR分解
    Q, R = A.QRdecomposition()
    # 断言Q的转置乘以Q等于单位矩阵
    assert Q.T * Q == eye(Q.cols)
    # 再次断言R是一个上三角矩阵
    assert R.is_upper
    # 断言A等于Q乘以R，验证QR分解的正确性
    assert A == Q*R

    # 创建一个3行2列的矩阵，其中第一列全为零，其余列为1, 2, 3
    A = Matrix([[0, 0, 0], [1, 2, 3]]).T
    # 对A进行QR分解
    Q, R = A.QRdecomposition()
    # 断言Q的转置乘以Q等于单位矩阵
    assert Q.T * Q == eye(Q.cols)
    # 断言R是一个上三角矩阵
    assert R.is_upper
    # 断言A等于Q乘以R，验证QR分解的正确性
    assert A == Q*R

    # 创建一个4行3列的矩阵，其中第一列全为零，第二列为1, 2, 3, 4，其余列为零
    A = Matrix([[0, 0, 0, 0], [1, 2, 3, 4], [0, 0, 0, 0]]).T
    # 对A进行QR分解
    Q, R = A.QRdecomposition()
    # 断言Q的转置乘以Q等于单位矩阵
    assert Q.T * Q == eye(Q.cols)
    # 断言R是一个上三角矩阵
    assert R.is_upper
    # 断言A等于Q乘以R，验证QR分解的正确性
    assert A == Q*R

    # 创建一个4行3列的矩阵，其中第一列全为零，第二列为1, 2, 3, 4，第四列为2, 4, 6, 8，其余列为零
    A = Matrix([[0, 0, 0, 0], [1, 2, 3, 4], [0, 0, 0, 0], [2, 4, 6, 8]]).T
    # 对A进行QR分解
    Q, R = A.QRdecomposition()
    # 断言Q的转置乘以Q等于单位矩阵
    assert Q.T * Q == eye(Q.cols)
    # 断言R是一个上三角矩阵
    assert R.is_upper
    # 断言A等于Q乘以R，验证QR分解的正确性
    assert A == Q*R

    # 创建一个4行3列的矩阵，其中前三列全为零，第四列为1, 2, 3，其余列为零
    A = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 3]]).T
    # 对A进行QR分解
    Q, R = A.QRdecomposition()
    # 断言Q的转置乘以Q等于单位矩阵
    assert Q.T * Q == eye(Q.cols)
    # 断言R是一个上三角矩阵
    assert R.is_upper
    # 断言A等于Q乘以R，验证QR分解的正确性
    assert A == Q*R
def test_LDLdecomposition():
    # 检查当输入非方阵时是否会引发异常
    raises(NonSquareMatrixError, lambda: Matrix((1, 2)).LDLdecomposition())
    # 检查当输入不符合LDL分解条件的矩阵时是否会引发异常
    raises(ValueError, lambda: Matrix(((1, 2), (3, 4))).LDLdecomposition())
    # 检查当输入包含复数且不是埃尔米特矩阵时是否会引发异常
    raises(ValueError, lambda: Matrix(((5 + I, 0), (0, 1))).LDLdecomposition())
    # 检查当输入不符合LDL分解条件的矩阵时是否会引发异常
    raises(ValueError, lambda: Matrix(((1, 5), (5, 1))).LDLdecomposition())
    # 检查当输入不符合LDL分解条件的矩阵且指定hermitian参数为False时是否会引发异常
    raises(ValueError, lambda: Matrix(((1, 2), (3, 4))).LDLdecomposition(hermitian=False))
    
    # 测试LDL分解正确性
    A = Matrix(((1, 5), (5, 1)))
    L, D = A.LDLdecomposition(hermitian=False)
    assert L * D * L.T == A
    
    # 进行更多LDL分解的测试
    A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L, D = A.LDLdecomposition()
    assert L * D * L.T == A
    assert L.is_lower
    assert L == Matrix([[1, 0, 0], [ Rational(3, 5), 1, 0], [Rational(-1, 5), Rational(1, 3), 1]])
    assert D.is_diagonal()
    assert D == Matrix([[25, 0, 0], [0, 9, 0], [0, 0, 9]])
    
    # 进行复数矩阵的LDL分解测试
    A = Matrix(((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11)))
    L, D = A.LDLdecomposition()
    assert expand_mul(L * D * L.H) == A
    assert L.expand() == Matrix([[1, 0, 0], [I/2, 1, 0], [S.Half - I/2, 0, 1]])
    assert D.expand() == Matrix(((4, 0, 0), (0, 1, 0), (0, 0, 9)))
    
    # 检查当输入稀疏矩阵且非方阵时是否会引发异常
    raises(NonSquareMatrixError, lambda: SparseMatrix((1, 2)).LDLdecomposition())
    # 检查当输入不符合LDL分解条件的稀疏矩阵时是否会引发异常
    raises(ValueError, lambda: SparseMatrix(((1, 2), (3, 4))).LDLdecomposition())
    # 检查当输入包含复数且不是埃尔米特矩阵的稀疏矩阵时是否会引发异常
    raises(ValueError, lambda: SparseMatrix(((5 + I, 0), (0, 1))).LDLdecomposition())
    # 检查当输入不符合LDL分解条件的稀疏矩阵时是否会引发异常
    raises(ValueError, lambda: SparseMatrix(((1, 5), (5, 1))).LDLdecomposition())
    # 检查当输入不符合LDL分解条件的稀疏矩阵且指定hermitian参数为False时是否会引发异常
    raises(ValueError, lambda: SparseMatrix(((1, 2), (3, 4))).LDLdecomposition(hermitian=False))
    # 创建一个稀疏矩阵 A，内容为 ((1, 5), (5, 1))
    A = SparseMatrix(((1, 5), (5, 1)))
    # 对矩阵 A 进行 LDL 分解，不要求矩阵是共轭的
    L, D = A.LDLdecomposition(hermitian=False)
    # 断言 LDL 分解后的重构矩阵与原始矩阵 A 相等
    assert L * D * L.T == A

    # 创建一个稀疏矩阵 A，内容为 ((25, 15, -5), (15, 18, 0), (-5, 0, 11))
    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    # 对矩阵 A 进行 LDL 分解
    L, D = A.LDLdecomposition()
    # 断言 LDL 分解后的重构矩阵与原始矩阵 A 相等
    assert L * D * L.T == A
    # 断言 L 是下三角矩阵
    assert L.is_lower
    # 断言 L 等于矩阵 [[1, 0, 0], [3/5, 1, 0], [-1/5, 1/3, 1]]
    assert L == Matrix([[1, 0, 0], [ Rational(3, 5), 1, 0], [Rational(-1, 5), Rational(1, 3), 1]])
    # 断言 D 是对角矩阵
    assert D.is_diagonal()
    # 断言 D 等于矩阵 [[25, 0, 0], [0, 9, 0], [0, 0, 9]]
    assert D == Matrix([[25, 0, 0], [0, 9, 0], [0, 0, 9]])

    # 创建一个稀疏矩阵 A，内容为 ((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11))
    A = SparseMatrix(((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11)))
    # 对矩阵 A 进行 LDL 分解
    L, D = A.LDLdecomposition()
    # 断言 LDL 分解后的重构矩阵与原始矩阵 A 相等，使用 Hermite 运算符表示共轭转置
    assert expand_mul(L * D * L.H) == A
    # 断言 L 等于矩阵 ((1, 0, 0), (I/2, 1, 0), (1/2 - I/2, 0, 1))
    assert L == Matrix(((1, 0, 0), (I/2, 1, 0), (S.Half - I/2, 0, 1)))
    # 断言 D 等于矩阵 ((4, 0, 0), (0, 1, 0), (0, 0, 9))
    assert D == Matrix(((4, 0, 0), (0, 1, 0), (0, 0, 9)))
def test_pinv_succeeds_with_rank_decomposition_method():
    # 测试伪逆的秩分解方法是否成功
    As = [Matrix([
        [61, 89, 55, 20, 71, 0],
        [62, 96, 85, 85, 16, 0],
        [69, 56, 17,  4, 54, 0],
        [10, 54, 91, 41, 71, 0],
        [ 7, 30, 10, 48, 90, 0],
        [0,0,0,0,0,0]])]
    for A in As:
        # 计算伪逆矩阵，使用秩分解方法
        A_pinv = A.pinv(method="RD")
        # 计算 A * A_pinv
        AAp = A * A_pinv
        # 计算 A_pinv * A
        ApA = A_pinv * A
        # 断言 AAp * A == A
        assert simplify(AAp * A) == A
        # 断言 ApA * A_pinv == A_pinv
        assert simplify(ApA * A_pinv) == A_pinv
        # 断言 AAp 的共轭转置等于自身
        assert AAp.H == AAp
        # 断言 ApA 的共轭转置等于自身
        assert ApA.H == ApA

def test_rank_decomposition():
    # 测试矩阵的秩分解方法
    a = Matrix(0, 0, [])
    c, f = a.rank_decomposition()
    # 断言 f 是阶梯形
    assert f.is_echelon
    # 断言 c 的列数等于 f 的行数等于 a 的秩
    assert c.cols == f.rows == a.rank()
    # 断言 c * f 等于 a
    assert c * f == a

    a = Matrix(1, 1, [5])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a

    a = Matrix(3, 3, [1, 2, 3, 1, 2, 3, 1, 2, 3])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a

    a = Matrix([
        [0, 0, 1, 2, 2, -5, 3],
        [-1, 5, 2, 2, 1, -7, 5],
        [0, 0, -2, -3, -3, 8, -5],
        [-1, 5, 0, -1, -2, 1, 0]])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a


@slow
def test_upper_hessenberg_decomposition():
    # 测试上 Hessenberg 分解方法
    A = Matrix([
        [1, 0, sqrt(3)],
        [sqrt(2), Rational(1, 2), 2],
        [1, Rational(1, 4), 3],
    ])
    # 进行上 Hessenberg 分解
    H, P = A.upper_hessenberg_decomposition()
    # 断言 P * P 的共轭转置等于单位矩阵
    assert simplify(P * P.H) == eye(P.cols)
    # 断言 P 的共轭转置 * P 等于单位矩阵
    assert simplify(P.H * P) == eye(P.cols)
    # 断言 H 是上 Hessenberg 形式
    assert H.is_upper_hessenberg
    # 断言 P * H * P 的共轭转置等于 A
    assert (simplify(P * H * P.H)) == A


    B = Matrix([
        [1, 2, 10],
        [8, 2, 5],
        [3, 12, 34],
    ])
    H, P = B.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == B

    C = Matrix([
        [1, sqrt(2), 2, 3],
        [0, 5, 3, 4],
        [1, 1, 4, sqrt(5)],
        [0, 2, 2, 3]
    ])

    H, P = C.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == C

    D = Matrix([
        [1, 2, 3],
        [-3, 5, 6],
        [4, -8, 9],
    ])
    H, P = D.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == D

    E = Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ])

    H, P = E.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == E
```
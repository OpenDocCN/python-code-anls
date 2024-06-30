# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_blockmatrix.py`

```
from sympy.matrices.expressions.trace import Trace  # 导入 Trace 类
from sympy.testing.pytest import raises, slow  # 导入测试相关的函数和装饰器
from sympy.matrices.expressions.blockmatrix import (  # 导入块矩阵相关的函数和类
    block_collapse, bc_matmul, bc_block_plus_ident, BlockDiagMatrix,
    BlockMatrix, bc_dist, bc_matadd, bc_transpose, bc_inverse,
    blockcut, reblock_2x2, deblock)
from sympy.matrices.expressions import (  # 导入矩阵表达式相关的类和函数
    MatrixSymbol, Identity, trace, det, ZeroMatrix, OneMatrix)
from sympy.matrices.expressions.inverse import Inverse  # 导入求逆相关的类
from sympy.matrices.expressions.matpow import MatPow  # 导入矩阵幂相关的类
from sympy.matrices.expressions.transpose import Transpose  # 导入转置相关的类
from sympy.matrices.exceptions import NonInvertibleMatrixError  # 导入非可逆矩阵异常
from sympy.matrices import (  # 导入矩阵相关的类和函数
    Matrix, ImmutableMatrix, ImmutableSparseMatrix, zeros)
from sympy.core import Tuple, Expr, S, Function  # 导入核心类和函数
from sympy.core.symbol import Symbol, symbols  # 导入符号相关类和函数
from sympy.functions import transpose, im, re  # 导入转置、虚部和实部函数

i, j, k, l, m, n, p = symbols('i:n, p', integer=True)  # 创建整数符号变量
A = MatrixSymbol('A', n, n)  # 定义一个 n x n 的矩阵符号
B = MatrixSymbol('B', n, n)  # 定义一个 n x n 的矩阵符号
C = MatrixSymbol('C', n, n)  # 定义一个 n x n 的矩阵符号
D = MatrixSymbol('D', n, n)  # 定义一个 n x n 的矩阵符号
G = MatrixSymbol('G', n, n)  # 定义一个 n x n 的矩阵符号
H = MatrixSymbol('H', n, n)  # 定义一个 n x n 的矩阵符号
b1 = BlockMatrix([[G, H]])  # 创建一个块矩阵 b1 包含 G 和 H
b2 = BlockMatrix([[G], [H]])  # 创建一个块矩阵 b2 包含 G 和 H

def test_bc_matmul():
    assert bc_matmul(H*b1*b2*G) == BlockMatrix([[(H*G*G + H*H*H)*G]])  # 测试块矩阵乘法的功能

def test_bc_matadd():
    assert bc_matadd(BlockMatrix([[G, H]]) + BlockMatrix([[H, H]])) == \
            BlockMatrix([[G+H, H+H]])  # 测试块矩阵加法的功能

def test_bc_transpose():
    assert bc_transpose(Transpose(BlockMatrix([[A, B], [C, D]]))) == \
            BlockMatrix([[A.T, C.T], [B.T, D.T]])  # 测试块矩阵转置的功能

def test_bc_dist_diag():
    A = MatrixSymbol('A', n, n)  # 定义一个 n x n 的矩阵符号
    B = MatrixSymbol('B', m, m)  # 定义一个 m x m 的矩阵符号
    C = MatrixSymbol('C', l, l)  # 定义一个 l x l 的矩阵符号
    X = BlockDiagMatrix(A, B, C)  # 创建一个块对角矩阵 X 包含 A、B 和 C

    assert bc_dist(X+X).equals(BlockDiagMatrix(2*A, 2*B, 2*C))  # 测试块矩阵分布的功能

def test_block_plus_ident():
    A = MatrixSymbol('A', n, n)  # 定义一个 n x n 的矩阵符号
    B = MatrixSymbol('B', n, m)  # 定义一个 n x m 的矩阵符号
    C = MatrixSymbol('C', m, n)  # 定义一个 m x n 的矩阵符号
    D = MatrixSymbol('D', m, m)  # 定义一个 m x m 的矩阵符号
    X = BlockMatrix([[A, B], [C, D]])  # 创建一个块矩阵 X 包含 A、B、C 和 D
    Z = MatrixSymbol('Z', n + m, n + m)  # 定义一个 (n+m) x (n+m) 的矩阵符号
    assert bc_block_plus_ident(X + Identity(m + n) + Z) == \
            BlockDiagMatrix(Identity(n), Identity(m)) + X + Z  # 测试块矩阵加单位矩阵的功能

def test_BlockMatrix():
    A = MatrixSymbol('A', n, m)  # 定义一个 n x m 的矩阵符号
    B = MatrixSymbol('B', n, k)  # 定义一个 n x k 的矩阵符号
    C = MatrixSymbol('C', l, m)  # 定义一个 l x m 的矩阵符号
    D = MatrixSymbol('D', l, k)  # 定义一个 l x k 的矩阵符号
    M = MatrixSymbol('M', m + k, p)  # 定义一个 (m+k) x p 的矩阵符号
    N = MatrixSymbol('N', l + n, k + m)  # 定义一个 (l+n) x (k+m) 的矩阵符号
    X = BlockMatrix(Matrix([[A, B], [C, D]]))  # 通过矩阵创建一个块矩阵 X

    assert X.__class__(*X.args) == X  # 检查块矩阵的类定义与实例化时一致

    # block_collapse 在普通输入上不做任何操作
    E = MatrixSymbol('E', n, m)  # 定义一个 n x m 的矩阵符号
    assert block_collapse(A + 2*E) == A + 2*E  # 测试块矩阵折叠的功能
    F = MatrixSymbol('F', m, m)  # 定义一个 m x m 的矩阵符号
    assert block_collapse(E.T*A*F) == E.T*A*F  # 测试块矩阵折叠的功能

    assert X.shape == (l + n, k + m)  # 检查块矩阵 X 的形状
    assert X.blockshape == (2, 2)  # 检查块矩阵 X 的块形状
    assert transpose(X) == BlockMatrix(Matrix([[A.T, C.T], [B.T, D.T]]))  # 测试块矩阵转置的功能
    assert transpose(X).shape == X.shape[::-1]  # 检查转置后块矩阵的形状与原始矩阵的转置形状一致

    # 测试块矩阵和矩阵符号的混合使用
    assert (X*M).is_MatMul  # 检查块矩阵与矩阵相乘后是否为 MatMul 类型
    assert X._blockmul(M).is_MatMul  # 检查块矩阵与矩阵相乘后是否为 MatMul 类型
    assert (X*M).shape == (n + l, p)  # 检查块矩阵与矩阵相乘后的形状
    # 确认 X + N 是矩阵加法
    assert (X + N).is_MatAdd
    # 确认 X._blockadd(N) 是矩阵加法
    assert X._blockadd(N).is_MatAdd
    # 确认 (X + N) 的形状与 X 的形状相同
    assert (X + N).shape == X.shape

    # 创建一个 m 行 1 列的矩阵符号 E
    E = MatrixSymbol('E', m, 1)
    # 创建一个 k 行 1 列的矩阵符号 F
    F = MatrixSymbol('F', k, 1)

    # 创建一个块矩阵 Y，包含两个矩阵 E 和 F
    Y = BlockMatrix(Matrix([[E], [F]]))

    # 确认 X*Y 的形状为 (l + n, 1)
    assert (X*Y).shape == (l + n, 1)
    # 确认 block_collapse(X*Y) 的第一个块等于 A*E + B*F
    assert block_collapse(X*Y).blocks[0, 0] == A*E + B*F
    # 确认 block_collapse(X*Y) 的第二个块等于 C*E + D*F
    assert block_collapse(X*Y).blocks[1, 0] == C*E + D*F

    # 确保 block_collapse 能够处理包含容器对象、转置和逆操作
    assert block_collapse(transpose(X*Y)) == transpose(block_collapse(X*Y))
    # 确保 block_collapse 能够处理 Tuple(X*Y, 2*X) 返回值
    assert block_collapse(Tuple(X*Y, 2*X)) == (
        block_collapse(X*Y), block_collapse(2*X))

    # 确保 MatrixSymbols 在简化后可以进入 1x1 的 BlockMatrix
    Ab = BlockMatrix([[A]])
    Z = MatrixSymbol('Z', *A.shape)
    assert block_collapse(Ab + Z) == A + Z
# 定义一个测试函数，用于验证块矩阵合并操作的正确性
def test_block_collapse_explicit_matrices():
    # 创建一个普通矩阵 A
    A = Matrix([[1, 2], [3, 4]])
    # 断言块矩阵 BlockMatrix([[A]]) 被合并后与 A 相等
    assert block_collapse(BlockMatrix([[A]])) == A

    # 创建一个不可变稀疏矩阵 A
    A = ImmutableSparseMatrix([[1, 2], [3, 4]])
    # 断言块矩阵 BlockMatrix([[A]]) 被合并后与 A 相等
    assert block_collapse(BlockMatrix([[A]])) == A

# 定义一个测试函数，用于验证问题编号 17624 的情况
def test_issue_17624():
    # 创建一个符号矩阵 a
    a = MatrixSymbol("a", 2, 2)
    # 创建一个 2x2 的零矩阵 z
    z = ZeroMatrix(2, 2)
    # 创建一个块矩阵 b，其中包含两个块：[a, z] 和 [z, z]
    b = BlockMatrix([[a, z], [z, z]])
    # 断言块矩阵 b*b 合并后等于 BlockMatrix([[a**2, z], [z, z]])
    assert block_collapse(b * b) == BlockMatrix([[a**2, z], [z, z]])
    # 断言块矩阵 b*b*b 合并后等于 BlockMatrix([[a**3, z], [z, z]])
    assert block_collapse(b * b * b) == BlockMatrix([[a**3, z], [z, z]])

# 定义一个测试函数，用于验证问题编号 18618 的情况
def test_issue_18618():
    # 创建一个普通矩阵 A
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 断言矩阵 A 等于其作为 BlockDiagMatrix(A) 的结果
    assert A == Matrix(BlockDiagMatrix(A))

# 定义一个测试函数，用于验证块矩阵的迹操作
def test_BlockMatrix_trace():
    # 创建符号矩阵 A, B, C, D
    A, B, C, D = [MatrixSymbol(s, 3, 3) for s in 'ABCD']
    # 创建一个块矩阵 X，包含块 [A, B], [C, D]
    X = BlockMatrix([[A, B], [C, D]])
    # 断言块矩阵 X 的迹等于块 A 和 D 的迹之和
    assert trace(X) == trace(A) + trace(D)
    # 断言块矩阵 BlockMatrix([ZeroMatrix(n, n)]) 的迹为 0
    assert trace(BlockMatrix([ZeroMatrix(n, n)])) == 0

# 定义一个测试函数，用于验证块矩阵的行列式操作
def test_BlockMatrix_Determinant():
    # 创建符号矩阵 A, B, C, D
    A, B, C, D = [MatrixSymbol(s, 3, 3) for s in 'ABCD']
    # 创建一个块矩阵 X，包含块 [A, B], [C, D]
    X = BlockMatrix([[A, B], [C, D]])
    # 导入相关模块用于假设和推理
    from sympy.assumptions.ask import Q
    from sympy.assumptions.assume import assuming
    # 在假设下，断言 X 的行列式等于 A 的行列式乘以 X.schur('A') 的行列式
    with assuming(Q.invertible(A)):
        assert det(X) == det(A) * det(X.schur('A'))
    # 断言 det(X) 返回的结果是 Expr 类型
    assert isinstance(det(X), Expr)
    # 断言块矩阵 BlockMatrix([A]) 的行列式等于 A 的行列式
    assert det(BlockMatrix([A])) == det(A)
    # 断言块矩阵 BlockMatrix([ZeroMatrix(n, n)]) 的行列式为 0
    assert det(BlockMatrix([ZeroMatrix(n, n)])) == 0

# 定义一个测试函数，用于验证块矩阵的各种特性和操作
def test_squareBlockMatrix():
    # 创建符号矩阵 A, B, C, D
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', m, m)
    # 创建一个块矩阵 X，包含块 [A, B], [C, D]
    X = BlockMatrix([[A, B], [C, D]])
    # 创建一个只包含块 [A] 的块矩阵 Y
    Y = BlockMatrix([[A]])

    # 断言 X 是方阵
    assert X.is_square

    # 创建一个新的块矩阵 Q，其中包含 X 和 Identity(m+n) 的和
    Q = X + Identity(m + n)
    # 断言 block_collapse(Q) 的结果等于块矩阵 [[A + Identity(n), B], [C, D + Identity(m)]]
    assert (block_collapse(Q) ==
        BlockMatrix([[A + Identity(n), B], [C, D + Identity(m)]]))

    # 断言 X + MatrixSymbol('Q', n + m, n + m) 是 MatAdd 类型
    assert (X + MatrixSymbol('Q', n + m, n + m)).is_MatAdd
    # 断言 X * MatrixSymbol('Q', n + m, n + m) 是 MatMul 类型
    assert (X * MatrixSymbol('Q', n + m, n + m)).is_MatMul

    # 断言块矩阵 Y 的逆矩阵合并后等于 A 的逆矩阵
    assert block_collapse(Y.I) == A.I

    # 断言 X 的逆矩阵的类型是 Inverse
    assert isinstance(X.inverse(), Inverse)

    # 断言 X 不是单位矩阵
    assert not X.is_Identity

    # 创建一个新的块矩阵 Z，其中包含块 [Identity(n), B], [C, D]
    Z = BlockMatrix([[Identity(n), B], [C, D]])
    # 断言 Z 不是单位矩阵
    assert not Z.is_Identity

# 定义一个测试函数，用于验证块矩阵的符号逆矩阵操作
def test_BlockMatrix_2x2_inverse_symbolic():
    # 创建符号矩阵 A, B, C, D
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', n, k - m)
    C = MatrixSymbol('C', k - n, m)
    D = MatrixSymbol('D', k - n, k - m)
    # 创建一个块矩阵 X，包含块 [A, B], [C, D]
    X = BlockMatrix([[A, B], [C, D]])
    # 断言 X 是方阵且形状为 (k, k)
    assert X.is_square and X.shape == (k, k)
    # 断言 block_collapse(X.I) 的结果是 Inverse 类型，因为没有一个块是方阵，无法求逆

    # 测试路径，其中只有 A 是可逆的情况
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = ZeroMatrix(m, m)
    X = BlockMatrix([[A, B], [C, D]])
    # 断言 block_collapse(X.inverse()) 的结果等于以下块矩阵：
    # [[A.I + A.I * B * X.schur('A').I * C * A.I, -A.I * B * X.schur('A').I],
    #  [-X.schur('A').I * C * A.I, X.schur('A
    # 断言：测试对 X 的逆矩阵进行块状矩阵合并后的结果是否正确
    assert block_collapse(X.inverse()) == BlockMatrix([
        # 第一个块矩阵
        [-X.schur('B').I * D * B.I, X.schur('B').I],
        # 第二个块矩阵
        [B.I + B.I * A * X.schur('B').I * D * B.I, -B.I * A * X.schur('B').I],
    ])

    # 测试路径：仅 C 可逆的情况
    # 定义符号矩阵
    A = MatrixSymbol('A', n, m)
    B = ZeroMatrix(n, n)
    C = MatrixSymbol('C', m, m)
    D = MatrixSymbol('D', m, n)
    # 构造块矩阵 X
    X = BlockMatrix([[A, B], [C, D]])
    # 断言：测试对 X 的逆矩阵进行块状矩阵合并后的结果是否正确
    assert block_collapse(X.inverse()) == BlockMatrix([
        # 第一个块矩阵
        [-C.I * D * X.schur('C').I, C.I + C.I * D * X.schur('C').I * A * C.I],
        # 第二个块矩阵
        [X.schur('C').I, -X.schur('C').I * A * C.I],
    ])

    # 测试路径：仅 D 可逆的情况
    A = ZeroMatrix(n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', m, m)
    X = BlockMatrix([[A, B], [C, D]])
    # 断言：测试对 X 的逆矩阵进行块状矩阵合并后的结果是否正确
    assert block_collapse(X.inverse()) == BlockMatrix([
        # 第一个块矩阵
        [X.schur('D').I, -X.schur('D').I * B * D.I],
        # 第二个块矩阵
        [-D.I * C * X.schur('D').I, D.I + D.I * C * X.schur('D').I * B * D.I],
    ])
def test_BlockMatrix_2x2_inverse_numeric():
    """Test 2x2 block matrix inversion numerically for all 4 formulas"""
    # 创建一个2x2的数值矩阵 M
    M = Matrix([[1, 2], [3, 4]])

    # 创建三个秩为1的矩阵 D1, D2, D3
    D1 = Matrix([[1, 2], [2, 4]])
    D2 = Matrix([[1, 3], [3, 9]])
    D3 = Matrix([[1, 4], [4, 16]])

    # 断言 D1, D2, D3 的秩均为1
    assert D1.rank() == D2.rank() == D3.rank() == 1

    # 断言 D1 + D2, D2 + D3, D3 + D1 的秩均为2
    assert (D1 + D2).rank() == (D2 + D3).rank() == (D3 + D1).rank() == 2

    # 创建一个块矩阵 K，测试其逆矩阵的数值化简是否与直接求逆后的结果相同
    K = BlockMatrix([[M, D1], [D2, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()

    K = BlockMatrix([[D1, M], [D2, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()

    K = BlockMatrix([[D1, D2], [M, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()

    K = BlockMatrix([[D1, D2], [D3, M]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()


@slow
def test_BlockMatrix_3x3_symbolic():
    # 由于速度较慢，只测试其中一种情况而不是所有排列组合
    rowblocksizes = (n, m, k)
    colblocksizes = (m, k, n)
    # 创建一个3x3的符号块矩阵 K，每个块由符号矩阵 Mij 组成
    K = BlockMatrix([
        [MatrixSymbol('M%s%s' % (rows, cols), rows, cols) for cols in colblocksizes]
        for rows in rowblocksizes
    ])
    # 对块矩阵 K 求逆并断言结果是一个块矩阵
    collapse = block_collapse(K.I)
    assert isinstance(collapse, BlockMatrix)


def test_BlockDiagMatrix():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', m, m)
    C = MatrixSymbol('C', l, l)
    M = MatrixSymbol('M', n + m + l, n + m + l)

    # 创建三个块对角矩阵 X, Y，测试其属性和运算
    X = BlockDiagMatrix(A, B, C)
    Y = BlockDiagMatrix(A, 2*B, 3*C)

    assert X.blocks[1, 1] == B
    assert X.shape == (n + m + l, n + m + l)
    assert all(X.blocks[i, j].is_ZeroMatrix if i != j else X.blocks[i, j] in [A, B, C]
            for i in range(3) for j in range(3))
    assert X.__class__(*X.args) == X
    assert X.get_diag_blocks() == (A, B, C)

    # 断言块对角矩阵 X 的逆矩阵与其乘积的块矩阵是单位矩阵
    assert isinstance(block_collapse(X.I * X), Identity)

    assert bc_matmul(X*X) == BlockDiagMatrix(A*A, B*B, C*C)
    assert block_collapse(X*X) == BlockDiagMatrix(A*A, B*B, C*C)
    #XXX: should be == ??
    assert block_collapse(X + X).equals(BlockDiagMatrix(2*A, 2*B, 2*C))
    assert block_collapse(X*Y) == BlockDiagMatrix(A*A, 2*B*B, 3*C*C)
    assert block_collapse(X + Y) == BlockDiagMatrix(2*A, 3*B, 4*C)

    # 确保块对角矩阵 X 可以与普通矩阵表达式进行交互
    assert (X*(2*M)).is_MatMul
    assert (X + (2*M)).is_MatAdd

    assert (X._blockmul(M)).is_MatMul
    assert (X._blockadd(M)).is_MatAdd


def test_BlockDiagMatrix_nonsquare():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', k, l)
    # 创建一个非方阵块对角矩阵 X，测试其属性
    X = BlockDiagMatrix(A, B)
    assert X.shape == (n + k, m + l)
    assert X.shape == (n + k, m + l)
    assert X.rowblocksizes == [n, k]
    assert X.colblocksizes == [m, l]
    C = MatrixSymbol('C', n, m)
    D = MatrixSymbol('D', k, l)
    # 创建另一个非方阵块对角矩阵 Y
    Y = BlockDiagMatrix(C, D)
    # 断言，验证 block_collapse(X + Y) 的结果是否等于 BlockDiagMatrix(A + C, B + D)
    assert block_collapse(X + Y) == BlockDiagMatrix(A + C, B + D)
    
    # 断言，验证 block_collapse(X * Y.T) 的结果是否等于 BlockDiagMatrix(A * C.T, B * D.T)
    assert block_collapse(X * Y.T) == BlockDiagMatrix(A * C.T, B * D.T)
    
    # 断言，验证 BlockDiagMatrix(A, C.T) 的逆矩阵运算是否会抛出 NonInvertibleMatrixError 异常
    raises(NonInvertibleMatrixError, lambda: BlockDiagMatrix(A, C.T).inverse())
def test_BlockDiagMatrix_determinant():
    # 创建矩阵符号 A 和 B，分别为 n x n 和 m x m 的矩阵
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', m, m)
    # 检查空块对角矩阵的行列式是否为 1
    assert det(BlockDiagMatrix()) == 1
    # 检查只包含 A 的块对角矩阵的行列式是否等于 A 的行列式
    assert det(BlockDiagMatrix(A)) == det(A)
    # 检查包含 A 和 B 的块对角矩阵的行列式是否等于 A 和 B 的行列式的乘积
    assert det(BlockDiagMatrix(A, B)) == det(A) * det(B)

    # 非方阵块的情况
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', n, m)
    # 检查包含非方阵 C 和 D 的块对角矩阵的行列式是否为 0
    assert det(BlockDiagMatrix(C, D)) == 0

def test_BlockDiagMatrix_trace():
    # 检查空块对角矩阵的迹是否为 0
    assert trace(BlockDiagMatrix()) == 0
    # 检查只包含 n x n 的零矩阵的块对角矩阵的迹是否为 0
    assert trace(BlockDiagMatrix(ZeroMatrix(n, n))) == 0
    A = MatrixSymbol('A', n, n)
    # 检查只包含 A 的块对角矩阵的迹是否等于 A 的迹
    assert trace(BlockDiagMatrix(A)) == trace(A)
    B = MatrixSymbol('B', m, m)
    # 检查包含 A 和 B 的块对角矩阵的迹是否等于 A 和 B 的迹的和
    assert trace(BlockDiagMatrix(A, B)) == trace(A) + trace(B)

    # 非方阵块的情况
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', n, m)
    # 检查包含非方阵 C 和 D 的块对角矩阵的迹是否为 Trace 类型
    assert isinstance(trace(BlockDiagMatrix(C, D)), Trace)

def test_BlockDiagMatrix_transpose():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', k, l)
    # 检查空块对角矩阵的转置是否仍为空块对角矩阵
    assert transpose(BlockDiagMatrix()) == BlockDiagMatrix()
    # 检查只包含 A 的块对角矩阵的转置是否为包含 A 转置的块对角矩阵
    assert transpose(BlockDiagMatrix(A)) == BlockDiagMatrix(A.T)
    # 检查包含 A 和 B 的块对角矩阵的转置是否为包含 A 和 B 转置的块对角矩阵
    assert transpose(BlockDiagMatrix(A, B)) == BlockDiagMatrix(A.T, B.T)

def test_issue_2460():
    # 创建包含单元素矩阵 [i] 和 [j] 的块对角矩阵 bdm1 和 bdm2
    bdm1 = BlockDiagMatrix(Matrix([i]), Matrix([j]))
    bdm2 = BlockDiagMatrix(Matrix([k]), Matrix([l]))
    # 检查两个块对角矩阵相加后坍塌成包含 [i+k] 和 [j+l] 的块对角矩阵
    assert block_collapse(bdm1 + bdm2) == BlockDiagMatrix(Matrix([i + k]), Matrix([j + l]))

def test_blockcut():
    A = MatrixSymbol('A', n, m)
    # 对 A 进行块分割，切分成四块子矩阵并重新组合成块对角矩阵 B
    B = blockcut(A, (n/2, n/2), (m/2, m/2))
    assert B == BlockMatrix([[A[:n/2, :m/2], A[:n/2, m/2:]],
                             [A[n/2:, :m/2], A[n/2:, m/2:]]])

    M = ImmutableMatrix(4, 4, range(16))
    # 使用 ImmutableMatrix 创建矩阵 M，将其分割成四块子矩阵并重新组合成块对角矩阵 B
    B = blockcut(M, (2, 2), (2, 2))
    assert M == ImmutableMatrix(B)

    # 将 M 按 (1, 3) 和 (2, 2) 块分割，获取其第一行第三列的子矩阵并检查
    B = blockcut(M, (1, 3), (2, 2))
    assert ImmutableMatrix(B.blocks[0, 1]) == ImmutableMatrix([[2, 3]])

def test_reblock_2x2():
    # 创建一个 3x3 的块矩阵 B，每个块都是 2x2 的矩阵符号 A_ij
    B = BlockMatrix([[MatrixSymbol('A_%d%d'%(i,j), 2, 2)
                            for j in range(3)]
                            for i in range(3)])
    assert B.blocks.shape == (3, 3)

    # 对 B 进行 2x2 块重新分组并返回 BB
    BB = reblock_2x2(B)
    assert BB.blocks.shape == (2, 2)

    assert B.shape == BB.shape
    assert B.as_explicit() == BB.as_explicit()

def test_deblock():
    # 创建一个 4x4 的块矩阵 B，每个块都是 n x n 的矩阵符号 A_ij
    B = BlockMatrix([[MatrixSymbol('A_%d%d'%(i,j), n, n)
                    for j in range(4)]
                    for i in range(4)])

    # 对 B 进行 2x2 块重新分组后进行反向操作 deblock
    assert deblock(reblock_2x2(B)) == B

def test_block_collapse_type():
    # 创建包含单元素矩阵 [1], [2], [3], [4] 的块对角矩阵 bm1 和 bm2
    bm1 = BlockDiagMatrix(ImmutableMatrix([1]), ImmutableMatrix([2]))
    bm2 = BlockDiagMatrix(ImmutableMatrix([3]), ImmutableMatrix([4]))

    # 检查 bm1 转置的类型是否为 BlockDiagMatrix
    assert bm1.T.__class__ == BlockDiagMatrix
    # 检查 bm1 - bm2 坍塌后的类型是否为 BlockDiagMatrix
    assert block_collapse(bm1 - bm2).__class__ == BlockDiagMatrix
    # 检查 Inverse(bm1) 坍塌后的类型是否为 BlockDiagMatrix
    assert block_collapse(Inverse(bm1)).__class__ == BlockDiagMatrix
    # 检查 Transpose(bm1) 坍塌后的类型是否为 BlockDiagMatrix
    assert block_collapse(Transpose(bm1)).__class__ == BlockDiagMatrix
    # 检查 bc_transpose(Transpose(bm1)) 的类型是否为 BlockDiagMatrix
    assert bc_transpose(Transpose(bm1)).__class__ == BlockDiagMatrix
    # 检查 bc_inverse(Inverse(bm1)) 的类型是否为 BlockDiagMatrix
    assert bc_inverse(Inverse(bm1)).__class__ == BlockDiagMatrix

def test_invalid_block_matrix():
    # 这个测试函数当前是空的，可以根据需要添加测试
    pass
    # 使用 raises 函数检查是否抛出 ValueError 异常，lambda 函数中的表达式是调用 BlockMatrix 构造函数并传入参数
    raises(ValueError, lambda: BlockMatrix([
        # 第一个测试用例：创建一个包含两个子矩阵的 BlockMatrix，每个子矩阵都是 2x2 的单位矩阵
        [Identity(2), Identity(5)],
    ]))
    raises(ValueError, lambda: BlockMatrix([
        # 第二个测试用例：创建一个包含两个子矩阵的 BlockMatrix，每个子矩阵都是大小未知的单位矩阵
        [Identity(n), Identity(m)],
    ]))
    raises(ValueError, lambda: BlockMatrix([
        # 第三个测试用例：创建一个包含两行两列子矩阵的 BlockMatrix，其中包括 n x n 的零矩阵和不同大小的零矩阵
        [ZeroMatrix(n, n), ZeroMatrix(n, n)],
        [ZeroMatrix(n, n - 1), ZeroMatrix(n, n + 1)],
    ]))
    raises(ValueError, lambda: BlockMatrix([
        # 第四个测试用例：创建一个包含两行两列子矩阵的 BlockMatrix，其中包括不同大小的零矩阵
        [ZeroMatrix(n - 1, n), ZeroMatrix(n, n)],
        [ZeroMatrix(n + 1, n), ZeroMatrix(n, n)],
    ]))
def test_block_lu_decomposition():
    A = MatrixSymbol('A', n, n)  # 定义一个 n x n 的符号矩阵 A
    B = MatrixSymbol('B', n, m)  # 定义一个 n x m 的符号矩阵 B
    C = MatrixSymbol('C', m, n)  # 定义一个 m x n 的符号矩阵 C
    D = MatrixSymbol('D', m, m)  # 定义一个 m x m 的符号矩阵 D
    X = BlockMatrix([[A, B], [C, D]])  # 创建一个块矩阵 X，由 A, B, C, D 组成

    #LDU分解
    L, D, U = X.LDUdecomposition()  # 对 X 进行LDU分解，返回 L, D, U 三个矩阵
    assert block_collapse(L*D*U) == X  # 断言 L*D*U 的块矩阵收缩等于 X

    #UDL分解
    U, D, L = X.UDLdecomposition()  # 对 X 进行UDL分解，返回 U, D, L 三个矩阵
    assert block_collapse(U*D*L) == X  # 断言 U*D*L 的块矩阵收缩等于 X

    #LU分解
    L, U = X.LUdecomposition()  # 对 X 进行LU分解，返回 L, U 两个矩阵
    assert block_collapse(L*U) == X  # 断言 L*U 的块矩阵收缩等于 X

def test_issue_21866():
    n = 10  # 设置变量 n 的值为 10
    I = Identity(n)  # 创建一个 n x n 的单位矩阵 I
    O = ZeroMatrix(n, n)  # 创建一个 n x n 的零矩阵 O
    A = BlockMatrix([[I, O, O, O], [O, I, O, O], [O, O, I, O], [I, O, O, I]])  # 创建一个块矩阵 A

    Ainv = block_collapse(A.inv())  # 计算 A 的逆矩阵并收缩
    AinvT = BlockMatrix([[I, O, O, O], [O, I, O, O], [O, O, I, O], [-I, O, O, I]])  # 创建预期的 A 逆矩阵的转置
    assert Ainv == AinvT  # 断言 Ainv 等于 AinvT

def test_adjoint_and_special_matrices():
    A = Identity(3)  # 创建一个 3 x 3 的单位矩阵 A
    B = OneMatrix(3, 2)  # 创建一个 3 x 2 的全1矩阵 B
    C = ZeroMatrix(2, 3)  # 创建一个 2 x 3 的零矩阵 C
    D = Identity(2)  # 创建一个 2 x 2 的单位矩阵 D
    X = BlockMatrix([[A, B], [C, D]])  # 创建一个块矩阵 X

    X2 = BlockMatrix([[A, S.ImaginaryUnit*B], [C, D]])  # 创建另一个块矩阵 X2
    assert X.adjoint() == BlockMatrix([[A, ZeroMatrix(3, 2)], [OneMatrix(2, 3), D]])  # 断言 X 的伴随矩阵
    assert re(X) == X  # 断言 X 的实部是 X
    assert X2.adjoint() == BlockMatrix([[A, ZeroMatrix(3, 2)], [-S.ImaginaryUnit*OneMatrix(2, 3), D]])  # 断言 X2 的伴随矩阵
    assert im(X2) == BlockMatrix([[ZeroMatrix(3, 3), OneMatrix(3, 2)], [ZeroMatrix(2, 3), ZeroMatrix(2, 2)]])  # 断言 X2 的虚部

def test_block_matrix_derivative():
    x = symbols('x')  # 声明符号变量 x
    A = Matrix(3, 3, [Function(f'a{i}')(x) for i in range(9)])  # 创建一个 3x3 的函数矩阵 A
    bc = BlockMatrix([[A[:2, :2], A[:2, 2]], [A[2, :2], A[2:, 2]]])  # 创建一个块矩阵 bc

    assert Matrix(bc.diff(x)) - A.diff(x) == zeros(3, 3)  # 断言 bc 对 x 的导数矩阵等于 A 对 x 的导数矩阵

def test_transpose_inverse_commute():
    n = Symbol('n')  # 声明符号变量 n
    I = Identity(n)  # 创建一个 n x n 的单位矩阵 I
    Z = ZeroMatrix(n, n)  # 创建一个 n x n 的零矩阵 Z
    A = BlockMatrix([[I, Z], [Z, I]])  # 创建一个块矩阵 A

    assert block_collapse(A.transpose().inverse()) == A  # 断言 A 的转置的逆矩阵等于 A
    assert block_collapse(A.inverse().transpose()) == A  # 断言 A 的逆矩阵的转置等于 A

    assert block_collapse(MatPow(A.transpose(), -2)) == MatPow(A, -2)  # 断言 A 的转置的负二次幂等于 A 的负二次幂
    assert block_collapse(MatPow(A, -2).transpose()) == MatPow(A, -2)  # 断言 A 的负二次幂的转置等于 A 的负二次幂
```
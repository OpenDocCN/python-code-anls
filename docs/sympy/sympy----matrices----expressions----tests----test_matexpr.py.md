# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_matexpr.py`

```
from sympy.concrete.summations import Sum  # 导入求和符号的模块
from sympy.core.exprtools import gcd_terms  # 导入表达式工具模块中的最大公因式项合并函数
from sympy.core.function import (diff, expand)  # 导入函数操作模块中的求导和展开函数
from sympy.core.relational import Eq  # 导入关系表达式模块中的相等关系符号
from sympy.core.symbol import (Dummy, Symbol, Str)  # 导入符号模块中的虚拟符号、普通符号和字符串符号
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入特殊张量函数模块中的克罗内克δ函数
from sympy.matrices.dense import zeros  # 导入稠密矩阵模块中的零矩阵生成函数
from sympy.polys.polytools import factor  # 导入多项式工具模块中的因式分解函数

from sympy.core import (S, symbols, Add, Mul, SympifyError, Rational,  # 导入核心模块中的各种符号、表达式类型和异常
                    Function)
from sympy.functions import sin, cos, tan, sqrt, cbrt, exp  # 导入数学函数模块中的各种函数
from sympy.simplify import simplify  # 导入简化模块中的简化函数
from sympy.matrices import (ImmutableMatrix, Inverse, MatAdd, MatMul,  # 导入矩阵模块中的各种矩阵运算和特殊矩阵
        MatPow, Matrix, MatrixExpr, MatrixSymbol,
        SparseMatrix, Transpose, Adjoint, MatrixSet)
from sympy.matrices.exceptions import NonSquareMatrixError  # 导入矩阵异常模块中的非方阵异常
from sympy.matrices.expressions.determinant import Determinant, det  # 导入矩阵表达式模块中的行列式和行列式计算函数
from sympy.matrices.expressions.matexpr import MatrixElement  # 导入矩阵表达式模块中的矩阵元素
from sympy.matrices.expressions.special import ZeroMatrix, Identity  # 导入矩阵表达式模块中的零矩阵和单位矩阵
from sympy.testing.pytest import raises, XFAIL, skip  # 导入测试模块中的异常测试、标记测试失败和跳过测试
from importlib.metadata import version  # 导入元数据模块中的版本信息

n, m, l, k, p = symbols('n m l k p', integer=True)  # 定义整数类型的符号变量
x = symbols('x')  # 定义普通符号变量
A = MatrixSymbol('A', n, m)  # 定义一个 n × m 的矩阵符号
B = MatrixSymbol('B', m, l)  # 定义一个 m × l 的矩阵符号
C = MatrixSymbol('C', n, n)  # 定义一个 n × n 的矩阵符号
D = MatrixSymbol('D', n, n)  # 定义一个 n × n 的矩阵符号
E = MatrixSymbol('E', m, n)  # 定义一个 m × n 的矩阵符号
w = MatrixSymbol('w', n, 1)  # 定义一个 n × 1 的列向量符号


def test_matrix_symbol_creation():
    assert MatrixSymbol('A', 2, 2)  # 检查创建 2 × 2 矩阵符号是否成功
    assert MatrixSymbol('A', 0, 0)  # 检查创建 0 × 0 矩阵符号是否成功
    raises(ValueError, lambda: MatrixSymbol('A', -1, 2))  # 检查创建非法尺寸矩阵符号是否引发异常
    raises(ValueError, lambda: MatrixSymbol('A', 2.0, 2))  # 检查创建非整数尺寸矩阵符号是否引发异常
    raises(ValueError, lambda: MatrixSymbol('A', 2j, 2))  # 检查创建非整数尺寸矩阵符号是否引发异常
    raises(ValueError, lambda: MatrixSymbol('A', 2, -1))  # 检查创建非法尺寸矩阵符号是否引发异常
    raises(ValueError, lambda: MatrixSymbol('A', 2, 2.0))  # 检查创建非整数尺寸矩阵符号是否引发异常
    raises(ValueError, lambda: MatrixSymbol('A', 2, 2j))  # 检查创建非整数尺寸矩阵符号是否引发异常

    n = symbols('n')
    assert MatrixSymbol('A', n, n)  # 检查创建 n × n 的矩阵符号是否成功
    n = symbols('n', integer=False)
    raises(ValueError, lambda: MatrixSymbol('A', n, n))  # 检查创建非整数类型尺寸矩阵符号是否引发异常
    n = symbols('n', negative=True)
    raises(ValueError, lambda: MatrixSymbol('A', n, n))  # 检查创建负数尺寸矩阵符号是否引发异常


def test_matexpr_properties():
    assert A.shape == (n, m)  # 检查矩阵符号 A 的形状是否为 (n, m)
    assert (A * B).shape == (n, l)  # 检查矩阵表达式 A * B 的形状是否为 (n, l)
    assert A[0, 1].indices == (0, 1)  # 检查矩阵元素 A[0, 1] 的索引是否为 (0, 1)
    assert A[0, 0].symbol == A  # 检查矩阵元素 A[0, 0] 的符号是否为 A
    assert A[0, 0].symbol.name == 'A'  # 检查矩阵元素 A[0, 0] 的符号名称是否为 'A'


def test_matexpr():
    assert (x*A).shape == A.shape  # 检查数乘表达式 x*A 的形状是否与 A 相同
    assert (x*A).__class__ == MatMul  # 检查数乘表达式 x*A 的类型是否为 MatMul
    assert 2*A - A - A == ZeroMatrix(*A.shape)  # 检查矩阵表达式 2*A - A - A 是否等于零矩阵
    assert (A*B).shape == (n, l)  # 检查矩阵表达式 A * B 的形状是否为 (n, l)


def test_matexpr_subs():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, l)
    C = MatrixSymbol('C', m, l)

    assert A.subs(n, m).shape == (m, m)  # 检查替换符号 n 为 m 后矩阵 A 的形状是否为 (m, m)
    assert (A*B).subs(B, C) == A*C  # 检查替换矩阵符号 B 为 C 后矩阵表达式 A * B 是否等于 A * C
    assert (A*B).subs(l, n).is_square  # 检查替换符号 l 为 n 后矩阵表达式 A * B 是否为方阵

    W = MatrixSymbol("W", 3, 3)
    X = MatrixSymbol("X", 2, 2)
    Y = MatrixSymbol("Y", 1, 2)
    Z = MatrixSymbol("Z", n, 2)
    assert X.subs(X, Y) == Y  # 检查替换矩阵符号 X 为 Y 后是否等于 Y
    y = Str('
    # 断言：用符号替换后，X[1, 1]应该等于W[1, 1]
    assert X[1, 1].subs(X, W) == W[1, 1]
    
    # 断言：如果在新形状上进行索引是有效的，才更改名称。
    # 这里，X 是 2x2 的矩阵；Y 是 1x2 的矩阵，而Y[1, 1]超出范围，因此会引发错误。
    raises(IndexError, lambda: X[1, 1].subs(X, Y))
    
    # 断言：在这里，[0, 1] 在范围内，因此替换成功。
    assert X[0, 1].subs(X, Y) == Y[0, 1]
    
    # 断言：在这里，n 的大小将接受第一个位置的任何索引。
    assert W[2, 1].subs(W, Z) == Z[2, 1]
    
    # 断言：但是第二个位置的索引无效，会引发 IndexError。
    raises(IndexError, lambda: W[2, 2].subs(W, Z))
    
    # 断言：任何矩阵如果无效都应该引发 IndexError。
    raises(IndexError, lambda: W[2, 2].subs(W, zeros(2)))

    # 创建稀疏矩阵 A 和密集矩阵 B
    A = SparseMatrix([[1, 2], [3, 4]])
    B = Matrix([[1, 2], [3, 4]])
    
    # 创建矩阵符号 C 和 D，都是2x2的矩阵符号
    C, D = MatrixSymbol('C', 2, 2), MatrixSymbol('D', 2, 2)
    
    # 断言：在用 A 替换 C，B 替换 D 后，C*D 应该等于 MatMul(A, B)
    assert (C*D).subs({C: A, D: B}) == MatMul(A, B)
def test_addition():
    # 创建符号矩阵 A 和 B，分别是 n 行 m 列的矩阵符号
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', n, m)

    # 断言 A + B 是 MatAdd 类型的对象
    assert isinstance(A + B, MatAdd)
    # 断言 (A + B) 的形状与 A 的形状相同
    assert (A + B).shape == A.shape
    # 断言 A - A + 2*B 是 MatMul 类型的对象
    assert isinstance(A - A + 2*B, MatMul)

    # 断言 A + 1 抛出 TypeError 异常
    raises(TypeError, lambda: A + 1)
    # 断言 5 + A 抛出 TypeError 异常
    raises(TypeError, lambda: 5 + A)
    # 断言 5 - A 抛出 TypeError 异常
    raises(TypeError, lambda: 5 - A)

    # 断言 A + ZeroMatrix(n, m) - A 等于 ZeroMatrix(n, m)
    assert A + ZeroMatrix(n, m) - A == ZeroMatrix(n, m)
    # 断言 ZeroMatrix(n, m) + S.Zero 抛出 TypeError 异常
    raises(TypeError, lambda: ZeroMatrix(n, m) + S.Zero)


def test_multiplication():
    # 创建符号矩阵 A、B 和 C，分别是 n 行 m 列、m 行 l 列、n 行 n 列的矩阵符号
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, l)
    C = MatrixSymbol('C', n, n)

    # 断言 (2*A*B) 的形状为 (n, l)
    assert (2*A*B).shape == (n, l)
    # 断言 (A*0*B) 等于 ZeroMatrix(n, l)
    assert (A*0*B) == ZeroMatrix(n, l)
    # 断言 (2*A) 的形状与 A 的形状相同
    assert (2*A).shape == A.shape

    # 断言 A * ZeroMatrix(m, m) * B 等于 ZeroMatrix(n, l)
    assert A * ZeroMatrix(m, m) * B == ZeroMatrix(n, l)

    # 断言 C * Identity(n) * C.I 等于 Identity(n)
    assert C * Identity(n) * C.I == Identity(n)

    # 断言 B/2 等于 S.Half * B
    assert B/2 == S.Half*B
    # 断言 2/B 抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: 2/B)

    # 重新定义符号矩阵 A 和 B，均为 n 行 n 列的矩阵符号
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    # 断言 Identity(n) * (A + B) 等于 A + B
    assert Identity(n) * (A + B) == A + B

    # 断言 A**2*A 等于 A**3
    assert A**2*A == A**3
    # 断言 A**2*(A.I)**3 等于 A.I
    assert A**2*(A.I)**3 == A.I
    # 断言 A**3*(A.I)**2 等于 A
    assert A**3*(A.I)**2 == A


def test_MatPow():
    # 创建符号矩阵 A，是 n 行 n 列的矩阵符号
    A = MatrixSymbol('A', n, n)

    # 创建 A 的平方 MatPow 对象 AA
    AA = MatPow(A, 2)
    # 断言 AA 的指数为 2
    assert AA.exp == 2
    # 断言 AA 的基础为 A
    assert AA.base == A
    # 断言 (A**n).exp 等于 n
    assert (A**n).exp == n

    # 断言 A**0 等于 Identity(n)
    assert A**0 == Identity(n)
    # 断言 A**1 等于 A
    assert A**1 == A
    # 断言 A**2 等于 AA
    assert A**2 == AA
    # 断言 A**-1 等于 Inverse(A)
    assert A**-1 == Inverse(A)
    # 断言 (A**-1)**-1 等于 A
    assert (A**-1)**-1 == A
    # 断言 (A**2)**3 等于 A**6
    assert (A**2)**3 == A**6
    # 断言 A**S.Half 等于 sqrt(A)
    assert A**S.Half == sqrt(A)
    # 断言 A**Rational(1, 3) 等于 cbrt(A)
    assert A**Rational(1, 3) == cbrt(A)
    # 断言 MatrixSymbol('B', 3, 2)**2 抛出 NonSquareMatrixError 异常
    raises(NonSquareMatrixError, lambda: MatrixSymbol('B', 3, 2)**2)


def test_MatrixSymbol():
    # 定义符号变量 n、m、t
    n, m, t = symbols('n,m,t')
    # 创建符号矩阵 X，是 n 行 m 列的矩阵符号
    X = MatrixSymbol('X', n, m)
    # 断言 X 的形状为 (n, m)
    assert X.shape == (n, m)
    # 断言 MatrixSymbol('X', n, m)(t) 抛出 TypeError 异常
    raises(TypeError, lambda: MatrixSymbol('X', n, m)(t))  # issue 5855
    # 断言 X.doit() 等于 X
    assert X.doit() == X


def test_dense_conversion():
    # 创建符号矩阵 X，是 2 行 2 列的矩阵符号
    X = MatrixSymbol('X', 2, 2)
    # 断言 ImmutableMatrix(X) 等于 ImmutableMatrix(2, 2, lambda i, j: X[i, j])
    assert ImmutableMatrix(X) == ImmutableMatrix(2, 2, lambda i, j: X[i, j])
    # 断言 Matrix(X) 等于 Matrix(2, 2, lambda i, j: X[i, j])


def test_free_symbols():
    # 断言 (C*D).free_symbols 等于 {C, D}
    assert (C*D).free_symbols == {C, D}


def test_zero_matmul():
    # 断言 S.Zero * MatrixSymbol('X', 2, 2) 是 MatrixExpr 类型的对象
    assert isinstance(S.Zero * MatrixSymbol('X', 2, 2), MatrixExpr)


def test_matadd_simplify():
    # 创建符号矩阵 A，是 1 行 1 列的矩阵符号
    A = MatrixSymbol('A', 1, 1)
    # 断言 simplify(MatAdd(A, ImmutableMatrix([[sin(x)**2 + cos(x)**2]]))) 等于 MatAdd(A, Matrix([[1]]))
    assert simplify(MatAdd(A, ImmutableMatrix([[sin(x)**2 + cos(x)**2]]))) == \
        MatAdd(A, Matrix([[1]]))


def test_matmul_simplify():
    # 创建符号矩阵 A，是 1 行 1 列的矩阵符号
    A = MatrixSymbol('A', 1, 1)
    # 断言 simplify(MatMul(A, ImmutableMatrix([[sin(x)**2 + cos(x)**2]]))) 等于 MatMul(A, Matrix([[1]]))
    assert simplify(MatMul(A, ImmutableMatrix([[sin(x)**2 + cos(x)**2]]))) == \
        MatMul(A, Matrix([[1]]))


def test_invariants():
    # 创建符号矩阵 A、B、X，分别是 n 行 m 列、m 行 l 列、n 行 n 列的矩阵符号
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, l)
    X = MatrixSymbol('X', n, n)
    # 定义对象列表 objs
    objs = [Identity(n), ZeroMatrix(m, n), A, MatMul(A, B), MatAdd(A, A),
            Transpose(A), Adjoint(A), Inverse(X), MatPow(X, 2), MatPow
    # 使用嵌套的循环遍历二维数组 A 的特定区域
    for i in range(-2, 2):  # 对 i 从 -2 到 1 进行迭代
        for j in range(-1, 1):  # 对 j 从 -1 到 0 进行迭代
            A[i, j]  # 访问二维数组 A 中坐标为 (i, j) 的元素
# 定义一个测试函数，用于单个元素的索引测试
def test_single_indexing():
    # 创建一个符号矩阵符号'A'，维度为2行3列
    A = MatrixSymbol('A', 2, 3)
    # 断言第1个元素与索引(0, 1)处的元素相同
    assert A[1] == A[0, 1]
    # 断言整数1索引与索引(0, 1)处的元素相同
    assert A[int(1)] == A[0, 1]
    # 断言索引3处的元素与索引(1, 0)处的元素相同
    assert A[3] == A[1, 0]
    # 断言列表形式的切片[0:2, 0:2]与具体元素的列表相同
    assert list(A[:2, :2]) == [A[0, 0], A[0, 1], A[1, 0], A[1, 1]]
    # 断言索引6超出范围时会引发IndexError异常
    raises(IndexError, lambda: A[6])
    # 断言未定义的符号'n'索引时会引发IndexError异常
    raises(IndexError, lambda: A[n])
    # 创建一个符号矩阵符号'B'，行数为符号'n'，列数为符号'm'，此处应引发IndexError异常
    raises(IndexError, lambda: B[1])
    # 创建一个符号矩阵符号'B'，行数为符号'n'，列数为3，断言索引3处的元素与索引(1, 0)处的元素相同
    B = MatrixSymbol('B', n, 3)
    assert B[3] == B[1, 0]


# 定义一个测试函数，测试矩阵元素的交换性质
def test_MatrixElement_commutative():
    # 断言矩阵元素A[0, 1]*A[1, 0]等于A[1, 0]*A[0, 1]
    assert A[0, 1]*A[1, 0] == A[1, 0]*A[0, 1]


# 定义一个测试函数，测试矩阵符号的行列式计算
def test_MatrixSymbol_determinant():
    # 创建一个4x4的符号矩阵符号'A'
    A = MatrixSymbol('A', 4, 4)
    # 断言矩阵A的显式行列式等于给定表达式的值
    assert A.as_explicit().det() == A[0, 0]*A[1, 1]*A[2, 2]*A[3, 3] - \
        A[0, 0]*A[1, 1]*A[2, 3]*A[3, 2] - A[0, 0]*A[1, 2]*A[2, 1]*A[3, 3] + \
        A[0, 0]*A[1, 2]*A[2, 3]*A[3, 1] + A[0, 0]*A[1, 3]*A[2, 1]*A[3, 2] - \
        A[0, 0]*A[1, 3]*A[2, 2]*A[3, 1] - A[0, 1]*A[1, 0]*A[2, 2]*A[3, 3] + \
        A[0, 1]*A[1, 0]*A[2, 3]*A[3, 2] + A[0, 1]*A[1, 2]*A[2, 0]*A[3, 3] - \
        A[0, 1]*A[1, 2]*A[2, 3]*A[3, 0] - A[0, 1]*A[1, 3]*A[2, 0]*A[3, 2] + \
        A[0, 1]*A[1, 3]*A[2, 2]*A[3, 0] + A[0, 2]*A[1, 0]*A[2, 1]*A[3, 3] - \
        A[0, 2]*A[1, 0]*A[2, 3]*A[3, 1] - A[0, 2]*A[1, 1]*A[2, 0]*A[3, 3] + \
        A[0, 2]*A[1, 1]*A[2, 3]*A[3, 0] + A[0, 2]*A[1, 3]*A[2, 0]*A[3, 1] - \
        A[0, 2]*A[1, 3]*A[2, 1]*A[3, 0] - A[0, 3]*A[1, 0]*A[2, 1]*A[3, 2] + \
        A[0, 3]*A[1, 0]*A[2, 2]*A[3, 1] + A[0, 3]*A[1, 1]*A[2, 0]*A[3, 2] - \
        A[0, 3]*A[1, 1]*A[2, 2]*A[3, 0] - A[0, 3]*A[1, 2]*A[2, 0]*A[3, 1] + \
        A[0, 3]*A[1, 2]*A[2, 1]*A[3, 0]
    
    # 创建一个4x4的符号矩阵符号'B'
    B = MatrixSymbol('B', 4, 4)
    # 断言矩阵A+B的行列式结果等于A+B的行列式
    assert Determinant(A + B).doit() == det(A + B) == (A + B).det()


# 定义一个测试函数，测试矩阵元素的微分
def test_MatrixElement_diff():
    # 计算表达式(D*w)[k,0]对w[p,0]的偏导数
    dexpr = diff((D*w)[k,0], w[p,0])
    
    # 断言w[k, p]对w[k, p]的偏导数为1
    assert w[k, p].diff(w[k, p]) == 1
    # 断言w[k, p]对w[0, 0]的偏导数为KroneckerDelta(0, k, (0, n-1))*KroneckerDelta(0, p, (0, 0))
    assert w[k, p].diff(w[0, 0]) == KroneckerDelta(0, k, (0, n-1))*KroneckerDelta(0, p, (0, 0))
    # 创建一个Dummy变量'_i_1'
    _i_1 = Dummy("_i_1")
    # 使用 assert 断言来验证 dexpr.dummy_eq(...) 的结果是否为真
    assert dexpr.dummy_eq(Sum(KroneckerDelta(_i_1, p, (0, n-1))*D[k, _i_1], (_i_1, 0, n - 1)))
    # 使用 assert 断言来验证 dexpr.doit() 的结果是否等于 D[k, p]
    assert dexpr.doit() == D[k, p]
# 定义一个测试函数，用于测试 MatrixElement 类的各种用例
def test_MatrixElement_with_values():
    # 定义符号变量 x, y, z, w
    x, y, z, w = symbols("x y z w")
    # 创建一个 2x2 的符号矩阵 M
    M = Matrix([[x, y], [z, w]])
    # 定义符号变量 i, j
    i, j = symbols("i, j")
    # 获取矩阵 M 的元素 M[i, j]
    Mij = M[i, j]
    # 断言 Mij 是 MatrixElement 类的实例
    assert isinstance(Mij, MatrixElement)
    
    # 创建一个稀疏矩阵 Ms
    Ms = SparseMatrix([[2, 3], [4, 5]])
    # 获取稀疏矩阵 Ms 的元素 Ms[i, j]
    msij = Ms[i, j]
    # 断言 msij 是 MatrixElement 类的实例
    assert isinstance(msij, MatrixElement)
    
    # 遍历矩阵 M 和稀疏矩阵 Ms 的所有元素，并进行断言比较
    for oi, oj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert Mij.subs({i: oi, j: oj}) == M[oi, oj]
        assert msij.subs({i: oi, j: oj}) == Ms[oi, oj]
    
    # 定义一个 2x2 的符号矩阵符号 A
    A = MatrixSymbol("A", 2, 2)
    # 断言 A 的元素 A[0, 0] 在将 A 替换为 M 后等于 x
    assert A[0, 0].subs(A, M) == x
    # 断言 A 的元素 A[i, j] 在将 A 替换为 M 后等于 M[i, j]
    assert A[i, j].subs(A, M) == M[i, j]
    # 断言 M 的元素 M[i, j] 在将 M 替换为 A 后等于 A[i, j]
    assert M[i, j].subs(M, A) == A[i, j]

    # 断言 M 的元素 M[3*i - 2, j] 是 MatrixElement 类的实例
    assert isinstance(M[3*i - 2, j], MatrixElement)
    # 断言将 M 的元素 M[3*i - 2, j] 中的 i 替换为 1，j 替换为 0 后等于 M[1, 0]
    assert M[3*i - 2, j].subs({i: 1, j: 0}) == M[1, 0]
    
    # 断言 M 的元素 M[i, 0] 是 MatrixElement 类的实例
    assert isinstance(M[i, 0], MatrixElement)
    # 断言将 M 的元素 M[i, 0] 中的 i 替换为 0 后等于 M[0, 0]
    assert M[i, 0].subs(i, 0) == M[0, 0]
    
    # 断言将 M 的元素 M[0, i] 中的 i 替换为 1 后等于 M[0, 1]
    assert M[0, i].subs(i, 1) == M[0, 1]
    
    # 断言 M 的元素 M[i, j] 对 x 的偏导数等于单位矩阵的相应元素 [1, 0; 0, 0][i, j]
    assert M[i, j].diff(x) == Matrix([[1, 0], [0, 0]])[i, j]

    # 断言访问超出矩阵维度会引发 ValueError 异常
    raises(ValueError, lambda: M[i, 2])
    raises(ValueError, lambda: M[i, -1])
    raises(ValueError, lambda: M[2, i])
    raises(ValueError, lambda: M[-1, i])


# 定义测试逆矩阵函数
def test_inv():
    # 定义一个 3x3 的符号矩阵符号 B
    B = MatrixSymbol('B', 3, 3)
    # 断言 B 的逆矩阵等于 B 的负一次方
    assert B.inv() == B**-1

    # 创建一个 1x1 和 2x2 的符号矩阵符号 X，并转换为显式矩阵
    X = MatrixSymbol('X', 1, 1).as_explicit()
    # 断言 X 的逆矩阵等于 [[1/X[0, 0]]]
    assert X.inv() == Matrix([[1/X[0, 0]]])

    X = MatrixSymbol('X', 2, 2).as_explicit()
    # 计算 X 的行列式 detX
    detX = X[0, 0]*X[1, 1] - X[0, 1]*X[1, 0]
    # 计算 X 的逆矩阵 invX
    invX = Matrix([[ X[1, 1], -X[0, 1]],
                   [-X[1, 0],  X[0, 0]]]) / detX
    # 断言 X 的逆矩阵等于 invX
    assert X.inv() == invX


# 定义测试使用 NumPy 转换矩阵的函数
def test_numpy_conversion():
    try:
        from numpy import array, array_equal
    except ImportError:
        skip('NumPy must be available to test creating matrices from ndarrays')
    
    # 定义一个 2x2 的符号矩阵符号 A
    A = MatrixSymbol('A', 2, 2)
    # 创建一个 NumPy 数组 np_array，包含 MatrixElement(A, 0, 0) 等符号元素
    np_array = array([[MatrixElement(A, 0, 0), MatrixElement(A, 0, 1)],
                      [MatrixElement(A, 1, 0), MatrixElement(A, 1, 1)]])
    # 断言将 A 转换为 NumPy 数组后与 np_array 相等
    assert array_equal(array(A), np_array)
    # 断言将 A 转换为 NumPy 数组（复制）后与 np_array 相等
    assert array_equal(array(A, copy=True), np_array)
    
    # 如果 NumPy 版本大于等于 2，则运行以下测试（确保 copy 参数正常传递）
    if(int(version('numpy').split('.')[0]) >= 2):
        raises(TypeError, lambda: array(A, copy=False))


# 定义解决问题 2749 的测试函数
def test_issue_2749():
    # 定义一个 5x2 的符号矩阵符号 A
    A = MatrixSymbol("A", 5, 2)
    # 断言 (A.T * A) 的逆矩阵转换为显式矩阵等于 [[(A.T * A).I[0, 0], (A.T * A).I[0, 1]], [(A.T * A).I[1, 0], (A.T * A).I[1, 1]]]
    assert (A.T * A).I.as_explicit() == Matrix([[(A.T * A).I[0, 0], (A.T * A).I[0, 1]], \
                                                [(A.T * A).I[1, 0], (A.T * A).I[1, 1]]])


# 定义解决问题 2750 的测试函数
def test_issue_2750():
    # 定义一个 1x1 的符号矩阵符号 x
    x = MatrixSymbol('x', 1, 1)
    # 断言 (x.T * x).as_explicit() 的负二次方等于 [[x[0, 0]**(-2)]]
    assert (x.T * x).as_explicit()**-1 == Matrix([[x[0, 0]**(-2)]])


# 定义解决问题 7842 的测试函数
def test_issue_7842():
    # 定义一个 3x1 的符号矩阵符号 A
    A = MatrixSymbol('A', 3, 1)
    # 创建一个名为 B 的符号矩阵，形状为 2x1
    B = MatrixSymbol('B', 2, 1)
    
    # 断言矩阵 A 不等于矩阵 B
    assert Eq(A, B) == False
    
    # 断言矩阵 A 的某个元素与矩阵 B 的相同位置的元素相等，并检查其函数是否为 Eq
    assert Eq(A[1,0], B[1, 0]).func is Eq
    
    # 创建一个形状为 2x3 的零矩阵 A
    A = ZeroMatrix(2, 3)
    
    # 创建一个形状为 2x3 的零矩阵 B
    B = ZeroMatrix(2, 3)
    
    # 断言矩阵 A 等于矩阵 B
    assert Eq(A, B) == True
# 定义测试函数，用于检验问题21195
def test_issue_21195():
    # 符号t作为符号t的符号
    t = symbols('t')
    # x为t的函数x的函数
    x = Function('x')(t)
    # dx为x相对t的导数
    dx = x.diff(t)
    # exp1为cos(x) + cos(x)*dx
    exp1 = cos(x) + cos(x)*dx
    # exp2为sin(x) + tan(x)*(dx相对t的导数)
    exp2 = sin(x) + tan(x)*(dx.diff(t))
    # exp3为sin(x)*sin(t)*(dx相对t的导数的相对t的导数)
    exp3 = sin(x)*sin(t)*(dx.diff(t)).diff(t)
    # A为一个包含exp1, exp2, exp3的矩阵
    A = Matrix([[exp1], [exp2], [exp3]])
    # B为A对x的导数的矩阵
    B = Matrix([[exp1.diff(x)], [exp2.diff(x)], [exp3.diff(x)]])
    # 断言A对x的导数等于B
    assert A.diff(x) == B


# 定义测试函数，用于检验问题24859
def test_issue_24859():
    # 定义一个2x3的矩阵符号A
    A = MatrixSymbol('A', 2, 3)
    # 定义一个3x2的矩阵符号B
    B = MatrixSymbol('B', 3, 2)
    # J为A和B的矩阵乘积
    J = A*B
    # Jinv为J的伴随矩阵
    Jinv = Matrix(J).adjugate()
    # 定义一个2x3的矩阵符号u
    u = MatrixSymbol('u', 2, 3)
    # Jk为Jinv中用A替换成A + x*u后的结果
    Jk = Jinv.subs(A, A + x*u)
    # 计算期望值，expected为B[0, 1]*u[1, 0] + B[1, 1]*u[1, 1] + B[2, 1]*u[1, 2]
    expected = B[0, 1]*u[1, 0] + B[1, 1]*u[1, 1] + B[2, 1]*u[1, 2]
    # 断言Jk的第(0, 0)元素对x的导数等于期望值
    assert Jk[0, 0].diff(x) == expected
    # 断言Jk的第(0, 0)元素对x的导数的计算结果等于期望值
    assert diff(Jk[0, 0], x).doit() == expected


# 定义测试函数，用于检验MatMul后处理器
def test_MatMul_postprocessor():
    # z为一个2x2的零矩阵
    z = zeros(2)
    # z1为一个2x2的零矩阵
    z1 = ZeroMatrix(2, 2)
    # 断言Mul(0, z)等于Mul(z, 0)等于[z, z1]中的一个
    assert Mul(0, z) == Mul(z, 0) in [z, z1]

    # M为一个2x2的矩阵
    M = Matrix([[1, 2], [3, 4]])
    # Mx为M中的每个元素乘以x得到的矩阵
    Mx = Matrix([[x, 2*x], [3*x, 4*x]])
    # 断言Mul(x, M)等于Mul(M, x)等于Mx
    assert Mul(x, M) == Mul(M, x) == Mx

    # A为一个2x2的矩阵符号A
    A = MatrixSymbol("A", 2, 2)
    # 断言Mul(A, M)等于MatMul(A, M)
    assert Mul(A, M) == MatMul(A, M)
    # 断言Mul(M, A)等于MatMul(M, A)
    assert Mul(M, A) == MatMul(M, A)

    # 标量应该吸收到常数矩阵中
    a = Mul(x, M, A)
    b = Mul(M, x, A)
    c = Mul(M, A, x)
    # 断言a等于b等于c等于MatMul(Mx, A)
    assert a == b == c == MatMul(Mx, A)
    a = Mul(x, A, M)
    b = Mul(A, x, M)
    c = Mul(A, M, x)
    # 断言a等于b等于c等于MatMul(A, Mx)
    assert a == b == c == MatMul(A, Mx)
    # 断言Mul(M, M)等于M的平方
    assert Mul(M, M) == M**2
    # 断言Mul(A, M, M)等于MatMul(A, M的平方)
    assert Mul(A, M, M) == MatMul(A, M**2)
    # 断言Mul(M, M, A)等于MatMul(M的平方, A)
    assert Mul(M, M, A) == MatMul(M**2, A)
    # 断言Mul(M, A, M)等于MatMul(M, A, M)
    assert Mul(M, A, M) == MatMul(M, A, M)

    # 断言Mul(A, x, M, M, x)等于MatMul(A, Mx的平方)
    assert Mul(A, x, M, M, x) == MatMul(A, Mx**2)


@XFAIL
# 定义XFAIL测试函数，用于MatAdd后处理器
def test_MatAdd_postprocessor_xfail():
    # 这很难工作，因为Add处理其参数的方式。
    z = zeros(2)
    assert Add(z, S.NaN) == Add(S.NaN, z)


# 定义测试函数，用于MatAdd后处理器
def test_MatAdd_postprocessor():
    # 一些是无意义的，但我们不会为Add引发错误，因为这会破坏要用虚拟符号替换矩阵的算法。

    z = zeros(2)

    # 断言Add(0, z)等于Add(z, 0)等于z
    assert Add(0, z) == Add(z, 0) == z

    a = Add(S.Infinity, z)
    # 断言a等于Add(z, S.Infinity)
    assert a == Add(z, S.Infinity)
    # 断言a是Add类型
    assert isinstance(a, Add)
    # 断言a的参数为(S.Infinity, z)
    assert a.args == (S.Infinity, z)

    a = Add(S.ComplexInfinity, z)
    # 断言a等于Add(z, S.ComplexInfinity)
    assert a == Add(z, S.ComplexInfinity)
    # 断言a是Add类型
    assert isinstance(a, Add)
    # 断言a的参数为(S.ComplexInfinity, z)
    assert a.args == (S.ComplexInfinity, z)

    a = Add(z, S.NaN)
    # 断言a是Add类型
    assert isinstance(a, Add)
    # 断言a的参数为(S.NaN, z)
    assert a.args == (S.NaN, z)

    # M为一个2x2的矩阵
    M = Matrix([[1, 2], [3, 4]])
    a = Add(x, M)
    # 断言a等于Add(M, x)
    assert a == Add(M, x)
    # 断言a是Add类型
    assert isinstance(a, Add)
    # 断言a的参数为(x, M)

    # A为一个2x2的矩阵符号A
    A = MatrixSymbol("A", 2, 2)
    # 断言Add(A, M)等于A + M
    assert Add(A, M) == Add(M, A) == A + M

    # 标量应该吸收到常数矩阵中（产生错误）
    a = Add(x, M, A)
    assert a == Add(M, x, A) == Add(M, A, x) == Add(x, A, M) == Add(A, x, M) == Add(A, M, x)
    # 断
    # 断言a是Add类的实例，确保类型正确
    assert isinstance(a, Add)
    # 断言a的参数是一个包含两个元素的元组，分别是2*x和A + 2*M
    assert a.args == (2*x, A + 2*M)
# 测试简化矩阵表达式的函数
def test_simplify_matrix_expressions():
    # 断言：gcd_terms(C*D + D*C) 的返回类型是 MatAdd
    assert type(gcd_terms(C*D + D*C)) == MatAdd
    # 对于表达式 2*C*D + 4*D*C 进行简化，断言其类型为 MatAdd
    a = gcd_terms(2*C*D + 4*D*C)
    assert type(a) == MatAdd
    # 断言简化后的表达式 a 的参数是 (2*C*D, 4*D*C)

# 测试指数函数
def test_exp():
    # 定义符号矩阵 A 和 B
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 2, 2)
    # 创建指数表达式 expr1 和 expr2
    expr1 = exp(A)*exp(B)
    expr2 = exp(B)*exp(A)
    # 断言 expr1 不等于 expr2
    assert expr1 != expr2
    # 断言 expr1 - expr2 不等于零
    assert expr1 - expr2 != 0
    # 断言 expr1 不是 exp 类型的实例
    assert not isinstance(expr1, exp)
    # 断言 expr2 不是 exp 类型的实例
    assert not isinstance(expr2, exp)

# 测试无效参数
def test_invalid_args():
    # 断言当传递无效参数给 MatrixSymbol 时，会抛出 SympifyError 异常
    raises(SympifyError, lambda: MatrixSymbol(1, 2, 'A'))

# 测试从符号创建的 MatrixSymbol
def test_matrixsymbol_from_symbol():
    # 定义复数符号 A_label
    A_label = Symbol('A', complex=True)
    # 使用 A_label 创建 MatrixSymbol A
    A = MatrixSymbol(A_label, 2, 2)

    # 对 A 进行 doit() 操作，断言结果的参数与 A 相同
    A_1 = A.doit()
    # 对 A 进行 subs(2, 3) 操作，断言结果的第一个参数与 A 的第一个参数相同
    A_2 = A.subs(2, 3)
    assert A_1.args == A.args
    assert A_2.args[0] == A.args[0]

# 测试 as_explicit() 方法
def test_as_explicit():
    # 定义矩阵符号 Z
    Z = MatrixSymbol('Z', 2, 3)
    # 断言 Z 的 as_explicit() 结果与指定的矩阵匹配
    assert Z.as_explicit() == ImmutableMatrix([
        [Z[0, 0], Z[0, 1], Z[0, 2]],
        [Z[1, 0], Z[1, 1], Z[1, 2]],
    ])
    # 断言调用未定义的矩阵 A 的 as_explicit() 会引发 ValueError 异常
    raises(ValueError, lambda: A.as_explicit())

# 测试 MatrixSet 类
def test_MatrixSet():
    # 创建实数域上的 2x2 矩阵集合 M
    M = MatrixSet(2, 2, set=S.Reals)
    # 断言 M 的形状是 (2, 2)
    assert M.shape == (2, 2)
    # 断言 M 的集合是 S.Reals
    assert M.set == S.Reals
    # 定义矩阵 X，断言 X 在 M 中
    X = Matrix([[1, 2], [3, 4]])
    assert X in M
    # 创建零矩阵 X，断言 X 在 M 中
    X = ZeroMatrix(2, 2)
    assert X in M
    # 断言矩阵 A 不在 M 中会引发 TypeError 异常
    raises(TypeError, lambda: A in M)
    raises(TypeError, lambda: 1 in M)

    # 测试使用未定义的 n 和 m 创建 MatrixSet 会引发异常
    M = MatrixSet(n, m, set=S.Reals)
    assert A in M  # 断言矩阵 A 在 M 中
    raises(TypeError, lambda: C in M)  # 断言矩阵 C 不在 M 中会引发 TypeError 异常
    raises(TypeError, lambda: X in M)  # 断言矩阵 X 不在 M 中会引发 TypeError 异常

    # 创建集合为 {1, 2, 3} 的 2x2 矩阵集合 M
    M = MatrixSet(2, 2, set={1, 2, 3})
    # 定义矩阵 X 和 Y
    X = Matrix([[1, 2], [3, 4]])
    Y = Matrix([[1, 2]])
    # 断言 X 和 Y 都不在 M 中
    assert (X in M) == S.false
    assert (Y in M) == S.false
    # 测试创建非法形状的 MatrixSet 会引发 ValueError 异常
    raises(ValueError, lambda: MatrixSet(2, -2, S.Reals))
    raises(ValueError, lambda: MatrixSet(2.4, -1, S.Reals))
    # 测试使用非法集合的 MatrixSet 会引发 TypeError 异常
    raises(TypeError, lambda: MatrixSet(2, 2, (1, 2, 3)))

# 测试解矩阵符号方程
def test_matrixsymbol_solving():
    # 定义矩阵符号 A 和 B，以及零矩阵 Z
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 2, 2)
    Z = ZeroMatrix(2, 2)
    # 断言 -(-A + B) - A + B 等于零矩阵 Z
    assert -(-A + B) - A + B == Z
    # 断言对 -(-A + B) - A + B 进行简化后等于零矩阵 Z
    assert (-(-A + B) - A + B).simplify() == Z
    # 断言对 -(-A + B) - A + B 进行展开后等于零矩阵 Z
    assert (-(-A + B) - A + B).expand() == Z
    # 断言对 -(-A + B) - A + B - Z 进行简化后等于零矩阵 Z
    assert (-(-A + B) - A + B - Z).simplify() == Z
    # 断言对 -(-A + B) - A + B - Z 进行展开后等于零矩阵 Z
    assert (-(-A + B) - A + B - Z).expand() == Z
    # 断言 (A*(A + B) + B*(A.T + B.T)) 进行展开后等于 A^2 + A*B + B*A.T + B*B.T
    assert (A*(A + B) + B*(A.T + B.T)).expand() == A**2 + A*B + B*A.T + B*B.T
```
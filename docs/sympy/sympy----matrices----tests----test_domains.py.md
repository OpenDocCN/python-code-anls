# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_domains.py`

```
# 导入必要的库和模块
from sympy import GF, ZZ, QQ, EXRAW
from sympy.polys.matrices import DomainMatrix, DM
from sympy import (
    Matrix,
    MutableMatrix,
    ImmutableMatrix,
    SparseMatrix,
    MutableDenseMatrix,
    ImmutableDenseMatrix,
    MutableSparseMatrix,
    ImmutableSparseMatrix,
)
from sympy import symbols, S, sqrt
from sympy.testing.pytest import raises

# 定义符号变量 x 和 y
x, y = symbols('x y')

# 定义所有可能的矩阵类型的元组
MATRIX_TYPES = (
    Matrix,
    MutableMatrix,
    ImmutableMatrix,
    SparseMatrix,
    MutableDenseMatrix,
    ImmutableDenseMatrix,
    MutableSparseMatrix,
    ImmutableSparseMatrix,
)

# 定义不可变矩阵类型的元组
IMMUTABLE = (
    ImmutableMatrix,
    ImmutableDenseMatrix,
    ImmutableSparseMatrix,
)

# 定义一个函数 DMs，用于创建域矩阵对象并转换为稀疏表示
def DMs(items, domain):
    return DM(items, domain).to_sparse()

# 定义测试函数 test_Matrix_rep_domain
def test_Matrix_rep_domain():

    # 遍历所有矩阵类型
    for Mat in MATRIX_TYPES:

        # 创建一个矩阵对象 M
        M = Mat([[1, 2], [3, 4]])
        # 断言矩阵对象的内部表示与预期的域矩阵对象相等
        assert M._rep == DMs([[1, 2], [3, 4]], ZZ)
        # 断言矩阵对象除以 2 后的内部表示与预期的域矩阵对象相等
        assert (M / 2)._rep == DMs([[(1,2), 1], [(3,2), 2]], QQ)
        # 如果矩阵对象不是不可变类型，则修改其元素并断言内部表示的变化
        if not isinstance(M, IMMUTABLE):
            M[0, 0] = x
            assert M._rep == DMs([[x, 2], [3, 4]], EXRAW)

        # 创建一个新的矩阵对象 M
        M = Mat([[S(1)/2, 2], [3, 4]])
        # 断言矩阵对象的内部表示与预期的域矩阵对象相等
        assert M._rep == DMs([[(1,2), 2], [3, 4]], QQ)
        # 如果矩阵对象不是不可变类型，则修改其元素并断言内部表示的变化
        if not isinstance(M, IMMUTABLE):
            M[0, 0] = x
            assert M._rep == DMs([[x, 2], [3, 4]], EXRAW)

        # 创建一个域矩阵对象 dM
        dM = DMs([[1, 2], [3, 4]], ZZ)
        # 断言矩阵类型的 _fromrep 方法返回的域矩阵对象与预期的 dM 相等
        assert Mat._fromrep(dM)._rep == dM

    # XXX: 这部分代码不是预期的行为。也许应该强制转换为 EXRAW？
    # 永远不会像这样调用私有的 _fromrep 方法，但也许应该有防护措施。
    #
    # 不清楚如何将除了 ZZ、QQ 和 EXRAW 之外的其他域集成到 Matrix 中，
    # 或者这个类的公共类型是否需要与 Matrix 有所不同。
    K = QQ.algebraic_field(sqrt(2))
    dM = DM([[1, 2], [3, 4]], K)
    assert Mat._fromrep(dM)._rep.domain == K

# 定义测试函数 test_Matrix_to_DM
def test_Matrix_to_DM():

    # 创建一个矩阵对象 M
    M = Matrix([[1, 2], [3, 4]])
    # 断言矩阵对象转换为域矩阵对象后与预期的稀疏表示相等
    assert M.to_DM() == DMs([[1, 2], [3, 4]], ZZ)
    # 断言矩阵对象转换为域矩阵对象后不是同一个对象
    assert M.to_DM() is not M._rep
    # 断言矩阵对象转换为域矩阵对象后指定 field=True 与预期的稀疏表示相等
    assert M.to_DM(field=True) == DMs([[1, 2], [3, 4]], QQ)
    # 断言矩阵对象转换为域矩阵对象后指定 domain=QQ 与预期的稀疏表示相等
    assert M.to_DM(domain=QQ) == DMs([[1, 2], [3, 4]], QQ)
    # 断言矩阵对象转换为域矩阵对象后指定 domain=QQ[x] 与预期的稀疏表示相等
    assert M.to_DM(domain=QQ[x]) == DMs([[1, 2], [3, 4]], QQ[x])
    # 断言矩阵对象转换为域矩阵对象后指定 domain=GF(3) 与预期的稀疏表示相等
    assert M.to_DM(domain=GF(3)) == DMs([[1, 2], [0, 1]], GF(3))

    # 创建一个新的矩阵对象 M
    M = Matrix([[1, 2], [3, 4]])
    # 修改矩阵对象元素并断言其内部表示的域为 EXRAW
    M[0, 0] = x
    assert M._rep.domain == EXRAW
    # 将矩阵对象元素恢复为原始值并断言转换为域矩阵对象后与预期的稀疏表示相等
    M[0, 0] = 1
    assert M.to_DM() == DMs([[1, 2], [3, 4]], ZZ)

    # 创建一个新的矩阵对象 M
    M = Matrix([[S(1)/2, 2], [3, 4]])
    # 断言矩阵对象转换为域矩阵对象后与预期的稀疏表示相等
    assert M.to_DM() == DMs([[QQ(1,2), 2], [3, 4]], QQ)

    # 创建一个新的矩阵对象 M
    M = Matrix([[x, 2], [3, 4]])
    # 断言矩阵对象转换为域矩阵对象后与预期的稀疏表示相等
    assert M.to_DM() == DMs([[x, 2], [3, 4]], ZZ[x])
    # 断言矩阵对象转换为域矩阵对象后指定 field=True 与预期的稀疏表示相等
    assert M.to_DM(field=True) == DMs([[x, 2], [3, 4]], ZZ.frac_field(x))

    # 创建一个新的矩阵对象 M
    M = Matrix([[1/x, 2], [3, 4]])
    # 断言矩阵对象转换为域矩阵对象后与预期的稀疏表示相等
    assert M.to_DM() == DMs([[1/x, 2], [3, 4]], ZZ.frac_field(x))

    # 创建一个新的矩阵对象 M
    M = Matrix([[1, sqrt(2)], [3, 4]])
    # 创建一个有理数域 K
    K = QQ.algebraic_field(sqrt(2))
    # 将 sqrt(2) 转换为 K 中的元素，并断言是否应该工作
    sqrt2 = K.from_sympy(sqrt(2))  # XXX: Maybe K(sqrt(2)) should work
    # 创建一个 DomainMatrix 对象 M_K，使用给定的元素和域参数 K
    M_K = DomainMatrix([[K(1), sqrt2], [K(3), K(4)]], (2, 2), K)
    # 使用断言检查 M 对象转换为 DomainMatrix 后是否与预期的 DomainMatrix 对象相等
    assert M.to_DM() == DMs([[1, sqrt(2)], [3, 4]], EXRAW)
    # 使用断言检查带有扩展选项的 M 对象转换为 DomainMatrix 是否与预期的稀疏矩阵数据相等
    assert M.to_DM(extension=True) == M_K.to_sparse()

    # 尝试将 Matrix 对象 M 转换为 DomainMatrix 时，使用了不兼容的 domain 参数和 field 参数，预期引发 TypeError 异常
    M = Matrix([[1, 2], [3, 4]])
    raises(TypeError, lambda: M.to_DM(domain=QQ, field=True))
```
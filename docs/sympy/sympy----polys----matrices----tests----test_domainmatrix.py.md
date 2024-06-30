# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_domainmatrix.py`

```
# 从 sympy.external.gmpy 导入 GROUND_TYPES
from sympy.external.gmpy import GROUND_TYPES

# 从 sympy 中导入需要的类和函数
from sympy import Integer, Rational, S, sqrt, Matrix, symbols
from sympy import FF, ZZ, QQ, QQ_I, EXRAW

# 导入 DomainMatrix 相关的类和异常
from sympy.polys.matrices.domainmatrix import DomainMatrix, DomainScalar, DM
from sympy.polys.matrices.exceptions import (
    DMBadInputError, DMDomainError, DMShapeError, DMFormatError, DMNotAField,
    DMNonSquareMatrixError, DMNonInvertibleMatrixError,
)
# 导入 DDM 和 SDM 类
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM

# 导入测试所需的 raises 函数
from sympy.testing.pytest import raises


# 定义测试函数 test_DM
def test_DM():
    # 创建 DDM 对象 ddm
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建 DM 对象 A
    A = DM([[1, 2], [3, 4]], ZZ)
    # 如果 GROUND_TYPES 不等于 'flint'，则断言 A.rep 等于 ddm
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        # 否则断言 A.rep 等于 ddm.to_dfm()
        assert A.rep == ddm.to_dfm()
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的 domain 为 ZZ


# 定义测试函数 test_DomainMatrix_init
def test_DomainMatrix_init():
    # 初始化 lol 和 dod
    lol = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    dod = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    # 创建 DDM 和 SDM 对象
    ddm = DDM(lol, (2, 2), ZZ)
    sdm = SDM(dod, (2, 2), ZZ)

    # 创建 DomainMatrix 对象 A，使用 lol 初始化
    A = DomainMatrix(lol, (2, 2), ZZ)
    # 如果 GROUND_TYPES 不等于 'flint'，则断言 A.rep 等于 ddm
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        # 否则断言 A.rep 等于 ddm.to_dfm()
        assert A.rep == ddm.to_dfm()
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的 domain 为 ZZ

    # 使用 dod 初始化 DomainMatrix 对象 A
    A = DomainMatrix(dod, (2, 2), ZZ)
    # 断言 A.rep 等于 sdm
    assert A.rep == sdm
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的 domain 为 ZZ

    # 使用 ddm 初始化 DomainMatrix 对象 A，预期抛出 TypeError 异常
    raises(TypeError, lambda: DomainMatrix(ddm, (2, 2), ZZ))
    # 使用 sdm 初始化 DomainMatrix 对象 A，预期抛出 TypeError 异常
    raises(TypeError, lambda: DomainMatrix(sdm, (2, 2), ZZ))
    # 使用不符合形状要求的 Matrix 对象初始化 DomainMatrix 对象 A，预期抛出 TypeError 异常
    raises(TypeError, lambda: DomainMatrix(Matrix([[1]]), (1, 1), ZZ))

    # 对于每一种格式 fmt，检查初始化 DomainMatrix 的结果
    for fmt, rep in [('sparse', sdm), ('dense', ddm)]:
        # 如果 fmt 是 'dense' 并且 GROUND_TYPES 是 'flint'，则将 rep 转换为 ddm
        if fmt == 'dense' and GROUND_TYPES == 'flint':
            rep = rep.to_dfm()
        # 使用 lol 和 dod 初始化 DomainMatrix 对象 A，指定 fmt
        A = DomainMatrix(lol, (2, 2), ZZ, fmt=fmt)
        # 断言 A.rep 等于 rep
        assert A.rep == rep
        # 使用 dod 初始化 DomainMatrix 对象 A，指定 fmt
        A = DomainMatrix(dod, (2, 2), ZZ, fmt=fmt)
        # 断言 A.rep 等于 rep
        assert A.rep == rep

    # 使用无效的 fmt 参数初始化 DomainMatrix 对象 A，预期抛出 ValueError 异常
    raises(ValueError, lambda: DomainMatrix(lol, (2, 2), ZZ, fmt='invalid'))

    # 使用不符合形状要求的 lol 初始化 DomainMatrix 对象 A，预期抛出 DMBadInputError 异常
    raises(DMBadInputError, lambda: DomainMatrix([[ZZ(1), ZZ(2)]], (2, 2), ZZ))


# 定义测试函数 test_DomainMatrix_from_rep
def test_DomainMatrix_from_rep():
    # 创建 DDM 对象 ddm
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 使用 ddm 初始化 DomainMatrix 对象 A
    A = DomainMatrix.from_rep(ddm)
    # 断言 A.rep 等于 ddm
    assert A.rep == ddm
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的 domain 为 ZZ

    # 创建 SDM 对象 sdm
    sdm = SDM({0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    # 使用 sdm 初始化 DomainMatrix 对象 A
    A = DomainMatrix.from_rep(sdm)
    # 断言 A.rep 等于 sdm
    assert A.rep == sdm
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的 domain 为 ZZ

    # 使用不符合形状要求的 Matrix 对象初始化 DomainMatrix 对象 A，预期抛出 TypeError 异常
    A = DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    raises(TypeError, lambda: DomainMatrix.from_rep(A))


# 定义测试函数 test_DomainMatrix_from_list
def test_DomainMatrix_from_list():
    # 创建 DDM 对象 ddm
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 使用 lol 初始化 DomainMatrix 对象 A
    A = DomainMatrix.from_list([[1, 2], [3, 4]], ZZ)
    # 如果 GROUND_TYPES 不等于 'flint'，则断言 A.rep 等于 ddm
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        # 否则断言 A.rep 等于 ddm.to_dfm()
        assert A.rep == ddm.to_dfm()
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的 domain 为 ZZ

    # 创建 FF(7) 域对象 dom
    dom = FF(7)
    # 创建 FF(7) 域中的 DDM 对象 ddm
    ddm = DDM([[dom(1), dom(2)], [dom(3), dom(4)]], (2, 2), dom)
    # 使用 lol 和 dom 初始化 DomainMatrix 对象 A
    A = DomainMatrix.from_list([[1, 2], [3, 4]], dom)
    # 确保 A 的表示形式等于 ddm
    assert A.rep == ddm
    # 确保 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 确保 A 的定义域等于 dom

    # 创建一个新的 DDM 对象 ddm，包含有理数 QQ(1/2), QQ(3), QQ(1/4), QQ(5) 的 2x2 矩阵
    ddm = DDM([[QQ(1, 2), QQ(3, 1)], [QQ(1, 4), QQ(5, 1)]], (2, 2), QQ)
    # 使用 DomainMatrix 类的 from_list 方法创建 A 对象，包含有理数 (1/2), (3), (1/4), (5) 的 2x2 矩阵
    A = DomainMatrix.from_list([[(1, 2), (3, 1)], [(1, 4), (5, 1)]], QQ)
    # 如果 GROUND_TYPES 不等于 'flint'，则确保 A 的表示形式等于 ddm
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        # 如果 GROUND_TYPES 等于 'flint'，则确保 A 的表示形式等于 ddm 转换为 DFM 后的结果
        assert A.rep == ddm.to_dfm()
    # 确保 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 确保 A 的定义域等于 QQ
    assert A.domain == QQ
# 定义一个测试函数，用于测试 DomainMatrix 类的 from_list_sympy 方法
def test_DomainMatrix_from_list_sympy():
    # 创建一个 ZZ 域上的 2x2 DomainMatrix 对象 ddm
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 使用 from_list_sympy 方法创建一个 DomainMatrix 对象 A，传入一个普通的二维列表作为输入
    A = DomainMatrix.from_list_sympy(2, 2, [[1, 2], [3, 4]])
    # 如果 GROUND_TYPES 不等于 'flint'，则断言 A 的内部表示与 ddm 相同
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        # 如果 GROUND_TYPES 等于 'flint'，则断言 A 的内部表示与 ddm 转换为 dfm 后的表示相同
        assert A.rep == ddm.to_dfm()
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的定义域为 ZZ

    # 创建一个 QQ 域的代数扩展域 K，用于后续测试
    K = QQ.algebraic_field(sqrt(2))
    # 创建一个 K 域上的 2x2 DomainMatrix 对象 ddm
    ddm = DDM(
        [[K.convert(1 + sqrt(2)), K.convert(2 + sqrt(2))],
         [K.convert(3 + sqrt(2)), K.convert(4 + sqrt(2))]],
        (2, 2),
        K
    )
    # 使用 from_list_sympy 方法创建一个 DomainMatrix 对象 A，传入一个包含 sqrt(2) 的表达式的列表作为输入，并标记为扩展模式
    A = DomainMatrix.from_list_sympy(
        2, 2, [[1 + sqrt(2), 2 + sqrt(2)], [3 + sqrt(2), 4 + sqrt(2)]],
        extension=True)
    # 断言 A 的内部表示与 ddm 相同
    assert A.rep == ddm
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的定义域为 K


# 定义一个测试函数，用于测试 DomainMatrix 类的 from_dict_sympy 方法
def test_DomainMatrix_from_dict_sympy():
    # 创建一个 QQ 域上的 2x2 SparseDomainMatrix 对象 sdm
    sdm = SDM({0: {0: QQ(1, 2)}, 1: {1: QQ(2, 3)}}, (2, 2), QQ)
    # 创建一个 sympy 字典
    sympy_dict = {0: {0: Rational(1, 2)}, 1: {1: Rational(2, 3)}}
    # 使用 from_dict_sympy 方法创建一个 DomainMatrix 对象 A，传入 sympy 字典作为输入
    A = DomainMatrix.from_dict_sympy(2, 2, sympy_dict)
    # 断言 A 的内部表示与 sdm 相同
    assert A.rep == sdm
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的定义域为 QQ

    # 定义 fds 作为 from_dict_sympy 方法的别名
    fds = DomainMatrix.from_dict_sympy
    # 使用 lambda 表达式测试输入不合法时的异常处理，预期抛出 DMBadInputError 异常
    raises(DMBadInputError, lambda: fds(2, 2, {3: {0: Rational(1, 2)}}))
    raises(DMBadInputError, lambda: fds(2, 2, {0: {3: Rational(1, 2)}}))


# 定义一个测试函数，用于测试 DomainMatrix 类的 from_Matrix 方法
def test_DomainMatrix_from_Matrix():
    # 创建一个 ZZ 域上的 2x2 SparseDomainMatrix 对象 sdm
    sdm = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
    # 使用 from_Matrix 方法创建一个 DomainMatrix 对象 A，传入 sympy Matrix 对象作为输入
    A = DomainMatrix.from_Matrix(Matrix([[1, 2], [3, 4]]))
    # 断言 A 的内部表示与 sdm 相同
    assert A.rep == sdm
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的定义域为 ZZ

    # 创建一个 QQ 域的代数扩展域 K，用于后续测试
    K = QQ.algebraic_field(sqrt(2))
    # 创建一个 K 域上的 2x2 SparseDomainMatrix 对象 sdm
    sdm = SDM(
        {0: {0: K.convert(1 + sqrt(2)), 1: K.convert(2 + sqrt(2))},
         1: {0: K.convert(3 + sqrt(2)), 1: K.convert(4 + sqrt(2))}},
        (2, 2),
        K
    )
    # 使用 from_Matrix 方法创建一个 DomainMatrix 对象 A，传入 sympy Matrix 对象作为输入，并标记为扩展模式
    A = DomainMatrix.from_Matrix(
        Matrix([[1 + sqrt(2), 2 + sqrt(2)], [3 + sqrt(2), 4 + sqrt(2)]]),
        extension=True)
    # 断言 A 的内部表示与 sdm 相同
    assert A.rep == sdm
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的定义域为 K

    # 使用 from_Matrix 方法创建一个 DomainMatrix 对象 A，传入 sympy Matrix 对象作为输入，并指定格式为 'dense'
    A = DomainMatrix.from_Matrix(Matrix([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]]), fmt='dense')
    # 创建一个 QQ 域上的 2x2 DomainMatrix 对象 ddm
    ddm = DDM([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]], (2, 2), QQ)
    # 如果 GROUND_TYPES 不等于 'flint'，则断言 A 的内部表示与 ddm 相同
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        # 如果 GROUND_TYPES 等于 'flint'，则断言 A 的内部表示与 ddm 转换为 dfm 后的表示相同
        assert A.rep == ddm.to_dfm()
    # 断言 A 的形状为 (2, 2)
    assert A.shape == (2, 2)
    # 断言 A 的定义域为 QQ


# 定义一个测试函数，用于测试 DomainMatrix 类的 __eq__ 方法
def test_DomainMatrix_eq():
    # 创建一个 ZZ 域上的 2x2 DomainMatrix 对象 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言 A 与自身相等
    assert A == A
    # 创建一个 ZZ 域上的 2x2 DomainMatrix 对象 B，与 A 的部分元素值不同
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(1)]], (2, 2), ZZ)
    # 断言 A 与 B 不相等
    assert A != B
    # 创建一个二维列表 C
    C = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    # 断言 A 与 C 不相等
    assert A != C


# 定义一个测试函数，用于测试 DomainMatrix 类的 unify_eq 方法
def test_DomainMatrix_unify_eq():
    # 创建一个 ZZ 域上的 2x2 DomainMatrix 对象 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建一个 QQ 域上的 2x2
    # 断言，验证 items 列表是否与期望值相同，期望值是包含 ZZ 类的实例对象的列表
    assert items == [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]
    
    # 断言，验证 K 是否等于 ZZ 类型
    assert K == ZZ
    
    # 调用 DomainMatrix 类的 get_domain 方法，传入列表 [1, 2, 3, Rational(1, 2)]，获取返回值 K 和 items
    K, items = DomainMatrix.get_domain([1, 2, 3, Rational(1, 2)])
    
    # 断言，验证 items 列表是否与期望值相同，期望值是包含 QQ 类的实例对象的列表
    assert items == [QQ(1), QQ(2), QQ(3), QQ(1, 2)]
    
    # 断言，验证 K 是否等于 QQ 类型
    assert K == QQ
def test_DomainMatrix_convert_to():
    # 创建一个整数类型的域矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 将 A 转换为有理数类型的域矩阵 Aq
    Aq = A.convert_to(QQ)
    # 断言 Aq 是否等于转换后的有理数类型域矩阵
    assert Aq == DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)


def test_DomainMatrix_choose_domain():
    # 创建一个整数类型的列表 A
    A = [[1, 2], [3, 0]]
    # 断言使用 QQ 域的 DM 对象是否等于使用整数类型 ZZ 的 DM 对象
    assert DM(A, QQ).choose_domain() == DM(A, ZZ)
    # 断言使用 QQ 域的 DM 对象是否等于自身，因为 field=True 时不改变域
    assert DM(A, QQ).choose_domain(field=True) == DM(A, QQ)
    # 断言使用 ZZ 域的 DM 对象是否能成功转换为 QQ 域
    assert DM(A, ZZ).choose_domain(field=True) == DM(A, QQ)

    # 创建一个包含符号 x 的列表 B
    x = symbols('x')
    B = [[1, x], [x**2, x**3]]
    # 断言使用 QQ[x] 域的 DM 对象是否能成功转换为 QQ(x) 域
    assert DM(B, QQ[x]).choose_domain(field=True) == DM(B, ZZ.frac_field(x))


def test_DomainMatrix_to_flat_nz():
    # 创建一个整数类型的域矩阵 Adm
    Adm = DM([[1, 2], [3, 0]], ZZ)
    # 将 Adm 转换为 DDM 和 SDM 类型
    Addm = Adm.rep.to_ddm()
    Asdm = Adm.rep.to_sdm()
    # 遍历转换后的域矩阵，测试转换和反转换是否成功
    for A in [Adm, Addm, Asdm]:
        elems, data = A.to_flat_nz()
        assert A.from_flat_nz(elems, data, A.domain) == A
        elemsq = [QQ(e) for e in elems]
        assert A.from_flat_nz(elemsq, data, QQ) == A.convert_to(QQ)
        elems2 = [2*e for e in elems]
        assert A.from_flat_nz(elems2, data, A.domain) == 2*A


def test_DomainMatrix_to_sympy():
    # 创建一个整数类型的域矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言将 A 转换为 SymPy 对象后是否与转换为 EXRAW 后的结果相等
    assert A.to_sympy() == A.convert_to(EXRAW)


def test_DomainMatrix_to_field():
    # 创建一个整数类型的域矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 将 A 转换为有理数类型的域矩阵 Aq
    Aq = A.to_field()
    # 断言 Aq 是否等于转换后的有理数类型域矩阵
    assert Aq == DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)


def test_DomainMatrix_to_sparse():
    # 创建一个整数类型的域矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 将 A 转换为稀疏表示 A_sparse
    A_sparse = A.to_sparse()
    # 断言 A_sparse 的稀疏表示是否正确
    assert A_sparse.rep == {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}


def test_DomainMatrix_to_dense():
    # 创建一个稀疏表示的域矩阵 A
    A = DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)
    # 将稀疏表示的 A 转换为密集表示 A_dense
    A_dense = A.to_dense()
    ddm = DDM([[1, 2], [3, 4]], (2, 2), ZZ)
    # 根据不同的环境选择验证方式
    if GROUND_TYPES != 'flint':
        assert A_dense.rep == ddm
    else:
        assert A_dense.rep == ddm.to_dfm()


def test_DomainMatrix_unify():
    # 创建整数类型的域矩阵 Az 和有理数类型的域矩阵 Aq
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    # 测试统一操作的各种情况
    assert Az.unify(Az) == (Az, Az)
    assert Az.unify(Aq) == (Aq, Aq)
    assert Aq.unify(Az) == (Aq, Aq)
    assert Aq.unify(Aq) == (Aq, Aq)

    # 创建整数类型的稀疏域矩阵 As 和整数类型的密集域矩阵 Ad
    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    # 测试稀疏和密集域矩阵的统一操作
    Bs, Bd = As.unify(Ad, fmt='dense')
    assert Bs.rep == DDM([[0, 1], [2, 0]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Bd.rep == DDM([[1, 2],[3, 4]], (2, 2), ZZ).to_dfm_or_ddm()

    Bs, Bd = As.unify(Ad, fmt='sparse')
    assert Bs.rep == SDM({0: {1: 1}, 1: {0: 2}}, (2, 2), ZZ)
    assert Bd.rep == SDM({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

    # 断言不支持的格式会引发 ValueError 异常
    raises(ValueError, lambda: As.unify(Ad, fmt='invalid'))


def test_DomainMatrix_to_Matrix():
    # 创建一个整数类型的域矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 将 A 转换为 SymPy 的 Matrix 对象
    A_Matrix = Matrix([[1, 2], [3, 4]])
    # 断言 A 转换为 Matrix 对象后是否与预期的 A_Matrix 相等
    assert A.to_Matrix() == A_Matrix
    # 确保将稀疏表示的 A 转换为稠密矩阵，并检查是否与 A_Matrix 相等
    assert A.to_sparse().to_Matrix() == A_Matrix
    
    # 确保将 A 转换为有理数域 QQ 上的矩阵，并检查是否与 A_Matrix 相等
    assert A.convert_to(QQ).to_Matrix() == A_Matrix
    
    # 确保将 A 转换为 QQ 有理数域的代数扩展（例如加入了 sqrt(2) 的域）上的矩阵，并检查是否与 A_Matrix 相等
    assert A.convert_to(QQ.algebraic_field(sqrt(2))).to_Matrix() == A_Matrix
def test_DomainMatrix_to_list():
    # 创建一个 DomainMatrix 对象 A，使用整数矩阵和指定的形状 (2, 2)，并指定整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言将 DomainMatrix 对象 A 转换为普通的 Python 列表
    assert A.to_list() == [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]


def test_DomainMatrix_to_list_flat():
    # 创建一个 DomainMatrix 对象 A，使用整数矩阵和指定的形状 (2, 2)，并指定整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言将 DomainMatrix 对象 A 扁平化为一个一维的 Python 列表
    assert A.to_list_flat() == [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]


def test_DomainMatrix_flat():
    # 创建一个 DomainMatrix 对象 A，使用整数矩阵和指定的形状 (2, 2)，并指定整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言将 DomainMatrix 对象 A 扁平化为一个一维的 Python 列表
    assert A.flat() == [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]


def test_DomainMatrix_from_list_flat():
    # 创建一个包含 ZZ 对象的列表 nums
    nums = [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]
    # 创建一个 DomainMatrix 对象 A，使用整数矩阵和指定的形状 (2, 2)，并指定整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    # 断言从给定的一维列表 nums 创建的 DomainMatrix 对象与预期的 A 相等
    assert DomainMatrix.from_list_flat(nums, (2, 2), ZZ) == A
    # 断言将 nums 列表转换为 DDM 对象后与 A 的 DDM 表示相等
    assert DDM.from_list_flat(nums, (2, 2), ZZ) == A.rep.to_ddm()
    # 断言将 nums 列表转换为 SDM 对象后与 A 的 SDM 表示相等
    assert SDM.from_list_flat(nums, (2, 2), ZZ) == A.rep.to_sdm()

    # 断言从 DomainMatrix 对象 A 自身的扁平列表再次创建 DomainMatrix 对象与 A 相等
    assert A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)

    # 断言使用不匹配的形状 (2, 3) 创建 DomainMatrix 对象会引发 DMBadInputError 异常
    raises(DMBadInputError, DomainMatrix.from_list_flat, nums, (2, 3), ZZ)
    raises(DMBadInputError, DDM.from_list_flat, nums, (2, 3), ZZ)
    raises(DMBadInputError, SDM.from_list_flat, nums, (2, 3), ZZ)


def test_DomainMatrix_to_dod():
    # 创建一个 DomainMatrix 对象 A，使用整数矩阵和指定的形状 (2, 2)，并指定整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言将 DomainMatrix 对象 A 转换为字典的字典 (Dictionary of Dictionaries)
    assert A.to_dod() == {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}
    
    # 创建一个 DomainMatrix 对象 A，使用整数矩阵和指定的形状 (2, 2)，并指定整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(4)]], (2, 2), ZZ)
    # 断言将 DomainMatrix 对象 A 转换为字典的字典 (Dictionary of Dictionaries)
    assert A.to_dod() == {0: {0: ZZ(1)}, 1: {1: ZZ(4)}}


def test_DomainMatrix_from_dod():
    # 创建一个字典的字典 items
    items = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}
    # 创建一个具有相同内容的 DomainMatrix 对象 A，使用整数环 ZZ
    A = DM([[1, 2], [3, 4]], ZZ)
    # 断言从字典的字典 items 创建的 DomainMatrix 对象与 A 的稀疏表示相等
    assert DomainMatrix.from_dod(items, (2, 2), ZZ) == A.to_sparse()
    # 断言从 items 创建的 DomainMatrix 对象与 A 相等
    assert A.from_dod_like(items) == A
    # 断言从 items 创建的 DomainMatrix 对象转换为 QQ 环后与 A 相等
    assert A.from_dod_like(items, QQ) == A.convert_to(QQ)


def test_DomainMatrix_to_dok():
    # 创建一个 DomainMatrix 对象 A，使用整数矩阵和指定的形状 (2, 2)，并指定整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言将 DomainMatrix 对象 A 转换为字典的键值对 (Dictionary of Keys) 表示
    assert A.to_dok() == {(0, 0): ZZ(1), (0, 1): ZZ(2), (1, 0): ZZ(3), (1, 1): ZZ(4)}
    
    # 创建一个 DomainMatrix 对象 A，使用整数矩阵和指定的形状 (2, 2)，并指定整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(4)]], (2, 2), ZZ)
    dok = {(0, 0): ZZ(1), (1, 1): ZZ(4)}
    # 断言将 DomainMatrix 对象 A 转换为字典的键值对 (Dictionary of Keys) 表示
    assert A.to_dok() == dok
    # 断言将 DomainMatrix 对象 A 转换为稠密表示后再转换为 DOK 格式与预期的 dok 相等
    assert A.to_dense().to_dok() == dok
    # 断言将 DomainMatrix 对象 A 转换为稀疏表示后再转换为 DOK 格式与预期的 dok 相等
    assert A.to_sparse().to_dok() == dok
    # 断言将 DomainMatrix 对象 A 的 DDM 表示转换为 DOK 格式与预期的 dok 相等
    assert A.rep.to_ddm().to_dok() == dok
    # 断言将 DomainMatrix 对象 A 的 SDM 表示转换为 DOK 格式与预期的 dok 相等
    assert A.rep.to_sdm().to_dok() == dok


def test_DomainMatrix_from_dok():
    # 创建一个字典的键值对 items
    items = {(0, 0): ZZ(1), (1, 1): ZZ(2)}
    # 创建一个具有相同内容的 DomainMatrix 对象 A，使用整数环 ZZ
    A = DM([[1, 0], [0, 2]], ZZ)
    # 断言从字典的键值对 items 创建的 DomainMatrix 对象与 A 的稀疏表示相等
    assert DomainMatrix.from_dok(items, (2, 2), ZZ) == A.to_sparse()
    # 断言从 items 创建的 DomainMatrix 对象与 A 的 DDM 表示相等
    assert DDM.from_dok(items, (2, 2), ZZ) == A.rep.to_ddm()
    # 断言从 items 创建的 DomainMatrix 对象与 A 的 SDM 表示相等
    assert
    # 断言：验证矩阵 A 不是零矩阵
    assert A.is_zero_matrix is False
    
    # 断言：验证矩阵 B 是零矩阵
    assert B.is_zero_matrix is True
def test_DomainMatrix_is_upper():
    # 创建两个 DomainMatrix 对象 A 和 B，分别初始化为上三角和非上三角矩阵
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(0), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言 A 是否为上三角矩阵
    assert A.is_upper is True
    # 断言 B 是否为上三角矩阵
    assert B.is_upper is False


def test_DomainMatrix_is_lower():
    # 创建两个 DomainMatrix 对象 A 和 B，分别初始化为下三角和非下三角矩阵
    A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言 A 是否为下三角矩阵
    assert A.is_lower is True
    # 断言 B 是否为下三角矩阵
    assert B.is_lower is False


def test_DomainMatrix_is_diagonal():
    # 创建两个 DomainMatrix 对象 A 和 B，分别初始化为对角线和非对角线矩阵
    A = DM([[1, 0], [0, 4]], ZZ)
    B = DM([[1, 2], [3, 4]], ZZ)
    # 断言 A 是否为对角线矩阵
    assert A.is_diagonal is A.to_sparse().is_diagonal is True
    # 断言 B 是否为对角线矩阵
    assert B.is_diagonal is B.to_sparse().is_diagonal is False


def test_DomainMatrix_is_square():
    # 创建两个 DomainMatrix 对象 A 和 B，分别初始化为方阵和非方阵
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)], [ZZ(5), ZZ(6)]], (3, 2), ZZ)
    # 断言 A 是否为方阵
    assert A.is_square is True
    # 断言 B 是否为方阵
    assert B.is_square is False


def test_DomainMatrix_diagonal():
    # 创建 DomainMatrix 对象 A，调用 diagonal 方法并断言结果与稀疏矩阵的对角元素一致
    A = DM([[1, 2], [3, 4]], ZZ)
    assert A.diagonal() == A.to_sparse().diagonal() == [ZZ(1), ZZ(4)]
    # 创建多个 DomainMatrix 对象 A，分别调用 diagonal 方法并断言结果与稀疏矩阵的对角元素一致
    A = DM([[1, 2], [3, 4], [5, 6]], ZZ)
    assert A.diagonal() == A.to_sparse().diagonal() == [ZZ(1), ZZ(4)]
    A = DM([[1, 2, 3], [4, 5, 6]], ZZ)
    assert A.diagonal() == A.to_sparse().diagonal() == [ZZ(1), ZZ(5)]


def test_DomainMatrix_rank():
    # 创建 DomainMatrix 对象 A，调用 rank 方法并断言结果为预期值
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(6), QQ(8)]], (3, 2), QQ)
    assert A.rank() == 2


def test_DomainMatrix_add():
    # 创建两个 DomainMatrix 对象 A 和 B，并进行加法操作，断言结果是否符合预期
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    assert A + A == A.add(A) == B

    # 创建 DomainMatrix 对象 A，尝试与非 DomainMatrix 类型进行加法操作，断言是否抛出 TypeError 异常
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    L = [[2, 3], [3, 4]]
    raises(TypeError, lambda: A + L)
    raises(TypeError, lambda: L + A)

    # 创建多个 DomainMatrix 对象 A1 和 A2，尝试形状不匹配的加法操作，断言是否抛出 DMShapeError 异常
    A1 = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A1 + A2)
    raises(DMShapeError, lambda: A2 + A1)
    raises(DMShapeError, lambda: A1.add(A2))
    raises(DMShapeError, lambda: A2.add(A1))

    # 创建 DomainMatrix 对象 Az 和 Aq，尝试使用不同的域进行加法操作，断言是否抛出 DMDomainError 异常
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Asum = DomainMatrix([[QQ(2), QQ(4)], [QQ(6), QQ(8)]], (2, 2), QQ)
    assert Az + Aq == Asum
    assert Aq + Az == Asum
    raises(DMDomainError, lambda: Az.add(Aq))
    raises(DMDomainError, lambda: Aq.add(Az))

    # 创建多个 DomainMatrix 对象 As 和 Ad，进行加法操作，验证结果并检查异常情况
    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    Asd = As + Ad
    Ads = Ad + As
    assert Asd == DomainMatrix([[1, 3], [5, 4]], (2, 2), ZZ)
    assert Asd.rep == DDM([[1, 3], [5, 4]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Ads == DomainMatrix([[1, 3], [5, 4]], (2, 2), ZZ)
    assert Ads.rep == DDM([[1, 3], [5, 4]], (2, 2), ZZ).to_dfm_or_ddm()
    raises(DMFormatError, lambda: As.add(Ad))


def test_DomainMatrix_sub():
    # 待实现的测试函数，暂无代码
    # 创建一个整数域上的 2x2 矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建一个整数域上的 2x2 零矩阵 B
    B = DomainMatrix([[ZZ(0), ZZ(0)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    # 断言 A - A 和 A.sub(A) 均等于 B
    assert A - A == A.sub(A) == B

    # 创建一个整数域上的 2x2 矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建一个普通的 Python 列表 L
    L = [[2, 3], [3, 4]]
    # 使用 lambda 函数检查 A - L 是否引发 TypeError 异常
    raises(TypeError, lambda: A - L)
    # 使用 lambda 函数检查 L - A 是否引发 TypeError 异常
    raises(TypeError, lambda: L - A)

    # 创建两个不同形状的整数域矩阵 A1 和 A2
    A1 = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 使用 lambda 函数检查 A1 - A2 是否引发 DMShapeError 异常
    raises(DMShapeError, lambda: A1 - A2)
    # 使用 lambda 函数检查 A2 - A1 是否引发 DMShapeError 异常
    raises(DMShapeError, lambda: A2 - A1)
    # 使用 lambda 函数检查 A1.sub(A2) 是否引发 DMShapeError 异常
    raises(DMShapeError, lambda: A1.sub(A2))
    # 使用 lambda 函数检查 A2.sub(A1) 是否引发 DMShapeError 异常
    raises(DMShapeError, lambda: A2.sub(A1))

    # 创建两个不同类型的矩阵 Az 和 Aq，分别是整数域和有理数域上的 2x2 矩阵
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    # 创建一个有理数域上的 2x2 零矩阵 Adiff
    Adiff = DomainMatrix([[QQ(0), QQ(0)], [QQ(0), QQ(0)]], (2, 2), QQ)
    # 断言 Az - Aq 和 Aq - Az 均等于 Adiff
    assert Az - Aq == Adiff
    assert Aq - Az == Adiff
    # 使用 lambda 函数检查 Az.sub(Aq) 是否引发 DMDomainError 异常
    raises(DMDomainError, lambda: Az.sub(Aq))
    # 使用 lambda 函数检查 Aq.sub(Az) 是否引发 DMDomainError 异常
    raises(DMDomainError, lambda: Aq.sub(Az))

    # 创建两个不同的整数域矩阵 As 和 Ad
    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 计算 As - Ad 和 Ad - As 的结果
    Asd = As - Ad
    Ads = Ad - As
    # 断言 Asd 等于特定的整数域矩阵
    assert Asd == DomainMatrix([[-1, -1], [-1, -4]], (2, 2), ZZ)
    # 断言 Asd 的表示等于特定的矩阵对象的表示
    assert Asd.rep == DDM([[-1, -1], [-1, -4]], (2, 2), ZZ).to_dfm_or_ddm()
    # 断言 Asd 等于其相反数
    assert Asd == -Ads
    # 断言 Asd 的表示等于其相反数的表示
    assert Asd.rep == -Ads.rep
# 定义测试函数，用于测试 DomainMatrix 类的 neg 方法
def test_DomainMatrix_neg():
    # 创建一个 DomainMatrix 对象 A，表示一个整数矩阵
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建一个 DomainMatrix 对象 Aneg，表示 A 的每个元素取负值后的矩阵
    Aneg = DomainMatrix([[ZZ(-1), ZZ(-2)], [ZZ(-3), ZZ(-4)]], (2, 2), ZZ)
    # 断言 -A 等于 A.neg() 等于 Aneg
    assert -A == A.neg() == Aneg


# 定义测试函数，用于测试 DomainMatrix 类的乘法方法
def test_DomainMatrix_mul():
    # 创建一个 DomainMatrix 对象 A，表示一个整数矩阵
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建一个 DomainMatrix 对象 A2，表示 A 乘以自身的结果矩阵
    A2 = DomainMatrix([[ZZ(7), ZZ(10)], [ZZ(15), ZZ(22)]], (2, 2), ZZ)
    # 断言 A*A 等于 A.matmul(A) 等于 A2
    assert A * A == A.matmul(A) == A2

    # 使用普通列表 L 尝试与 A 相乘，预期会引发 TypeError 异常
    L = [[1, 2], [3, 4]]
    raises(TypeError, lambda: A * L)
    raises(TypeError, lambda: L * A)

    # 创建一个整数矩阵 Az 和一个有理数矩阵 Aq
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    # 创建一个有理数矩阵 Aprod，表示 Az 与 Aq 相乘的结果矩阵
    Aprod = DomainMatrix([[QQ(7), QQ(10)], [QQ(15), QQ(22)]], (2, 2), QQ)
    # 断言 Az * Aq 等于 Aprod，以及 Aq * Az 等于 Aprod
    assert Az * Aq == Aprod
    assert Aq * Az == Aprod
    # 使用 Az 和 Aq 尝试执行矩阵乘法，预期会引发 DMDomainError 异常
    raises(DMDomainError, lambda: Az.matmul(Aq))
    raises(DMDomainError, lambda: Aq.matmul(Az))

    # 创建一个整数矩阵 A 和一个整数标量 x
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    AA = DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    x = ZZ(2)
    # 断言 A * x 等于 x * A 等于 A.mul(x) 等于 AA
    assert A * x == x * A == A.mul(x) == AA

    # 创建一个整数矩阵 A 和一个整数标量 x=0
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    AA = DomainMatrix.zeros((2, 2), ZZ)
    x = ZZ(0)
    # 断言 A * x 等于 x * A 等于 A.mul(x).to_sparse() 等于 AA
    assert A * x == x * A == A.mul(x).to_sparse() == AA

    # 创建两个稀疏矩阵 As 和 Ad
    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 计算 As 与 Ad 的乘积，预期结果为指定的整数矩阵
    Asd = As * Ad
    Ads = Ad * As
    assert Asd == DomainMatrix([[3, 4], [2, 4]], (2, 2), ZZ)
    assert Asd.rep == DDM([[3, 4], [2, 4]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Ads == DomainMatrix([[4, 1], [8, 3]], (2, 2), ZZ)
    assert Ads.rep == DDM([[4, 1], [8, 3]], (2, 2), ZZ).to_dfm_or_ddm()


# 定义测试函数，用于测试 DomainMatrix 类的元素级乘法方法
def test_DomainMatrix_mul_elementwise():
    # 创建两个整数矩阵 A 和 B
    A = DomainMatrix([[ZZ(2), ZZ(2)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(4), ZZ(0)], [ZZ(3), ZZ(0)]], (2, 2), ZZ)
    C = DomainMatrix([[ZZ(8), ZZ(0)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    # 断言 A 与 B 的元素级乘法结果等于 C
    assert A.mul_elementwise(B) == C
    assert B.mul_elementwise(A) == C


# 定义测试函数，用于测试 DomainMatrix 类的幂运算方法
def test_DomainMatrix_pow():
    # 创建单位矩阵 eye 和整数矩阵 A
    eye = DomainMatrix.eye(2, ZZ)
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(7), ZZ(10)], [ZZ(15), ZZ(22)]], (2, 2), ZZ)
    A3 = DomainMatrix([[ZZ(37), ZZ(54)], [ZZ(81), ZZ(118)]], (2, 2), ZZ)
    # 断言 A**0 等于 A.pow(0) 等于 eye
    assert A**0 == A.pow(0) == eye
    # 断言 A**1 等于 A.pow(1) 等于 A
    assert A**1 == A.pow(1) == A
    # 断言 A**2 等于 A.pow(2) 等于 A2
    assert A**2 == A.pow(2) == A2
    # 断言 A**3 等于 A.pow(3) 等于 A3

    # 尝试对 A 进行非整数幂运算，预期会引发 TypeError 异常
    raises(TypeError, lambda: A ** Rational(1, 2))
    # 尝试对 A 进行负幂运算，预期会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: A ** -1)
    raises(NotImplementedError, lambda: A.pow(-1))

    # 创建一个非方阵 A
    A = DomainMatrix.zeros((2, 1), ZZ)
    # 尝试对非方阵 A 进行幂运算，预期会引发 DMNonSquareMatrixError 异常
    raises(DMNonSquareMatrixError, lambda: A ** 1)


# 定义测试函数，用于测试 DomainMatrix 类的清除分母方法
def test_DomainMatrix_clear_denoms():
    # 创建一个有理数矩阵 A
    A = DM([[(1,2),(1,3)],[(1,4),(1,5)]], QQ)

    # 创建一个整数标量 den_Z 和一个整数矩阵 Anum_Z
    den_Z = DomainScalar(ZZ(60), ZZ)
    Anum_Z = DM([[30, 20], [15, 12]], ZZ)
    # 将 Anum_Z 转换为有理数矩阵 Anum_Q
    Anum_Q = Anum_Z.convert_to(QQ)

    # 断言执行 A 的清除分母操作后返回的结果
    assert A.clear_denoms() == (den_Z, Anum_Q)
    # 调用 A 对象的 clear_denoms 方法，传入 convert=True 参数，并断言其返回值与 (den_Z, Anum_Z) 相等
    assert A.clear_denoms(convert=True) == (den_Z, Anum_Z)
    # 断言 A 对象乘以 den_Z 后的结果与 Anum_Q 相等
    assert A * den_Z == Anum_Q
    # 断言 A 对象与 Anum_Q 除以 den_Z 后的结果相等
    assert A == Anum_Q / den_Z
# 定义测试函数 test_DomainMatrix_clear_denoms_rowwise
def test_DomainMatrix_clear_denoms_rowwise():
    # 创建域矩阵 A，元素为有理数 QQ
    A = DM([[(1,2),(1,3)],[(1,4),(1,5)]], QQ)

    # 创建整数域矩阵 den_Z，并将其转换为稀疏表示
    den_Z = DM([[6, 0], [0, 20]], ZZ).to_sparse()
    # 创建整数域矩阵 Anum_Z
    Anum_Z = DM([[3, 2], [5, 4]], ZZ)
    # 创建有理数域矩阵 Anum_Q
    Anum_Q = DM([[3, 2], [5, 4]], QQ)

    # 断言 A.clear_denoms_rowwise() 返回的结果与 den_Z, Anum_Q 相等
    assert A.clear_denoms_rowwise() == (den_Z, Anum_Q)
    # 断言 A.clear_denoms_rowwise(convert=True) 返回的结果与 den_Z, Anum_Z 相等
    assert A.clear_denoms_rowwise(convert=True) == (den_Z, Anum_Z)
    # 断言 den_Z 乘以 A 等于 Anum_Q
    assert den_Z * A == Anum_Q
    # 断言 A 等于 den_Z 的逆乘以 Anum_Q
    assert A == den_Z.to_field().inv() * Anum_Q

    # 重新赋值 A
    A = DM([[(1,2),(1,3),0,0],[0,0,0,0], [(1,4),(1,5),(1,6),(1,7)]], QQ)
    # 创建稀疏表示的整数域矩阵 den_Z
    den_Z = DM([[6, 0, 0], [0, 1, 0], [0, 0, 420]], ZZ).to_sparse()
    # 创建整数域矩阵 Anum_Z
    Anum_Z = DM([[3, 2, 0, 0], [0, 0, 0, 0], [105, 84, 70, 60]], ZZ)
    # 将 Anum_Z 转换为有理数域矩阵 Anum_Q
    Anum_Q = Anum_Z.convert_to(QQ)

    # 断言 A.clear_denoms_rowwise() 返回的结果与 den_Z, Anum_Q 相等
    assert A.clear_denoms_rowwise() == (den_Z, Anum_Q)
    # 断言 A.clear_denoms_rowwise(convert=True) 返回的结果与 den_Z, Anum_Z 相等
    assert A.clear_denoms_rowwise(convert=True) == (den_Z, Anum_Z)
    # 断言 den_Z 乘以 A 等于 Anum_Q
    assert den_Z * A == Anum_Q
    # 断言 A 等于 den_Z 的逆乘以 Anum_Q
    assert A == den_Z.to_field().inv() * Anum_Q


# 定义测试函数 test_DomainMatrix_cancel_denom
def test_DomainMatrix_cancel_denom():
    # 创建整数域矩阵 A
    A = DM([[2, 4], [6, 8]], ZZ)
    # 断言 A.cancel_denom(ZZ(1)) 返回的结果为 (A, ZZ(1))
    assert A.cancel_denom(ZZ(1)) == (DM([[2, 4], [6, 8]], ZZ), ZZ(1))
    # 断言 A.cancel_denom(ZZ(3)) 返回的结果为 (A, ZZ(3))
    assert A.cancel_denom(ZZ(3)) == (DM([[2, 4], [6, 8]], ZZ), ZZ(3))
    # 断言 A.cancel_denom(ZZ(4)) 返回的结果为 (已约分的 A, ZZ(2))
    assert A.cancel_denom(ZZ(4)) == (DM([[1, 2], [3, 4]], ZZ), ZZ(2))

    # 重新赋值 A
    A = DM([[1, 2], [3, 4]], ZZ)
    # 断言 A.cancel_denom(ZZ(2)) 返回的结果为 (A, ZZ(2))
    assert A.cancel_denom(ZZ(2)) == (A, ZZ(2))
    # 断言 A.cancel_denom(ZZ(-2)) 返回的结果为 (-A, ZZ(2))
    assert A.cancel_denom(ZZ(-2)) == (-A, ZZ(2))

    # 测试在高斯有理数域上分母的规范化
    A = DM([[1, 2], [3, 4]], QQ_I)
    # 断言 A.cancel_denom(QQ_I(0,2)) 返回的结果为 (QQ_I(0,-1)*A, QQ_I(2))
    assert A.cancel_denom(QQ_I(0,2)) == (QQ_I(0,-1)*A, QQ_I(2))
    # 断言在分母为零时引发 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: A.cancel_denom(ZZ(0)))


# 定义测试函数 test_DomainMatrix_cancel_denom_elementwise
def test_DomainMatrix_cancel_denom_elementwise():
    # 创建整数域矩阵 A
    A = DM([[2, 4], [6, 8]], ZZ)
    # 对 A 中的每个元素分别约分，返回分子 numers 和分母 denoms
    numers, denoms = A.cancel_denom_elementwise(ZZ(1))
    # 断言 numers 与 A 相等
    assert numers == DM([[2, 4], [6, 8]], ZZ)
    # 断言 denoms 是一个全为1的整数域矩阵
    assert denoms == DM([[1, 1], [1, 1]], ZZ)
    # 对 A 中的每个元素分别约分，返回分子 numers 和分母 denoms
    numers, denoms = A.cancel_denom_elementwise(ZZ(4))
    # 断言 numers 是已约分的 A
    assert numers == DM([[1, 1], [3, 2]], ZZ)
    # 断言 denoms 是一个全为2的整数域矩阵
    assert denoms == DM([[2, 1], [2, 1]], ZZ)

    # 断言在分母为零时引发 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: A.cancel_denom_elementwise(ZZ(0)))


# 定义测试函数 test_DomainMatrix_content_primitive
def test_DomainMatrix_content_primitive():
    # 创建整数域矩阵 A
    A = DM([[2, 4], [6, 8]], ZZ)
    # 计算 A 的内容 content
    A_content = ZZ(2)
    # 断言 A.content() 返回的结果与 A_content 相等
    assert A.content() == A_content
    # 计算 A 的原胞 primitive
    A_primitive = DM([[1, 2], [3, 4]], ZZ)
    # 断言 A.primitive() 返回的结果为 (A_content, A_primitive)
    assert A.primitive() == (A_content, A_primitive)


# 定义测试函数 test_DomainMatrix_scc
def test_DomainMatrix_scc():
    # 创建整数域矩阵 Ad
    Ad = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)],
                       [ZZ(0), ZZ(1), ZZ(0)],
                       [ZZ(2), ZZ(0), ZZ(4)]], (3, 3), ZZ)
    # 将 Ad 转换为稀疏表示 As
    As = Ad.to_sparse()
    # 获取 Ad 和 As 的底层表示 Addm 和 Asdm
    Addm = Ad.rep
    Asdm = As.rep
    # 对于每个 A 在 [Ad, As, Addm, Asdm] 中，断言 Ad.scc() 的结果为 [[1], [0, 2]]

    for A in [Ad, As, Addm, Asdm]:
        assert Ad.scc() == [[1], [0, 2]]

    # 创建整数域矩阵 A，不是方阵
    A = DM([[ZZ(1), ZZ(2), ZZ(3)]], ZZ)
    # 断言在非方阵上调用 Ad.scc() 会引发 DMNonSquareMatrixError 异常
    raises(DMNonSquareMatrixError, lambda: A.scc())


# 定义测试函数 test_DomainMatrix_rref
    # 断言检查行阶梯形矩阵和主元素索引是否正确
    assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    assert pivots == (0, 1)

    # 创建一个有理数域上的矩阵 A，然后计算其行阶梯形式和主元素索引
    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Ar, pivots = A.rref()
    # 断言检查行阶梯形矩阵和主元素索引是否正确
    assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    assert pivots == (0, 1)

    # 创建另一个有理数域上的矩阵 A，然后计算其行阶梯形式和主元素索引
    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(0), QQ(4)]], (2, 2), QQ)
    Ar, pivots = A.rref()
    # 断言检查行阶梯形矩阵和主元素索引是否正确
    assert Ar == DomainMatrix([[QQ(0), QQ(1)], [QQ(0), QQ(0)]], (2, 2), QQ)
    assert pivots == (1,)

    # 创建整数环上的矩阵 Az，然后计算其行阶梯形式和主元素索引
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Ar, pivots = Az.rref()
    # 断言检查行阶梯形矩阵和主元素索引是否正确
    assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    assert pivots == (0, 1)

    # 定义多种行阶梯形计算方法
    methods = ('auto', 'GJ', 'FF', 'CD', 'GJ_dense', 'FF_dense', 'CD_dense')
    # 对整数环上的矩阵 Az 使用不同的方法计算其行阶梯形式和主元素索引
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    for method in methods:
        Ar, pivots = Az.rref(method=method)
        # 断言检查行阶梯形矩阵和主元素索引是否正确
        assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
        assert pivots == (0, 1)

    # 检查错误情况：尝试使用不存在的方法引发 ValueError 异常
    raises(ValueError, lambda: Az.rref(method='foo'))
    raises(ValueError, lambda: Az.rref_den(method='foo'))
def test_DomainMatrix_columnspace():
    # 创建 DomainMatrix 对象 A，表示一个有理数域上的矩阵
    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ)
    # 创建期望的列空间 DomainMatrix 对象 Acol
    Acol = DomainMatrix([[QQ(1), QQ(1)], [QQ(2), QQ(3)]], (2, 2), QQ)
    # 断言 A 的列空间应该等于 Acol
    assert A.columnspace() == Acol

    # 创建 DomainMatrix 对象 Az，表示整数域上的矩阵
    Az = DomainMatrix([[ZZ(1), ZZ(-1), ZZ(1)], [ZZ(2), ZZ(-2), ZZ(3)]], (2, 3), ZZ)
    # 调用列空间函数应该引发 DMNotAField 异常
    raises(DMNotAField, lambda: Az.columnspace())

    # 创建稀疏格式的 DomainMatrix 对象 A
    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ, fmt='sparse')
    # 创建期望的列空间 DomainMatrix 对象 Acol
    Acol = DomainMatrix({0: {0: QQ(1), 1: QQ(1)}, 1: {0: QQ(2), 1: QQ(3)}}, (2, 2), QQ)
    # 断言 A 的列空间应该等于 Acol
    assert A.columnspace() == Acol


def test_DomainMatrix_rowspace():
    # 创建 DomainMatrix 对象 A，表示一个有理数域上的矩阵
    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ)
    # 断言 A 的行空间应该等于 A 自身
    assert A.rowspace() == A

    # 创建 DomainMatrix 对象 Az，表示整数域上的矩阵
    Az = DomainMatrix([[ZZ(1), ZZ(-1), ZZ(1)], [ZZ(2), ZZ(-2), ZZ(3)]], (2, 3), ZZ)
    # 调用行空间函数应该引发 DMNotAField 异常
    raises(DMNotAField, lambda: Az.rowspace())

    # 创建稀疏格式的 DomainMatrix 对象 A
    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ, fmt='sparse')
    # 断言 A 的行空间应该等于 A 自身
    assert A.rowspace() == A


def test_DomainMatrix_nullspace():
    # 创建 DomainMatrix 对象 A，表示一个有理数域上的矩阵
    A = DomainMatrix([[QQ(1), QQ(1)], [QQ(1), QQ(1)]], (2, 2), QQ)
    # 创建期望的零空间 DomainMatrix 对象 Anull
    Anull = DomainMatrix([[QQ(-1), QQ(1)]], (1, 2), QQ)
    # 断言 A 的零空间应该等于 Anull
    assert A.nullspace() == Anull

    # 创建 DomainMatrix 对象 A，表示一个整数域上的矩阵
    A = DomainMatrix([[ZZ(1), ZZ(1)], [ZZ(1), ZZ(1)]], (2, 2), ZZ)
    # 创建期望的零空间 DomainMatrix 对象 Anull
    Anull = DomainMatrix([[ZZ(-1), ZZ(1)]], (1, 2), ZZ)
    # 断言 A 的零空间应该等于 Anull
    assert A.nullspace() == Anull

    # 调用具有 divide_last=True 参数的 nullspace 函数应该引发 DMNotAField 异常
    raises(DMNotAField, lambda: A.nullspace(divide_last=True))

    # 创建 DomainMatrix 对象 A，表示一个整数域上的矩阵
    A = DomainMatrix([[ZZ(2), ZZ(2)], [ZZ(2), ZZ(2)]], (2, 2), ZZ)
    # 创建期望的零空间 DomainMatrix 对象 Anull
    Anull = DomainMatrix([[ZZ(-2), ZZ(2)]], (1, 2), ZZ)

    # 调用 rref_den 函数得到 Arref, den, pivots
    Arref, den, pivots = A.rref_den()
    # 断言 den 应该等于 2
    assert den == ZZ(2)
    # 使用 rref 函数的结果 Arref 计算其零空间应该等于 Anull
    assert Arref.nullspace_from_rref() == Anull
    assert Arref.nullspace_from_rref(pivots) == Anull
    # 将 Arref 转换为稀疏格式后计算其零空间应该等于 Anull 的稀疏格式
    assert Arref.to_sparse().nullspace_from_rref() == Anull.to_sparse()
    assert Arref.to_sparse().nullspace_from_rref(pivots) == Anull.to_sparse()


def test_DomainMatrix_solve():
    # 创建 DomainMatrix 对象 A，表示一个有理数域上的矩阵
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    # 创建 DomainMatrix 对象 b，表示一个有理数域上的列向量
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    # 创建期望的特解 DomainMatrix 对象 particular 和零空间 DomainMatrix 对象 nullspace
    particular = DomainMatrix([[1, 0]], (1, 2), QQ)
    nullspace = DomainMatrix([[-2, 1]], (1, 2), QQ)
    # 断言 A._solve(b) 返回的结果应该等于 (particular, nullspace)
    assert A._solve(b) == (particular, nullspace)

    # 创建 DomainMatrix 对象 b3，表示一个有理数域上的列向量
    b3 = DomainMatrix([[QQ(1)], [QQ(1)], [QQ(1)]], (3, 1), QQ)
    # 调用 _solve 函数应该引发 DMShapeError 异常
    raises(DMShapeError, lambda: A._solve(b3))

    # 创建 DomainMatrix 对象 bz，表示一个整数域上的列向量
    bz = DomainMatrix([[ZZ(1)], [ZZ(1)]], (2, 1), ZZ)
    # 调用 _solve 函数应该引发 DMNotAField 异常
    raises(DMNotAField, lambda: A._solve(bz))


def test_DomainMatrix_inv():
    # 创建空矩阵 DomainMatrix 对象 A
    A = DomainMatrix([], (0, 0), QQ)
    # 断言 A 的逆矩阵应该等于 A 自身
    assert A.inv() == A

    # 创建 DomainMatrix 对象 A，表示一个有理数域上的矩阵
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    # 创建期望的逆矩阵 DomainMatrix 对象 Ainv
    Ainv = DomainMatrix([[QQ(-2), QQ(1)], [QQ(3, 2), QQ(-1, 2)]], (2, 2), QQ)
    # 断言 A 的逆矩阵应该等于 Ainv
    assert A.inv() == Ainv

    # 创建 DomainMatrix 对象 Az，表示一个整数域上的矩阵
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 调用 inv 函数应该引发 DMNotAField 异常
    raises(DMNotAField, lambda: Az.inv())

    # 创建 DomainMatrix 对象 Ans，表示一个有理数域上的矩阵
    Ans = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    # 调用 inv 函数应该引发 DMNonSquareMatrixError 异常
    raises(DMNonSquareMatrix
    # 调用 raises 函数，验证是否会引发 DMNonInvertibleMatrixError 异常，
    # 使用 lambda 匿名函数调用 Aninv.inv() 方法来执行测试
    raises(DMNonInvertibleMatrixError, lambda: Aninv.inv())
def test_DomainMatrix_det():
    # 创建一个空的 DomainMatrix 对象，使用整数环 ZZ
    A = DomainMatrix([], (0, 0), ZZ)
    # 断言计算该矩阵的行列式为 1
    assert A.det() == 1

    # 创建一个包含单个元素的 DomainMatrix 对象，使用整数环 ZZ
    A = DomainMatrix([[1]], (1, 1), ZZ)
    # 断言计算该矩阵的行列式为 1
    assert A.det() == 1

    # 创建一个2x2的 DomainMatrix 对象，使用整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言计算该矩阵的行列式为 -2
    assert A.det() == ZZ(-2)

    # 创建一个3x3的 DomainMatrix 对象，使用整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(3), ZZ(5)]], (3, 3), ZZ)
    # 断言计算该矩阵的行列式为 -1
    assert A.det() == ZZ(-1)

    # 创建一个3x3的 DomainMatrix 对象，使用整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(2), ZZ(5)]], (3, 3), ZZ)
    # 断言计算该矩阵的行列式为 0
    assert A.det() == ZZ(0)

    # 创建一个包含单个元素的 DomainMatrix 对象，使用有理数环 QQ
    Ans = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    # 断言计算该矩阵的行列式会引发 DMNonSquareMatrixError 异常
    raises(DMNonSquareMatrixError, lambda: Ans.det())

    # 创建一个2x2的 DomainMatrix 对象，使用有理数环 QQ
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    # 断言计算该矩阵的行列式为 -2
    assert A.det() == QQ(-2)


def test_DomainMatrix_eval_poly():
    # 创建一个2x2的 DomainMatrix 对象，使用整数环 ZZ
    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 定义多项式系数
    p = [ZZ(1), ZZ(2), ZZ(3)]
    # 创建一个2x2的 DomainMatrix 对象，使用整数环 ZZ
    result = DomainMatrix([[ZZ(12), ZZ(14)], [ZZ(21), ZZ(33)]], (2, 2), ZZ)
    # 断言计算 dM 对多项式 p 的求值结果等于 result
    assert dM.eval_poly(p) == result == p[0]*dM**2 + p[1]*dM + p[2]*dM**0
    # 断言计算 dM 对空多项式的求值结果为全零矩阵
    assert dM.eval_poly([]) == dM.zeros(dM.shape, dM.domain)
    # 断言计算 dM 对包含单个元素的多项式的求值结果
    assert dM.eval_poly([ZZ(2)]) == 2*dM.eye(2, dM.domain)

    # 创建一个1x2的 DomainMatrix 对象，使用整数环 ZZ
    dM2 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 断言计算 dM2 对多项式 [ZZ(1)] 的求值会引发 DMNonSquareMatrixError 异常
    raises(DMNonSquareMatrixError, lambda: dM2.eval_poly([ZZ(1)]))


def test_DomainMatrix_eval_poly_mul():
    # 创建一个2x2的 DomainMatrix 对象，使用整数环 ZZ
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建一个2x1的 DomainMatrix 对象，使用整数环 ZZ
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    # 定义多项式系数
    p = [ZZ(1), ZZ(2), ZZ(3)]
    # 创建一个2x1的 DomainMatrix 对象，使用整数环 ZZ
    result = DomainMatrix([[ZZ(40)], [ZZ(87)]], (2, 1), ZZ)
    # 断言计算 A * p[0] * A^2 * b + p[1] * A * b + p[2] * b 的结果等于 result
    assert A.eval_poly_mul(p, b) == result == p[0]*A**2*b + p[1]*A*b + p[2]*b

    # 创建一个2x2的 DomainMatrix 对象，使用整数环 ZZ
    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建一个2x1的 DomainMatrix 对象，使用整数环 ZZ
    dM1 = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    # 断言计算 dM1 对多项式 [ZZ(1)] 的乘法求值会引发 DMNonSquareMatrixError 异常
    raises(DMNonSquareMatrixError, lambda: dM1.eval_poly_mul([ZZ(1)], b))
    # 创建一个1x2的 DomainMatrix 对象，使用整数环 ZZ
    b1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 断言计算 dM 对多项式 [ZZ(1)] 的乘法求值会引发 DMShapeError 异常
    raises(DMShapeError, lambda: dM.eval_poly_mul([ZZ(1)], b1))
    # 创建一个2x1的 DomainMatrix 对象，使用有理数环 QQ
    bq = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    # 断言计算 dM 对多项式 [ZZ(1)] 的乘法求值会引发 DMDomainError 异常
    raises(DMDomainError, lambda: dM.eval_poly_mul([ZZ(1)], bq))


def _check_solve_den(A, b, xnum, xden):
    # 用于 solve_den、solve_den_charpoly、solve_den_rref 的例子应使用此函数，
    # 以确保测试所有方法和类型。
    case1 = (A, xnum, b)
    case2 = (A.to_sparse(), xnum.to_sparse(), b.to_sparse())
    # 遍历列表中的每个元组(case1, case2)，分别将元组解包为Ai, xnum_i, b_i三个变量
    for Ai, xnum_i, b_i in [case1, case2]:
        # 解决方程 solve_den 的关键不变式检查
        assert Ai * xnum_i == xden * b_i

        # solve_den_rref 的结果可能仅相差一个负号
        answers = [(xnum_i, xden), (-xnum_i, -xden)]
        # 验证 solve_den 方法返回的结果在预期的答案列表中
        assert Ai.solve_den(b) in answers
        # 验证 solve_den 方法在 'rref' 模式下返回的结果在预期的答案列表中
        assert Ai.solve_den(b, method='rref') in answers
        # 验证 solve_den_rref 方法返回的结果在预期的答案列表中
        assert Ai.solve_den_rref(b) in answers

        # 如果矩阵 Ai 是方阵，则 charpoly 方法可以使用，并保证以实际行列式作为分母返回
        m, n = Ai.shape
        if m == n:
            # 验证 solve_den 方法在 'charpoly' 模式下返回的结果符合预期的 (xnum_i, xden)
            assert Ai.solve_den(b_i, method='charpoly') == (xnum_i, xden)
            # 验证 solve_den_charpoly 方法返回的结果符合预期的 (xnum_i, xden)
            assert Ai.solve_den_charpoly(b_i) == (xnum_i, xden)
        else:
            # 如果矩阵 Ai 不是方阵，预期会引发 DMNonSquareMatrixError 异常
            raises(DMNonSquareMatrixError, lambda: Ai.solve_den_charpoly(b))
            raises(DMNonSquareMatrixError, lambda: Ai.solve_den(b, method='charpoly'))
def test_DomainMatrix_solve_den():
    # 创建 DomainMatrix A，表示一个 2x2 的整数矩阵
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建 DomainMatrix b，表示一个 2x1 的整数矩阵
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    # 创建 DomainMatrix result，表示一个 2x1 的整数矩阵
    result = DomainMatrix([[ZZ(0)], [ZZ(-1)]], (2, 1), ZZ)
    # 创建整数 den，值为 -2
    den = ZZ(-2)
    # 调用 _check_solve_den 函数，检查解决方案是否符合预期
    _check_solve_den(A, b, result, den)

    # 创建另一个 DomainMatrix A，表示一个 3x3 的整数矩阵
    A = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(1), ZZ(2), ZZ(4)],
        [ZZ(1), ZZ(3), ZZ(5)]], (3, 3), ZZ)
    # 创建 DomainMatrix b，表示一个 3x1 的整数矩阵
    b = DomainMatrix([[ZZ(1)], [ZZ(2)], [ZZ(3)]], (3, 1), ZZ)
    # 创建 DomainMatrix result，表示一个 3x1 的整数矩阵
    result = DomainMatrix([[ZZ(2)], [ZZ(0)], [ZZ(-1)]], (3, 1), ZZ)
    # 创建整数 den，值为 -1
    den = ZZ(-1)
    # 调用 _check_solve_den 函数，检查解决方案是否符合预期
    _check_solve_den(A, b, result, den)

    # 创建 DomainMatrix A，表示一个 2x1 的整数矩阵
    A = DomainMatrix([[ZZ(2)], [ZZ(2)]], (2, 1), ZZ)
    # 创建 DomainMatrix b，表示一个 2x1 的整数矩阵
    b = DomainMatrix([[ZZ(3)], [ZZ(3)]], (2, 1), ZZ)
    # 创建 DomainMatrix result，表示一个 1x1 的整数矩阵
    result = DomainMatrix([[ZZ(3)]], (1, 1), ZZ)
    # 创建整数 den，值为 2
    den = ZZ(2)
    # 调用 _check_solve_den 函数，检查解决方案是否符合预期
    _check_solve_den(A, b, result, den)


def test_DomainMatrix_solve_den_charpoly():
    # 创建 DomainMatrix A，表示一个 2x2 的整数矩阵
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建 DomainMatrix b，表示一个 2x1 的整数矩阵
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    # 创建 DomainMatrix A1，表示一个 1x2 的整数矩阵
    A1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 断言 A1.solve_den_charpoly(b) 抛出 DMNonSquareMatrixError 异常
    raises(DMNonSquareMatrixError, lambda: A1.solve_den_charpoly(b))

    # 创建 DomainMatrix b1，表示一个 1x2 的整数矩阵
    b1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 断言 A.solve_den_charpoly(b1) 抛出 DMShapeError 异常
    raises(DMShapeError, lambda: A.solve_den_charpoly(b1))

    # 创建 DomainMatrix bq，表示一个 2x1 的有理数矩阵
    bq = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    # 断言 A.solve_den_charpoly(bq) 抛出 DMDomainError 异常
    raises(DMDomainError, lambda: A.solve_den_charpoly(bq))


def test_DomainMatrix_solve_den_charpoly_check():
    # 创建 DomainMatrix A，表示一个 2x2 的整数矩阵
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(2), ZZ(4)]], (2, 2), ZZ)
    # 创建 DomainMatrix b，表示一个 2x1 的整数矩阵
    b = DomainMatrix([[ZZ(1)], [ZZ(3)]], (2, 1), ZZ)
    # 断言 A.solve_den_charpoly(b) 抛出 DMNonInvertibleMatrixError 异常
    raises(DMNonInvertibleMatrixError, lambda: A.solve_den_charpoly(b))

    # 创建 DomainMatrix adjAb，表示一个 2x1 的整数矩阵
    adjAb = DomainMatrix([[ZZ(-2)], [ZZ(1)]], (2, 1), ZZ)
    # 断言 A.adjugate() * b 等于 adjAb
    assert A.adjugate() * b == adjAb
    # 断言 A.solve_den_charpoly(b, check=False) 返回 (adjAb, ZZ(0))
    assert A.solve_den_charpoly(b, check=False) == (adjAb, ZZ(0))


def test_DomainMatrix_solve_den_errors():
    # 创建 DomainMatrix A，表示一个 1x2 的整数矩阵
    A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 创建 DomainMatrix b，表示一个 2x1 的整数矩阵
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    # 断言 A.solve_den(b) 抛出 DMShapeError 异常
    raises(DMShapeError, lambda: A.solve_den(b))
    # 断言 A.solve_den_rref(b) 抛出 DMShapeError 异常
    raises(DMShapeError, lambda: A.solve_den_rref(b))

    # 创建 DomainMatrix A，表示一个 1x2 的整数矩阵
    A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 创建 DomainMatrix b，表示一个 1x2 的整数矩阵
    b = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 断言 A.solve_den(b) 抛出 DMShapeError 异常
    raises(DMShapeError, lambda: A.solve_den(b))
    # 断言 A.solve_den_rref(b) 抛出 DMShapeError 异常
    raises(DMShapeError, lambda: A.solve_den_rref(b))

    # 创建 DomainMatrix A，表示一个 2x2 的整数矩阵
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建 DomainMatrix b1，表示一个 1x2 的整数矩阵
    b1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 断言 A.solve_den(b1) 抛出 DMShapeError 异常
    raises(DMShapeError, lambda: A.solve_den(b1))

    # 创建 DomainMatrix A，表示一个 1x1 的整数矩阵
    A = DomainMatrix([[ZZ(2)]], (1, 1), ZZ)
    # 创建 DomainMatrix b，表示一个 1x1 的整数矩阵
    b = DomainMatrix([[ZZ(2)]], (1, 1), ZZ)
    # 断言 A.solve_den(b1, method='invalid') 抛出 DMBadInputError 异常
    raises(DMBadInputError, lambda: A.solve_den(b1, method='invalid'))

    # 创建 DomainMatrix A，表示一个 2x1 的整数矩阵
    A = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    # 创建 DomainMatrix b，表示一个 2x1 的整数矩阵
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    # 断言 A.solve_den_charpoly(b) 抛出 DMNonSquareMatrixError 异常
    raises(DMNonSquareMatrixError, lambda: A.solve_den_charpoly(b))


def test_DomainMatrix_solve_den
    # 调用函数 raises，并传入两个参数：
    # 1. DMNonInvertibleMatrixError，表示期望的异常类型是非可逆矩阵错误。
    # 2. lambda 函数：lambda: A.solve_den_rref(b)，这是一个匿名函数，用于调用 A 对象的 solve_den_rref 方法，并传入参数 b。
    # lambda 函数的作用是在 raises 函数内部定义一个可以调用 solve_den_rref 方法的匿名函数。
    raises(DMNonInvertibleMatrixError, lambda: A.solve_den_rref(b))
def test_DomainMatrix_adj_poly_det():
    # 创建一个 3x3 的整数域矩阵 A
    A = DM([[ZZ(1), ZZ(2), ZZ(3)],
            [ZZ(4), ZZ(5), ZZ(6)],
            [ZZ(7), ZZ(8), ZZ(9)]], ZZ)
    # 调用 adj_poly_det 方法，返回多项式 p 和行列式 detA
    p, detA = A.adj_poly_det()
    # 断言 p 的值为 [1, -15, -18]
    assert p == [ZZ(1), ZZ(-15), ZZ(-18)]
    # 断言 A 的伴随矩阵等于多项式的线性组合
    assert A.adjugate() == p[0]*A**2 + p[1]*A**1 + p[2]*A**0 == A.eval_poly(p)
    # 断言 A 的行列式等于 detA
    assert A.det() == detA

    # 创建一个 2x3 的整数域矩阵 A（非方阵）
    A = DM([[ZZ(1), ZZ(2), ZZ(3)],
            [ZZ(7), ZZ(8), ZZ(9)]], ZZ)
    # 断言调用 adj_poly_det 抛出非方阵异常
    raises(DMNonSquareMatrixError, lambda: A.adj_poly_det())


def test_DomainMatrix_inv_den():
    # 创建一个 2x2 的整数域矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 设定 den 的值为 -2
    den = ZZ(-2)
    # 创建一个期望的结果矩阵 result 和 den
    result = DomainMatrix([[ZZ(4), ZZ(-2)], [ZZ(-3), ZZ(1)]], (2, 2), ZZ)
    # 断言 A 的 inv_den 方法返回 (result, den)
    assert A.inv_den() == (result, den)


def test_DomainMatrix_adjugate():
    # 创建一个 2x2 的整数域矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建一个期望的结果矩阵 result
    result = DomainMatrix([[ZZ(4), ZZ(-2)], [ZZ(-3), ZZ(1)]], (2, 2), ZZ)
    # 断言 A 的 adjugate 方法返回 result
    assert A.adjugate() == result


def test_DomainMatrix_adj_det():
    # 创建一个 2x2 的整数域矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建一个期望的结果矩阵 adjA 和 detA
    adjA = DomainMatrix([[ZZ(4), ZZ(-2)], [ZZ(-3), ZZ(1)]], (2, 2), ZZ)
    # 断言 A 的 adj_det 方法返回 (adjA, -2)
    assert A.adj_det() == (adjA, ZZ(-2))


def test_DomainMatrix_lu():
    # 创建一个空矩阵 A
    A = DomainMatrix([], (0, 0), QQ)
    # 断言 A 的 lu 方法返回 (A, A, [])
    assert A.lu() == (A, A, [])

    # 创建一个 2x2 的有理数域矩阵 A
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    # 创建期望的下三角矩阵 L、上三角矩阵 U 和交换列表 swaps
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(3), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(1), QQ(2)], [QQ(0), QQ(-2)]], (2, 2), QQ)
    swaps = []
    # 断言 A 的 lu 方法返回 (L, U, swaps)
    assert A.lu() == (L, U, swaps)

    # 更多的 lu 测试用例，略去详细注释

    A = DomainMatrix([[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]], (2, 3), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(4), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(1), QQ(2), QQ(3)], [QQ(0), QQ(-3), QQ(-6)]], (2, 3), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    L = DomainMatrix([
        [QQ(1), QQ(0), QQ(0)],
        [QQ(3), QQ(1), QQ(0)],
        [QQ(5), QQ(2), QQ(1)]], (3, 3), QQ)
    U = DomainMatrix([[QQ(1), QQ(2)], [QQ(0), QQ(-2)], [QQ(0), QQ(0)]], (3, 2), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 2]]
    L = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
    # 定义一个 4x4 的上三角矩阵 U
    U = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    # 定义一个将矩阵每个元素应用 dom 函数后的结果矩阵的函数
    to_dom = lambda rows, dom: [[dom(e) for e in row] for row in rows]
    # 创建一个 DomainMatrix 对象 A，使用 QQ 作为其域，将原始矩阵 A 转换成对应的域矩阵
    A = DomainMatrix(to_dom(A, QQ), (4, 4), QQ)
    # 创建一个 DomainMatrix 对象 L，使用 QQ 作为其域，将原始矩阵 L 转换成对应的域矩阵
    L = DomainMatrix(to_dom(L, QQ), (4, 4), QQ)
    # 创建一个 DomainMatrix 对象 U，使用 QQ 作为其域，将原始矩阵 U 转换成对应的域矩阵
    U = DomainMatrix(to_dom(U, QQ), (4, 4), QQ)
    # 断言 A 的 LU 分解结果为 (L, U, [])，即无置换向量
    assert A.lu() == (L, U, [])

    # 创建一个 DomainMatrix 对象 A，使用 ZZ 作为其域，包含整数元素的矩阵
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言 A 对象不是一个域对象，即它不具备域属性，会引发 DMNotAField 异常
    raises(DMNotAField, lambda: A.lu())
def test_DomainMatrix_lu_solve():
    # Base case
    A = b = x = DomainMatrix([], (0, 0), QQ)
    assert A.lu_solve(b) == x

    # Basic example
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DomainMatrix([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    assert A.lu_solve(b) == x

    # Example with swaps
    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DomainMatrix([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    assert A.lu_solve(b) == x

    # Non-invertible
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    raises(DMNonInvertibleMatrixError, lambda: A.lu_solve(b))

    # Overdetermined, consistent
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    x = DomainMatrix([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    assert A.lu_solve(b) == x

    # Overdetermined, inconsistent
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)], [QQ(4)]], (3, 1), QQ)
    raises(DMNonInvertibleMatrixError, lambda: A.lu_solve(b))

    # Underdetermined
    A = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    b = DomainMatrix([[QQ(1)]], (1, 1), QQ)
    raises(NotImplementedError, lambda: A.lu_solve(b))

    # Non-field
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMNotAField, lambda: A.lu_solve(b))

    # Shape mismatch
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMShapeError, lambda: A.lu_solve(b))



def test_DomainMatrix_charpoly():
    A = DomainMatrix([], (0, 0), ZZ)
    p = [ZZ(1)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[1]], (1, 1), ZZ)
    p = [ZZ(1), ZZ(-1)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    p = [ZZ(1), ZZ(-5), ZZ(-2)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    p = [ZZ(1), ZZ(-15), ZZ(-18), ZZ(0)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[ZZ(0), ZZ(1), ZZ(0)],
                      [ZZ(1), ZZ(0), ZZ(1)],
                      [ZZ(0), ZZ(1), ZZ(0)]], (3, 3), ZZ)
    p = [ZZ(1), ZZ(0), ZZ(-2), ZZ(0)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p
    # 创建一个整数类型的域矩阵 A
    A = DM([[17, 0, 30,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [69, 0,  0,  0,  0, 86, 0,  0, 0, 0],
            [23, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0, 13,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0, 32, 0, 0],
            [ 0, 0,  0,  0, 37, 67, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0]], ZZ)
    
    # 创建一个整数类型的域矩阵 A 的特征多项式 p
    p = ZZ.map([1, -17, -2070, 0, -771420, 0, 0, 0, 0, 0, 0])
    
    # 断言 A 的特征多项式与 p 相等
    assert A.charpoly() == p
    
    # 将矩阵 A 转换为稀疏矩阵后，断言其特征多项式与 p 相等
    assert A.to_sparse().charpoly() == p
    
    # 创建有理数类型的域矩阵 Ans
    Ans = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    
    # 断言非方阵 Ans 计算特征多项式时会引发 DMNonSquareMatrixError 异常
    raises(DMNonSquareMatrixError, lambda: Ans.charpoly())
def test_DomainMatrix_charpoly_factor_list():
    # 创建一个空的域矩阵 A，其维度为 (0, 0)，使用整数环 ZZ 作为元素域
    A = DomainMatrix([], (0, 0), ZZ)
    # 断言 A 的特征多项式的因式分解结果为空列表
    assert A.charpoly_factor_list() == []

    # 创建一个 1x1 的域矩阵 A，元素为整数环 ZZ 中的 1
    A = DM([[1]], ZZ)
    # 断言 A 的特征多项式的因式分解结果为 [([1, -1], 1)]
    assert A.charpoly_factor_list() == [
        ([ZZ(1), ZZ(-1)], 1)
    ]

    # 创建一个 2x2 的域矩阵 A，元素为整数环 ZZ 中的 [[1, 2], [3, 4]]
    A = DM([[1, 2], [3, 4]], ZZ)
    # 断言 A 的特征多项式的因式分解结果为 [([1, -5, -2], 1)]
    assert A.charpoly_factor_list() == [
        ([ZZ(1), ZZ(-5), ZZ(-2)], 1)
    ]

    # 创建一个 3x3 的域矩阵 A，元素为整数环 ZZ 中的 [[1, 2, 0], [3, 4, 0], [0, 0, 1]]
    A = DM([[1, 2, 0], [3, 4, 0], [0, 0, 1]], ZZ)
    # 断言 A 的特征多项式的因式分解结果为 [([1, -1], 1), ([1, -5, -2], 1)]
    assert A.charpoly_factor_list() == [
        ([ZZ(1), ZZ(-1)], 1),
        ([ZZ(1), ZZ(-5), ZZ(-2)], 1)
    ]


def test_DomainMatrix_eye():
    # 创建一个大小为 3x3 的单位矩阵 A，元素域为有理数域 QQ
    A = DomainMatrix.eye(3, QQ)
    # 断言 A 的表示等于大小为 (3, 3) 的有理数域单位矩阵的表示
    assert A.rep == SDM.eye((3, 3), QQ)
    # 断言 A 的形状为 (3, 3)
    assert A.shape == (3, 3)
    # 断言 A 的元素域为有理数域 QQ
    assert A.domain == QQ


def test_DomainMatrix_zeros():
    # 创建一个大小为 (1, 2) 的零矩阵 A，元素域为有理数域 QQ
    A = DomainMatrix.zeros((1, 2), QQ)
    # 断言 A 的表示等于大小为 (1, 2) 的有理数域零矩阵的表示
    assert A.rep == SDM.zeros((1, 2), QQ)
    # 断言 A 的形状为 (1, 2)
    assert A.shape == (1, 2)
    # 断言 A 的元素域为有理数域 QQ
    assert A.domain == QQ


def test_DomainMatrix_ones():
    # 创建一个大小为 (2, 3) 的全 1 矩阵 A，元素域为有理数域 QQ
    A = DomainMatrix.ones((2, 3), QQ)
    # 如果 GROUND_TYPES 不等于 'flint'，则断言 A 的表示等于大小为 (2, 3) 的有理数域全 1 矩阵的表示
    if GROUND_TYPES != 'flint':
        assert A.rep == DDM.ones((2, 3), QQ)
    else:
        # 如果 GROUND_TYPES 等于 'flint'，则断言 A 的表示等于大小为 (2, 3) 的有理数域全 1 矩阵转换成 DFM 后的表示
        assert A.rep == SDM.ones((2, 3), QQ).to_dfm()
    # 断言 A 的形状为 (2, 3)
    assert A.shape == (2, 3)
    # 断言 A 的元素域为有理数域 QQ
    assert A.domain == QQ


def test_DomainMatrix_diag():
    # 创建一个以 [2, 3] 为对角线元素的大小为 (2, 2) 的整数环 ZZ 域矩阵 A
    A = DomainMatrix({0:{0:ZZ(2)}, 1:{1:ZZ(3)}}, (2, 2), ZZ)
    # 断言使用 DomainMatrix.diag 函数生成的以 [2, 3] 为对角线元素的大小为 (2, 2) 的整数环 ZZ 域矩阵与 A 相等
    assert DomainMatrix.diag([ZZ(2), ZZ(3)], ZZ) == A

    # 创建一个以 [2, 3] 为对角线元素的大小为 (3, 4) 的整数环 ZZ 域矩阵 A
    A = DomainMatrix({0:{0:ZZ(2)}, 1:{1:ZZ(3)}}, (3, 4), ZZ)
    # 断言使用 DomainMatrix.diag 函数生成的以 [2, 3] 为对角线元素的大小为 (3, 4) 的整数环 ZZ 域矩阵与 A 相等
    assert DomainMatrix.diag([ZZ(2), ZZ(3)], ZZ, (3, 4)) == A


def test_DomainMatrix_hstack():
    # 创建大小为 (2, 2) 的整数环 ZZ 域矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建大小为 (2, 2) 的整数环 ZZ 域矩阵 B
    B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
    # 创建大小为 (2, 2) 的整数环 ZZ 域矩阵 C
    C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)

    # 创建 A 和 B 水平堆叠后的大小为 (2, 4) 的整数环 ZZ 域矩阵 AB
    AB = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(5), ZZ(6)],
        [ZZ(3), ZZ(4), ZZ(7), ZZ(8)]], (2, 4), ZZ)
    # 创建 A、B 和 C 水平堆叠后的大小为 (2, 6) 的整数环 ZZ 域矩阵 ABC
    ABC = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(5), ZZ(6), ZZ(9), ZZ(10)],
        [ZZ(3), ZZ(4), ZZ(7), ZZ(8), ZZ(11), ZZ(12)]], (2, 6), ZZ)
    # 断言 A 和 B 的水平堆叠结果等于 AB
    assert A.hstack(B) == AB
    # 断言 A、B 和 C 的水平堆叠结果等于 ABC
    assert A.hstack(B, C) == ABC


def test_DomainMatrix_vstack():
    # 创建大小为 (2, 2) 的整数环 ZZ 域矩阵 A
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建大小为 (2, 2) 的整数环 ZZ 域矩阵 B
    B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
    # 创建大小为 (2, 2) 的整数环 ZZ 域矩阵 C
    C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)

    # 创建 A 和 B 垂直堆叠后的大小为 (4, 2) 的整数环 ZZ 域矩阵 AB
    AB = DomainMatrix([
        [ZZ(1), ZZ(2)],
        [ZZ(3
    # 断言：A 乘以 2 等于给定的域矩阵
    assert A * 2 == DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    
    # 断言：2 乘以 A 等于给定的域矩阵
    assert 2 * A == DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    
    # 断言：A 乘以 DomainScalar(ZZ(0), ZZ) 等于一个空的域矩阵
    assert A * DomainScalar(ZZ(0), ZZ) == DomainMatrix({}, (2, 2), ZZ)
    
    # 断言：A 乘以 DomainScalar(ZZ(1), ZZ) 等于 A 自身
    assert A * DomainScalar(ZZ(1), ZZ) == A
    
    # 断言：使用 lambda 表达式捕获 TypeError 异常，验证 A 乘以浮点数会引发异常
    raises(TypeError, lambda: A * 1.5)
def test_DomainMatrix_truediv():
    # 创建一个 DomainMatrix 对象 A，从给定的 Matrix 对象转换而来
    A = DomainMatrix.from_Matrix(Matrix([[1, 2], [3, 4]]))
    # 创建一个 DomainScalar 对象 lamda，表示有理数 QQ(3/2)
    lamda = DomainScalar(QQ(3)/QQ(2), QQ)
    # 断言 A 除以 lamda 得到的结果是否符合预期
    assert A / lamda == DomainMatrix({0: {0: QQ(2, 3), 1: QQ(4, 3)}, 1: {0: QQ(2), 1: QQ(8, 3)}}, (2, 2), QQ)
    # 创建一个 DomainScalar 对象 b，表示整数 ZZ(1)
    b = DomainScalar(ZZ(1), ZZ)
    # 断言 A 除以 b 得到的结果是否符合预期
    assert A / b == DomainMatrix({0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}, (2, 2), QQ)

    # 断言 A 除以整数 1 得到的结果是否符合预期
    assert A / 1 == DomainMatrix({0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}, (2, 2), QQ)
    # 断言 A 除以整数 2 得到的结果是否符合预期
    assert A / 2 == DomainMatrix({0: {0: QQ(1, 2), 1: QQ(1)}, 1: {0: QQ(3, 2), 1: QQ(2)}}, (2, 2), QQ)

    # 断言除以 0 会抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: A / 0)
    # 断言除以浮点数 1.5 会抛出 TypeError 异常
    raises(TypeError, lambda: A / 1.5)
    # 断言 A 除以 DomainScalar(ZZ(0), ZZ) 会抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: A / DomainScalar(ZZ(0), ZZ))

    # 创建一个 DomainMatrix 对象 A，从给定的整数矩阵转换而来
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 断言 A 转换为域上的矩阵后再除以整数 2 的结果是否符合预期
    assert A.to_field() / 2 == DomainMatrix([[QQ(1, 2), QQ(1)], [QQ(3, 2), QQ(2)]], (2, 2), QQ)
    # 断言 A 除以整数 2 的结果是否符合预期
    assert A / 2 == DomainMatrix([[QQ(1, 2), QQ(1)], [QQ(3, 2), QQ(2)]], (2, 2), QQ)
    # 断言 A 转换为域上的矩阵后再除以有理数 QQ(2/3) 的结果是否符合预期
    assert A.to_field() / QQ(2,3) == DomainMatrix([[QQ(3, 2), QQ(3)], [QQ(9, 2), QQ(6)]], (2, 2), QQ)


def test_DomainMatrix_getitem():
    # 创建一个 DomainMatrix 对象 dM，从给定的整数矩阵转换而来
    dM = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)

    # 断言对 dM 进行切片操作得到的结果是否符合预期
    assert dM[1:,:-2] == DomainMatrix([[ZZ(4)], [ZZ(7)]], (2, 1), ZZ)
    assert dM[2,:-2] == DomainMatrix([[ZZ(7)]], (1, 1), ZZ)
    assert dM[:-2,:-2] == DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    assert dM[:-1,0:2] == DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(4), ZZ(5)]], (2, 2), ZZ)
    assert dM[:, -1] == DomainMatrix([[ZZ(3)], [ZZ(6)], [ZZ(9)]], (3, 1), ZZ)
    assert dM[-1, :] == DomainMatrix([[ZZ(7), ZZ(8), ZZ(9)]], (1, 3), ZZ)
    assert dM[::-1, :] == DomainMatrix([
                            [ZZ(7), ZZ(8), ZZ(9)],
                            [ZZ(4), ZZ(5), ZZ(6)],
                            [ZZ(1), ZZ(2), ZZ(3)]], (3, 3), ZZ)

    # 断言超出索引范围会抛出 IndexError 异常
    raises(IndexError, lambda: dM[4, :-2])
    raises(IndexError, lambda: dM[:-2, 4])

    # 断言对 dM 的单个元素访问得到的结果是否符合预期
    assert dM[1, 2] == DomainScalar(ZZ(6), ZZ)
    assert dM[-2, 2] == DomainScalar(ZZ(6), ZZ)
    assert dM[1, -2] == DomainScalar(ZZ(5), ZZ)
    assert dM[-1, -3] == DomainScalar(ZZ(7), ZZ)

    # 断言超出索引范围会抛出 IndexError 异常
    raises(IndexError, lambda: dM[3, 3])
    raises(IndexError, lambda: dM[1, 4])
    raises(IndexError, lambda: dM[-1, -4])

    # 创建一个 DomainMatrix 对象 dM，从给定的字典表示转换而来
    dM = DomainMatrix({0: {0: ZZ(1)}}, (10, 10), ZZ)
    # 断言对 dM 的特定元素访问得到的结果是否符合预期
    assert dM[5, 5] == DomainScalar(ZZ(0), ZZ)
    assert dM[0, 0] == DomainScalar(ZZ(1), ZZ)

    # 创建一个 DomainMatrix 对象 dM，从给定的字典表示转换而来
    dM = DomainMatrix({1: {0: 1}}, (2,1), ZZ)
    # 断言对 dM 的特定元素切片访问得到的结果是否符合预期
    assert dM[0:, 0] == DomainMatrix({1: {0: 1}}, (2, 1), ZZ)
    # 断言超出索引范围会抛出 IndexError 异常
    raises(IndexError, lambda: dM[3, 0])

    # 创建一个 DomainMatrix 对象 dM，从给定的字典表示转换而来
    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    # 断言对 dM 的特定元素切片访问得到的结果是否符合预期
    assert dM[:2,:2] == DomainMatrix({}, (2, 2), ZZ)
    assert dM[2:,2:] == DomainMatrix({0: {0: 1}, 2: {2: 1}}, (3, 3), ZZ)
    assert dM[3:,3:] == DomainMatrix({1: {1: 1}}, (2, 2), ZZ)
    assert dM[2:, 6:] == DomainMatrix({}, (3, 0), ZZ)


def test_DomainMatrix_getitem_sympy():
    # 创建一个 DomainMatrix 对象 dM，从给定的字典表示转换而来
    dM = DomainMatrix({2: {2: ZZ(2)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    # 从 dM 中获取元素 (0, 0) 的值
    val1 = dM.getitem_sympy(0, 0)
    # 使用断言检查 val1 是否为 SymPy 的零值 S.Zero
    assert val1 is S.Zero
    # 从 dM 中获取元素 (2, 2) 的值
    val2 = dM.getitem_sympy(2, 2)
    # 使用断言检查 val2 是否等于整数 2，并且 val2 是否是 Integer 类型的实例
    assert val2 == 2 and isinstance(val2, Integer)
# 定义测试函数 test_DomainMatrix_extract
def test_DomainMatrix_extract():
    # 创建 DomainMatrix 对象 dM1，包含一个 3x3 的整数矩阵
    dM1 = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    # 创建 DomainMatrix 对象 dM2，包含一个 2x2 的整数矩阵
    dM2 = DomainMatrix([
        [ZZ(1), ZZ(3)],
        [ZZ(7), ZZ(9)]], (2, 2), ZZ)
    # 断言从 dM1 中提取特定行和列的部分等于 dM2
    assert dM1.extract([0, 2], [0, 2]) == dM2
    # 断言将 dM1 转换为稀疏矩阵后，提取特定行和列的部分等于 dM2 的稀疏形式
    assert dM1.to_sparse().extract([0, 2], [0, 2]) == dM2.to_sparse()
    # 断言从 dM1 中提取特定行和列的部分等于 dM2，支持负数索引
    assert dM1.extract([0, -1], [0, -1]) == dM2
    # 断言将 dM1 转换为稀疏矩阵后，提取特定行和列的部分等于 dM2 的稀疏形式，支持负数索引
    assert dM1.to_sparse().extract([0, -1], [0, -1]) == dM2.to_sparse()

    # 创建 DomainMatrix 对象 dM3，包含一个 3x3 的整数矩阵
    dM3 = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(2)],
        [ZZ(4), ZZ(5), ZZ(5)],
        [ZZ(4), ZZ(5), ZZ(5)]], (3, 3), ZZ)
    # 断言从 dM1 中提取特定行和列的部分等于 dM3
    assert dM1.extract([0, 1, 1], [0, 1, 1]) == dM3
    # 断言将 dM1 转换为稀疏矩阵后，提取特定行和列的部分等于 dM3 的稀疏形式
    assert dM1.to_sparse().extract([0, 1, 1], [0, 1, 1]) == dM3.to_sparse()

    # 创建一个空的测试集 empty，包含三种情况：空行、空列和空矩阵
    empty = [
        ([], [], (0, 0)),
        ([1], [], (1, 0)),
        ([], [1], (0, 1)),
    ]
    # 对每种情况进行断言：从 dM1 中提取指定的行和列应该得到对应大小的零矩阵
    for rows, cols, size in empty:
        assert dM1.extract(rows, cols) == DomainMatrix.zeros(size, ZZ).to_dense()
        assert dM1.to_sparse().extract(rows, cols) == DomainMatrix.zeros(size, ZZ)

    # 创建 DomainMatrix 对象 dM，包含一个 2x2 的整数矩阵
    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    # 创建一个包含错误索引的测试集 bad_indices
    bad_indices = [([2], [0]), ([0], [2]), ([-3], [0]), ([0], [-3])]
    # 对每个错误的索引组合进行断言：设置这些索引时应该引发 IndexError 异常
    for rows, cols in bad_indices:
        raises(IndexError, lambda: dM.extract(rows, cols))
        raises(IndexError, lambda: dM.to_sparse().extract(rows, cols))


# 定义测试函数 test_DomainMatrix_setitem
def test_DomainMatrix_setitem():
    # 创建 DomainMatrix 对象 dM，包含一个字典形式的 5x5 矩阵
    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    # 修改 dM 的一个元素
    dM[2, 2] = ZZ(2)
    # 断言修改后 dM 的内容符合预期
    assert dM == DomainMatrix({2: {2: ZZ(2)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    # 定义一个用于设置元素的函数 setitem，并测试不支持的数据类型
    def setitem(i, j, val):
        dM[i, j] = val
    raises(TypeError, lambda: setitem(2, 2, QQ(1, 2)))
    # 测试对切片设置元素时应该引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: setitem(slice(1, 2), 2, ZZ(1)))


# 定义测试函数 test_DomainMatrix_pickling
def test_DomainMatrix_pickling():
    # 导入 pickle 模块
    import pickle
    # 创建 DomainMatrix 对象 dM1，序列化并反序列化后应该得到原始对象
    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    assert pickle.loads(pickle.dumps(dM)) == dM
    # 创建 DomainMatrix 对象 dM2，序列化并反序列化后应该得到原始对象
    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert pickle.loads(pickle.dumps(dM)) == dM
```
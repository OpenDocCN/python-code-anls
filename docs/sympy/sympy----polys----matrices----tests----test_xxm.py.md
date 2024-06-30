# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_xxm.py`

```
# 导入所需的模块和类
from sympy.external.gmpy import GROUND_TYPES
from sympy import ZZ, QQ, GF, ZZ_I, symbols
from sympy.polys.matrices.exceptions import (
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError,
    DMShapeError,
)
from sympy.polys.matrices.domainmatrix import DM, DomainMatrix, DDM, SDM, DFM
from sympy.testing.pytest import raises, skip
import pytest

# 定义测试函数，测试不同类型（DDM、SDM、DFM）的构造函数
def test_XXM_constructors():
    """Test the DDM, etc constructors."""
    
    # 定义测试数据结构
    lol = [
        [ZZ(1), ZZ(2)],
        [ZZ(3), ZZ(4)],
        [ZZ(5), ZZ(6)],
    ]
    dod = {
        0: {0: ZZ(1), 1: ZZ(2)},
        1: {0: ZZ(3), 1: ZZ(4)},
        2: {0: ZZ(5), 1: ZZ(6)},
    }

    lol_0x0 = []
    lol_0x2 = []
    lol_2x0 = [[], []]
    dod_0x0 = {}
    dod_0x2 = {}
    dod_2x0 = {}

    lol_bad = [
        [ZZ(1), ZZ(2)],
        [ZZ(3), ZZ(4)],
        [ZZ(5), ZZ(6), ZZ(7)],
    ]
    dod_bad = {
        0: {0: ZZ(1), 1: ZZ(2)},
        1: {0: ZZ(3), 1: ZZ(4)},
        2: {0: ZZ(5), 1: ZZ(6), 2: ZZ(7)},
    }

    XDM_dense = [DDM]
    XDM_sparse = [SDM]

    # 如果底层类型为'flint'，支持更多的稀疏矩阵类型
    if GROUND_TYPES == 'flint':
        XDM_dense.append(DFM)

    # 开始测试不同类型的矩阵构造和属性
    for XDM in XDM_dense:

        # 使用 lol 构造矩阵 A，检查基本属性
        A = XDM(lol, (3, 2), ZZ)
        assert A.rows == 3
        assert A.cols == 2
        assert A.domain == ZZ
        assert A.shape == (3, 2)
        
        # 针对不同类型的矩阵，检查元素类型判断
        if XDM is not DFM:
            assert ZZ.of_type(A[0][0]) is True
        else:
            assert ZZ.of_type(A.rep[0, 0]) is True

        # 使用 DomainMatrix 构造，并进行比较
        Adm = DomainMatrix(lol, (3, 2), ZZ)
        if XDM is DFM:
            assert Adm.rep == A
            assert Adm.rep.to_ddm() != A
        elif GROUND_TYPES == 'flint':
            assert Adm.rep.to_ddm() == A
            assert Adm.rep != A
        else:
            assert Adm.rep == A
            assert Adm.rep.to_ddm() == A

        # 测试各种特殊情况的矩阵构造
        assert XDM(lol_0x0, (0, 0), ZZ).shape == (0, 0)
        assert XDM(lol_0x2, (0, 2), ZZ).shape == (0, 2)
        assert XDM(lol_2x0, (2, 0), ZZ).shape == (2, 0)
        raises(DMBadInputError, lambda: XDM(lol, (2, 3), ZZ))
        raises(DMBadInputError, lambda: XDM(lol_bad, (3, 2), ZZ))
        raises(DMBadInputError, lambda: XDM(dod, (3, 2), ZZ))
    # 遍历 XDM_sparse 中的每个 XDM 对象
    for XDM in XDM_sparse:
        # 创建一个 (3, 2) 的 XDM 对象 A，使用 dod 作为数据域，ZZ 作为环
        A = XDM(dod, (3, 2), ZZ)
        # 断言 A 的行数为 3
        assert A.rows == 3
        # 断言 A 的列数为 2
        assert A.cols == 2
        # 断言 A 的数据域为 ZZ
        assert A.domain == ZZ
        # 断言 A 的形状为 (3, 2)
        assert A.shape == (3, 2)
        # 断言 A[0][0] 的类型属于 ZZ
        assert ZZ.of_type(A[0][0]) is True

        # 断言使用 dod 创建的 (3, 2) 的 DomainMatrix 对象的内部表示应该等于 A
        assert DomainMatrix(dod, (3, 2), ZZ).rep == A

        # 断言使用 dod_0x0 创建的 (0, 0) 的 XDM 对象的形状为 (0, 0)
        assert XDM(dod_0x0, (0, 0), ZZ).shape == (0, 0)
        # 断言使用 dod_0x2 创建的 (0, 2) 的 XDM 对象的形状为 (0, 2)
        assert XDM(dod_0x2, (0, 2), ZZ).shape == (0, 2)
        # 断言使用 dod_2x0 创建的 (2, 0) 的 XDM 对象的形状为 (2, 0)
        assert XDM(dod_2x0, (2, 0), ZZ).shape == (2, 0)
        # 断言尝试使用 (2, 3) 的形状创建 XDM 对象会抛出 DMBadInputError 异常
        raises(DMBadInputError, lambda: XDM(dod, (2, 3), ZZ))
        # 断言尝试使用 lol 创建 XDM 对象会抛出 DMBadInputError 异常
        raises(DMBadInputError, lambda: XDM(lol, (3, 2), ZZ))
        # 断言尝试使用 dod_bad 创建 XDM 对象会抛出 DMBadInputError 异常
        raises(DMBadInputError, lambda: XDM(dod_bad, (3, 2), ZZ))

    # 断言尝试使用 lol 创建 DomainMatrix 对象会抛出 DMBadInputError 异常
    raises(DMBadInputError, lambda: DomainMatrix(lol, (2, 3), ZZ))
    # 断言尝试使用 lol_bad 创建 DomainMatrix 对象会抛出 DMBadInputError 异常
    raises(DMBadInputError, lambda: DomainMatrix(lol_bad, (3, 2), ZZ))
    # 断言尝试使用 dod_bad 创建 DomainMatrix 对象会抛出 DMBadInputError 异常
    raises(DMBadInputError, lambda: DomainMatrix(dod_bad, (3, 2), ZZ))
def test_XXM_eq():
    """Test equality for DDM, SDM, DFM and DomainMatrix."""

    lol1 = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]  # 创建一个包含整数对象 ZZ 的嵌套列表 lol1
    dod1 = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}  # 创建一个包含整数对象 ZZ 的嵌套字典 dod1

    lol2 = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(5)]]  # 创建另一个包含整数对象 ZZ 的嵌套列表 lol2
    dod2 = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(5)}}  # 创建另一个包含整数对象 ZZ 的嵌套字典 dod2

    # 创建 DDM、SDM、DomainMatrix 对象，并初始化为 A1 和 A2 的对应值
    A1_ddm = DDM(lol1, (2, 2), ZZ)
    A1_sdm = SDM(dod1, (2, 2), ZZ)
    A1_dm_d = DomainMatrix(lol1, (2, 2), ZZ)
    A1_dm_s = DomainMatrix(dod1, (2, 2), ZZ)

    A2_ddm = DDM(lol2, (2, 2), ZZ)
    A2_sdm = SDM(dod2, (2, 2), ZZ)
    A2_dm_d = DomainMatrix(lol2, (2, 2), ZZ)
    A2_dm_s = DomainMatrix(dod2, (2, 2), ZZ)

    A1_all = [A1_ddm, A1_sdm, A1_dm_d, A1_dm_s]  # 将 A1 的所有对象存入列表 A1_all
    A2_all = [A2_ddm, A2_sdm, A2_dm_d, A2_dm_s]  # 将 A2 的所有对象存入列表 A2_all

    # 如果 GROUND_TYPES 等于 'flint'，则创建 DFM 对象 A1_dfm 和 A2_dfm，并添加到对应列表
    if GROUND_TYPES == 'flint':
        A1_dfm = DFM([[1, 2], [3, 4]], (2, 2), ZZ)
        A2_dfm = DFM([[1, 2], [3, 5]], (2, 2), ZZ)
        A1_all.append(A1_dfm)
        A2_all.append(A2_dfm)

    # 检查 A1_all 中的对象是否相等或不相等
    for n, An in enumerate(A1_all):
        for m, Am in enumerate(A1_all):
            if n == m:
                assert (An == Am) is True
                assert (An != Am) is False
            else:
                assert (An == Am) is False
                assert (An != Am) is True

    # 检查 A2_all 中的对象是否相等或不相等
    for n, An in enumerate(A2_all):
        for m, Am in enumerate(A2_all):
            if n == m:
                assert (An == Am) is True
                assert (An != Am) is False
            else:
                assert (An == Am) is False
                assert (An != Am) is True

    # 检查 A1_all 和 A2_all 中的对象是否不相等
    for n, A1 in enumerate(A1_all):
        for m, A2 in enumerate(A2_all):
            assert (A1 == A2) is False
            assert (A1 != A2) is True


def test_to_XXM():
    """Test to_ddm etc. for DDM, SDM, DFM and DomainMatrix."""

    lol = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]  # 创建一个包含整数对象 ZZ 的嵌套列表 lol
    dod = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}  # 创建一个包含整数对象 ZZ 的嵌套字典 dod

    # 创建 DDM、SDM、DomainMatrix 对象，并初始化为 A 的对应值
    A_ddm = DDM(lol, (2, 2), ZZ)
    A_sdm = SDM(dod, (2, 2), ZZ)
    A_dm_d = DomainMatrix(lol, (2, 2), ZZ)
    A_dm_s = DomainMatrix(dod, (2, 2), ZZ)

    A_all = [A_ddm, A_sdm, A_dm_d, A_dm_s]  # 将 A 的所有对象存入列表 A_all

    # 如果 GROUND_TYPES 不等于 'flint'，则测试未实现的方法调用和 A.to_dfm_or_ddm() 方法
    if GROUND_TYPES != 'flint':
        raises(NotImplementedError, lambda: A.to_dfm())
        assert A.to_dfm_or_ddm() == A_ddm

    # 循环遍历 A_all 中的对象，并检查其转换方法的正确性
    for A in A_all:
        assert A.to_ddm() == A_ddm
        assert A.to_sdm() == A_sdm

        # Add e.g. DDM.to_DM()?
        # assert A.to_DM() == A_dm
    # 如果 GROUND_TYPES 的值等于 'flint'
    if GROUND_TYPES == 'flint':
        # 对于 A_all 中的每个 A
        for A in A_all:
            # 断言 A 对象调用 to_dfm() 方法后的结果等于 A_dfm
            assert A.to_dfm() == A_dfm
            # 对于以下四种环域类型 K
            for K in [ZZ, QQ, GF(5), ZZ_I]:
                # 如果 A 是 DFM 类型并且不支持域 K，预期会抛出 NotImplementedError 异常
                if isinstance(A, DFM) and not DFM._supports_domain(K):
                    raises(NotImplementedError, lambda: A.convert_to(K))
                else:
                    # 否则，将 A 转换为域 K 得到 A_K
                    A_K = A.convert_to(K)
                    # 如果 DFM 支持域 K
                    if DFM._supports_domain(K):
                        # 将 A_dfm 转换为域 K 得到 A_dfm_K
                        A_dfm_K = A_dfm.convert_to(K)
                        # 断言 A_K 调用 to_dfm() 方法后的结果等于 A_dfm_K
                        assert A_K.to_dfm() == A_dfm_K
                        # 断言 A_K 调用 to_dfm_or_ddm() 方法后的结果等于 A_dfm_K
                        assert A_K.to_dfm_or_ddm() == A_dfm_K
                    else:
                        # 如果 DFM 不支持域 K，预期会抛出 NotImplementedError 异常
                        raises(NotImplementedError, lambda: A_K.to_dfm())
                        # 断言 A_K 调用 to_dfm_or_ddm() 方法后的结果等于将 A_ddm 转换为域 K 后的结果
                        assert A_K.to_dfm_or_ddm() == A_ddm.convert_to(K)
def test_DFM_domains():
    """Test which domains are supported by DFM."""

    # 导入符号变量 x 和 y
    x, y = symbols('x, y')

    # 根据 GROUND_TYPES 的取值进行条件判断
    if GROUND_TYPES in ('python', 'gmpy'):
        # Python 和 gmpy 地面类型下的设置

        # 初始化空列表 supported 和空字典 flint_funcs
        supported = []
        flint_funcs = {}

        # 初始化 not_supported 列表，包含各种不支持的域类型
        not_supported = [ZZ, QQ, GF(5), QQ[x], QQ[x,y]]

    elif GROUND_TYPES == 'flint':
        # flint 地面类型下的设置

        # 导入 flint 库
        import flint

        # 初始化 supported 列表，包含支持的域类型 ZZ 和 QQ
        supported = [ZZ, QQ]

        # 初始化 flint_funcs 字典，映射 ZZ 和 QQ 到 flint 库中的函数
        flint_funcs = {
            ZZ: flint.fmpz_mat,
            QQ: flint.fmpq_mat,
        }

        # 初始化 not_supported 列表，包含不支持的域类型及原因说明
        not_supported = [
            # 可能会支持但尚未在 SymPy 中实现
            GF(5),
            # 其他域类型在 python-flint 中作为矩阵尚未实现
            QQ[x],
            QQ[x,y],
            QQ.frac_field(x,y),
            # 其他可能永远不会被 python-flint 支持的域类型
            ZZ_I,
        ]

    else:
        # 如果 GROUND_TYPES 的值未知，则断言错误
        assert False, "Unknown GROUND_TYPES: %s" % GROUND_TYPES

    # 遍历 supported 列表中的每个域类型，进行支持断言和 flint 函数比较
    for domain in supported:
        assert DFM._supports_domain(domain) is True
        assert DFM._get_flint_func(domain) == flint_funcs[domain]

    # 遍历 not_supported 列表中的每个域类型，进行不支持断言和异常检查
    for domain in not_supported:
        assert DFM._supports_domain(domain) is False
        raises(NotImplementedError, lambda: DFM._get_flint_func(domain))


def _DM(lol, typ, K):
    """Make a DM of type typ over K from lol."""

    # 利用 lol 和 K 创建一个 DM 对象 A
    A = DM(lol, K)

    # 根据 typ 类型进行条件分支处理
    if typ == 'DDM':
        # 如果 typ 是 'DDM'，则将 A 转换为 DDM 类型并返回
        return A.to_ddm()
    elif typ == 'SDM':
        # 如果 typ 是 'SDM'，则将 A 转换为 SDM 类型并返回
        return A.to_sdm()
    elif typ == 'DFM':
        # 如果 typ 是 'DFM'，则检查当前 GROUND_TYPES 是否为 'flint'
        if GROUND_TYPES != 'flint':
            # 如果不是 'flint'，则跳过并提示 DFM 在当前地面类型下不支持
            skip("DFM not supported in this ground type")
        # 如果是 'flint'，则将 A 转换为 DFM 类型并返回
        return A.to_dfm()
    else:
        # 如果 typ 类型未知，则断言错误
        assert False, "Unknown type %s" % typ


def _DMZ(lol, typ):
    """Make a DM of type typ over ZZ from lol."""

    # 调用 _DM 函数创建一个 ZZ 类型的 DM 对象并返回
    return _DM(lol, typ, ZZ)


def _DMQ(lol, typ):
    """Make a DM of type typ over QQ from lol."""

    # 调用 _DM 函数创建一个 QQ 类型的 DM 对象并返回
    return _DM(lol, typ, QQ)


def DM_ddm(lol, K):
    """Make a DDM over K from lol."""

    # 调用 _DM 函数创建一个 DDM 类型的 DM 对象并返回
    return _DM(lol, 'DDM', K)


def DM_sdm(lol, K):
    """Make a SDM over K from lol."""

    # 调用 _DM 函数创建一个 SDM 类型的 DM 对象并返回
    return _DM(lol, 'SDM', K)


def DM_dfm(lol, K):
    """Make a DFM over K from lol."""

    # 调用 _DM 函数创建一个 DFM 类型的 DM 对象并返回
    return _DM(lol, 'DFM', K)


def DMZ_ddm(lol):
    """Make a DDM from lol."""

    # 调用 _DMZ 函数创建一个 ZZ 类型的 DDM 对象并返回
    return _DMZ(lol, 'DDM')


def DMZ_sdm(lol):
    """Make a SDM from lol."""

    # 调用 _DMZ 函数创建一个 ZZ 类型的 SDM 对象并返回
    return _DMZ(lol, 'SDM')


def DMZ_dfm(lol):
    """Make a DFM from lol."""

    # 调用 _DMZ 函数创建一个 ZZ 类型的 DFM 对象并返回
    return _DMZ(lol, 'DFM')


def DMQ_ddm(lol):
    """Make a DDM from lol."""

    # 调用 _DMQ 函数创建一个 QQ 类型的 DDM 对象并返回
    return _DMQ(lol, 'DDM')


def DMQ_sdm(lol):
    """Make a SDM from lol."""

    # 调用 _DMQ 函数创建一个 QQ 类型的 SDM 对象并返回
    return _DMQ(lol, 'SDM')


def DMQ_dfm(lol):
    """Make a DFM from lol."""

    # 调用 _DMQ 函数创建一个 QQ 类型的 DFM 对象并返回
    return _DMQ(lol, 'DFM')


# 将所有 DM 相关函数放入列表 DM_all 中
DM_all = [DM_ddm, DM_sdm, DM_dfm]

# 将所有 DMZ 相关函数放入列表 DMZ_all 中
DMZ_all = [DMZ_ddm, DMZ_sdm, DMZ_dfm]

# 将所有 DMQ 相关函数放入列表 DMQ_all 中
DMQ_all = [DMQ_ddm, DMQ_sdm, DMQ_dfm]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XDM_getitem(DM):
    """Test getitem for DDM, etc."""

    # 创建一个 lol 列表作为 DM 的输入数据
    lol = [[0, 1], [2, 0]]

    # 调用 DM 函数创建一个 DM 对象 A
    A = DM(lol)

    # 获取 A 的形状尺寸 m 和 n
    m, n = A.shape

    # 定义索引列表 indices
    indices = [-3, -2, -1, 0, 1, 2]
    # 对于每个 i 在 indices 中循环
    for i in indices:
        # 对于每个 j 在 indices 中循环
        for j in indices:
            # 如果 i 和 j 都在有效范围内
            if -2 <= i < m and -2 <= j < n:
                # 断言矩阵 A 中 (i, j) 处的元素等于 lol[i][j] 转换为整数后的值
                assert A.getitem(i, j) == ZZ(lol[i][j])
            else:
                # 如果 i 或 j 超出有效范围，断言会引发 IndexError 异常
                raises(IndexError, lambda: A.getitem(i, j))
# 使用 pytest.mark.parametrize 装饰器为 test_XDM_setitem 函数参数化，DM 是从 DMZ_all 中取出的参数之一
@pytest.mark.parametrize('DM', DMZ_all)
def test_XDM_setitem(DM):
    """Test setitem for DDM, etc."""

    # 创建一个 DM 对象 A，传入一个包含列表的列表作为初始化参数
    A = DM([[0, 1, 2], [3, 4, 5]])

    # 调用 A 的 setitem 方法，设置位置 (0, 0) 处的值为 ZZ(6)
    A.setitem(0, 0, ZZ(6))
    # 断言 A 等于给定的 DM 对象
    assert A == DM([[6, 1, 2], [3, 4, 5]])

    # 继续类似的测试设置不同位置的值并进行断言
    A.setitem(0, 1, ZZ(7))
    assert A == DM([[6, 7, 2], [3, 4, 5]])

    A.setitem(0, 2, ZZ(8))
    assert A == DM([[6, 7, 8], [3, 4, 5]])

    A.setitem(0, -1, ZZ(9))
    assert A == DM([[6, 7, 9], [3, 4, 5]])

    A.setitem(0, -2, ZZ(10))
    assert A == DM([[6, 10, 9], [3, 4, 5]])

    A.setitem(0, -3, ZZ(11))
    assert A == DM([[11, 10, 9], [3, 4, 5]])

    # 使用 lambda 函数来测试设置超出边界的索引会抛出 IndexError 异常
    raises(IndexError, lambda: A.setitem(0, 3, ZZ(12)))
    raises(IndexError, lambda: A.setitem(0, -4, ZZ(13)))

    # 继续测试其他的 setitem 操作和断言
    A.setitem(1, 0, ZZ(14))
    assert A == DM([[11, 10, 9], [14, 4, 5]])

    A.setitem(1, 1, ZZ(15))
    assert A == DM([[11, 10, 9], [14, 15, 5]])

    A.setitem(-1, 1, ZZ(16))
    assert A == DM([[11, 10, 9], [14, 16, 5]])

    A.setitem(-2, 1, ZZ(17))
    assert A == DM([[11, 17, 9], [14, 16, 5]])

    raises(IndexError, lambda: A.setitem(2, 0, ZZ(18)))
    raises(IndexError, lambda: A.setitem(-3, 0, ZZ(19)))

    A.setitem(1, 2, ZZ(0))
    assert A == DM([[11, 17, 9], [14, 16, 0]])

    A.setitem(1, -2, ZZ(0))
    assert A == DM([[11, 17, 9], [14, 0, 0]])

    A.setitem(1, -3, ZZ(0))
    assert A == DM([[11, 17, 9], [0, 0, 0]])

    A.setitem(0, 0, ZZ(0))
    assert A == DM([[0, 17, 9], [0, 0, 0]])

    A.setitem(0, -1, ZZ(0))
    assert A == DM([[0, 17, 0], [0, 0, 0]])

    A.setitem(0, 0, ZZ(0))
    assert A == DM([[0, 17, 0], [0, 0, 0]])

    A.setitem(0, -2, ZZ(0))
    assert A == DM([[0, 0, 0], [0, 0, 0]])

    A.setitem(0, -3, ZZ(1))
    assert A == DM([[1, 0, 0], [0, 0, 0]])


# 定义一个名为 _Sliced 的类，实现 __getitem__ 方法，返回 item 本身
class _Sliced:
    def __getitem__(self, item):
        return item


# 创建一个 _Sliced 的实例，命名为 _slice
_slice = _Sliced()


# 使用 pytest.mark.parametrize 装饰器为 test_XXM_extract_slice 函数参数化，DM 是从 DMZ_all 中取出的参数之一
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_extract_slice(DM):
    # 创建一个 DM 对象 A，传入一个包含列表的列表作为初始化参数
    A = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # 断言 A.extract_slice 方法的结果等于 A 本身
    assert A.extract_slice(*_slice[:,:]) == A
    # 继续测试不同的 slice 操作并进行断言
    assert A.extract_slice(*_slice[1:,:]) == DM([[4, 5, 6], [7, 8, 9]])
    assert A.extract_slice(*_slice[1:,1:]) == DM([[5, 6], [8, 9]])
    assert A.extract_slice(*_slice[1:,:-1]) == DM([[4, 5], [7, 8]])
    assert A.extract_slice(*_slice[1:,:-1:2]) == DM([[4], [7]])
    assert A.extract_slice(*_slice[:,::2]) == DM([[1, 3], [4, 6], [7, 9]])
    assert A.extract_slice(*_slice[::2,:]) == DM([[1, 2, 3], [7, 8, 9]])
    assert A.extract_slice(*_slice[::2,::2]) == DM([[1, 3], [7, 9]])
    assert A.extract_slice(*_slice[::2,::-2]) == DM([[3, 1], [9, 7]])
    assert A.extract_slice(*_slice[::-2,::2]) == DM([[7, 9], [1, 3]])
    assert A.extract_slice(*_slice[::-2,::-2]) == DM([[9, 7], [3, 1]])
    assert A.extract_slice(*_slice[:,::-1]) == DM([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
    assert A.extract_slice(*_slice[::-1,:]) == DM([[7, 8, 9], [4, 5, 6], [1, 2, 3]])


# 使用 pytest.mark.parametrize 装饰器为 test_XXM_extract 函数参数化，DM 是从 DMZ_all 中取出的参数之一
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_extract(DM):
    # 创建一个 DM 对象 A，传入一个包含列表的列表作为初始化参数
    A = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # 断言 A.extract 方法的结果等于 A 本身
    assert A.extract([0, 1, 2], [0, 1, 2]) == A
    # 断言语句：验证 A.extract 方法的返回结果是否符合预期
    assert A.extract([1, 2], [1, 2]) == DM([[5, 6], [8, 9]])
    # 断言语句：验证 A.extract 方法的返回结果是否符合预期
    assert A.extract([1, 2], [0, 1]) == DM([[4, 5], [7, 8]])
    # 断言语句：验证 A.extract 方法的返回结果是否符合预期
    assert A.extract([1, 2], [0, 2]) == DM([[4, 6], [7, 9]])
    # 断言语句：验证 A.extract 方法的返回结果是否符合预期
    assert A.extract([1, 2], [0]) == DM([[4], [7]])
    # 断言语句：验证 A.extract 方法的返回结果是否符合预期
    assert A.extract([1, 2], []) == DM([[1]]).zeros((2, 0), ZZ)
    # 断言语句：验证 A.extract 方法的返回结果是否符合预期
    assert A.extract([], [0, 1, 2]) == DM([[1]]).zeros((0, 3), ZZ)

    # raises 函数调用：验证 A.extract 方法在特定条件下是否抛出 IndexError 异常
    raises(IndexError, lambda: A.extract([1, 2], [0, 3]))
    # raises 函数调用：验证 A.extract 方法在特定条件下是否抛出 IndexError 异常
    raises(IndexError, lambda: A.extract([1, 2], [0, -4]))
    # raises 函数调用：验证 A.extract 方法在特定条件下是否抛出 IndexError 异常
    raises(IndexError, lambda: A.extract([3, 1], [0, 1]))
    # raises 函数调用：验证 A.extract 方法在特定条件下是否抛出 IndexError 异常
    raises(IndexError, lambda: A.extract([-4, 2], [3, 1]))

    # 变量赋值语句：创建一个大小为 3x3 的零矩阵 B
    B = DM([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # 断言语句：验证 B.extract 方法的返回结果是否符合预期
    assert B.extract([1, 2], [1, 2]) == DM([[0, 0], [0, 0]])
# 定义名为 test_XXM_str 的测试函数
def test_XXM_str():

    # 创建一个 DomainMatrix 对象 A，包含整数矩阵及其维度信息
    A = DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)

    # 断言 A 对象的字符串表示是否符合预期
    assert str(A) == \
        'DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
    
    # 断言将 A 转换为 DDM 格式的字符串表示是否符合预期
    assert str(A.to_ddm()) == \
        '[[1, 2, 3], [4, 5, 6], [7, 8, 9]]'
    
    # 断言将 A 转换为 SDM 格式的字符串表示是否符合预期
    assert str(A.to_sdm()) == \
        '{0: {0: 1, 1: 2, 2: 3}, 1: {0: 4, 1: 5, 2: 6}, 2: {0: 7, 1: 8, 2: 9}}'

    # 断言 A 对象的详细表示是否符合预期
    assert repr(A) == \
        'DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
    
    # 断言将 A 转换为 DDM 格式的详细表示是否符合预期
    assert repr(A.to_ddm()) == \
        'DDM([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
    
    # 断言将 A 转换为 SDM 格式的详细表示是否符合预期
    assert repr(A.to_sdm()) == \
        'SDM({0: {0: 1, 1: 2, 2: 3}, 1: {0: 4, 1: 5, 2: 6}, 2: {0: 7, 1: 8, 2: 9}}, (3, 3), ZZ)'

    # 创建一个 DomainMatrix 对象 B，包含字典格式的矩阵及其维度信息
    B = DomainMatrix({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3)}}, (2, 2), ZZ)

    # 断言 B 对象的字符串表示是否符合预期
    assert str(B) == \
        'DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3}}, (2, 2), ZZ)'
    
    # 断言将 B 转换为 DDM 格式的字符串表示是否符合预期
    assert str(B.to_ddm()) == \
        '[[1, 2], [3, 0]]'
    
    # 断言将 B 转换为 SDM 格式的字符串表示是否符合预期
    assert str(B.to_sdm()) == \
        '{0: {0: 1, 1: 2}, 1: {0: 3}}'

    # 断言 B 对象的详细表示是否符合预期
    assert repr(B) == \
        'DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3}}, (2, 2), ZZ)'

    # 根据 GROUND_TYPES 的设置条件，进行不同的详细表示断言
    if GROUND_TYPES != 'gmpy':
        assert repr(B.to_ddm()) == \
            'DDM([[1, 2], [3, 0]], (2, 2), ZZ)'
        assert repr(B.to_sdm()) == \
            'SDM({0: {0: 1, 1: 2}, 1: {0: 3}}, (2, 2), ZZ)'
    else:
        assert repr(B.to_ddm()) == \
            'DDM([[mpz(1), mpz(2)], [mpz(3), mpz(0)]], (2, 2), ZZ)'
        assert repr(B.to_sdm()) == \
            'SDM({0: {0: mpz(1), 1: mpz(2)}, 1: {0: mpz(3)}}, (2, 2), ZZ)'

    # 根据 GROUND_TYPES 的设置条件，进行不同类型的测试
    if GROUND_TYPES == 'flint':

        # 断言将 A 转换为 DFM 格式的字符串表示是否符合预期
        assert str(A.to_dfm()) == \
            '[[1, 2, 3], [4, 5, 6], [7, 8, 9]]'
        
        # 断言将 B 转换为 DFM 格式的字符串表示是否符合预期
        assert str(B.to_dfm()) == \
            '[[1, 2], [3, 0]]'

        # 断言将 A 转换为 DFM 格式的详细表示是否符合预期
        assert repr(A.to_dfm()) == \
            'DFM([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
        
        # 断言将 B 转换为 DFM 格式的详细表示是否符合预期
        assert repr(B.to_dfm()) == \
            'DFM([[1, 2], [3, 0]], (2, 2), ZZ)'


# 使用 pytest 的参数化装饰器，对 DMZ_all 中的每个 DM 进行测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_list(DM):
    # 获取 DM 类型的对象 T
    T = type(DM([[0]]))

    # 定义一个嵌套列表 lol 和对应的 ZZ 类型的嵌套列表 lol_ZZ
    lol = [[1, 2, 4], [4, 5, 6]]
    lol_ZZ = [[ZZ(1), ZZ(2), ZZ(4)], [ZZ(4), ZZ(5), ZZ(6)]]
    lol_ZZ_bad = [[ZZ(1), ZZ(2), ZZ(4)], [ZZ(4), ZZ(5), ZZ(6), ZZ(7)]]

    # 断言 T 类型对象根据 lol_ZZ 和指定维度信息能够正确创建 DM(lol) 对象
    assert T.from_list(lol_ZZ, (2, 3), ZZ) == DM(lol)
    
    # 断言输入不合规的 lol_ZZ_bad 会引发 DMBadInputError 异常
    raises(DMBadInputError, lambda: T.from_list(lol_ZZ_bad, (3, 2), ZZ))


# 使用 pytest 的参数化装饰器，对 DMZ_all 中的每个 DM 进行测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_list(DM):
    # 定义一个嵌套列表 lol
    lol = [[1, 2, 4]]
    
    # 断言 DM(lol) 对象能够正确转换为 ZZ 类型的嵌套列表
    assert DM(lol).to_list() == [[ZZ(1), ZZ(2), ZZ(4)]]


# 使用 pytest 的参数化装饰器，对 DMZ_all 中的每个 DM 进行测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_list_flat(DM):
    # 定义一个嵌套列表 lol
    lol = [[1, 2, 4]]
    
    # 断言 DM(lol) 对象能够正确转换为扁平化的 ZZ 类型列表
    assert DM(lol).to_list_flat() == [ZZ(1), ZZ(2), ZZ(4)]


# 使用 pytest 的参数化装饰器，对 DMZ_all 中的每个 DM 进行测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_list_flat(DM):
    # 获取 DM 类型的对象 T
    T = type(DM([[0]]))
    
    # 定义一个扁平化的 ZZ 类型列表 flat
    flat = [ZZ(1), ZZ(2), ZZ(4)]
    
    # 断言 T 类
# 定义一个测试函数，用于测试 DM 类型对象的 to_flat_nz 方法
def test_XXM_to_flat_nz(DM):
    # 创建一个稀疏矩阵 M
    M = DM([[1, 2, 0], [0, 0, 0], [0, 0, 3]])
    # 指定稀疏元素的值和位置
    elements = [ZZ(1), ZZ(2), ZZ(3)]
    indices = ((0, 0), (0, 1), (2, 2))
    # 断言调用 to_flat_nz 方法后返回的结果是否符合预期
    assert M.to_flat_nz() == (elements, (indices, M.shape))


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_flat_nz(DM):
    # 获取 DM 对象的类型 T
    T = type(DM([[0]]))
    # 指定稀疏元素的值和位置
    elements = [ZZ(1), ZZ(2), ZZ(3)]
    indices = ((0, 0), (0, 1), (2, 2))
    data = (indices, (3, 3))
    # 构造期望的结果矩阵
    result = DM([[1, 2, 0], [0, 0, 0], [0, 0, 3]])
    # 断言调用 from_flat_nz 方法后返回的结果是否符合预期
    assert T.from_flat_nz(elements, data, ZZ) == result
    # 测试输入错误数据时是否会抛出 DMBadInputError 异常
    raises(DMBadInputError, lambda: T.from_flat_nz(elements, (indices, (2, 3)), ZZ))


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_dod(DM):
    # 指定期望的字典形式稀疏矩阵
    dod = {0: {0: ZZ(1), 2: ZZ(4)}, 1: {0: ZZ(4), 1: ZZ(5), 2: ZZ(6)}}
    # 断言调用 to_dod 方法后返回的结果是否符合预期
    assert DM([[1, 0, 4], [4, 5, 6]]).to_dod() == dod


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_dod(DM):
    # 获取 DM 对象的类型 T
    T = type(DM([[0]]))
    # 指定期望的字典形式稀疏矩阵
    dod = {0: {0: ZZ(1), 2: ZZ(4)}, 1: {0: ZZ(4), 1: ZZ(5), 2: ZZ(6)}}
    # 构造期望的结果矩阵
    result = DM([[1, 0, 4], [4, 5, 6]])
    # 断言调用 from_dod 方法后返回的结果是否符合预期
    assert T.from_dod(dod, (2, 3), ZZ) == result


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_dok(DM):
    # 指定期望的字典形式稀疏矩阵
    dod = {(0, 0): ZZ(1), (0, 2): ZZ(4),
           (1, 0): ZZ(4), (1, 1): ZZ(5), (1, 2): ZZ(6)}
    # 断言调用 to_dok 方法后返回的结果是否符合预期
    assert DM([[1, 0, 4], [4, 5, 6]]).to_dok() == dod


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_dok(DM):
    # 获取 DM 对象的类型 T
    T = type(DM([[0]]))
    # 指定期望的字典形式稀疏矩阵
    dod = {(0, 0): ZZ(1), (0, 2): ZZ(4),
           (1, 0): ZZ(4), (1, 1): ZZ(5), (1, 2): ZZ(6)}
    # 构造期望的结果矩阵
    result = DM([[1, 0, 4], [4, 5, 6]])
    # 断言调用 from_dok 方法后返回的结果是否符合预期
    assert T.from_dok(dod, (2, 3), ZZ) == result


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_iter_values(DM):
    # 指定期望的值迭代结果
    values = [ZZ(1), ZZ(4), ZZ(4), ZZ(5), ZZ(6)]
    # 断言调用 iter_values 方法后返回的结果是否符合预期
    assert sorted(DM([[1, 0, 4], [4, 5, 6]]).iter_values()) == values


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_iter_items(DM):
    # 指定期望的键值对迭代结果
    items = [((0, 0), ZZ(1)), ((0, 2), ZZ(4)),
             ((1, 0), ZZ(4)), ((1, 1), ZZ(5)), ((1, 2), ZZ(6))]
    # 断言调用 iter_items 方法后返回的结果是否符合预期
    assert sorted(DM([[1, 0, 4], [4, 5, 6]]).iter_items()) == items


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_ddm(DM):
    # 获取 DM 对象的类型 T
    T = type(DM([[0]]))
    # 构造输入的 DDM 对象
    ddm = DDM([[1, 2, 4], [4, 5, 6]], (2, 3), ZZ)
    # 构造期望的结果矩阵
    result = DM([[1, 2, 4], [4, 5, 6]])
    # 断言调用 from_ddm 方法后返回的结果是否符合预期
    assert T.from_ddm(ddm) == result


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_zeros(DM):
    # 获取 DM 对象的类型 T
    T = type(DM([[0]]))
    # 断言调用 zeros 方法后返回的结果是否符合预期
    assert T.zeros((2, 3), ZZ) == DM([[0, 0, 0], [0, 0, 0]])


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_ones(DM):
    # 获取 DM 对象的类型 T
    T = type(DM([[0]]))
    # 断言调用 ones 方法后返回的结果是否符合预期
    assert T.ones((2, 3), ZZ) == DM([[1, 1, 1], [1, 1, 1]])


# 使用 pytest 的参数化功能，对 DMZ_all 中的每个 DM 进行参数化测试
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_eye(DM):
    # 获取 DM 对象的类型 T
    T = type(DM([[0]]))
    # 断言调用 eye 方法后返回的结果是否符合预期
    assert T.eye(3, ZZ) == DM([[1, 0, 0], [0, 1,
@pytest.mark.parametrize('DM', DMZ_all)
# 使用 pytest 的 parametrize 装饰器，对测试函数进行参数化，DM 是参数化的变量，取自 DMZ_all 列表
def test_XXM_add(DM):
    # 创建矩阵 A、B 和期望结果矩阵 C，用于测试矩阵加法
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[2, 4, 6], [8, 10, 12]])
    # 断言矩阵 A 加上矩阵 B 等于矩阵 C
    assert A.add(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
# 使用 pytest 的 parametrize 装饰器，对测试函数进行参数化，DM 是参数化的变量，取自 DMZ_all 列表
def test_XXM_sub(DM):
    # 创建矩阵 A、B 和期望结果矩阵 C，用于测试矩阵减法
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[0, 0, 0], [0, 0, 0]])
    # 断言矩阵 A 减去矩阵 B 等于矩阵 C
    assert A.sub(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
# 使用 pytest 的 parametrize 装饰器，对测试函数进行参数化，DM 是参数化的变量，取自 DMZ_all 列表
def test_XXM_mul(DM):
    # 创建矩阵 A 和标量 b，用于测试矩阵乘法
    A = DM([[1, 2, 3], [4, 5, 6]])
    b = ZZ(2)
    # 断言矩阵 A 乘以标量 b 等于预期的结果矩阵
    assert A.mul(b) == DM([[2, 4, 6], [8, 10, 12]])
    assert A.rmul(b) == DM([[2, 4, 6], [8, 10, 12]])


@pytest.mark.parametrize('DM', DMZ_all)
# 使用 pytest 的 parametrize 装饰器，对测试函数进行参数化，DM 是参数化的变量，取自 DMZ_all 列表
def test_XXM_matmul(DM):
    # 创建矩阵 A 和 B，用于测试矩阵乘法（矩阵乘）
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2], [3, 4], [5, 6]])
    C = DM([[22, 28], [49, 64]])
    # 断言矩阵 A 乘以矩阵 B 等于矩阵 C
    assert A.matmul(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
# 使用 pytest 的 parametrize 装饰器，对测试函数进行参数化，DM 是参数化的变量，取自 DMZ_all 列表
def test_XXM_mul_elementwise(DM):
    # 创建矩阵 A 和 B，用于测试元素级乘法
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[1, 4, 9], [16, 25, 36]])
    # 断言矩阵 A 和矩阵 B 的元素级乘法等于矩阵 C
    assert A.mul_elementwise(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
# 使用 pytest 的 parametrize 装饰器，对测试函数进行参数化，DM 是参数化的变量，取自 DMZ_all 列表
def test_XXM_neg(DM):
    # 创建矩阵 A 和期望结果矩阵 C，用于测试矩阵取负
    A = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[-1, -2, -3], [-4, -5, -6]])
    # 断言矩阵 A 取负后等于矩阵 C
    assert A.neg() == C


@pytest.mark.parametrize('DM', DM_all)
# 使用 pytest 的 parametrize 装饰器，对测试函数进行参数化，DM 是参数化的变量，取自 DM_all 列表
def test_XXM_convert_to(DM):
    # 创建矩阵 A 和 B，分别用 ZZ 和 QQ 类型初始化，用于测试矩阵类型转换
    A = DM([[1, 2, 3], [4, 5, 6]], ZZ)
    B = DM([[1, 2, 3], [4, 5, 6]], QQ)
    # 断言将矩阵 A 转换为 QQ 类型后等于矩阵 B，反之亦然
    assert A.convert_to(QQ) == B
    assert B.convert_to(ZZ) == A


@pytest.mark.parametrize('DM', DMZ_all)
# 使用 pytest 的 parametrize 装饰器，对测试函数进行参数化，DM 是参数化的变量，取自 DMZ_all 列表
def test_XXM_scc(DM):
    # 创建邻接矩阵 A 和预期结果列表，用于测试强连通分量算法
    A = DM([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1]])
    # 断言矩阵 A 的强连通分量算法结果等于预期的结果列表
    assert A.scc() == [[0, 1], [2], [3, 5], [4]]


@pytest.mark.parametrize('DM', DMZ_all)
# 使用 pytest 的 parametrize 装饰器，对测试函数进行参数化，DM 是参数化的变量，取自 DMZ_all 列表
def test_XXM_hstack(DM):
    # 创建矩阵 A 和 B，用于测试水平拼接
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[7, 8], [9, 10]])
    C = DM([[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]])
    ABC = DM([[1, 2, 3, 7, 8, 1, 2, 3, 7, 8],
              [4, 5, 6, 9, 10, 4, 5, 6, 9, 10]])
    # 断言矩阵 A 和 B 的水平拼接结果等于预期的结果矩阵 C 和 ABC
    assert A.hstack(B) == C
    assert A.hstack(B, C) == ABC


@pytest.mark.parametrize('DM', DMZ_all)
# 使用 pytest 的 parametrize 装饰器，对测试函数进行参数化，DM 是参数化的变量，取自 DMZ_all 列表
def test_XXM_vstack(DM):
    # 创建矩阵 A 和 B，用于测试垂直拼接
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[7, 8, 9]])
    C = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ABC = DM([[1, 2, 3], [4, 5, 6], [7,
    # 对给定的二维列表创建一个 DiagonalMatrix 对象，并断言其为对角矩阵（返回True）
    assert DM([[1, 0, 0], [0, 5, 0]]).is_diagonal() is True
    # 对给定的二维列表创建一个 DiagonalMatrix 对象，并断言其不是对角矩阵（返回False）
    assert DM([[1, 2, 3], [4, 5, 6]]).is_diagonal() is False
@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_diagonal(DM):
    # 断言对角线方法是否返回正确的对角线元素
    assert DM([[1, 0, 0], [0, 5, 0]]).diagonal() == [1, 5]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_is_zero_matrix(DM):
    # 断言是否正确识别零矩阵和非零矩阵
    assert DM([[0, 0, 0], [0, 0, 0]]).is_zero_matrix() is True
    assert DM([[1, 0, 0], [0, 0, 0]]).is_zero_matrix() is False


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_det_ZZ(DM):
    # 断言整数矩阵的行列式计算是否正确
    assert DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).det() == 0
    assert DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).det() == -3


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_det_QQ(DM):
    # 断言有理数矩阵的行列式计算是否正确
    dM1 = DM([[(1,2), (2,3)], [(3,4), (4,5)]])
    assert dM1.det() == QQ(-1,10)


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_inv_QQ(DM):
    # 断言有理数矩阵的求逆运算是否正确
    dM1 = DM([[(1,2), (2,3)], [(3,4), (4,5)]])
    dM2 = DM([[(-8,1), (20,3)], [(15,2), (-5,1)]])
    assert dM1.inv() == dM2
    assert dM1.matmul(dM2) == DM([[1, 0], [0, 1]])

    # 测试非可逆矩阵抛出异常
    dM3 = DM([[(1,2), (2,3)], [(1,4), (1,3)]])
    raises(DMNonInvertibleMatrixError, lambda: dM3.inv())

    # 测试非方阵矩阵抛出异常
    dM4 = DM([[(1,2), (2,3), (3,4)], [(1,4), (1,3), (1,2)]])
    raises(DMNonSquareMatrixError, lambda: dM4.inv())


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_inv_ZZ(DM):
    # 测试整数矩阵求逆抛出异常
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    # XXX: 可能应该返回一个有理数域上的矩阵？
    # XXX: 处理单模块矩阵？
    raises(DMDomainError, lambda: dM1.inv())


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_charpoly_ZZ(DM):
    # 断言整数矩阵的特征多项式计算是否正确
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    assert dM1.charpoly() == [1, -16, -12, 3]


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_charpoly_QQ(DM):
    # 断言有理数矩阵的特征多项式计算是否正确
    dM1 = DM([[(1,2), (2,3)], [(3,4), (4,5)]])
    assert dM1.charpoly() == [QQ(1,1), QQ(-13,10), QQ(-1,10)]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_lu_solve_ZZ(DM):
    # 测试整数矩阵的LU分解求解线性方程组抛出异常
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    dM2 = DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    raises(DMDomainError, lambda: dM1.lu_solve(dM2))


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_lu_solve_QQ(DM):
    # 断言有理数矩阵的LU分解求解线性方程组是否正确
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    dM2 = DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dM3 = DM([[(-2,3),(-4,3),(1,1)],[(-2,3),(11,3),(-2,1)],[(1,1),(-2,1),(1,1)]])
    assert dM1.lu_solve(dM2) == dM3 == dM1.inv()

    # 测试形状不匹配的矩阵抛出异常
    dM4 = DM([[1, 2, 3], [4, 5, 6]])
    dM5 = DM([[1, 0], [0, 1], [0, 0]])
    raises(DMShapeError, lambda: dM4.lu_solve(dM5))


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_nullspace_QQ(DM):
    # 断言有理数矩阵的零空间计算是否正确
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # XXX: 修改签名，只返回零空间。可能返回秩或零度使得意义，但非枢轴列表则没有用。
    assert dM1.nullspace() == (DM([[1, -2, 1]]), [2])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_lll(DM):
    # 断言整数矩阵的LLL约简算法是否正确
    M = DM([[1, 2, 3], [4, 5, 20]])
    M_lll = DM([[1, 2, 3], [-1, -5, 5]])
    T = DM([[1, 0], [-5, 1]])
    assert M.lll() == M_lll
    assert M.lll_transform() == (M_lll, T)
    # 使用 assert 语句来检查矩阵 T 乘以矩阵 M 的结果是否等于矩阵 M_lll。
    assert T.matmul(M) == M_lll
```
# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_lll.py`

```
# 导入所需的符号数学库中的特定模块和函数
from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices import DM
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMRankError, DMValueError, DMShapeError, DMDomainError
from sympy.polys.matrices.lll import _ddm_lll, ddm_lll, ddm_lll_transform
# 导入用于测试的 pytest 模块中的 raises 函数

# 定义一个测试函数 test_lll，用于测试 LLL 算法的各个方面
def test_lll():
    # 正常测试数据集，包含多个元组，每个元组包含两个 DomainMatrix 对象
    normal_test_data = [
        (
            DM([[1, 0, 0, 0, -20160],
                [0, 1, 0, 0, 33768],
                [0, 0, 1, 0, 39578],
                [0, 0, 0, 1, 47757]], ZZ),
            DM([[10, -3, -2, 8, -4],
                [3, -9, 8, 1, -11],
                [-3, 13, -9, -3, -9],
                [-12, -7, -11, 9, -1]], ZZ)
        ),
        (
            DM([[20, 52, 3456],
                [14, 31, -1],
                [34, -442, 0]], ZZ),
            DM([[14, 31, -1],
                [188, -101, -11],
                [236, 13, 3443]], ZZ)
        ),
        (
            DM([[34, -1, -86, 12],
                [-54, 34, 55, 678],
                [23, 3498, 234, 6783],
                [87, 49, 665, 11]], ZZ),
            DM([[34, -1, -86, 12],
                [291, 43, 149, 83],
                [-54, 34, 55, 678],
                [-189, 3077, -184, -223]], ZZ)
        )
    ]
    
    # 定义 delta 值为有理数 QQ(5, 6)
    delta = QQ(5, 6)
    
    # 遍历正常测试数据集
    for basis_dm, reduced_dm in normal_test_data:
        # 使用 _ddm_lll 函数对基础矩阵进行 LLL 算法约简，返回结果中取第一个元素
        reduced = _ddm_lll(basis_dm.rep.to_ddm(), delta=delta)[0]
        # 断言约简后的结果与预期的约简矩阵相等
        assert reduced == reduced_dm.rep.to_ddm()

        # 使用 ddm_lll 函数对基础矩阵进行 LLL 算法约简
        reduced = ddm_lll(basis_dm.rep.to_ddm(), delta=delta)
        # 断言约简后的结果与预期的约简矩阵相等
        assert reduced == reduced_dm.rep.to_ddm()

        # 使用 _ddm_lll 函数同时返回变换矩阵，进行约简操作
        reduced, transform = _ddm_lll(basis_dm.rep.to_ddm(), delta=delta, return_transform=True)
        # 断言约简后的结果与预期的约简矩阵相等
        assert reduced == reduced_dm.rep.to_ddm()
        # 断言变换矩阵乘以基础矩阵等于预期的约简矩阵
        assert transform.matmul(basis_dm.rep.to_ddm()) == reduced_dm.rep.to_ddm()

        # 使用 ddm_lll_transform 函数对基础矩阵进行 LLL 算法约简，并返回变换矩阵
        reduced, transform = ddm_lll_transform(basis_dm.rep.to_ddm(), delta=delta)
        # 断言约简后的结果与预期的约简矩阵相等
        assert reduced == reduced_dm.rep.to_ddm()
        # 断言变换矩阵乘以基础矩阵等于预期的约简矩阵
        assert transform.matmul(basis_dm.rep.to_ddm()) == reduced_dm.rep.to_ddm()

        # 使用 DomainMatrix 对象的 lll 方法进行约简
        reduced = basis_dm.rep.lll(delta=delta)
        # 断言约简后的结果与预期的约简矩阵相等
        assert reduced == reduced_dm.rep

        # 使用 DomainMatrix 对象的 lll_transform 方法进行约简，并返回变换矩阵
        reduced, transform = basis_dm.rep.lll_transform(delta=delta)
        # 断言约简后的结果与预期的约简矩阵相等
        assert reduced == reduced_dm.rep
        # 断言变换矩阵乘以基础矩阵等于预期的约简矩阵
        assert transform.matmul(basis_dm.rep) == reduced_dm.rep

        # 将 DomainMatrix 对象转换为 sdm 格式后，再进行 LLL 算法约简
        reduced = basis_dm.rep.to_sdm().lll(delta=delta)
        # 断言约简后的结果与预期的约简矩阵相等
        assert reduced == reduced_dm.rep.to_sdm()

        # 将 DomainMatrix 对象转换为 sdm 格式后，使用 lll_transform 方法进行约简，并返回变换矩阵
        reduced, transform = basis_dm.rep.to_sdm().lll_transform(delta=delta)
        # 断言约简后的结果与预期的约简矩阵相等
        assert reduced == reduced_dm.rep.to_sdm()
        # 断言变换矩阵乘以基础矩阵等于预期的约简矩阵
        assert transform.matmul(basis_dm.rep.to_sdm()) == reduced_dm.rep.to_sdm()

        # 使用 DomainMatrix 对象的 lll 方法进行约简
        reduced = basis_dm.lll(delta=delta)
        # 断言约简后的结果与预期的约简矩阵相等
        assert reduced == reduced_dm

        # 使用 DomainMatrix 对象的 lll_transform 方法进行约简，并返回变换矩阵
        reduced, transform = basis_dm.lll_transform(delta=delta)
        # 断言约简后的结果与预期的约简矩阵相等
        assert reduced == reduced_dm
        # 断言变换矩阵乘以基础矩阵等于预期的约简矩阵
        assert transform.matmul(basis_dm) == reduced_dm


def test_lll_linear_dependent():
    # 定义包含线性相关测试数据的列表
    linear_dependent_test_data = [
        # 创建整数矩阵，用于测试线性相关性
        DM([[0, -1, -2, -3],
            [1, 0, -1, -2],
            [2, 1, 0, -1],
            [3, 2, 1, 0]], ZZ),
        # 创建整数矩阵，用于测试线性相关性
        DM([[1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 2, 3, 6]], ZZ),
        # 创建整数矩阵，用于测试线性相关性
        DM([[3, -5, 1],
            [4, 6, 0],
            [10, -4, 2]], ZZ)
    ]
    # 对每个测试数据进行循环
    for not_basis in linear_dependent_test_data:
        # 检查是否抛出 DMRankError 异常，期望情况下 _ddm_lll 函数不能处理线性相关矩阵
        raises(DMRankError, lambda: _ddm_lll(not_basis.rep.to_ddm()))
        # 检查是否抛出 DMRankError 异常，期望情况下 ddm_lll 函数不能处理线性相关矩阵
        raises(DMRankError, lambda: ddm_lll(not_basis.rep.to_ddm()))
        # 检查是否抛出 DMRankError 异常，期望情况下 not_basis.rep.lll() 不能处理线性相关矩阵
        raises(DMRankError, lambda: not_basis.rep.lll())
        # 检查是否抛出 DMRankError 异常，期望情况下 not_basis.rep.to_sdm().lll() 不能处理线性相关矩阵
        raises(DMRankError, lambda: not_basis.rep.to_sdm().lll())
        # 检查是否抛出 DMRankError 异常，期望情况下 not_basis.lll() 不能处理线性相关矩阵
        raises(DMRankError, lambda: not_basis.lll())
        # 检查是否抛出 DMRankError 异常，期望情况下 _ddm_lll 函数不能处理线性相关矩阵，并返回变换
        raises(DMRankError, lambda: _ddm_lll(not_basis.rep.to_ddm(), return_transform=True))
        # 检查是否抛出 DMRankError 异常，期望情况下 ddm_lll_transform 函数不能处理线性相关矩阵
        raises(DMRankError, lambda: ddm_lll_transform(not_basis.rep.to_ddm()))
        # 检查是否抛出 DMRankError 异常，期望情况下 not_basis.rep.lll_transform() 不能处理线性相关矩阵
        raises(DMRankError, lambda: not_basis.rep.lll_transform())
        # 检查是否抛出 DMRankError 异常，期望情况下 not_basis.rep.to_sdm().lll_transform() 不能处理线性相关矩阵
        raises(DMRankError, lambda: not_basis.rep.to_sdm().lll_transform())
        # 检查是否抛出 DMRankError 异常，期望情况下 not_basis.lll_transform() 不能处理线性相关矩阵
        raises(DMRankError, lambda: not_basis.lll_transform())
# 测试函数，用于检查在给定特定条件下是否会引发异常
def test_lll_wrong_delta():
    # 创建一个 3x3 的整数矩阵
    dummy_matrix = DomainMatrix.ones((3, 3), ZZ)
    # 对于不同的分数对象，执行以下操作：
    for wrong_delta in [QQ(-1, 4), QQ(0, 1), QQ(1, 4), QQ(1, 1), QQ(100, 1)]:
        # 检查是否会抛出 DMValueError 异常，并且不返回任何转换
        raises(DMValueError, lambda: _ddm_lll(dummy_matrix.rep, delta=wrong_delta))
        # 检查是否会抛出 DMValueError 异常，并且不返回任何转换
        raises(DMValueError, lambda: ddm_lll(dummy_matrix.rep, delta=wrong_delta))
        # 检查是否会抛出 DMValueError 异常，并且不返回任何转换
        raises(DMValueError, lambda: dummy_matrix.rep.lll(delta=wrong_delta))
        # 检查是否会抛出 DMValueError 异常，并且不返回任何转换
        raises(DMValueError, lambda: dummy_matrix.rep.to_sdm().lll(delta=wrong_delta))
        # 检查是否会抛出 DMValueError 异常，并且不返回任何转换
        raises(DMValueError, lambda: dummy_matrix.lll(delta=wrong_delta))
        # 检查是否会抛出 DMValueError 异常，并且返回转换
        raises(DMValueError, lambda: _ddm_lll(dummy_matrix.rep, delta=wrong_delta, return_transform=True))
        # 检查是否会抛出 DMValueError 异常，并且返回转换
        raises(DMValueError, lambda: ddm_lll_transform(dummy_matrix.rep, delta=wrong_delta))
        # 检查是否会抛出 DMValueError 异常，并且返回转换
        raises(DMValueError, lambda: dummy_matrix.rep.lll_transform(delta=wrong_delta))
        # 检查是否会抛出 DMValueError 异常，并且返回转换
        raises(DMValueError, lambda: dummy_matrix.rep.to_sdm().lll_transform(delta=wrong_delta))
        # 检查是否会抛出 DMValueError 异常，并且返回转换
        raises(DMValueError, lambda: dummy_matrix.lll_transform(delta=wrong_delta))


# 测试函数，用于检查在给定特定条件下是否会引发异常
def test_lll_wrong_shape():
    # 创建一个 4x3 的整数矩阵
    wrong_shape_matrix = DomainMatrix.ones((4, 3), ZZ)
    # 对于该矩阵执行以下操作：
    raises(DMShapeError, lambda: _ddm_lll(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: ddm_lll(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.lll())
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.to_sdm().lll())
    raises(DMShapeError, lambda: wrong_shape_matrix.lll())
    raises(DMShapeError, lambda: _ddm_lll(wrong_shape_matrix.rep, return_transform=True))
    raises(DMShapeError, lambda: ddm_lll_transform(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.lll_transform())
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.to_sdm().lll_transform())
    raises(DMShapeError, lambda: wrong_shape_matrix.lll_transform())


# 测试函数，用于检查在给定特定条件下是否会引发异常
def test_lll_wrong_domain():
    # 创建一个 3x3 的有理数矩阵
    wrong_domain_matrix = DomainMatrix.ones((3, 3), QQ)
    # 对于该矩阵执行以下操作：
    raises(DMDomainError, lambda: _ddm_lll(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: ddm_lll(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.lll())
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.to_sdm().lll())
    raises(DMDomainError, lambda: wrong_domain_matrix.lll())
    raises(DMDomainError, lambda: _ddm_lll(wrong_domain_matrix.rep, return_transform=True))
    raises(DMDomainError, lambda: ddm_lll_transform(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.lll_transform())
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.to_sdm().lll_transform())
    raises(DMDomainError, lambda: wrong_domain_matrix.lll_transform())
```
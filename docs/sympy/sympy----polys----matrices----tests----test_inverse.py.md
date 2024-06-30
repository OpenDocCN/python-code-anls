# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_inverse.py`

```
# 从 sympy 库导入需要的模块和函数
from sympy import ZZ, Matrix
from sympy.polys.matrices import DM, DomainMatrix
from sympy.polys.matrices.dense import ddm_iinv
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.matrices.exceptions import NonInvertibleMatrixError

# 导入 pytest 的必要模块和函数
import pytest
from sympy.testing.pytest import raises
from sympy.core.numbers import all_close

# 从 sympy.abc 模块导入变量 x
from sympy.abc import x

# 定义一个包含不同示例的列表，用于测试矩阵求逆的几种情况
# 每个元组包含矩阵名字、原矩阵 A、逆矩阵 A_inv、除数 den
INVERSE_EXAMPLES = [

    (
        'zz_1',
        DomainMatrix([], (0, 0), ZZ),
        DomainMatrix([], (0, 0), ZZ),
        ZZ(1),
    ),

    (
        'zz_2',
        DM([[2]], ZZ),
        DM([[1]], ZZ),
        ZZ(2),
    ),

    (
        'zz_3',
        DM([[2, 0],
            [0, 2]], ZZ),
        DM([[2, 0],
            [0, 2]], ZZ),
        ZZ(4),
    ),

    (
        'zz_4',
        DM([[1, 2],
            [3, 4]], ZZ),
        DM([[ 4, -2],
            [-3,  1]], ZZ),
        ZZ(-2),
    ),

    (
        'zz_5',
        DM([[2, 2, 0],
            [0, 2, 2],
            [0, 0, 2]], ZZ),
        DM([[4, -4, 4],
            [0, 4, -4],
            [0, 0,  4]], ZZ),
        ZZ(8),
    ),

    (
        'zz_6',
        DM([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], ZZ),
        DM([[-3,   6, -3],
            [ 6, -12,  6],
            [-3,   6, -3]], ZZ),
        ZZ(0),
    ),
]

# 使用 pytest.mark.parametrize 装饰器对测试函数 test_Matrix_inv 进行参数化测试
@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_Matrix_inv(name, A, A_inv, den):

    # 定义内部函数 _check，用于检查矩阵求逆是否正确
    def _check(**kwargs):
        # 如果除数 den 不为 0，则断言 A 的逆矩阵应该等于预期的 A_inv
        if den != 0:
            assert A.inv(**kwargs) == A_inv
        # 否则，预期会抛出 NonInvertibleMatrixError 异常
        else:
            raises(NonInvertibleMatrixError, lambda: A.inv(**kwargs))

    # 获取 A 的定义域，并将 A 和 A_inv 转换为 Matrix 类型，并除以 den 转换为符号类型
    K = A.domain
    A = A.to_Matrix()
    A_inv = A_inv.to_Matrix() / K.to_sympy(den)
    
    # 调用 _check 函数进行检查，使用不同的求逆方法进行测试
    _check()
    for method in ['GE', 'LU', 'ADJ', 'CH', 'LDL', 'QR']:
        _check(method=method)


# 使用 pytest.mark.parametrize 装饰器对测试函数 test_dm_inv_den 进行参数化测试
@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dm_inv_den(name, A, A_inv, den):
    # 如果 den 不为 0，则获取 A 的分数逆 A_inv_f 和 den_f，并断言其取消分母后应与预期 A_inv 相等
    if den != 0:
        A_inv_f, den_f = A.inv_den()
        assert A_inv_f.cancel_denom(den_f) == A_inv.cancel_denom(den)
    # 否则，预期会抛出 DMNonInvertibleMatrixError 异常
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv_den())


# 使用 pytest.mark.parametrize 装饰器对测试函数 test_dm_inv 进行参数化测试
@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dm_inv(name, A, A_inv, den):
    # 将 A 转换为字段类型，并在 den 不为 0 的情况下，计算其逆 A_inv，并断言其等于预期的 A_inv
    A = A.to_field()
    if den != 0:
        A_inv = A_inv.to_field() / den
        assert A.inv() == A_inv
    # 否则，预期会抛出 DMNonInvertibleMatrixError 异常
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv())


# 使用 pytest.mark.parametrize 装饰器对测试函数 test_ddm_inv 进行参数化测试
@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_ddm_inv(name, A, A_inv, den):
    # 将 A 转换为字段类型和 DDM 类型，并在 den 不为 0 的情况下，计算其逆 A_inv，并断言其等于预期的 A_inv
    A = A.to_field().to_ddm()
    if den != 0:
        A_inv = (A_inv.to_field() / den).to_ddm()
        assert A.inv() == A_inv
    # 否则，预期会抛出 DMNonInvertibleMatrixError 异常
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv())


# 使用 pytest.mark.parametrize 装饰器对测试函数 test_sdm_inv 进行参数化测试
@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_sdm_inv(name, A, A_inv, den):
    # 将 A 转换为字段类型和 SDM 类型（分数 DDM 类型）进行测试
    A = A.to_field().to_sdm()
    # 如果 den 不等于 0，则进行以下操作
    A_inv = (A_inv.to_field() / den).to_sdm()
    # 断言 A 的逆矩阵等于 A_inv
    assert A.inv() == A_inv
    # 如果 den 等于 0，则抛出 DMNonInvertibleMatrixError 异常，lambda 函数为抛出异常的方式
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv())
@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dense_ddm_iinv(name, A, A_inv, den):
    # 将 A 转换为域矩阵，再转换为密集矩阵，并复制副本
    A = A.to_field().to_ddm().copy()
    # 获取域 K
    K = A.domain
    # 复制 A 到 A_result
    A_result = A.copy()
    # 如果 den 不为零
    if den != 0:
        # 计算 A_inv / den 并转换为域矩阵
        A_inv = (A_inv.to_field() / den).to_ddm()
        # 使用 ddm_iinv 函数计算 A 的逆，并将结果存储在 A_result 中
        ddm_iinv(A_result, A, K)
        # 断言 A_result 等于预期的 A_inv
        assert A_result == A_inv
    else:
        # 如果 den 为零，预期会引发 DMNonInvertibleMatrixError 错误
        raises(DMNonInvertibleMatrixError, lambda: ddm_iinv(A_result, A, K))


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_Matrix_adjugate(name, A, A_inv, den):
    # 将 A 和 A_inv 转换为一般矩阵形式
    A = A.to_Matrix()
    A_inv = A_inv.to_Matrix()
    # 断言 A 的伴随矩阵等于预期的 A_inv
    assert A.adjugate() == A_inv
    # 针对不同的方法（"bareiss", "berkowitz", "bird", "laplace", "lu"），断言 A 的伴随矩阵等于预期的 A_inv
    for method in ["bareiss", "berkowitz", "bird", "laplace", "lu"]:
        assert A.adjugate(method=method) == A_inv


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dm_adj_det(name, A, A_inv, den):
    # 断言 A 的 adj_det 方法返回预期的 (A_inv, den) 元组
    assert A.adj_det() == (A_inv, den)


def test_inverse_inexact():
    # 创建一个具有变量 x 的矩阵 M
    M = Matrix([[x-0.3, -0.06, -0.22],
                [-0.46, x-0.48, -0.41],
                [-0.14, -0.39, x-0.64]])

    # 创建 M 的伴随矩阵 Mn
    Mn = Matrix([[1.0*x**2 - 1.12*x + 0.1473, 0.06*x + 0.0474, 0.22*x - 0.081],
                 [0.46*x - 0.237, 1.0*x**2 - 0.94*x + 0.1612, 0.41*x - 0.0218],
                 [0.14*x + 0.1122, 0.39*x - 0.1086, 1.0*x**2 - 0.78*x + 0.1164]])

    # 创建 M 的分母 d
    d = 1.0*x**3 - 1.42*x**2 + 0.4249*x - 0.0546540000000002

    # 计算 Mi，即 Mn / d
    Mi = Mn / d

    # 将 M 转换为域矩阵，再转换为密集域矩阵
    M_dm = M.to_DM()
    M_dmd = M_dm.to_dense()
    # 获取 M_dm 的分子和分母
    M_dm_num, M_dm_den = M_dm.inv_den()
    M_dmd_num, M_dmd_den = M_dmd.inv_den()

    # 注释：我们不检查 M_dm().to_field().inv()，该方法目前使用除法，并且在取消最大公因数时产生更复杂的结果。
    # DomainMatrix.inv() 在 RR(x) 上应更改为清除分母并使用 DomainMatrix.inv_den()。

    # 创建 Minvs 列表，其中包含不同方法得到的 M 的逆矩阵
    Minvs = [
        M.inv(),
        (M_dm_num.to_field() / M_dm_den).to_Matrix(),
        (M_dmd_num.to_field() / M_dmd_den).to_Matrix(),
        M_dm_num.to_Matrix() / M_dm_den.as_expr(),
        M_dmd_num.to_Matrix() / M_dmd_den.as_expr(),
    ]

    # 断言每个 Minv 中的每个元素与预期的 Mi 中的元素在接近范围内相等
    for Minv in Minvs:
        for Mi1, Mi2 in zip(Minv.flat(), Mi.flat()):
            assert all_close(Mi2, Mi1)
```
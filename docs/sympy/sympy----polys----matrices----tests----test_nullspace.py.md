# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_nullspace.py`

```
from sympy import ZZ, Matrix  # 导入 ZZ 和 Matrix 类
from sympy.polys.matrices import DM, DomainMatrix  # 导入 DM 和 DomainMatrix 类
from sympy.polys.matrices.ddm import DDM  # 导入 DDM 类
from sympy.polys.matrices.sdm import SDM  # 导入 SDM 类

import pytest  # 导入 pytest 测试框架

# 定义一个函数 zeros，返回指定形状和域的零矩阵
zeros = lambda shape, K: DomainMatrix.zeros(shape, K).to_dense()

# 定义一个函数 eye，返回指定大小和域的单位矩阵
eye = lambda n, K: DomainMatrix.eye(n, K).to_dense()

#
# DomainMatrix.nullspace 可以返回一个分割的答案或者一个未分割的非规范化的答案。
# 未规范化的答案不唯一，但可以通过使其原始化（移除最大公约数）使其唯一。
# 这些测试展示了所有的原始形式。我们测试两件事：
#
#   A.nullspace().primitive()[1] == answer.
#   A.nullspace(divide_last=True) == _divide_last(answer).
#
# DomainMatrix 和相关类返回的 nullspace 是 Matrix 返回的 nullspace 的转置。
# Matrix 返回一个列向量列表，而 DomainMatrix 返回其行向量为 nullspace 向量的矩阵。
#

NULLSPACE_EXAMPLES = [

    (
        'zz_1',
        DM([[ 1, 2, 3]], ZZ),  # 第一个例子的系数矩阵 A
        DM([[-2, 1, 0],        # 第一个例子的期望的原始化 nullspace
           [-3, 0, 1]], ZZ),
    ),

    (
        'zz_2',
        zeros((0, 0), ZZ),     # 第二个例子的系数矩阵 A 是零矩阵
        zeros((0, 0), ZZ),     # 第二个例子的期望的原始化 nullspace 是零矩阵
    ),

    (
        'zz_3',
        zeros((2, 0), ZZ),     # 第三个例子的系数矩阵 A 是 2x0 的零矩阵
        zeros((0, 0), ZZ),     # 第三个例子的期望的原始化 nullspace 是零矩阵
    ),

    (
        'zz_4',
        zeros((0, 2), ZZ),     # 第四个例子的系数矩阵 A 是 0x2 的零矩阵
        eye(2, ZZ),            # 第四个例子的期望的原始化 nullspace 是 2x2 的单位矩阵
    ),

    (
        'zz_5',
        zeros((2, 2), ZZ),     # 第五个例子的系数矩阵 A 是 2x2 的零矩阵
        eye(2, ZZ),            # 第五个例子的期望的原始化 nullspace 是 2x2 的单位矩阵
    ),

    (
        'zz_6',
        DM([[1, 2],            # 第六个例子的系数矩阵 A
            [3, 4]], ZZ),
        zeros((0, 2), ZZ),     # 第六个例子的期望的原始化 nullspace 是 0x2 的零矩阵
    ),

    (
        'zz_7',
        DM([[1, 1],            # 第七个例子的系数矩阵 A
            [1, 1]], ZZ),
        DM([[-1, 1]], ZZ),     # 第七个例子的期望的原始化 nullspace
    ),

    (
        'zz_8',
        DM([[1],               # 第八个例子的系数矩阵 A
            [1]], ZZ),
        zeros((0, 1), ZZ),     # 第八个例子的期望的原始化 nullspace 是 0x1 的零矩阵
    ),

    (
        'zz_9',
        DM([[1, 1]], ZZ),      # 第九个例子的系数矩阵 A
        DM([[-1, 1]], ZZ),     # 第九个例子的期望的原始化 nullspace
    ),

    (
        'zz_10',
        DM([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   # 第十个例子的系数矩阵 A
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]], ZZ),
        DM([[ 0,  0, 1,  0,  0, 0, 0, 0, 0, 0],  # 第十个例子的期望的原始化 nullspace
            [-1,  0, 0,  0,  0, 0, 1, 0, 0, 0],
            [ 0, -1, 0,  0,  0, 0, 0, 1, 0, 0],
            [ 0,  0, 0, -1,  0, 0, 0, 0, 1, 0],
            [ 0,  0, 0,  0, -1, 0, 0, 0, 0, 1]], ZZ),
    ),

]


def _to_DM(A, ans):
    """将答案转换为 DomainMatrix 类型."""
    if isinstance(A, DomainMatrix):
        return A.to_dense()
    elif isinstance(A, DDM):
        return DomainMatrix(list(A), A.shape, A.domain).to_dense()
    elif isinstance(A, SDM):
        return DomainMatrix(dict(A), A.shape, A.domain).to_dense()
    else:
        assert False # pragma: no cover


def _divide_last(null):
    """通过最右边的非零元素来规范化 nullspace."""
    null = null.to_field()

    if null.is_zero_matrix:
        return null

    rows = []
    # 遍历二维数组 null 的每一行
    for i in range(null.shape[0]):
        # 反向遍历当前行的每一个元素
        for j in reversed(range(null.shape[1])):
            # 检查当前元素是否为真值（非零）
            if null[i, j]:
                # 将当前行除以当前元素，将结果添加到 rows 列表中
                rows.append(null[i, :] / null[i, j])
                # 跳出内层循环
                break
        else:
            # 如果内层循环未触发 break，说明当前行全为零，引发断言错误
            assert False # pragma: no cover

    # 使用 DomainMatrix.vstack 方法将 rows 列表中的元素垂直堆叠成一个矩阵，并返回该矩阵
    return DomainMatrix.vstack(*rows)
# 检查答案的原始性质是否匹配
def _check_primitive(null, null_ans):
    # 将答案转换为域矩阵对象，以便处理
    null = _to_DM(null, null_ans)
    # 获取域矩阵对象的原始性质
    cont, null_prim = null.primitive()
    # 断言获取的原始性质与预期答案相等
    assert null_prim == null_ans


# 检查分解后的答案
def _check_divided(null, null_ans):
    # 将答案转换为域矩阵对象，以便处理
    null = _to_DM(null, null_ans)
    # 对答案进行最后一步的分解处理
    null_ans_norm = _divide_last(null_ans)
    # 断言处理后的答案与转换后的对象相等
    assert null == null_ans_norm


# 使用 NULLSPACE_EXAMPLES 参数化测试矩阵的零空间
@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_Matrix_nullspace(name, A, A_null):
    # 将矩阵 A 转换为 Matrix 类型
    A = A.to_Matrix()

    # 获取矩阵 A 的零空间列向量
    A_null_cols = A.nullspace()

    # 处理零空间为空的情况
    if A_null_cols:
        # 将零空间列向量水平拼接成新的矩阵 A_null_found
        A_null_found = Matrix.hstack(*A_null_cols)
    else:
        # 创建一个全零矩阵，列数与 A 相同
        A_null_found = Matrix.zeros(A.cols, 0)

    # 将 A_null_found 转换为域矩阵，再转换为域对象，最后转换为密集矩阵
    A_null_found = A_null_found.to_DM().to_field().to_dense()

    # 矩阵结果是 DomainMatrix 结果的转置
    A_null_found = A_null_found.transpose()

    # 检查分解后的结果是否与预期答案匹配
    _check_divided(A_null_found, A_null)


# 使用 NULLSPACE_EXAMPLES 参数化测试域矩阵转换为密集矩阵后的零空间
@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_dense_nullspace(name, A, A_null):
    # 将域矩阵 A 转换为域对象，再转换为密集矩阵
    A = A.to_field().to_dense()
    # 获取密集矩阵 A 的零空间
    A_null_found = A.nullspace(divide_last=True)
    # 检查分解后的结果是否与预期答案匹配
    _check_divided(A_null_found, A_null)


# 使用 NULLSPACE_EXAMPLES 参数化测试域矩阵转换为稀疏矩阵后的零空间
@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_sparse_nullspace(name, A, A_null):
    # 将域矩阵 A 转换为域对象，再转换为稀疏矩阵
    A = A.to_field().to_sparse()
    # 获取稀疏矩阵 A 的零空间
    A_null_found = A.nullspace(divide_last=True)
    # 检查分解后的结果是否与预期答案匹配
    _check_divided(A_null_found, A_null)


# 使用 NULLSPACE_EXAMPLES 参数化测试域矩阵转换为 DDM 后的零空间
@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_ddm_nullspace(name, A, A_null):
    # 将域矩阵 A 转换为域对象，再转换为 DDM
    A = A.to_field().to_ddm()
    # 获取 DDM 对象 A 的零空间
    A_null_found, _ = A.nullspace()
    # 检查分解后的结果是否与预期答案匹配
    _check_divided(A_null_found, A_null)


# 使用 NULLSPACE_EXAMPLES 参数化测试域矩阵转换为 SDM 后的零空间
@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_sdm_nullspace(name, A, A_null):
    # 将域矩阵 A 转换为域对象，再转换为 SDM
    A = A.to_field().to_sdm()
    # 获取 SDM 对象 A 的零空间
    A_null_found, _ = A.nullspace()
    # 检查分解后的结果是否与预期答案匹配
    _check_divided(A_null_found, A_null)


# 使用 NULLSPACE_EXAMPLES 参数化测试密集矩阵零空间的原始性质
@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_dense_nullspace_fracfree(name, A, A_null):
    # 将矩阵 A 转换为密集矩阵
    A = A.to_dense()
    # 获取密集矩阵 A 的零空间
    A_null_found = A.nullspace()
    # 检查答案的原始性质是否匹配
    _check_primitive(A_null_found, A_null)


# 使用 NULLSPACE_EXAMPLES 参数化测试稀疏矩阵零空间的原始性质
@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_sparse_nullspace_fracfree(name, A, A_null):
    # 将矩阵 A 转换为稀疏矩阵
    A = A.to_sparse()
    # 获取稀疏矩阵 A 的零空间
    A_null_found = A.nullspace()
    # 检查答案的原始性质是否匹配
    _check_primitive(A_null_found, A_null)
```
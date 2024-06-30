# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_coo.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_equal  # 导入 NumPy 的断言函数 assert_equal，用于数组比较
import pytest  # 导入 Pytest 测试框架，用于编写和运行测试
from scipy.sparse import coo_array  # 从 SciPy 稀疏矩阵模块中导入 coo_array 函数

# 测试稀疏矩阵的形状构造函数
def test_shape_constructor():
    empty1d = coo_array((3,))  # 创建一个 1 维稀疏矩阵，形状为 (3,)
    assert empty1d.shape == (3,)  # 断言稀疏矩阵的形状为 (3,)
    assert_equal(empty1d.toarray(), np.zeros((3,)))  # 断言稀疏矩阵转换为普通数组后全为零

    empty2d = coo_array((3, 2))  # 创建一个 2 维稀疏矩阵，形状为 (3, 2)
    assert empty2d.shape == (3, 2)  # 断言稀疏矩阵的形状为 (3, 2)
    assert_equal(empty2d.toarray(), np.zeros((3, 2)))  # 断言稀疏矩阵转换为普通数组后全为零

    with pytest.raises(TypeError, match='invalid input format'):  # 断言抛出 TypeError 异常，匹配错误信息 'invalid input format'
        coo_array((3, 2, 2))  # 使用无效的输入格式创建稀疏矩阵


# 测试稀疏矩阵的密集数组构造函数
def test_dense_constructor():
    res1d = coo_array([1, 2, 3])  # 创建一个 1 维稀疏矩阵，数据为 [1, 2, 3]
    assert res1d.shape == (3,)  # 断言稀疏矩阵的形状为 (3,)
    assert_equal(res1d.toarray(), np.array([1, 2, 3]))  # 断言稀疏矩阵转换为普通数组后与给定数组相等

    res2d = coo_array([[1, 2, 3], [4, 5, 6]])  # 创建一个 2 维稀疏矩阵，数据为 [[1, 2, 3], [4, 5, 6]]
    assert res2d.shape == (2, 3)  # 断言稀疏矩阵的形状为 (2, 3)
    assert_equal(res2d.toarray(), np.array([[1, 2, 3], [4, 5, 6]]))  # 断言稀疏矩阵转换为普通数组后与给定数组相等

    with pytest.raises(ValueError, match='shape must be a 1- or 2-tuple'):  # 断言抛出 ValueError 异常，匹配错误信息 'shape must be a 1- or 2-tuple'
        coo_array([[[3]], [[4]]])  # 使用无效形状创建稀疏矩阵


# 测试带形状参数的稀疏矩阵的密集数组构造函数
def test_dense_constructor_with_shape():
    res1d = coo_array([1, 2, 3], shape=(3,))  # 创建一个 1 维稀疏矩阵，数据为 [1, 2, 3]，形状为 (3,)
    assert res1d.shape == (3,)  # 断言稀疏矩阵的形状为 (3,)
    assert_equal(res1d.toarray(), np.array([1, 2, 3]))  # 断言稀疏矩阵转换为普通数组后与给定数组相等

    res2d = coo_array([[1, 2, 3], [4, 5, 6]], shape=(2, 3))  # 创建一个 2 维稀疏矩阵，数据为 [[1, 2, 3], [4, 5, 6]]，形状为 (2, 3)
    assert res2d.shape == (2, 3)  # 断言稀疏矩阵的形状为 (2, 3)
    assert_equal(res2d.toarray(), np.array([[1, 2, 3], [4, 5, 6]]))  # 断言稀疏矩阵转换为普通数组后与给定数组相等

    with pytest.raises(ValueError, match='shape must be a 1- or 2-tuple'):  # 断言抛出 ValueError 异常，匹配错误信息 'shape must be a 1- or 2-tuple'
        coo_array([[[3]], [[4]]], shape=(2, 1, 1))  # 使用无效形状创建稀疏矩阵


# 测试不一致形状的稀疏矩阵构造函数
def test_dense_constructor_with_inconsistent_shape():
    with pytest.raises(ValueError, match='inconsistent shapes'):  # 断言抛出 ValueError 异常，匹配错误信息 'inconsistent shapes'
        coo_array([1, 2, 3], shape=(4,))  # 使用不一致形状创建稀疏矩阵

    with pytest.raises(ValueError, match='inconsistent shapes'):  # 断言抛出 ValueError 异常，匹配错误信息 'inconsistent shapes'
        coo_array([1, 2, 3], shape=(3, 1))  # 使用不一致形状创建稀疏矩阵

    with pytest.raises(ValueError, match='inconsistent shapes'):  # 断言抛出 ValueError 异常，匹配错误信息 'inconsistent shapes'
        coo_array([[1, 2, 3]], shape=(3,))  # 使用不一致形状创建稀疏矩阵

    with pytest.raises(ValueError, match='axis 0 index 2 exceeds matrix dimension 2'):  # 断言抛出 ValueError 异常，匹配错误信息 'axis 0 index 2 exceeds matrix dimension 2'
        coo_array(([1], ([2],)), shape=(2,))  # 使用超出维度的索引创建稀疏矩阵

    with pytest.raises(ValueError, match='negative axis 0 index: -1'):  # 断言抛出 ValueError 异常，匹配错误信息 'negative axis 0 index: -1'
        coo_array(([1], ([-1],)))  # 使用负索引创建稀疏矩阵


# 测试 1 维稀疏矩阵的稀疏数组构造函数
def test_1d_sparse_constructor():
    empty1d = coo_array((3,))  # 创建一个 1 维稀疏矩阵，形状为 (3,)
    res = coo_array(empty1d)  # 使用另一个稀疏矩阵创建新的稀疏矩阵
    assert res.shape == (3,)  # 断言稀疏矩阵的形状为 (3,)
    assert_equal(res.toarray(), np.zeros((3,)))  # 断言稀疏矩阵转换为普通数组后全为零


# 测试带元组的 1 维稀疏矩阵的稀疏数组构造函数
def test_1d_tuple_constructor():
    res = coo_array(([9,8], ([1,2],)))  # 创建一个 1 维稀疏矩阵，数据为 [9, 8]，索引为 [1, 2]
    assert res.shape == (3,)  # 断言稀疏矩阵的形状为 (3,)
    # 断言列向量的稀疏表示是否等于指定的数组
    assert_equal(col_vec.toarray(), np.array([[1], [0], [3]]))
    
    # 将一维数组重塑为行向量，确保形状为 (1, 3)
    row_vec = arr1d.reshape((1, 3))
    assert row_vec.shape == (1, 3)
    # 断言行向量的稀疏表示是否等于指定的数组
    assert_equal(row_vec.toarray(), np.array([[1, 0, 3]]))
    
    # 创建一个 COO 格式的二维数组对象，包含指定的数据
    arr2d = coo_array([[1, 2, 0], [0, 0, 3]])
    # 断言数组的形状是否为 (2, 3)
    assert arr2d.shape == (2, 3)
    
    # 将二维数组展开为一维数组，确保形状为 (6,)
    flat = arr2d.reshape((6,))
    # 断言展开后的数组形状是否为 (6,)
    assert flat.shape == (6,)
    # 断言展开后的稀疏表示是否等于指定的数组
    assert_equal(flat.toarray(), np.array([1, 2, 0, 0, 0, 3]))
# 定义一个测试函数，用于验证 COO 数组的非零元素数量属性
def test_nnz():
    # 创建一个一维 COO 数组 [1, 0, 3]
    arr1d = coo_array([1, 0, 3])
    # 断言数组形状为 (3,)
    assert arr1d.shape == (3,)
    # 断言数组的非零元素数量为 2
    assert arr1d.nnz == 2

    # 创建一个二维 COO 数组 [[1, 2, 0], [0, 0, 3]]
    arr2d = coo_array([[1, 2, 0], [0, 0, 3]])
    # 断言数组形状为 (2, 3)
    assert arr2d.shape == (2, 3)
    # 断言数组的非零元素数量为 3
    assert arr2d.nnz == 3


# 定义一个测试函数，用于验证 COO 数组的转置操作
def test_transpose():
    # 创建一个一维 COO 数组 [1, 0, 3] 并进行转置
    arr1d = coo_array([1, 0, 3]).T
    # 断言数组形状为 (3,)
    assert arr1d.shape == (3,)
    # 断言转置后的数组内容与预期一致
    assert_equal(arr1d.toarray(), np.array([1, 0, 3]))

    # 创建一个二维 COO 数组 [[1, 2, 0], [0, 0, 3]] 并进行转置
    arr2d = coo_array([[1, 2, 0], [0, 0, 3]]).T
    # 断言数组形状为 (3, 2)
    assert arr2d.shape == (3, 2)
    # 断言转置后的数组内容与预期一致
    assert_equal(arr2d.toarray(), np.array([[1, 0], [2, 0], [0, 3]]))


# 定义一个测试函数，用于验证带有轴参数的 COO 数组转置操作
def test_transpose_with_axis():
    # 创建一个一维 COO 数组 [1, 0, 3] 并根据指定的轴进行转置
    arr1d = coo_array([1, 0, 3]).transpose(axes=(0,))
    # 断言数组形状为 (3,)
    assert arr1d.shape == (3,)
    # 断言转置后的数组内容与预期一致
    assert_equal(arr1d.toarray(), np.array([1, 0, 3]))

    # 创建一个二维 COO 数组 [[1, 2, 0], [0, 0, 3]] 并根据指定的轴进行转置
    arr2d = coo_array([[1, 2, 0], [0, 0, 3]]).transpose(axes=(0, 1))
    # 断言数组形状为 (2, 3)
    assert arr2d.shape == (2, 3)
    # 断言转置后的数组内容与预期一致
    assert_equal(arr2d.toarray(), np.array([[1, 2, 0], [0, 0, 3]]))

    # 测试当指定的轴与数组维度不匹配时是否抛出 ValueError 异常
    with pytest.raises(ValueError, match="axes don't match matrix dimensions"):
        coo_array([1, 0, 3]).transpose(axes=(0, 1))

    # 测试当指定的轴中存在重复时是否抛出 ValueError 异常
    with pytest.raises(ValueError, match="repeated axis in transpose"):
        coo_array([[1, 2, 0], [0, 0, 3]]).transpose(axes=(1, 1))


# 定义一个测试函数，用于验证一维 COO 数组的行和列属性
def test_1d_row_and_col():
    # 创建一个一维 COO 数组 [1, -2, -3]
    res = coo_array([1, -2, -3])
    # 断言列索引数组与预期一致
    assert_equal(res.col, np.array([0, 1, 2]))
    # 断言行索引数组为空，即全为零
    assert_equal(res.row, np.zeros_like(res.col))
    # 断言行索引数组的数据类型与列索引数组一致
    assert res.row.dtype == res.col.dtype
    # 断言行索引数组不可写入
    assert res.row.flags.writeable is False

    # 尝试修改列索引数组，预期抛出 ValueError 异常
    res.col = [1, 2, 3]
    assert len(res.coords) == 1
    assert_equal(res.col, np.array([1, 2, 3]))
    assert res.row.dtype == res.col.dtype

    with pytest.raises(ValueError, match="cannot set row attribute"):
        res.row = [1, 2, 3]


# 定义一个测试函数，用于验证一维 COO 数组转换到其他格式的操作
def test_1d_toformats():
    # 创建一个一维 COO 数组 [1, -2, -3]
    res = coo_array([1, -2, -3])
    # 遍历不支持的转换函数，预期抛出 ValueError 异常
    for f in [res.tobsr, res.tocsc, res.todia, res.tolil]:
        with pytest.raises(ValueError, match='Cannot convert'):
            f()
    # 遍历支持的转换函数，验证转换后的数组内容与原数组一致
    for f in [res.tocoo, res.tocsr, res.todok]:
        assert_equal(f().toarray(), res.toarray())


# 定义一个测试函数，用于验证一维 COO 数组的调整大小操作
@pytest.mark.parametrize('arg', [1, 2, 4, 5, 8])
def test_1d_resize(arg: int):
    # 创建一个一维 COO 数组 [1, -2, -3]
    den = np.array([1, -2, -3])
    res = coo_array(den)
    # 修改底层数组的大小
    den.resize(arg, refcheck=False)
    # 调整 COO 数组的大小
    res.resize(arg)
    # 断言调整后数组的形状与底层数组一致
    assert res.shape == den.shape
    # 断言调整后数组的内容与底层数组一致
    assert_equal(res.toarray(), den)


# 定义一个测试函数，用于验证一维 COO 数组转换到二维并调整大小的操作
@pytest.mark.parametrize('arg', zip([1, 2, 3, 4], [1, 2, 3, 4]))
def test_1d_to_2d_resize(arg: tuple[int, int]):
    # 创建一个一维 COO 数组 [1, 0, 3]
    den = np.array([1, 0, 3])
    res = coo_array(den)

    # 修改底层数组的大小
    den.resize(arg, refcheck=False)
    # 调整 COO 数组的大小
    res.resize(arg)
    # 断言调整后数组的形状与底层数组一致
    assert res.shape == den.shape
    # 断言调整后数组的内容与底层数组一致
    assert_equal(res.toarray(), den)


# 定义一个测试函数，用于验证二维 COO 数组转换到一维并调整大小的操作
@pytest.mark.parametrize('arg', [1, 4, 6, 8])
def test_2d_to_1d_resize(arg: int):
    # 创建一个二维 COO 数组 [[1, 0, 3], [4, 0, 0]]
    den = np.array([[1, 0, 3], [4, 0, 0]])
    res = coo_array(den)
    # 修改底层数组的大小
    den.resize(arg, refcheck=False)
    # 调整 COO 数组的大小
    res.resize(arg)
    # 断言调整后数组的形状与底层数组一致
    assert res.shape == den.shape
    # 断言调整后数组的内容与底层数组
    # 使用断言检查 arr1d 的稀疏矩阵表示是否与给定的数组 [2, 4] 相等
    assert_equal(arr1d.toarray(), np.array([2, 4]))
# 测试消除稀疏数组中的零元素
def test_eliminate_zeros():
    # 创建一个 COO 格式的稀疏数组，包含值 [0, 0, 1]，并指定行索引为 [1, 0, 1]
    arr1d = coo_array(([0, 0, 1], ([1, 0, 1],)))
    # 断言非零元素的数量为 3
    assert arr1d.nnz == 3
    # 断言计算非零元素的数量为 1
    assert arr1d.count_nonzero() == 1
    # 断言将稀疏数组转换为密集数组后的结果为 [0, 1]
    assert_equal(arr1d.toarray(), np.array([0, 1]))
    # 调用消除零元素的方法
    arr1d.eliminate_zeros()
    # 再次断言非零元素的数量为 1
    assert arr1d.nnz == 1
    # 再次断言计算非零元素的数量为 1
    assert arr1d.count_nonzero() == 1
    # 再次断言将稀疏数组转换为密集数组后的结果为 [0, 1]
    assert_equal(arr1d.toarray(), np.array([0, 1]))
    # 断言稀疏数组的列索引为 [1]
    assert_equal(arr1d.col, np.array([1]))
    # 断言稀疏数组的行索引为 [0]
    assert_equal(arr1d.row, np.array([0]))


# 测试稀疏数组与密集数组的一维加法
def test_1d_add_dense():
    # 创建两个密集数组 den_a 和 den_b
    den_a = np.array([0, -2, -3, 0])
    den_b = np.array([0, 1, 2, 3])
    # 计算 den_a 和 den_b 的加法结果
    exp = den_a + den_b
    # 将 den_a 转换为 COO 格式的稀疏数组，并与 den_b 执行加法
    res = coo_array(den_a) + den_b
    # 断言结果的类型与预期相同
    assert type(res) == type(exp)
    # 断言结果与预期相等
    assert_equal(res, exp)


# 测试稀疏数组与稀疏数组的一维加法
def test_1d_add_sparse():
    # 创建两个密集数组 den_a 和 den_b
    den_a = np.array([0, -2, -3, 0])
    den_b = np.array([0, 1, 2, 3])
    # 计算 den_a 和 den_b 的加法结果（密集数组）
    dense_sum = den_a + den_b
    # 将 den_a 和 den_b 转换为 COO 格式的稀疏数组，并执行稀疏矩阵加法
    sparse_sum = coo_array(den_a) + coo_array(den_b)
    # 断言稠密和稀疏矩阵加法结果相等
    assert_equal(dense_sum, sparse_sum.toarray())


# 测试稀疏数组与一维向量的矩阵乘法
def test_1d_matmul_vector():
    # 创建密集数组 den_a 和 den_b
    den_a = np.array([0, -2, -3, 0])
    den_b = np.array([0, 1, 2, 3])
    # 计算 den_a 和 den_b 的矩阵乘法结果
    exp = den_a @ den_b
    # 将 den_a 转换为 COO 格式的稀疏数组，并与 den_b 执行矩阵乘法
    res = coo_array(den_a) @ den_b
    # 断言结果的维度为 0
    assert np.ndim(res) == 0
    # 断言结果与预期相等
    assert_equal(res, exp)


# 测试稀疏数组与多个向量的矩阵乘法
def test_1d_matmul_multivector():
    # 创建密集数组 den 和 other
    den = np.array([0, -2, -3, 0])
    other = np.array([[0, 1, 2, 3], [3, 2, 1, 0]]).T
    # 计算 den 与 other 的矩阵乘法结果
    exp = den @ other
    # 将 den 转换为 COO 格式的稀疏数组，并与 other 执行矩阵乘法
    res = coo_array(den) @ other
    # 断言结果的类型与预期相同
    assert type(res) == type(exp)
    # 断言结果与预期相等
    assert_equal(res, exp)


# 测试稀疏数组的二维矩阵乘法
def test_2d_matmul_multivector():
    # 创建二维密集数组 den
    den = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    # 将 den 转换为 COO 格式的稀疏数组
    arr2d = coo_array(den)
    # 计算 den 与其转置矩阵的矩阵乘法结果
    exp = den @ den.T
    # 计算 arr2d 与其转置矩阵的矩阵乘法结果
    res = arr2d @ arr2d.T
    # 断言结果与预期相等
    assert_equal(res.toarray(), exp)


# 测试稀疏数组的一维对角线提取
def test_1d_diagonal():
    # 创建密集数组 den
    den = np.array([0, -2, -3, 0])
    # 将 den 转换为 COO 格式的稀疏数组，并尝试提取对角线
    with pytest.raises(ValueError, match='diagonal requires two dimensions'):
        coo_array(den).diagonal()
```
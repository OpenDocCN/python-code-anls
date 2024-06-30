# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_csr.py`

```
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_  # 导入需要的模块和函数
from scipy.sparse import csr_matrix, csc_matrix, csr_array, csc_array, hstack  # 导入稀疏矩阵相关模块和函数
from scipy import sparse  # 导入稀疏矩阵的主模块
import pytest  # 导入 pytest 模块


def _check_csr_rowslice(i, sl, X, Xcsr):
    np_slice = X[i, sl]  # 从密集矩阵 X 中提取指定行和切片的数据
    csr_slice = Xcsr[i, sl]  # 从 CSR 稀疏矩阵 Xcsr 中提取指定行和切片的数据
    assert_array_almost_equal(np_slice, csr_slice.toarray()[0])  # 断言：比较两个切片的数据是否几乎相等
    assert_(type(csr_slice) is csr_matrix)  # 断言：验证切片的类型是否为 csr_matrix


def test_csr_rowslice():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))  # 创建一个随机的 N × N 密集矩阵
    X[X > 0.7] = 0  # 将大于 0.7 的元素置为 0，稀疏化处理
    Xcsr = csr_matrix(X)  # 将密集矩阵转换为 CSR 稀疏矩阵

    slices = [slice(None, None, None),  # 切片列表：完整切片
              slice(None, None, -1),    # 切片列表：逆序切片
              slice(1, -2, 2),           # 切片列表：步长为 2 的切片
              slice(-2, 1, -2)]          # 切片列表：逆序步长为 -2 的切片

    for i in range(N):  # 遍历矩阵的每一行
        for sl in slices:  # 遍历切片列表
            _check_csr_rowslice(i, sl, X, Xcsr)  # 调用函数验证 CSR 行切片的功能


def test_csr_getrow():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))  # 创建一个随机的 N × N 密集矩阵
    X[X > 0.7] = 0  # 将大于 0.7 的元素置为 0，稀疏化处理
    Xcsr = csr_matrix(X)  # 将密集矩阵转换为 CSR 稀疏矩阵

    for i in range(N):  # 遍历矩阵的每一行
        arr_row = X[i:i + 1, :]  # 获取密集矩阵中的指定行
        csr_row = Xcsr.getrow(i)  # 获取 CSR 稀疏矩阵中的指定行

        assert_array_almost_equal(arr_row, csr_row.toarray())  # 断言：比较两行数据是否几乎相等
        assert_(type(csr_row) is csr_matrix)  # 断言：验证行的类型是否为 csr_matrix


def test_csr_getcol():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))  # 创建一个随机的 N × N 密集矩阵
    X[X > 0.7] = 0  # 将大于 0.7 的元素置为 0，稀疏化处理
    Xcsr = csr_matrix(X)  # 将密集矩阵转换为 CSR 稀疏矩阵

    for i in range(N):  # 遍历矩阵的每一列
        arr_col = X[:, i:i + 1]  # 获取密集矩阵中的指定列
        csr_col = Xcsr.getcol(i)  # 获取 CSR 稀疏矩阵中的指定列

        assert_array_almost_equal(arr_col, csr_col.toarray())  # 断言：比较两列数据是否几乎相等
        assert_(type(csr_col) is csr_matrix)  # 断言：验证列的类型是否为 csr_matrix


@pytest.mark.parametrize("matrix_input, axis, expected_shape",
    [(csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      0, (0, 4)),
     (csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      1, (3, 0)),
     (csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      'both', (0, 0)),
     (csr_matrix([[0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 3, 0]]),
      0, (0, 5))])
def test_csr_empty_slices(matrix_input, axis, expected_shape):
    # see gh-11127 for related discussion
    slice_1 = matrix_input.toarray().shape[0] - 1  # 计算切片的维度大小
    slice_2 = slice_1  # 复制切片维度大小
    slice_3 = slice_2 - 1  # 计算另一个切片的维度大小

    if axis == 0:  # 如果是按行切片
        actual_shape_1 = matrix_input[slice_1:slice_2, :].toarray().shape  # 获取切片后的数组形状
        actual_shape_2 = matrix_input[slice_1:slice_3, :].toarray().shape  # 获取另一个切片后的数组形状
    elif axis == 1:  # 如果是按列切片
        actual_shape_1 = matrix_input[:, slice_1:slice_2].toarray().shape  # 获取切片后的数组形状
        actual_shape_2 = matrix_input[:, slice_1:slice_3].toarray().shape  # 获取另一个切片后的数组形状
    elif axis == 'both':  # 如果是同时按行列切片
        actual_shape_1 = matrix_input[slice_1:slice_2, slice_1:slice_2].toarray().shape  # 获取切片后的数组形状
        actual_shape_2 = matrix_input[slice_1:slice_3, slice_1:slice_3].toarray().shape  # 获取另一个切片后的数组形状

    assert actual_shape_1 == expected_shape  # 断言：验证实际形状与期望形状是否相等
    assert actual_shape_1 == actual_shape_2  # 断言：验证两次切片的形状是否相等


def test_csr_bool_indexing():
    data = csr_matrix([[0, 1, 2], [3, 4, 5], [6, 7, 8]])  # 创建 CSR 稀疏矩阵
    list_indices1 = [False, True, False]  # 布尔索引列表
    array_indices1 = np.array(list_indices1)  # 将列表转换为 NumPy 数组
    # 创建一个包含布尔值的二维列表，表示要从数据中选择的切片索引
    list_indices2 = [[False, True, False], [False, True, False]]
    # 将二维列表转换为 NumPy 数组
    array_indices2 = np.array(list_indices2)
    # 创建一个包含布尔值的元组，表示另一个切片索引方式
    list_indices3 = ([False, True, False], [False, True, False])
    # 分别将元组中的每个列表转换为 NumPy 数组
    array_indices3 = (np.array(list_indices3[0]), np.array(list_indices3[1]))
    # 使用布尔列表 list_indices1 对数据进行切片并转换为稀疏矩阵表示
    slice_list1 = data[list_indices1].toarray()
    # 使用布尔数组 array_indices1 对数据进行切片并转换为稀疏矩阵表示
    slice_array1 = data[array_indices1].toarray()
    # 使用布尔列表 list_indices2 对数据进行切片
    slice_list2 = data[list_indices2]
    # 使用布尔数组 array_indices2 对数据进行切片
    slice_array2 = data[array_indices2]
    # 使用布尔列表 list_indices3 对数据进行切片
    slice_list3 = data[list_indices3]
    # 使用布尔数组 array_indices3 对数据进行切片
    slice_array3 = data[array_indices3]
    # 检查切片后的稀疏矩阵和数组切片是否完全相等
    assert (slice_list1 == slice_array1).all()
    assert (slice_list2 == slice_array2).all()
    assert (slice_list3 == slice_array3).all()
def test_csr_hstack_int64():
    """
    Tests if hstack properly promotes to indices and indptr arrays to np.int64
    when using np.int32 during concatenation would result in either array
    overflowing.
    """
    # 获取 np.int32 的最大值
    max_int32 = np.iinfo(np.int32).max

    # First case: indices would overflow with int32
    # 定义稀疏矩阵的数据和行索引
    data = [1.0]
    row = [0]

    # 计算最大索引值，确保在 int32 范围内
    max_indices_1 = max_int32 - 1
    max_indices_2 = 3

    # Individual indices arrays are representable with int32
    # 定义列索引，确保在 int32 范围内
    col_1 = [max_indices_1 - 1]
    col_2 = [max_indices_2 - 1]

    # 创建第一个稀疏矩阵 X_1 和 X_2
    X_1 = csr_matrix((data, (row, col_1)))
    X_2 = csr_matrix((data, (row, col_2)))

    # 断言确保索引和偏移数组的数据类型为 np.int32
    assert max(max_indices_1 - 1, max_indices_2 - 1) < max_int32
    assert X_1.indices.dtype == X_1.indptr.dtype == np.int32
    assert X_2.indices.dtype == X_2.indptr.dtype == np.int32

    # ... but when concatenating their CSR matrices, the resulting indices
    # array can't be represented with int32 and must be promoted to int64.
    # 合并两个稀疏矩阵 X_1 和 X_2，确保结果的索引数组为 np.int64
    X_hs = hstack([X_1, X_2], format="csr")

    assert X_hs.indices.max() == max_indices_1 + max_indices_2 - 1
    assert max_indices_1 + max_indices_2 - 1 > max_int32
    assert X_hs.indices.dtype == X_hs.indptr.dtype == np.int64

    # Even if the matrices are empty, we must account for their size
    # contribution so that we may safely set the final elements.
    # 创建空的稀疏矩阵 X_1_empty 和 X_2_empty，并合并它们，确保结果的索引数组为 np.int64
    X_1_empty = csr_matrix(X_1.shape)
    X_2_empty = csr_matrix(X_2.shape)
    X_hs_empty = hstack([X_1_empty, X_2_empty], format="csr")

    assert X_hs_empty.shape == X_hs.shape
    assert X_hs_empty.indices.dtype == np.int64

    # Should be just small enough to stay in int32 after stack. Note that
    # we theoretically could support indices.max() == max_int32, but due to an
    # edge-case in the underlying sparsetools code
    # (namely the `coo_tocsr` routine),
    # we require that max(X_hs_32.shape) < max_int32 as well.
    # Hence we can only support max_int32 - 1.
    # 创建第三个稀疏矩阵 X_3，确保其索引数组为 np.int32
    col_3 = [max_int32 - max_indices_1 - 1]
    X_3 = csr_matrix((data, (row, col_3)))
    X_hs_32 = hstack([X_1, X_3], format="csr")
    assert X_hs_32.indices.dtype == np.int32
    assert X_hs_32.indices.max() == max_int32 - 1

@pytest.mark.parametrize("cls", [csr_matrix, csr_array, csc_matrix, csc_array])
def test_mixed_index_dtype_int_indexing(cls):
    # https://github.com/scipy/scipy/issues/20182
    # 使用默认的随机数生成器创建稀疏矩阵 base_mtx
    rng = np.random.default_rng(0)
    base_mtx = cls(sparse.random(50, 50, random_state=rng, density=0.1))
    
    # 创建 int64 类型的 indptr 和 indices
    indptr_64bit = base_mtx.copy()
    indices_64bit = base_mtx.copy()
    indptr_64bit.indptr = base_mtx.indptr.astype(np.int64)
    indices_64bit.indices = base_mtx.indices.astype(np.int64)

    # 遍历稀疏矩阵，确保切片索引的结果与 base_mtx 相同
    for mtx in [base_mtx, indptr_64bit, indices_64bit]:
        np.testing.assert_array_equal(
            mtx[[1,2], :].toarray(),
            base_mtx[[1, 2], :].toarray()
        )
        np.testing.assert_array_equal(
            mtx[:, [1, 2]].toarray(),
            base_mtx[:, [1, 2]].toarray()
        )
```
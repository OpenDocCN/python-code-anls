# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_seq_dataset.py`

```
# 导入必要的模块和库
from itertools import product  # 导入 itertools 中的 product 函数

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 pytest 库用于单元测试
from numpy.testing import assert_array_equal  # 导入 NumPy 测试模块中的数组相等断言方法

from sklearn.datasets import load_iris  # 导入加载鸢尾花数据集的函数
from sklearn.utils._seq_dataset import (  # 导入序列数据集相关的类
    ArrayDataset32, ArrayDataset64, CSRDataset32, CSRDataset64,
)
from sklearn.utils._testing import assert_allclose  # 导入测试模块中的近似相等断言方法
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入稀疏矩阵容器修复方法

# 加载鸢尾花数据集
iris = load_iris()

# 将数据集转换为 float64 类型
X64 = iris.data.astype(np.float64)
y64 = iris.target.astype(np.float64)
sample_weight64 = np.arange(y64.size, dtype=np.float64)

# 将数据集转换为 float32 类型
X32 = iris.data.astype(np.float32)
y32 = iris.target.astype(np.float32)
sample_weight32 = np.arange(y32.size, dtype=np.float32)

# 浮点数类型的列表
floating = [np.float32, np.float64]

# 检查 CSR 矩阵的值是否相等的辅助函数
def assert_csr_equal_values(current, expected):
    current.eliminate_zeros()  # 移除当前 CSR 矩阵中的零元素
    expected.eliminate_zeros()  # 移除期望 CSR 矩阵中的零元素
    expected = expected.astype(current.dtype)  # 将期望的 CSR 矩阵转换为与当前相同的数据类型
    assert current.shape[0] == expected.shape[0]  # 断言当前 CSR 矩阵的行数与期望的相同
    assert current.shape[1] == expected.shape[1]  # 断言当前 CSR 矩阵的列数与期望的相同
    assert_array_equal(current.data, expected.data)  # 断言当前 CSR 矩阵的数据与期望的相同
    assert_array_equal(current.indices, expected.indices)  # 断言当前 CSR 矩阵的索引与期望的相同
    assert_array_equal(current.indptr, expected.indptr)  # 断言当前 CSR 矩阵的指针与期望的相同

# 创建稠密数据集的辅助函数，根据浮点数类型选择相应的数据集类
def _make_dense_dataset(float_dtype):
    if float_dtype == np.float32:
        return ArrayDataset32(X32, y32, sample_weight32, seed=42)
    return ArrayDataset64(X64, y64, sample_weight64, seed=42)

# 创建稀疏数据集的辅助函数，根据 CSR 矩阵容器和浮点数类型选择相应的数据集类
def _make_sparse_dataset(csr_container, float_dtype):
    if float_dtype == np.float32:
        X, y, sample_weight, csr_dataset = X32, y32, sample_weight32, CSRDataset32
    else:
        X, y, sample_weight, csr_dataset = X64, y64, sample_weight64, CSRDataset64
    X = csr_container(X)  # 使用指定的 CSR 矩阵容器处理输入数据
    return csr_dataset(X.data, X.indptr, X.indices, y, sample_weight, seed=42)

# 创建所有稠密数据集的辅助函数
def _make_dense_datasets():
    return [_make_dense_dataset(float_dtype) for float_dtype in floating]

# 创建所有稀疏数据集的辅助函数
def _make_sparse_datasets():
    return [
        _make_sparse_dataset(csr_container, float_dtype)
        for csr_container, float_dtype in product(CSR_CONTAINERS, floating)
    ]

# 创建融合类型数据集的辅助函数，将稠密和稀疏数据集分组为 (float32, float64) 元组
def _make_fused_types_datasets():
    all_datasets = _make_dense_datasets() + _make_sparse_datasets()
    return (all_datasets[idx : idx + 2] for idx in range(0, len(all_datasets), 2))

# 使用 pytest 标记和参数化，测试稠密和稀疏数据集的基本迭代
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("dataset", _make_dense_datasets() + _make_sparse_datasets())
def test_seq_dataset_basic_iteration(dataset, csr_container):
    NUMBER_OF_RUNS = 5  # 设置迭代运行的次数
    X_csr64 = csr_container(X64)  # 使用 CSR 矩阵容器处理 float64 类型的输入数据
    # 循环运行指定次数（NUMBER_OF_RUNS）
    for _ in range(NUMBER_OF_RUNS):
        # 从数据集中获取下一个样本数据
        xi_, yi, swi, idx = dataset._next_py()
        # 将 xi_ 转换为 CSR 格式的稀疏矩阵，并设定形状为 (1, X64.shape[1])
        xi = csr_container(xi_, shape=(1, X64.shape[1]))

        # 断言 xi 的值与 X_csr64[[idx]] 的值相等
        assert_csr_equal_values(xi, X_csr64[[idx]])
        # 断言 yi 的值与 y64[idx] 的值相等
        assert yi == y64[idx]
        # 断言 swi 的值与 sample_weight64[idx] 的值相等
        assert swi == sample_weight64[idx]

        # 从数据集中随机获取一个样本数据
        xi_, yi, swi, idx = dataset._random_py()
        # 将 xi_ 转换为 CSR 格式的稀疏矩阵，并设定形状为 (1, X64.shape[1])
        xi = csr_container(xi_, shape=(1, X64.shape[1]))

        # 断言 xi 的值与 X_csr64[[idx]] 的值相等
        assert_csr_equal_values(xi, X_csr64[[idx]])
        # 断言 yi 的值与 y64[idx] 的值相等
        assert yi == y64[idx]
        # 断言 swi 的值与 sample_weight64[idx] 的值相等
        assert swi == sample_weight64[idx]
@pytest.mark.parametrize(
    "dense_dataset,sparse_dataset",
    [  # 使用 pytest 的 parametrize 标记来定义测试参数化
        (
            _make_dense_dataset(float_dtype),  # 调用函数 _make_dense_dataset，传入 float_dtype 参数
            _make_sparse_dataset(csr_container, float_dtype),  # 调用函数 _make_sparse_dataset，传入 csr_container 和 float_dtype 参数
        )
        for float_dtype, csr_container in product(floating, CSR_CONTAINERS)  # 使用 product 函数生成参数组合
    ],
)
def test_seq_dataset_shuffle(dense_dataset, sparse_dataset):
    # not shuffled  # 测试未经过洗牌的情况
    for i in range(5):  # 循环5次
        _, _, _, idx1 = dense_dataset._next_py()  # 调用 dense_dataset 的 _next_py 方法，获取返回值的第四个元素到 idx1
        _, _, _, idx2 = sparse_dataset._next_py()  # 调用 sparse_dataset 的 _next_py 方法，获取返回值的第四个元素到 idx2
        assert idx1 == i  # 断言 idx1 等于 i
        assert idx2 == i  # 断言 idx2 等于 i

    for i in [132, 50, 9, 18, 58]:  # 遍历指定的索引列表
        _, _, _, idx1 = dense_dataset._random_py()  # 调用 dense_dataset 的 _random_py 方法，获取返回值的第四个元素到 idx1
        _, _, _, idx2 = sparse_dataset._random_py()  # 调用 sparse_dataset 的 _random_py 方法，获取返回值的第四个元素到 idx2
        assert idx1 == i  # 断言 idx1 等于 i
        assert idx2 == i  # 断言 idx2 等于 i

    seed = 77  # 设置随机种子为 77
    dense_dataset._shuffle_py(seed)  # 调用 dense_dataset 的 _shuffle_py 方法，使用指定的种子进行洗牌操作
    sparse_dataset._shuffle_py(seed)  # 调用 sparse_dataset 的 _shuffle_py 方法，使用指定的种子进行洗牌操作

    idx_next = [63, 91, 148, 87, 29]  # 设置下一个索引列表
    idx_shuffle = [137, 125, 56, 121, 127]  # 设置洗牌后的索引列表
    for i, j in zip(idx_next, idx_shuffle):  # 使用 zip 遍历 idx_next 和 idx_shuffle
        _, _, _, idx1 = dense_dataset._next_py()  # 调用 dense_dataset 的 _next_py 方法，获取返回值的第四个元素到 idx1
        _, _, _, idx2 = sparse_dataset._next_py()  # 调用 sparse_dataset 的 _next_py 方法，获取返回值的第四个元素到 idx2
        assert idx1 == i  # 断言 idx1 等于 i
        assert idx2 == i  # 断言 idx2 等于 i

        _, _, _, idx1 = dense_dataset._random_py()  # 调用 dense_dataset 的 _random_py 方法，获取返回值的第四个元素到 idx1
        _, _, _, idx2 = sparse_dataset._random_py()  # 调用 sparse_dataset 的 _random_py 方法，获取返回值的第四个元素到 idx2
        assert idx1 == j  # 断言 idx1 等于 j
        assert idx2 == j  # 断言 idx2 等于 j


@pytest.mark.parametrize("dataset_32,dataset_64", _make_fused_types_datasets())  # 使用 pytest 的 parametrize 标记来定义测试参数化
def test_fused_types_consistency(dataset_32, dataset_64):
    NUMBER_OF_RUNS = 5  # 设置循环次数为 5
    for _ in range(NUMBER_OF_RUNS):  # 循环指定次数
        # next sample  # 获取下一个样本
        (xi_data32, _, _), yi32, _, _ = dataset_32._next_py()  # 从 dataset_32 获取下一个样本的数据和标签
        (xi_data64, _, _), yi64, _, _ = dataset_64._next_py()  # 从 dataset_64 获取下一个样本的数据和标签

        assert xi_data32.dtype == np.float32  # 断言 xi_data32 的数据类型为 np.float32
        assert xi_data64.dtype == np.float64  # 断言 xi_data64 的数据类型为 np.float64

        assert_allclose(xi_data64, xi_data32, rtol=1e-5)  # 使用 assert_allclose 断言 xi_data64 和 xi_data32 在给定的相对误差下相等
        assert_allclose(yi64, yi32, rtol=1e-5)  # 使用 assert_allclose 断言 yi64 和 yi32 在给定的相对误差下相等


def test_buffer_dtype_mismatch_error():
    with pytest.raises(ValueError, match="Buffer dtype mismatch"):  # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配指定的错误消息
        ArrayDataset64(X32, y32, sample_weight32, seed=42),  # 调用 ArrayDataset64 构造函数，抛出异常时执行

    with pytest.raises(ValueError, match="Buffer dtype mismatch"):  # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配指定的错误消息
        ArrayDataset32(X64, y64, sample_weight64, seed=42),  # 调用 ArrayDataset32 构造函数，抛出异常时执行

    for csr_container in CSR_CONTAINERS:  # 遍历 CSR_CONTAINERS
        X_csr32 = csr_container(X32)  # 使用当前 csr_container 创建 X_csr32
        X_csr64 = csr_container(X64)  # 使用当前 csr_container 创建 X_csr64
        with pytest.raises(ValueError, match="Buffer dtype mismatch"):  # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配指定的错误消息
            CSRDataset64(  # 调用 CSRDataset64 构造函数，抛出异常时执行
                X_csr32.data,
                X_csr32.indptr,
                X_csr32.indices,
                y32,
                sample_weight32,
                seed=42,
            ),

        with pytest.raises(ValueError, match="Buffer dtype mismatch"):  # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配指定的错误消息
            CSRDataset32(  # 调用 CSRDataset32 构造函数，抛出异常时执行
                X_csr64.data,
                X_csr64.indptr,
                X_csr64.indices,
                y64,
                sample_weight64,
                seed=42,
            ),
```
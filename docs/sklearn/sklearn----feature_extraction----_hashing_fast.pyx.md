# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\_hashing_fast.pyx`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的 Cython 库函数和数据结构
from libc.stdlib cimport abs
from libcpp.vector cimport vector

# 导入 NumPy 和 Cython 中必要的类型和函数
cimport numpy as cnp
import numpy as np
from ..utils._typedefs cimport int32_t, int64_t
from ..utils.murmurhash cimport murmurhash3_bytes_s32
from ..utils._vector_sentinel cimport vector_to_nd_array

# 导入 NumPy 的数组操作接口
cnp.import_array()

# 定义 transform 函数，用于特征哈希转换
def transform(raw_X, Py_ssize_t n_features, dtype,
              bint alternate_sign=1, unsigned int seed=0):
    """Guts of FeatureHasher.transform.

    Returns
    -------
    n_samples : integer
        样本数
    indices, indptr, values : lists
        用于构建 scipy.sparse.csr_matrix 的索引和数值数组

    """
    # 声明 Cython 的变量
    cdef int32_t h
    cdef double value

    # 使用 Cython 的 vector 容器定义 indices 和 indptr 数组
    cdef vector[int32_t] indices
    cdef vector[int64_t] indptr
    indptr.push_back(0)

    # 创建一个 NumPy 数组用于存储值，初始容量为 8192
    cdef Py_ssize_t capacity = 8192     # arbitrary
    cdef int64_t size = 0
    cdef cnp.ndarray values = np.empty(capacity, dtype=dtype)

    # 遍历输入数据集 raw_X
    for x in raw_X:
        # 遍历每个特征-值对 (f, v)
        for f, v in x:
            # 如果值 v 是字符串，则将特征名 f 与字符串值 v 拼接
            if isinstance(v, (str, unicode)):
                f = "%s%s%s" % (f, '=', v)
                value = 1
            else:
                value = v

            # 如果值为零，则跳过当前特征-值对
            if value == 0:
                continue

            # 如果特征名 f 是 unicode 类型，则转换为 UTF-8 编码的字节串
            if isinstance(f, unicode):
                f = (<unicode>f).encode("utf-8")
            # 如果特征名 f 不是字节串类型，则抛出类型错误异常
            elif not isinstance(f, bytes):
                raise TypeError("feature names must be strings")

            # 使用 MurmurHash 算法计算特征名的哈希值 h
            h = murmurhash3_bytes_s32(<bytes>f, seed)

            # 处理 MurmurHash 返回的特殊情况，确保索引在 n_features 范围内
            if h == - 2147483648:
                # abs(-2**31) 是未定义行为，因为 h 是 np.int32 类型
                # 下面的计算等同于 abs(-2**31) % n_features
                indices.push_back((2147483647 - (n_features - 1)) % n_features)
            else:
                indices.push_back(abs(h) % n_features)

            # 如果 alternate_sign 为真，则根据哈希值 h 的正负调整值的符号
            if alternate_sign:
                value *= (h >= 0) * 2 - 1

            # 将值存入 values 数组，并增加 size 计数
            values[size] = value
            size += 1

            # 如果 values 数组达到容量上限，则扩展其容量
            if size == capacity:
                capacity *= 2
                # 不能使用 resize 方法，因为可能存在多个对数组的引用
                values = np.resize(values, capacity)

        # 在处理完一个样本后，将当前 size 添加到 indptr 数组中
        indptr.push_back(size)

    # 将 Cython 的 vector 转换为 NumPy 数组
    indices_array = vector_to_nd_array(&indices)
    indptr_array = vector_to_nd_array(&indptr)

    # 如果最后一个 indptr 值超过了 np.int32 的最大值，则将 indices_array 转换为 int64 类型
    if indptr_array[indptr_array.shape[0]-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
        indices_array = indices_array.astype(np.int64, copy=False)
    else:
        indptr_array = indptr_array.astype(np.int32, copy=False)
    # 返回一个元组，包含三个元素：indices_array、indptr_array 和 values[:size]
    # indices_array: 索引数组，用于存储非零元素在数据中的位置信息
    # indptr_array: 指针数组，指示每行（或每列）的起始位置在 values 数组中的索引
    # values[:size]: 值数组的部分切片，包含了矩阵中的非零元素的数值信息
    return (indices_array, indptr_array, values[:size])
```
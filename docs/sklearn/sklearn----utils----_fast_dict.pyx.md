# `D:\src\scipysrc\scikit-learn\sklearn\utils\_fast_dict.pyx`

```
"""
Uses C++ map containers for fast dict-like behavior with keys being
integers, and values float.
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# C++ 导入必要的 Cython 库
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.utility cimport pair
from libcpp.map cimport map as cpp_map

# 导入 NumPy 库
import numpy as np

# 导入 Cython 定义的类型
from ._typedefs cimport float64_t, intp_t

###############################################################################
# An object to be used in Python

# Lookup is faster than dict (up to 10 times), and so is full traversal
# (up to 50 times), and assignment (up to 6 times), but creation is
# slower (up to 3 times). Also, a large benefit is that memory
# consumption is reduced a lot compared to a Python dict

# 定义一个 Cython 类 IntFloatDict
cdef class IntFloatDict:

    def __init__(
        self,
        intp_t[:] keys,
        float64_t[:] values,
    ):
        cdef int i
        cdef int size = values.size
        # 检查 keys 和 values 的大小是否相等，并禁用边界检查
        for i in range(size):
            self.my_map[keys[i]] = values[i]

    def __len__(self):
        # 返回 my_map 的大小作为长度
        return self.my_map.size()

    def __getitem__(self, int key):
        # 查找给定 key 的值并返回，如果不存在则抛出 KeyError 异常
        cdef cpp_map[intp_t, float64_t].iterator it = self.my_map.find(key)
        if it == self.my_map.end():
            # 如果 key 不存在于字典中，则抛出 KeyError
            raise KeyError('%i' % key)
        return deref(it).second

    def __setitem__(self, int key, float value):
        # 设置给定 key 对应的值为指定的 value
        self.my_map[key] = value

    def __iter__(self):
        # 使用迭代器遍历 my_map 的键值对并逐个返回
        cdef int size = self.my_map.size()
        cdef intp_t [:] keys = np.empty(size, dtype=np.intp)
        cdef float64_t [:] values = np.empty(size, dtype=np.float64)
        self._to_arrays(keys, values)
        cdef int idx
        cdef intp_t key
        cdef float64_t value
        for idx in range(size):
            key = keys[idx]
            value = values[idx]
            yield key, value

    def to_arrays(self):
        """Return the key, value representation of the IntFloatDict
           object.

           Returns
           =======
           keys : ndarray, shape (n_items, ), dtype=int
                The indices of the data points
           values : ndarray, shape (n_items, ), dtype=float
                The values of the data points
        """
        # 将 my_map 中的键和值转换为 NumPy 数组并返回
        cdef int size = self.my_map.size()
        keys = np.empty(size, dtype=np.intp)
        values = np.empty(size, dtype=np.float64)
        self._to_arrays(keys, values)
        return keys, values
    cdef _to_arrays(self, intp_t [:] keys, float64_t [:] values):
        # 定义一个内部方法用于将 C++ map 转换为两个已初始化的数组
        # 开始迭代 C++ map 中的元素
        cdef cpp_map[intp_t, float64_t].iterator it = self.my_map.begin()
        # 获取 C++ map 的末尾迭代器
        cdef cpp_map[intp_t, float64_t].iterator end = self.my_map.end()
        # 初始化索引
        cdef int index = 0
        # 遍历 C++ map 中的所有元素
        while it != end:
            # 将当前迭代器指向的键放入 keys 数组中
            keys[index] = deref(it).first
            # 将当前迭代器指向的值放入 values 数组中
            values[index] = deref(it).second
            # 移动迭代器到下一个元素
            inc(it)
            # 增加索引
            index += 1

    def update(self, IntFloatDict other):
        # 从另一个 IntFloatDict 对象更新当前对象的 C++ map
        # 开始迭代另一个对象的 C++ map
        cdef cpp_map[intp_t, float64_t].iterator it = other.my_map.begin()
        # 获取另一个对象的 C++ map 的末尾迭代器
        cdef cpp_map[intp_t, float64_t].iterator end = other.my_map.end()
        # 遍历另一个对象的 C++ map 中的所有元素
        while it != end:
            # 将另一个对象当前迭代器指向的键值对复制到当前对象的 C++ map 中
            self.my_map[deref(it).first] = deref(it).second
            # 移动迭代器到下一个元素
            inc(it)

    def copy(self):
        # 创建当前对象的深拷贝副本
        cdef IntFloatDict out_obj = IntFloatDict.__new__(IntFloatDict)
        # 使用 C++ map 的 '=' 拷贝运算符来复制映射内容
        out_obj.my_map = self.my_map
        return out_obj

    def append(self, intp_t key, float64_t value):
        # 向当前对象的 C++ map 中插入一个新的键值对
        # 构造键值对参数
        cdef pair[intp_t, float64_t] args
        args.first = key
        args.second = value
        # 插入键值对到 C++ map 中
        self.my_map.insert(args)
###############################################################################
# operation on dict

# 定义一个函数用于在 IntFloatDict 字典对象中找到值最小的键值对
def argmin(IntFloatDict d):
    # 使用 Cython 定义 C++ 类型的迭代器 it，指向 d.my_map 的开始位置
    cdef cpp_map[intp_t, float64_t].iterator it = d.my_map.begin()
    # 使用 Cython 定义 C++ 类型的迭代器 end，指向 d.my_map 的结束位置
    cdef cpp_map[intp_t, float64_t].iterator end = d.my_map.end()
    # 初始化最小键和最小值
    cdef intp_t min_key = -1
    cdef float64_t min_value = np.inf
    # 遍历字典 d.my_map
    while it != end:
        # 如果当前迭代器指向的键值对的值比当前最小值小
        if deref(it).second < min_value:
            # 更新最小值和对应的键
            min_value = deref(it).second
            min_key = deref(it).first
        # 移动迭代器到下一个位置
        inc(it)
    # 返回找到的最小键和对应的最小值
    return min_key, min_value
```
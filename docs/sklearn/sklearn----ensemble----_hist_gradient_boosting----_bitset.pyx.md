# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\_bitset.pyx`

```
# 导入相关的 C 语言扩展定义
from .common cimport BITSET_INNER_DTYPE_C
from .common cimport BITSET_DTYPE_C
from .common cimport X_DTYPE_C
from .common cimport X_BINNED_DTYPE_C

# 位集（bitset）是一种数据结构，用于表示整数集合 [0, n]。在这里，它们用于表示特征索引的集合（例如，作为左子树的特征或分类特征）。
# 为了了解位集及位运算的使用，请参考以下链接：
# https://en.wikipedia.org/wiki/Bit_array
# https://en.wikipedia.org/wiki/Bitwise_operation


# 初始化位集的函数，将位集 bitset 中的所有位初始化为 0
cdef inline void init_bitset(BITSET_DTYPE_C bitset) noexcept nogil:  # OUT
    cdef:
        unsigned int i

    for i in range(8):
        bitset[i] = 0


# 设置位集的函数，将位集 bitset 中的某个位设置为 1，位置由 val 决定
cdef inline void set_bitset(BITSET_DTYPE_C bitset,  # OUT
                            X_BINNED_DTYPE_C val) noexcept nogil:
    bitset[val // 32] |= (1 << (val % 32))


# 检查位集中是否包含某个值 val，返回 0 或 1
cdef inline unsigned char in_bitset(BITSET_DTYPE_C bitset,
                                    X_BINNED_DTYPE_C val) noexcept nogil:

    return (bitset[val // 32] >> (val % 32)) & 1


# 使用 memoryview 检查位集中是否包含某个值 val，返回 0 或 1
cpdef inline unsigned char in_bitset_memoryview(const BITSET_INNER_DTYPE_C[:] bitset,
                                                X_BINNED_DTYPE_C val) noexcept nogil:
    return (bitset[val // 32] >> (val % 32)) & 1


# 在二维 memoryview 上检查位集中是否包含某个值 val，返回 0 或 1
cdef inline unsigned char in_bitset_2d_memoryview(const BITSET_INNER_DTYPE_C [:, :] bitset,
                                                  X_BINNED_DTYPE_C val,
                                                  unsigned int row) noexcept nogil:

    # 与上面的函数类似，但适用于二维 memoryview，避免创建一维 memoryview。详见 https://github.com/scikit-learn/scikit-learn/issues/17299
    return (bitset[row, val // 32] >> (val % 32)) & 1


# 使用 memoryview 设置位集的函数，将位集 bitset 中的某个位设置为 1
cpdef inline void set_bitset_memoryview(BITSET_INNER_DTYPE_C[:] bitset,  # OUT
                                        X_BINNED_DTYPE_C val):
    bitset[val // 32] |= (1 << (val % 32))


# 将 binned_bitset 的值转换为 raw_bitset
def set_raw_bitset_from_binned_bitset(BITSET_INNER_DTYPE_C[:] raw_bitset,  # OUT
                                      BITSET_INNER_DTYPE_C[:] binned_bitset,
                                      X_DTYPE_C[:] categories):
    """根据 binned_bitset 的值设置 raw_bitset

    categories 是从 binned 类别值到 raw 类别值的映射。
    """
    cdef:
        int binned_cat_value  # binned 类别值
        X_DTYPE_C raw_cat_value  # raw 类别值

    # 遍历 categories 中的 binned_cat_value 和对应的 raw_cat_value
    for binned_cat_value, raw_cat_value in enumerate(categories):
        # 如果 binned_bitset 中包含 binned_cat_value，则将对应的 raw_cat_value 设置到 raw_bitset 中
        if in_bitset_memoryview(binned_bitset, binned_cat_value):
            set_bitset_memoryview(raw_bitset, <X_BINNED_DTYPE_C>raw_cat_value)
```
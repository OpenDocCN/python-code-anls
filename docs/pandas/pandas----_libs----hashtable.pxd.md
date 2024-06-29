# `D:\src\scipysrc\pandas\pandas\_libs\hashtable.pxd`

```
# 导入必要的 C 扩展类型和对象
from numpy cimport (
    intp_t,         # C 扩展中定义的整数类型
    ndarray,        # C 扩展中定义的多维数组类型
)

from pandas._libs.khash cimport (
    complex64_t,    # 复数类型
    complex128_t,   # 双精度复数类型
    float32_t,      # 单精度浮点数类型
    float64_t,      # 双精度浮点数类型
    int8_t,         # 8 位整数类型
    int16_t,        # 16 位整数类型
    int32_t,        # 32 位整数类型
    int64_t,        # 64 位整数类型
    kh_complex64_t, # 复数的哈希表类型
    kh_complex128_t,# 双精度复数的哈希表类型
    kh_float32_t,   # 单精度浮点数的哈希表类型
    kh_float64_t,   # 双精度浮点数的哈希表类型
    kh_int8_t,      # 8 位整数的哈希表类型
    kh_int16_t,     # 16 位整数的哈希表类型
    kh_int32_t,     # 32 位整数的哈希表类型
    kh_int64_t,     # 64 位整数的哈希表类型
    kh_pymap_t,     # Python 字典的哈希表类型
    kh_str_t,       # 字符串的哈希表类型
    kh_uint8_t,     # 8 位无符号整数的哈希表类型
    kh_uint16_t,    # 16 位无符号整数的哈希表类型
    kh_uint32_t,    # 32 位无符号整数的哈希表类型
    kh_uint64_t,    # 64 位无符号整数的哈希表类型
    khcomplex64_t,  # 复数的哈希表类型（别名）
    khcomplex128_t, # 双精度复数的哈希表类型（别名）
    uint8_t,        # 8 位无符号整数类型
    uint16_t,       # 16 位无符号整数类型
    uint32_t,       # 32 位无符号整数类型
    uint64_t,       # 64 位无符号整数类型
)

# 哈希表类的原型定义

cdef class HashTable:
    pass

# 64 位无符号整数哈希表类
cdef class UInt64HashTable(HashTable):
    cdef kh_uint64_t *table     # 哈希表的指针
    cdef int64_t na_position    # 缺失值的位置
    cdef bint uses_mask         # 是否使用掩码

    cpdef get_item(self, uint64_t val)         # 获取哈希表中的条目
    cpdef set_item(self, uint64_t key, Py_ssize_t val)   # 设置哈希表中的条目
    cpdef get_na(self)                        # 获取缺失值
    cpdef set_na(self, Py_ssize_t val)        # 设置缺失值

# 64 位整数哈希表类
cdef class Int64HashTable(HashTable):
    cdef kh_int64_t *table       # 哈希表的指针
    cdef int64_t na_position      # 缺失值的位置
    cdef bint uses_mask           # 是否使用掩码

    cpdef get_item(self, int64_t val)         # 获取哈希表中的条目
    cpdef set_item(self, int64_t key, Py_ssize_t val)   # 设置哈希表中的条目
    cpdef get_na(self)                        # 获取缺失值
    cpdef set_na(self, Py_ssize_t val)        # 设置缺失值

# 32 位无符号整数哈希表类
cdef class UInt32HashTable(HashTable):
    cdef kh_uint32_t *table     # 哈希表的指针
    cdef int64_t na_position    # 缺失值的位置
    cdef bint uses_mask         # 是否使用掩码

    cpdef get_item(self, uint32_t val)         # 获取哈希表中的条目
    cpdef set_item(self, uint32_t key, Py_ssize_t val)   # 设置哈希表中的条目
    cpdef get_na(self)                        # 获取缺失值
    cpdef set_na(self, Py_ssize_t val)        # 设置缺失值

# 32 位整数哈希表类
cdef class Int32HashTable(HashTable):
    cdef kh_int32_t *table       # 哈希表的指针
    cdef int64_t na_position      # 缺失值的位置
    cdef bint uses_mask           # 是否使用掩码

    cpdef get_item(self, int32_t val)         # 获取哈希表中的条目
    cpdef set_item(self, int32_t key, Py_ssize_t val)   # 设置哈希表中的条目
    cpdef get_na(self)                        # 获取缺失值
    cpdef set_na(self, Py_ssize_t val)        # 设置缺失值

# 16 位无符号整数哈希表类
cdef class UInt16HashTable(HashTable):
    cdef kh_uint16_t *table     # 哈希表的指针
    cdef int64_t na_position    # 缺失值的位置
    cdef bint uses_mask         # 是否使用掩码

    cpdef get_item(self, uint16_t val)         # 获取哈希表中的条目
    cpdef set_item(self, uint16_t key, Py_ssize_t val)   # 设置哈希表中的条目
    cpdef get_na(self)                        # 获取缺失值
    cpdef set_na(self, Py_ssize_t val)        # 设置缺失值

# 16 位整数哈希表类
cdef class Int16HashTable(HashTable):
    cdef kh_int16_t *table       # 哈希表的指针
    cdef int64_t na_position      # 缺失值的位置
    cdef bint uses_mask           # 是否使用掩码

    cpdef get_item(self, int16_t val)         # 获取哈希表中的条目
    cpdef set_item(self, int16_t key, Py_ssize_t val)   # 设置哈希表中的条目
    cpdef get_na(self)                        # 获取缺失值
    cpdef set_na(self, Py_ssize_t val)        # 设置缺失值

# 8 位无符号整数哈希表类
cdef class UInt8HashTable(HashTable):
    cdef kh_uint8_t *table      # 哈希表的指针
    cdef int64_t na_position    # 缺失值的位置
    cdef bint uses_mask         # 是否使用掩码

    cpdef get_item(self, uint8_t val)         # 获取哈希表中的条目
    cpdef set_item(self, uint8_t key, Py_ssize_t val)   # 设置哈希表中的条目
    cpdef get_na(self)                        # 获取缺失值
    cpdef set_na(self, Py_ssize_t val)        # 设置缺失值

# 8 位整数哈希表类
cdef class Int8HashTable(HashTable):
    cdef kh_int8_t *table       # 哈希表的指针
    cdef int64_t na_position      # 缺失值的位置
    cdef bint uses_mask           # 是否使用掩码

    cpdef get_item(self, int8_t val)         # 获取哈希表中的条目
    cpdef set_item(self, int8_t key, Py_ssize_t val)   # 设置哈希表中的条目
    cpdef get_na(self)                        # 获取缺失值
    cpdef set_na(self, Py_ssize_t val)        # 设置缺失值

# 双精度浮点数哈希表类
cdef class Float64HashTable(HashTable):
    cdef kh_float64_t *table    # 哈希表的指针
    cdef int64_t na_position    # 缺失值的位置
    cdef bint uses_mask
    # 定义一个包含两个参数的 Cython cpdef 方法，用于设置键值对
    cpdef set_item(self, float64_t key, Py_ssize_t val)
    
    # 定义一个 Cython cpdef 方法，用于获取缺失值
    cpdef get_na(self)
    
    # 定义一个 Cython cpdef 方法，用于设置缺失值
    cpdef set_na(self, Py_ssize_t val)
# 定义一个名为 Float32HashTable 的 Cython cdef 类，继承自 HashTable
cdef class Float32HashTable(HashTable):
    # 声明一个指向 kh_float32_t 类型的指针 table
    cdef kh_float32_t *table
    # 声明一个 int64_t 类型的变量 na_position
    cdef int64_t na_position
    # 声明一个布尔型变量 uses_mask，表示是否使用掩码
    cdef bint uses_mask

    # 定义一个公共 cpdef 方法 get_item，接受一个 float32_t 类型的参数 val
    cpdef get_item(self, float32_t val)
    # 定义一个公共 cpdef 方法 set_item，接受一个 float32_t 类型的参数 key 和 Py_ssize_t 类型的参数 val
    cpdef set_item(self, float32_t key, Py_ssize_t val)
    # 定义一个公共 cpdef 方法 get_na，用于获取 na_position 的值
    cpdef get_na(self)
    # 定义一个公共 cpdef 方法 set_na，接受一个 Py_ssize_t 类型的参数 val，设置 na_position 的值
    cpdef set_na(self, Py_ssize_t val)

# 定义一个名为 Complex64HashTable 的 Cython cdef 类，继承自 HashTable
cdef class Complex64HashTable(HashTable):
    # 声明一个指向 kh_complex64_t 类型的指针 table
    cdef kh_complex64_t *table
    # 声明一个 int64_t 类型的变量 na_position
    cdef int64_t na_position
    # 声明一个布尔型变量 uses_mask
    cdef bint uses_mask

    # 定义一个公共 cpdef 方法 get_item，接受一个 complex64_t 类型的参数 val
    cpdef get_item(self, complex64_t val)
    # 定义一个公共 cpdef 方法 set_item，接受一个 complex64_t 类型的参数 key 和 Py_ssize_t 类型的参数 val
    cpdef set_item(self, complex64_t key, Py_ssize_t val)
    # 定义一个公共 cpdef 方法 get_na，用于获取 na_position 的值
    cpdef get_na(self)
    # 定义一个公共 cpdef 方法 set_na，接受一个 Py_ssize_t 类型的参数 val，设置 na_position 的值
    cpdef set_na(self, Py_ssize_t val)

# 定义一个名为 Complex128HashTable 的 Cython cdef 类，继承自 HashTable
cdef class Complex128HashTable(HashTable):
    # 声明一个指向 kh_complex128_t 类型的指针 table
    cdef kh_complex128_t *table
    # 声明一个 int64_t 类型的变量 na_position
    cdef int64_t na_position
    # 声明一个布尔型变量 uses_mask
    cdef bint uses_mask

    # 定义一个公共 cpdef 方法 get_item，接受一个 complex128_t 类型的参数 val
    cpdef get_item(self, complex128_t val)
    # 定义一个公共 cpdef 方法 set_item，接受一个 complex128_t 类型的参数 key 和 Py_ssize_t 类型的参数 val
    cpdef set_item(self, complex128_t key, Py_ssize_t val)
    # 定义一个公共 cpdef 方法 get_na，用于获取 na_position 的值
    cpdef get_na(self)
    # 定义一个公共 cpdef 方法 set_na，接受一个 Py_ssize_t 类型的参数 val，设置 na_position 的值
    cpdef set_na(self, Py_ssize_t val)

# 定义一个名为 PyObjectHashTable 的 Cython cdef 类，继承自 HashTable
cdef class PyObjectHashTable(HashTable):
    # 声明一个指向 kh_pymap_t 类型的指针 table
    cdef kh_pymap_t *table

    # 定义一个公共 cpdef 方法 get_item，接受一个 object 类型的参数 val
    cpdef get_item(self, object val)
    # 定义一个公共 cpdef 方法 set_item，接受一个 object 类型的参数 key 和 Py_ssize_t 类型的参数 val
    cpdef set_item(self, object key, Py_ssize_t val)

# 定义一个名为 StringHashTable 的 Cython cdef 类，继承自 HashTable
cdef class StringHashTable(HashTable):
    # 声明一个指向 kh_str_t 类型的指针 table
    cdef kh_str_t *table

    # 定义一个公共 cpdef 方法 get_item，接受一个 str 类型的参数 val
    cpdef get_item(self, str val)
    # 定义一个公共 cpdef 方法 set_item，接受一个 str 类型的参数 key 和 Py_ssize_t 类型的参数 val
    cpdef set_item(self, str key, Py_ssize_t val)

# 定义一个名为 Int64VectorData 的 Cython 结构体
cdef struct Int64VectorData:
    # 声明一个 int64_t 类型的指针 data
    int64_t *data
    # 声明 Py_ssize_t 类型的变量 size 和 capacity
    Py_ssize_t size, capacity

# 定义一个名为 Vector 的 Cython cdef 类
cdef class Vector:
    # 声明一个布尔型变量 external_view_exists
    cdef bint external_view_exists

# 定义一个名为 Int64Vector 的 Cython cdef 类，继承自 Vector
cdef class Int64Vector(Vector):
    # 声明一个 Int64VectorData 类型的变量 data
    cdef Int64VectorData data
    # 声明一个 ndarray 类型的变量 ao
    cdef ndarray ao

    # 定义一个私有 cdef 方法 resize，接受一个 Py_ssize_t 类型的参数 new_size，用于调整向量大小
    cdef resize(self, Py_ssize_t new_size)
    # 定义一个公共 cpdef 方法 to_array，将向量转换为 ndarray 类型返回
    cpdef ndarray to_array(self)
    # 定义一个私有 cdef 方法 append，接受一个 int64_t 类型的参数 x，用于向向量追加元素
    cdef void append(self, int64_t x) noexcept
    # 定义一个私有 cdef 方法 extend，接受一个 int64_t[:] 类型的参数 x，用于向向量扩展元素
    cdef extend(self, int64_t[:] x)
```
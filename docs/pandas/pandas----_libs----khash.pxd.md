# `D:\src\scipysrc\pandas\pandas\_libs\khash.pxd`

```
# 从cpython.object中导入PyObject对象，用于与Cython集成
# 从numpy中导入多个C语言类型，用于Cython中的类型声明
from cpython.object cimport PyObject
from numpy cimport (
    complex64_t,
    complex128_t,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

# 从指定路径导入外部C头文件中的定义
cdef extern from "pandas/vendored/klib/khash_python.h":
    # 定义常量KHASH_TRACE_DOMAIN，表示某种追踪域
    const int KHASH_TRACE_DOMAIN

    # 定义khuint_t和khiter_t为uint32_t类型，作为哈希表的键和迭代器
    ctypedef uint32_t khuint_t
    ctypedef khuint_t khiter_t

    # 定义khcomplex128_t结构体，包含双精度实部和虚部，用于复数运算
    ctypedef struct khcomplex128_t:
        double real
        double imag

    # 定义比较复数khcomplex128_t相等的函数声明
    bint are_equivalent_khcomplex128_t \
        "kh_complex_hash_equal" (khcomplex128_t a, khcomplex128_t b) nogil

    # 定义khcomplex64_t结构体，包含单精度实部和虚部，用于复数运算
    ctypedef struct khcomplex64_t:
        float real
        float imag

    # 定义比较复数khcomplex64_t相等的函数声明
    bint are_equivalent_khcomplex64_t \
        "kh_complex_hash_equal" (khcomplex64_t a, khcomplex64_t b) nogil

    # 定义比较双精度浮点数float64_t相等的函数声明
    bint are_equivalent_float64_t \
        "kh_floats_hash_equal" (float64_t a, float64_t b) nogil

    # 定义比较单精度浮点数float32_t相等的函数声明
    bint are_equivalent_float32_t \
        "kh_floats_hash_equal" (float32_t a, float32_t b) nogil

    # 定义Python对象的哈希计算函数和相等判断函数的声明
    uint32_t kh_python_hash_func(object key)
    bint kh_python_hash_equal(object a, object b)

    # 定义kh_pymap_t结构体，表示Python对象到size_t值的哈希映射表
    ctypedef struct kh_pymap_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        PyObject **keys
        size_t *vals

    # 声明操作kh_pymap_t结构体的初始化、销毁、清除、获取、调整大小、添加、删除、存在等函数
    kh_pymap_t* kh_init_pymap()
    void kh_destroy_pymap(kh_pymap_t*)
    void kh_clear_pymap(kh_pymap_t*)
    khuint_t kh_get_pymap(kh_pymap_t*, PyObject*)
    void kh_resize_pymap(kh_pymap_t*, khuint_t)
    khuint_t kh_put_pymap(kh_pymap_t*, PyObject*, int*)
    void kh_del_pymap(kh_pymap_t*, khuint_t)

    # 判断kh_pymap_t结构体中某个位置是否存在值的函数声明
    bint kh_exist_pymap(kh_pymap_t*, khiter_t)

    # 定义kh_pyset_t结构体，表示Python对象到size_t值的哈希集合表
    ctypedef struct kh_pyset_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        PyObject **keys
        size_t *vals

    # 声明操作kh_pyset_t结构体的初始化、销毁、清除、获取、调整大小、添加、删除、存在等函数
    kh_pyset_t* kh_init_pyset()
    void kh_destroy_pyset(kh_pyset_t*)
    void kh_clear_pyset(kh_pyset_t*)
    khuint_t kh_get_pyset(kh_pyset_t*, PyObject*)
    void kh_resize_pyset(kh_pyset_t*, khuint_t)
    khuint_t kh_put_pyset(kh_pyset_t*, PyObject*, int*)
    void kh_del_pyset(kh_pyset_t*, khuint_t)

    # 判断kh_pyset_t结构体中某个位置是否存在值的函数声明
    bint kh_exist_pyset(kh_pyset_t*, khiter_t)

    # 定义kh_cstr_t类型为char*，表示C字符串的哈希表键类型
    ctypedef char* kh_cstr_t

    # 定义kh_str_t结构体，表示C字符串到size_t值的哈希映射表
    ctypedef struct kh_str_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        kh_cstr_t *keys
        size_t *vals

    # 声明操作kh_str_t结构体的初始化、销毁、清除、获取、调整大小、添加、删除、存在等函数
    kh_str_t* kh_init_str() nogil
    void kh_destroy_str(kh_str_t*) nogil
    void kh_clear_str(kh_str_t*) nogil
    khuint_t kh_get_str(kh_str_t*, kh_cstr_t) nogil
    void kh_resize_str(kh_str_t*, khuint_t) nogil
    khuint_t kh_put_str(kh_str_t*, kh_cstr_t, int*) nogil
    void kh_del_str(kh_str_t*, khuint_t) nogil

    # 判断kh_str_t结构体中某个位置是否存在值的函数声明
    bint kh_exist_str(kh_str_t*, khiter_t) nogil

    # 定义kh_str_starts_t结构体，表示特定格式的C字符串哈希表
    ctypedef struct kh_str_starts_t:
        kh_str_t *table
        int starts[256]

    # 声明操作kh_str_starts_t结构体的初始化、添加、获取项的函数
    kh_str_starts_t* kh_init_str_starts() nogil
    khuint_t kh_put_str_starts_item(kh_str_starts_t* table, char* key,
                                    int* ret) nogil
    khuint_t kh_get_str_starts_item(kh_str_starts_t* table, char* key) nogil
    # 定义了一系列操作 kh_str_starts_t 结构的函数声明，这些函数都是在不需要全局解释器锁 (nogil) 的情况下运行的
    
    ctypedef struct kh_strbox_t:
        # 定义了一个结构体 kh_strbox_t，包含了几个整数字段和指针数组
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags  # 指向 uint32_t 类型的指针
        kh_cstr_t *keys  # 指向 kh_cstr_t 类型的指针
        PyObject **vals  # 指向 PyObject* 类型的指针数组
    
    kh_strbox_t* kh_init_strbox() nogil
    # 初始化一个 kh_strbox_t 结构体，并返回其指针，操作不需要全局解释器锁 (nogil)
    
    void kh_destroy_strbox(kh_strbox_t*) nogil
    # 销毁一个 kh_strbox_t 结构体，释放其内存，操作不需要全局解释器锁 (nogil)
    
    void kh_clear_strbox(kh_strbox_t*) nogil
    # 清空一个 kh_strbox_t 结构体，即移除所有元素，但保留内存空间，操作不需要全局解释器锁 (nogil)
    
    khuint_t kh_get_strbox(kh_strbox_t*, kh_cstr_t) nogil
    # 根据给定的键从 kh_strbox_t 结构体中获取对应的值，操作不需要全局解释器锁 (nogil)
    
    void kh_resize_strbox(kh_strbox_t*, khuint_t) nogil
    # 调整 kh_strbox_t 结构体的内部存储空间大小，操作不需要全局解释器锁 (nogil)
    
    khuint_t kh_put_strbox(kh_strbox_t*, kh_cstr_t, int*) nogil
    # 向 kh_strbox_t 结构体中插入或更新键值对，返回新插入元素的索引，操作不需要全局解释器锁 (nogil)
    
    void kh_del_strbox(kh_strbox_t*, khuint_t) nogil
    # 从 kh_strbox_t 结构体中删除指定索引位置的键值对，操作不需要全局解释器锁 (nogil)
    
    bint kh_exist_strbox(kh_strbox_t*, khiter_t) nogil
    # 检查在 kh_strbox_t 结构体中指定位置的元素是否存在，返回布尔值，操作不需要全局解释器锁 (nogil)
    
    khuint_t kh_needed_n_buckets(khuint_t element_n) nogil
    # 根据元素数量计算 kh_strbox_t 结构体所需的哈希表桶的数量，操作不需要全局解释器锁 (nogil)
# 导入名为 "khash_for_primitive_helper.pxi" 的外部模块
include "khash_for_primitive_helper.pxi"
```
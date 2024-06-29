# `D:\src\scipysrc\pandas\pandas\_libs\hashtable.pyx`

```
# 导入Cython模块声明
cimport cython
# 从CPython的引用模块中导入特定的标识符
from cpython.ref cimport (
    Py_INCREF,  # 增加Python对象的引用计数
    PyObject,   # Python对象的基类
)
# 从C标准库中导入特定的函数和变量声明
from libc.stdlib cimport (
    free,   # 释放动态分配的内存
    malloc, # 分配内存
)
# 从C标准库的字符串处理部分导入内存复制函数
from libc.string cimport memcpy

# 导入NumPy库，Python层面的接口
import numpy as np

# C级导入NumPy库
cimport numpy as cnp
# 从NumPy库中导入特定的类型
from numpy cimport ndarray

# 调用NumPy的C级函数导入数组接口
cnp.import_array()

# 从Pandas库的内部库中导入util模块
from pandas._libs cimport util
# 从Pandas库的内部数据类型模块中导入特定类型
from pandas._libs.dtypes cimport numeric_object_t
# 从Pandas库的内部哈希模块中导入特定类型和函数声明
from pandas._libs.khash cimport (
    KHASH_TRACE_DOMAIN,            # 哈希表追踪域
    are_equivalent_float32_t,      # 判断两个float32是否等效的函数
    are_equivalent_float64_t,      # 判断两个float64是否等效的函数
    are_equivalent_khcomplex64_t,  # 判断两个khcomplex64是否等效的函数
    are_equivalent_khcomplex128_t, # 判断两个khcomplex128是否等效的函数
    kh_needed_n_buckets,           # 哈希表所需的桶数
    kh_python_hash_equal,          # 判断两个Python对象哈希是否相等的函数
    kh_python_hash_func,           # 计算Python对象哈希值的函数
    khiter_t,                      # 哈希表迭代器类型
)
# 从Pandas库的内部缺失值处理模块中导入检查空值的函数声明
from pandas._libs.missing cimport checknull

# 定义函数返回哈希表的追踪域
def get_hashtable_trace_domain():
    return KHASH_TRACE_DOMAIN

# 定义函数返回对象的哈希值
def object_hash(obj):
    return kh_python_hash_func(obj)

# 定义函数判断两个对象是否相等
def objects_are_equal(a, b):
    return kh_python_hash_equal(a, b)

# 定义C级的int64_t类型变量NPY_NAT，其值为Pandas util模块的自然数常量
cdef int64_t NPY_NAT = util.get_nat()
# 定义SIZE_HINT_LIMIT常量，其值为2的20次方加7
SIZE_HINT_LIMIT = (1 << 20) + 7

# 定义C级的Py_ssize_t类型变量_INIT_VEC_CAP，初始值为128
cdef Py_ssize_t _INIT_VEC_CAP = 128

# 引入哈希表辅助类的PXI声明文件
include "hashtable_class_helper.pxi"
# 引入哈希表函数辅助类的PXI声明文件
include "hashtable_func_helper.pxi"

# 根据NumPy中整数指针类型选择不同的哈希表类型和标签索引函数
if np.dtype(np.intp) == np.dtype(np.int64):
    IntpHashTable = Int64HashTable       # 整数指针哈希表为Int64HashTable
    unique_label_indices = _unique_label_indices_int64  # 唯一标签索引为_int64
elif np.dtype(np.intp) == np.dtype(np.int32):
    IntpHashTable = Int32HashTable       # 整数指针哈希表为Int32HashTable
    unique_label_indices = _unique_label_indices_int32  # 唯一标签索引为_int32
else:
    raise ValueError(np.dtype(np.intp))  # 抛出值错误异常，如果整数指针类型不是int64或int32

# 定义C级的Factorizer类
cdef class Factorizer:
    cdef readonly:
        Py_ssize_t count  # 只读属性count，类型为Py_ssize_t

    def __cinit__(self, size_hint: int, uses_mask: bool = False):
        self.count = 0  # 初始化count为0

    # 返回计数count的方法
    def get_count(self) -> int:
        return self.count

    # 因子化方法的声明，抛出未实现错误
    def factorize(self, values, na_sentinel=-1, na_value=None, mask=None) -> np.ndarray:
        raise NotImplementedError

    # 哈希内连接方法的声明，抛出未实现错误
    def hash_inner_join(self, values, mask=None):
        raise NotImplementedError

# 定义C级的ObjectFactorizer类，继承自Factorizer类
cdef class ObjectFactorizer(Factorizer):
    cdef public:
        PyObjectHashTable table   # 公共成员table，类型为PyObjectHashTable
        ObjectVector uniques      # 公共成员uniques，类型为ObjectVector

    def __cinit__(self, size_hint: int, uses_mask: bool = False):
        self.table = PyObjectHashTable(size_hint)  # 初始化table为给定大小的PyObjectHashTable
        self.uniques = ObjectVector()              # 初始化uniques为空的ObjectVector

    # 重载factorize方法，接收object类型的ndarray数组
    def factorize(
        self, ndarray[object] values, na_sentinel=-1, na_value=None, mask=None
    ) -> np.ndarray:
        """
        返回值为一个 NumPy 数组，数组中的元素类型为 np.intp

        Examples
        --------
        将值因子化，并用 na_sentinel 替换 NaN 值

        >>> fac = ObjectFactorizer(3)
        >>> fac.factorize(np.array([1,2,np.nan], dtype='O'), na_sentinel=20)
        array([ 0,  1, 20])
        """
        cdef:
            ndarray[intp_t] labels  # 声明一个 Cython 的 C 数组 labels，元素类型为 intp_t

        if mask is not None:
            raise NotImplementedError("mask not supported for ObjectFactorizer.")  # 如果传入了 mask 参数，则抛出未实现错误

        if self.uniques.external_view_exists:
            uniques = ObjectVector()  # 创建一个 ObjectVector 对象 uniques
            uniques.extend(self.uniques.to_array())  # 将 self.uniques 转换为数组并扩展到 uniques
            self.uniques = uniques  # 将扩展后的 uniques 赋值给 self.uniques

        labels = self.table.get_labels(values, self.uniques,
                                       self.count, na_sentinel, na_value)
        # 调用 self.table.get_labels 方法获取标签，使用给定的参数 values, self.uniques,
        # self.count, na_sentinel, na_value，将结果赋给 labels
        self.count = len(self.uniques)  # 更新 self.count 为 self.uniques 的长度
        return labels  # 返回计算得到的 labels 数组
```
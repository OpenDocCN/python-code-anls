# `D:\src\scipysrc\pandas\pandas\_libs\index.pyx`

```
cimport cython
# 导入 Cython 扩展模块

from cpython.sequence cimport PySequence_GetItem
# 从 CPython 序列模块中导入 PySequence_GetItem 函数

import numpy as np
# 导入 NumPy 库并使用别名 np

cimport numpy as cnp
# 导入 NumPy C 扩展，并使用别名 cnp

from numpy cimport (
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
)
# 从 NumPy C 扩展中导入特定数据类型

cnp.import_array()
# 调用 NumPy C 扩展的 import_array 函数

from pandas._libs cimport util
# 从 pandas._libs 中导入 util 模块（Cython 版本）

from pandas._libs.hashtable cimport HashTable
# 从 pandas._libs.hashtable 中导入 HashTable 类（Cython 版本）

from pandas._libs.tslibs.nattype cimport c_NaT as NaT
# 从 pandas._libs.tslibs.nattype 中导入 c_NaT，并使用别名 NaT

from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    get_unit_from_dtype,
    import_pandas_datetime,
)
# 从 pandas._libs.tslibs.np_datetime 中导入多个函数和常量

import_pandas_datetime()
# 调用 import_pandas_datetime 函数

from pandas._libs.tslibs.period cimport is_period_object
# 从 pandas._libs.tslibs.period 中导入 is_period_object 函数（Cython 版本）

from pandas._libs.tslibs.timedeltas cimport _Timedelta
# 从 pandas._libs.tslibs.timedeltas 中导入 _Timedelta 类（Cython 版本）

from pandas._libs.tslibs.timestamps cimport _Timestamp
# 从 pandas._libs.tslibs.timestamps 中导入 _Timestamp 类（Cython 版本）

from pandas._libs import (
    algos,
    hashtable as _hash,
)
# 从 pandas._libs 中导入 algos 模块和 hashtable 模块，并使用别名 _hash

from pandas._libs.lib cimport eq_NA_compat
# 从 pandas._libs.lib 中导入 eq_NA_compat 函数（Cython 版本）

from pandas._libs.missing cimport (
    C_NA,
    checknull,
    is_matching_na,
)
# 从 pandas._libs.missing 中导入多个函数和常量（Cython 版本）

from decimal import InvalidOperation
# 从 decimal 模块中导入 InvalidOperation 异常类

# 定义 MultiIndex 代码的偏移量，避免负数代码（缺失值）
multiindex_nulls_shift = 2

cdef bint is_definitely_invalid_key(object val):
    # 检查是否可以对 val 进行哈希操作，如果不能则返回 True，否则返回 False
    try:
        hash(val)
    except TypeError:
        return True
    return False

cdef ndarray _get_bool_indexer(ndarray values, object val, ndarray mask = None):
    """
    Return a ndarray[bool] of locations where val matches self.values.

    If val is not NA, this is equivalent to `self.values == val`
    """
    # 返回一个 ndarray[bool]，标识出 val 与 self.values 匹配的位置

    # 调用者需确保已经调用了 _check_type 函数
    cdef:
        ndarray[uint8_t, ndim=1, cast=True] indexer
        Py_ssize_t i
        object item

    if values.descr.type_num == cnp.NPY_OBJECT:
        assert mask is None  # 对于对象类型没有掩码
        # 即 values.dtype == object
        if not checknull(val):
            indexer = eq_NA_compat(values, val)

        else:
            # 需要检查匹配的 NA 值
            indexer = np.empty(len(values), dtype=np.uint8)

            for i in range(len(values)):
                item = PySequence_GetItem(values, i)
                indexer[i] = is_matching_na(item, val)

    else:
        if mask is not None:
            if val is C_NA:
                indexer = mask == 1
            else:
                indexer = (values == val) & ~mask
        else:
            if util.is_nan(val):
                indexer = np.isnan(values)
            else:
                indexer = values == val

    return indexer.view(bool)

cdef _maybe_resize_array(ndarray values, Py_ssize_t loc, Py_ssize_t max_length):
    """
    Resize array if loc is out of bounds.
    """
    cdef:
        Py_ssize_t n = len(values)

    if loc >= n:
        while loc >= n:
            n *= 2
        values = np.resize(values, min(n, max_length))
    return values

# 在大于此大小的单调索引中不填充哈希表
_SIZE_CUTOFF = 1_000_000

cdef _unpack_bool_indexer(ndarray[uint8_t, ndim=1, cast=True] indexer, object val):
    """
    Possibly unpack a boolean mask to a single indexer.
    """
    # 返回类型为 ndarray[bool] 或 int
    cdef:
        # 定义一个一维的 ndarray，其元素类型为 intp_t（通常表示指针大小的整数）
        ndarray[intp_t, ndim=1] found
        # 定义一个整数变量 count
        int count
    
    # 使用 np.where 函数查找满足条件的索引，返回一个一维 ndarray
    found = np.where(indexer)[0]
    # 统计找到的索引数量
    count = len(found)
    
    # 如果找到的索引数量大于 1，返回原始的 indexer（假设 indexer 是 ndarray[bool]）
    if count > 1:
        return indexer
    # 如果找到的索引数量等于 1，返回找到的索引对应的整数值
    if count == 1:
        return int(found[0])
    
    # 如果找到的索引数量为 0，则抛出 KeyError 异常，val 作为异常信息的参数
    raise KeyError(val)
@cython.freelist(32)
cdef class IndexEngine:
    # 定义 Cython 类 IndexEngine

    cdef readonly:
        ndarray values  # 保存索引引擎的值数组
        ndarray mask  # 索引引擎的掩码数组
        HashTable mapping  # 哈希表用于值的映射
        bint over_size_threshold  # 是否超过大小阈值的布尔标志

    cdef:
        bint unique, monotonic_inc, monotonic_dec  # 布尔标志：唯一值，单调递增，单调递减
        bint need_monotonic_check, need_unique_check  # 布尔标志：需要检查单调性，需要检查唯一性
        object _np_type  # NumPy 数据类型

    def __init__(self, ndarray values):
        # IndexEngine 类的初始化方法
        self.values = values  # 初始化值数组
        self.mask = None  # 初始化掩码为 None

        self.over_size_threshold = len(values) >= _SIZE_CUTOFF  # 判断是否超过预设大小阈值
        self.clear_mapping()  # 调用清空映射方法
        self._np_type = values.dtype.type  # 初始化 NumPy 数据类型

    def __contains__(self, val: object) -> bool:
        # 检查值是否在索引引擎中
        hash(val)  # 计算值的哈希值
        try:
            self.get_loc(val)  # 尝试获取值的位置
        except KeyError:
            return False  # 如果发生 KeyError 则返回 False
        return True  # 否则返回 True

    cpdef get_loc(self, object val):
        # 获取值的位置方法
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t loc  # 声明 Py_ssize_t 类型的 loc 变量

        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")  # 如果值无效则抛出 TypeError

        val = self._check_type(val)  # 检查值的类型

        if self.over_size_threshold and self.is_monotonic_increasing:
            # 如果超过大小阈值并且是单调递增的情况
            if not self.is_unique:
                return self._get_loc_duplicates(val)  # 返回重复值的位置
            values = self.values

            loc = self._searchsorted_left(val)  # 使用左侧二分查找获取位置
            if loc >= len(values):
                raise KeyError(val)  # 如果位置超过数组长度则抛出 KeyError
            if values[loc] != val:
                raise KeyError(val)  # 如果找到的值与目标值不匹配则抛出 KeyError
            return loc  # 返回位置

        self._ensure_mapping_populated()  # 确保映射已填充
        if not self.unique:
            return self._get_loc_duplicates(val)  # 返回重复值的位置
        if self.mask is not None and val is C_NA:
            return self.mapping.get_na()  # 返回缺失值的位置

        try:
            return self.mapping.get_item(val)  # 尝试获取值的映射项
        except OverflowError as err:
            # 处理溢出错误
            # GH#41775 OverflowError e.g. if we are uint64 and val is -1
            #  or if we are int64 and value is np.iinfo(np.int64).max+1
            #  (the uint64 with -1 case should actually be excluded by _check_type)
            raise KeyError(val) from err  # 抛出 KeyError

    cdef Py_ssize_t _searchsorted_left(self, val) except? -1:
        """
        See ObjectEngine._searchsorted_left.__doc__.
        """
        # 查找左侧位置的方法，调用方负责确保已调用 _check_type
        loc = self.values.searchsorted(self._np_type(val), side="left")  # 使用左侧二分查找获取位置
        return loc  # 返回位置
    cdef _get_loc_duplicates(self, object val):
        # 定义一个 CPython 函数，用于获取重复的位置索引或者布尔数组
        # 返回类型可以是 Py_ssize_t、slice 或者 ndarray[bool]
        cdef:
            Py_ssize_t diff, left, right

        if self.is_monotonic_increasing:
            # 如果当前索引是单调递增的，获取其值
            values = self.values
            try:
                # 使用二分查找（binary search）获取值在索引中的左右位置
                left = values.searchsorted(val, side="left")
                right = values.searchsorted(val, side="right")
            except TypeError:
                # 抛出 KeyError 如果类型错误，例如对于 Float64Index 的 None 值查询
                # 2021-09-29 现在只有对象类型索引会触发这种情况
                raise KeyError(val)

            # 计算左右位置之间的差异
            diff = right - left
            if diff == 0:
                # 如果差异为 0，即未找到对应值，抛出 KeyError
                raise KeyError(val)
            elif diff == 1:
                # 如果差异为 1，返回左位置
                return left
            else:
                # 否则返回一个切片对象，表示找到的范围
                return slice(left, right)

        # 如果不是单调递增，尝试获取布尔型索引器
        return self._maybe_get_bool_indexer(val)

    cdef _maybe_get_bool_indexer(self, object val):
        # 尝试获取布尔型索引器，返回 ndarray[bool] 或者 int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer

        # 调用 C 函数获取布尔型索引器
        indexer = _get_bool_indexer(self.values, val, self.mask)
        # 解包布尔型索引器，返回结果
        return _unpack_bool_indexer(indexer, val)

    def sizeof(self, deep: bool = False) -> int:
        """ 返回映射的大小 """
        if not self.is_mapping_populated:
            # 如果映射未被填充，则大小为 0
            return 0
        # 否则返回映射的大小，可以选择是否递归深度计算
        return self.mapping.sizeof(deep=deep)

    def __sizeof__(self) -> int:
        # 返回对象的大小
        return self.sizeof()

    cpdef _update_from_sliced(self, IndexEngine other, reverse: bool):
        # 从切片的索引引擎更新当前索引对象
        self.unique = other.unique
        self.need_unique_check = other.need_unique_check
        if not other.need_monotonic_check and (
                other.is_monotonic_increasing or other.is_monotonic_decreasing):
            # 如果不需要单调性检查，但是索引是单调递增或递减的，设置相应的检查标志和单调性属性
            self.need_monotonic_check = other.need_monotonic_check
            # 如果 reverse=True 表示索引已经被反转
            self.monotonic_inc = other.monotonic_dec if reverse else other.monotonic_inc
            self.monotonic_dec = other.monotonic_inc if reverse else other.monotonic_dec

    @property
    def is_unique(self) -> bool:
        # 检查索引是否唯一
        # 查看为什么在这里需要检查 is_monotonic_increasing，请参阅：
        # https://github.com/pandas-dev/pandas/pull/55342#discussion_r1361405781
        if self.need_monotonic_check:
            self.is_monotonic_increasing
        if self.need_unique_check:
            # 进行唯一性检查
            self._do_unique_check()

        return self.unique == 1

    cdef _do_unique_check(self):
        # 执行唯一性检查前确保映射已被填充
        self._ensure_mapping_populated()

    @property
    def is_monotonic_increasing(self) -> bool:
        # 检查索引是否单调递增
        if self.need_monotonic_check:
            # 在需要单调性检查时执行
            self._do_monotonic_check()

        return self.monotonic_inc == 1

    @property
    def is_monotonic_decreasing(self) -> bool:
        # 检查索引是否单调递减
        if self.need_monotonic_check:
            # 在需要单调性检查时执行
            self._do_monotonic_check()

        return self.monotonic_dec == 1
    # 执行单调性检查的私有方法
    cdef _do_monotonic_check(self):
        cdef:
            bint is_strict_monotonic  # 是否严格单调的标志位
        
        # 如果存在掩码并且有任何非零元素
        if self.mask is not None and np.any(self.mask):
            self.monotonic_inc = 0  # 升序标志置为0
            self.monotonic_dec = 0  # 降序标志置为0
        else:
            try:
                values = self.values  # 获取数据值
                # 调用单调性检查函数，并返回升序、降序标志以及是否严格单调的结果
                self.monotonic_inc, self.monotonic_dec, is_strict_monotonic = \
                    self._call_monotonic(values)
            except (TypeError, InvalidOperation, ValueError):
                # 处理异常时，将升序、降序标志置为0，并将严格单调标志置为0
                self.monotonic_inc = 0
                self.monotonic_dec = 0
                is_strict_monotonic = 0

            self.need_monotonic_check = 0  # 不再需要单调性检查

            # 只有在严格单调时，我们才能确定唯一性
            if is_strict_monotonic:
                self.unique = 1  # 唯一性标志置为1
                self.need_unique_check = 0  # 不再需要唯一性检查

    # 调用算法函数检查数据是否单调
    cdef _call_monotonic(self, values):
        return algos.is_monotonic(values, timelike=False)

    # 生成哈希表的私有方法，抛出未实现异常
    cdef _make_hash_table(self, Py_ssize_t n):
        raise NotImplementedError  # pragma: no cover

    # 检查对象类型的私有方法，计算其哈希值并返回对象本身
    cdef _check_type(self, object val):
        hash(val)  # 计算对象的哈希值
        return val  # 返回对象本身

    @property
    def is_mapping_populated(self) -> bool:
        return self.mapping is not None  # 返回映射是否已填充的布尔值

    # 确保映射已填充的私有方法
    cdef _ensure_mapping_populated(self):
        # 如果映射未填充
        if not self.is_mapping_populated:
            values = self.values  # 获取数据值
            self.mapping = self._make_hash_table(len(values))  # 创建哈希表并映射数据值及其掩码
            self.mapping.map_locations(values, self.mask)  # 映射数据值及其掩码的位置

            # 如果哈希表长度与数据值长度相等，则唯一性标志置为1
            if len(self.mapping) == len(values):
                self.unique = 1

        self.need_unique_check = 0  # 不再需要唯一性检查

    # 清除映射的方法
    def clear_mapping(self):
        self.mapping = None  # 清空映射
        self.need_monotonic_check = 1  # 需要重新进行单调性检查
        self.need_unique_check = 1  # 需要重新进行唯一性检查

        self.unique = 0  # 唯一性标志置为0
        self.monotonic_inc = 0  # 升序标志置为0
        self.monotonic_dec = 0  # 降序标志置为0

    # 获取索引器的方法，确保映射已填充，并返回查找到的索引数组
    def get_indexer(self, ndarray values) -> np.ndarray:
        self._ensure_mapping_populated()  # 确保映射已填充
        return self.mapping.lookup(values)  # 返回查找到的索引数组
cdef Py_ssize_t _bin_search(ndarray values, object val) except -1:
    # GH#1757 ndarray.searchsorted is not safe to use with array of tuples
    # (treats a tuple `val` as a sequence of keys instead of a single key),
    # so we implement something similar.
    # This is equivalent to the stdlib's bisect.bisect_left

    cdef:
        Py_ssize_t mid = 0, lo = 0, hi = len(values) - 1
        object pval

    if hi == 0 or (hi > 0 and val > PySequence_GetItem(values, hi)):
        # If `values` is empty or `val` is greater than the last item in `values`,
        # return the length of `values`, indicating `val` should be inserted at the end.
        return len(values)

    while lo < hi:
        mid = (lo + hi) // 2
        pval = PySequence_GetItem(values, mid)
        if val < pval:
            hi = mid
        elif val > pval:
            lo = mid + 1
        else:
            # If `val` is found at `mid`, adjust `mid` to the first occurrence of `val`.
            while mid > 0 and val == PySequence_GetItem(values, mid - 1):
                mid -= 1
            return mid

    # If `val` is less than or equal to the value at `mid`, return `mid`.
    # Otherwise, return `mid + 1` indicating `val` should be inserted at `mid + 1`.
    if val <= PySequence_GetItem(values, mid):
        return mid
    else:
        return mid + 1


cdef class ObjectEngine(IndexEngine):
    """
    Index Engine for use with object-dtype Index, namely the base class Index.
    """
    cdef _make_hash_table(self, Py_ssize_t n):
        # Create a hash table of type PyObjectHashTable with size `n`.
        return _hash.PyObjectHashTable(n)

    cdef Py_ssize_t _searchsorted_left(self, val) except? -1:
        # using values.searchsorted here would treat a tuple `val` as a sequence
        # instead of a single key, so we use a different implementation
        try:
            # Perform binary search using `_bin_search` to find `val` in `self.values`.
            loc = _bin_search(self.values, val)
        except TypeError as err:
            # If a TypeError occurs during `_bin_search`, raise a KeyError with `val`.
            raise KeyError(val) from err
        return loc


cdef class StringEngine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        # Create a hash table of type StringHashTable with size `n`.
        return _hash.StringHashTable(n)

    cdef _check_type(self, object val):
        # Check if `val` is not a string, raise a KeyError with `val`.
        if not isinstance(val, str):
            raise KeyError(val)
        return str(val)


cdef class DatetimeEngine(Int64Engine):

    cdef:
        NPY_DATETIMEUNIT _creso

    def __init__(self, ndarray values):
        super().__init__(values.view("i8"))
        # Initialize `_creso` with the datetime unit derived from `values.dtype`.
        self._creso = get_unit_from_dtype(values.dtype)

    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        # NB: caller is responsible for ensuring tzawareness compat
        # before we get here
        if scalar is NaT:
            # If `scalar` is NaT (Not-a-Time), return its internal value `_value`.
            return NaT._value
        elif isinstance(scalar, _Timestamp):
            if scalar._creso == self._creso:
                # If `scalar` has the same resolution `_creso`, return its value `_value`.
                return scalar._value
            else:
                # If `scalar` resolution differs, convert `scalar` to current `_creso`
                # and return the resulting value.
                return (
                    (<_Timestamp>scalar)._as_creso(self._creso, round_ok=False)._value
                )
        # If `scalar` is neither NaT nor `_Timestamp`, raise a TypeError.
        raise TypeError(scalar)

    def __contains__(self, val: object) -> bool:
        # We assume before we get here:
        # - `val` is hashable
        try:
            # Attempt to unbox `val` using `_unbox_scalar`.
            self._unbox_scalar(val)
        except ValueError:
            # If `ValueError` occurs during unboxing, `val` is not in the engine.
            return False

        try:
            # Attempt to locate `val` using `get_loc`.
            self.get_loc(val)
            return True
        except KeyError:
            # If `KeyError` is raised during `get_loc`, `val` is not in the engine.
            return False
    # 定义一个 CPython 扩展函数，用于调用算法库判断给定的值是否单调递增或递减
    cdef _call_monotonic(self, values):
        return algos.is_monotonic(values, timelike=True)

    # 定义一个 CPython 扩展函数，用于获取对象在索引中的位置
    cpdef get_loc(self, object val):
        # 注意：调用者需确保传入的值是 Timestamp 或 NaT（对于 Timedelta 或 NaT 的 TimedeltaEngine）

        cdef:
            Py_ssize_t loc  # 声明一个 CPython 中的 ssize_t 类型变量 loc

        # 检查是否为无效键值，若是则抛出类型错误异常
        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")

        try:
            conv = self._unbox_scalar(val)  # 尝试解析标量值 conv
        except (TypeError, ValueError) as err:
            raise KeyError(val) from err  # 解析失败则抛出键错误异常

        # 欢迎来到意大利面厂
        # 如果超过阈值并且是单调递增的情况下
        if self.over_size_threshold and self.is_monotonic_increasing:
            # 如果不是唯一值，则调用 _get_loc_duplicates 方法
            if not self.is_unique:
                return self._get_loc_duplicates(conv)
            values = self.values  # 获取实例的值

            # 在值中搜索 conv 的位置，返回左侧最近的索引
            loc = values.searchsorted(conv, side="left")

            # 如果 loc 等于值的长度或者 values[loc] 不等于 conv，则抛出键错误异常
            if loc == len(values) or PySequence_GetItem(values, loc) != conv:
                raise KeyError(val)
            return loc  # 返回位置索引

        self._ensure_mapping_populated()  # 确保映射已填充
        if not self.unique:
            return self._get_loc_duplicates(conv)  # 如果不唯一，则调用 _get_loc_duplicates 方法

        try:
            return self.mapping.get_item(conv)  # 尝试从映射中获取项
        except KeyError:
            raise KeyError(val)  # 获取失败则抛出键错误异常
cdef class TimedeltaEngine(DatetimeEngine):

    # 定义私有方法 _unbox_scalar，用于将标量解包成整数值
    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        # 检查标量是否为 NaT（Not a Time）
        if scalar is NaT:
            return NaT._value
        # 如果标量是 _Timedelta 类型
        elif isinstance(scalar, _Timedelta):
            # 检查 _Timedelta 对象的 _creso 属性是否与当前对象的 _creso 相同
            if scalar._creso == self._creso:
                return scalar._value
            else:
                # 注意：调用方需处理可能从 _as_creso 方法中引发的 ValueError 异常
                #  使用 _as_creso 方法将 scalar 转换为当前对象的 _creso，不允许舍入
                return (
                    (<_Timedelta>scalar)._as_creso(self._creso, round_ok=False)._value
                )
        # 若标量不是 NaT 也不是 _Timedelta 类型，则引发 TypeError 异常
        raise TypeError(scalar)


cdef class PeriodEngine(Int64Engine):

    # 定义私有方法 _unbox_scalar，用于将标量解包成整数值
    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        # 检查标量是否为 NaT（Not a Time）
        if scalar is NaT:
            return scalar._value
        # 检查标量是否为期间对象（Period）
        if is_period_object(scalar):
            # 注意：假设此处标量的频率已正确设置
            #  返回标量的 ordinal 属性值
            return scalar.ordinal
        # 若标量既不是 NaT 也不是期间对象，则引发 TypeError 异常
        raise TypeError(scalar)

    # 定义公共方法 get_loc，用于获取对象的位置
    cpdef get_loc(self, object val):
        # 注意：调用方需确保以 Period 或 NaT 调用本方法
        cdef:
            int64_t conv

        try:
            # 尝试将 val 解包成整数值
            conv = self._unbox_scalar(val)
        except TypeError:
            # 若解包失败，则引发 KeyError 异常
            raise KeyError(val)

        # 调用父类方法 Int64Engine.get_loc 获取 conv 的位置
        return Int64Engine.get_loc(self, conv)

    # 定义私有方法 _call_monotonic，调用 algos.is_monotonic 方法
    cdef _call_monotonic(self, values):
        return algos.is_monotonic(values, timelike=True)


cdef class BaseMultiIndexCodesEngine:
    """
    MultiIndexUIntEngine 和 MultiIndexPyIntEngine 的基类，将 MultiIndex 中的每个标签
    表示为整数，通过拼接各级别编码的位并使用适当的偏移量。

    例如：如果有3个级别，分别具有3、6和1个可能的值，则它们的标签可以使用2、3和1位来表示，
    如下所示：
     _ _ _ _____ _ __ __ __
    |0|0|0| ... |0| 0|a1|a0| -> 偏移量0（第一级别）
     — — — ————— — —— —— ——
    |0|0|0| ... |0|b2|b1|b0| -> 偏移量2（第一级别所需的位数）
     — — — ————— — —— —— ——
    |0|0|0| ... |0| 0| 0|c0| -> 偏移量5（前两个级别所需的位数）
     ‾ ‾ ‾ ‾‾‾‾‾ ‾ ‾‾ ‾‾ ‾‾
    最终得到的无符号整数表示为：
     _ _ _ _____ _ __ __ __ __ __ __
    |0|0|0| ... |0|c0|b2|b1|b0|a1|a0|
     ‾ ‾ ‾ ‾‾‾‾‾ ‾ ‾‾ ‾‾ ‾‾ ‾‾ ‾‾ ‾‾

    初始化时计算偏移量，在方法 _codes_to_ints 中转换标签。

    通过首先定位每个组件到各自级别，然后定位（整数表示的）编码来定位键。
    """
    def __init__(self, levels, labels, offsets):
        """
        Parameters
        ----------
        levels : list-like of numpy arrays
            MultiIndex 的各个层级数据。
        labels : list-like of numpy arrays of integer dtype
            MultiIndex 的标签数据。
        offsets : numpy array of int dtype
            预先计算的偏移量，每个索引级别对应一个偏移量。
        """
        # 将传入的 levels 和 offsets 分配给对象的属性
        self.levels = levels
        self.offsets = offsets

        # 转换标签数据为一个单一的数组，并且加 2，确保处理的是正整数
        codes = np.array(labels).T
        codes += multiindex_nulls_shift  # 原地操作优化求和

        # 检查每个层级的标签中是否存在 NaN (-1)，并将结果存储在列表中
        self.level_has_nans = [-1 in lab for lab in labels]

        # 将标签组合映射为整数，确保无冲突的映射，基于预先计算的偏移量
        lab_ints = self._codes_to_ints(codes)

        # 使用整数标签初始化底层索引 (例如 libindex.UInt64Engine)
        # 这里将使用它的 get_loc 和 get_indexer 方法
        self._base.__init__(self, lab_ints)

    def _codes_to_ints(self, codes) -> np.ndarray:
        """
        将多个 uint 组合或单个 uint 或 Python 整数，按严格单调递增的方式转换为整数。

        Parameters
        ----------
        codes : 1- or 2-dimensional array of dtype uint
            整数组合 (每行一个)

        Returns
        -------
        scalar or 1-dimensional array, of dtype _codes_dtype
            表示组合的整数或整数数组。
        """
        # 确保使用正确的 dtype，避免溢出
        codes = codes.astype(self._codes_dtype, copy=False)

        # 根据预先计算的位移量，移动每个级别的表示
        codes <<= self.offsets  # 原地移位优化

        # 现在求和和按位或操作实际上是可以互换的。这是将每个级别的显著位 (即 "codes" 中的每列) 组合为单个正整数的简单方法
        if codes.ndim == 1:
            # 单个键
            return np.bitwise_or.reduce(codes)

        # 多个键
        return np.bitwise_or.reduce(codes, axis=1)
    def _extract_level_codes(self, target) -> np.ndarray:
        """
        将请求的多级索引键列表映射到其整数表示，以便在底层整数索引中进行搜索。

        Parameters
        ----------
        target : MultiIndex
            要映射的目标多级索引对象

        Returns
        ------
        int_keys : 1维数组，dtype为uint64或object
            每个组合的整数表示
        """
        # 获取目标多级索引的重新编码后的级别代码列表
        level_codes = list(target._recode_for_new_levels(self.levels))
        for i, codes in enumerate(level_codes):
            # 如果当前级别含有 NaN 值，则用 NaN 的索引替换对应的 -1
            if self.levels[i].hasnans:
                na_index = self.levels[i].isna().nonzero()[0][0]
                codes[target.codes[i] == -1] = na_index
            # 对每个代码加 1
            codes += 1
            # 将大于 0 的代码再加 1
            codes[codes > 0] += 1
            # 如果该级别含有 NaN 值，对应的代码再加 1
            if self.level_has_nans[i]:
                codes[target.codes[i] == -1] += 1
        # 将代码转换为整数并返回
        return self._codes_to_ints(np.array(level_codes, dtype=self._codes_dtype).T)

    def get_indexer(self, target: np.ndarray) -> np.ndarray:
        """
        返回一个数组，给出 `target` 中每个值在 `self.values` 中的位置，
        其中 -1 表示 `target` 中的某个值不在 `self.values` 中出现。

        Parameters
        ----------
        target : np.ndarray
            要查询索引的目标数组

        Returns
        -------
        np.ndarray[intp_t, ndim=1]
            `target` 在 `self.values` 中的索引数组
        """
        return self._base.get_indexer(self, target)

    def get_loc(self, object key):
        """
        返回指定键 `key` 对应的位置。

        Parameters
        ----------
        key : object
            要查找位置的键值

        Returns
        -------
        int
            `key` 对应的位置
        """
        # 如果 `key` 明确无效，则引发 TypeError
        if is_definitely_invalid_key(key):
            raise TypeError(f"'{key}' 是无效的键")
        # 如果 `key` 不是元组，则引发 KeyError
        if not isinstance(key, tuple):
            raise KeyError(key)
        try:
            # 尝试获取每个级别对应值的位置，并转换为单个整数
            indices = [1 if checknull(v) else lev.get_loc(v) + multiindex_nulls_shift
                       for lev, v in zip(self.levels, key)]
        except KeyError:
            # 如果键错误，则重新引发 KeyError
            raise KeyError(key)

        # 将索引转换为整数并返回
        lab_int = self._codes_to_ints(np.array(indices, dtype=self._codes_dtype))

        return self._base.get_loc(self, lab_int)

    def get_indexer_non_unique(self, target: np.ndarray) -> np.ndarray:
        """
        返回 `target` 在 `self.values` 中的索引数组，允许重复项。

        Parameters
        ----------
        target : np.ndarray
            要查询索引的目标数组

        Returns
        -------
        np.ndarray[intp_t, ndim=1]
            `target` 在 `self.values` 中的索引数组
        """
        # 调用基础类的非唯一索引方法并返回结果
        indexer = self._base.get_indexer_non_unique(self, target)

        return indexer

    def __contains__(self, val: object) -> bool:
        """
        检查对象是否存在于当前实例中。

        Parameters
        ----------
        val : object
            要检查的对象

        Returns
        -------
        bool
            如果对象存在，则返回 True；否则返回 False
        """
        # 假设 `val` 是可哈希的
        # 默认的 __contains__ 方法查找底层映射，这里仅包含整数表示
        try:
            # 尝试获取对象的位置，如果成功则返回 True
            self.get_loc(val)
            return True
        except (KeyError, TypeError, ValueError):
            # 如果出现键错误、类型错误或值错误，则返回 False
            return False
# Generated from template.
# 导入 index_class_helper.pxi 文件

include "index_class_helper.pxi"


cdef class BoolEngine(UInt8Engine):
    # BoolEngine 类继承自 UInt8Engine

    cdef _check_type(self, object val):
        # 检查传入的值是否为布尔类型对象
        if not util.is_bool_object(val):
            raise KeyError(val)
        return <uint8_t>val


cdef class MaskedBoolEngine(MaskedUInt8Engine):
    # MaskedBoolEngine 类继承自 MaskedUInt8Engine

    cdef _check_type(self, object val):
        # 检查传入的值是否为布尔类型对象，如果是 C_NA 则直接返回
        if val is C_NA:
            return val
        if not util.is_bool_object(val):
            raise KeyError(val)
        return <uint8_t>val


@cython.internal
@cython.freelist(32)
cdef class SharedEngine:
    # SharedEngine 类定义开始

    cdef readonly:
        object values  # ExtensionArray
        bint over_size_threshold

    cdef:
        bint unique, monotonic_inc, monotonic_dec
        bint need_monotonic_check, need_unique_check

    def __contains__(self, val: object) -> bool:
        # 检查对象是否包含指定值
        # 假设传入的值是可哈希的
        try:
            self.get_loc(val)
            return True
        except KeyError:
            return False

    def clear_mapping(self):
        # 与 IndexEngine 兼容，清空映射
        pass

    cpdef _update_from_sliced(self, ExtensionEngine other, reverse: bool):
        # 从切片的 ExtensionEngine 更新当前引擎的属性
        self.unique = other.unique
        self.need_unique_check = other.need_unique_check
        if not other.need_monotonic_check and (
                other.is_monotonic_increasing or other.is_monotonic_decreasing):
            self.need_monotonic_check = other.need_monotonic_check
            # 如果 reverse=True 表示索引已被反转
            self.monotonic_inc = other.monotonic_dec if reverse else other.monotonic_inc
            self.monotonic_dec = other.monotonic_inc if reverse else other.monotonic_dec

    @property
    def is_unique(self) -> bool:
        # 检查 values 是否唯一
        # 如果需要检查单调性，则调用 is_monotonic_increasing 方法
        if self.need_monotonic_check:
            self.is_monotonic_increasing
        if self.need_unique_check:
            arr = self.values.unique()
            self.unique = len(arr) == len(self.values)

            self.need_unique_check = False
        return self.unique

    cdef _do_monotonic_check(self):
        # 执行单调性检查的方法
        raise NotImplementedError

    @property
    def is_monotonic_increasing(self) -> bool:
        # 检查是否单调递增
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_inc == 1

    @property
    def is_monotonic_decreasing(self) -> bool:
        # 检查是否单调递减
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_dec == 1

    cdef _call_monotonic(self, values):
        # 调用算法检查给定的值是否单调
        return algos.is_monotonic(values, timelike=False)

    def sizeof(self, deep: bool = False) -> int:
        """ return the sizeof our mapping """
        # 返回映射的大小
        return 0

    def __sizeof__(self) -> int:
        # 返回对象的大小
        return self.sizeof()

    cdef _check_type(self, object obj):
        # 检查对象类型的方法
        raise NotImplementedError
    # 定义一个 Cython 函数，获取指定值在对象中的位置索引或切片，可能返回整数、切片或布尔数组
    cpdef get_loc(self, object val):
        # 返回类型声明为 Py_ssize_t、slice 或 ndarray[bool]
        cdef:
            Py_ssize_t loc  # 用于存储位置索引的变量

        # 如果传入的值被确定为无效键，则抛出类型错误异常
        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")

        # 检查传入值的类型是否符合对象的预期类型
        self._check_type(val)

        # 如果对象设置了超过阈值的大小和单调递增属性
        if self.over_size_threshold and self.is_monotonic_increasing:
            # 如果对象不唯一，则调用处理重复键的方法
            if not self.is_unique:
                return self._get_loc_duplicates(val)

            # 否则，获取对象的值
            values = self.values

            # 使用二分查找在对象的值中找到 val 的左侧位置索引
            loc = self._searchsorted_left(val)

            # 如果位置索引超出了对象值的长度，则抛出键错误异常
            if loc >= len(values):
                raise KeyError(val)

            # 如果找到的位置索引处的值不等于 val，则抛出键错误异常
            if values[loc] != val:
                raise KeyError(val)

            # 返回找到的位置索引
            return loc

        # 如果对象不是唯一的，则调用处理重复键的方法
        if not self.unique:
            return self._get_loc_duplicates(val)

        # 否则，调用处理重复键的方法
        return self._get_loc_duplicates(val)

    # Cython 方法：处理对象中存在重复值时获取位置索引或切片
    cdef _get_loc_duplicates(self, object val):
        # 返回类型声明为 Py_ssize_t、slice 或 ndarray[bool]
        cdef:
            Py_ssize_t diff  # 用于存储位置差异的变量

        # 如果对象是单调递增的，则执行以下操作
        if self.is_monotonic_increasing:
            # 获取对象的值
            values = self.values

            # 尝试在对象的值中查找 val 的左侧和右侧位置索引
            try:
                left = values.searchsorted(val, side="left")
                right = values.searchsorted(val, side="right")
            except TypeError:
                # 捕获类型错误异常，例如 GH#29189：在 Float64Index 中使用 get_loc(None)
                raise KeyError(val)

            # 计算位置索引之间的差异
            diff = right - left

            # 如果差异为 0，则抛出键错误异常
            if diff == 0:
                raise KeyError(val)
            # 如果差异为 1，则返回左侧位置索引
            elif diff == 1:
                return left
            # 否则，返回左侧到右侧的切片
            else:
                return slice(left, right)

        # 否则，调用处理可能返回布尔索引器的方法
        return self._maybe_get_bool_indexer(val)

    # Cython 方法：执行二分查找获取 val 的左侧位置索引
    cdef Py_ssize_t _searchsorted_left(self, val) except? -1:
        """
        See ObjectEngine._searchsorted_left.__doc__.
        """
        # 尝试使用二分查找在对象的值中获取 val 的左侧位置索引
        try:
            loc = self.values.searchsorted(val, side="left")
        except TypeError as err:
            # 捕获类型错误异常，例如 GH#35788：val=None 且对象值为 float64
            raise KeyError(val)

        # 返回找到的位置索引
        return loc

    # Cython 方法：抛出未实现错误，暂不支持处理布尔索引器
    cdef ndarray _get_bool_indexer(self, val):
        raise NotImplementedError

    # Cython 方法：处理可能返回布尔索引器的值，返回解包后的布尔索引器或整数
    cdef _maybe_get_bool_indexer(self, object val):
        # 返回类型声明为 ndarray[bool] 或 int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer  # 用于存储布尔索引器的变量

        # 获取布尔索引器
        indexer = self._get_bool_indexer(val)

        # 返回解包后的布尔索引器或整数
        return _unpack_bool_indexer(indexer, val)

    # 普通方法：获取指定值的索引器
    def get_indexer(self, values) -> np.ndarray:
        # values 的类型为 type(self.values)
        # 注意：只有当对象是唯一的时候才会进入此方法

        # 声明变量 i 和 N，分别表示 values 的长度和索引
        cdef:
            Py_ssize_t i, N = len(values)

        # 创建一个长度为 N 的空数组 res，数据类型为 np.intp
        res = np.empty(N, dtype=np.intp)

        # 遍历 values 中的每个元素
        for i in range(N):
            # 获取 values 中的第 i 个元素
            val = PySequence_GetItem(values, i)

            # 尝试获取 val 的位置索引，如果找不到则将 loc 设为 -1
            try:
                loc = self.get_loc(val)
                # 因为我们是唯一的，loc 应该总是一个整数
            except KeyError:
                loc = -1
            else:
                # 断言 loc 是整数对象，如果不是则抛出 AssertionError
                assert util.is_integer_object(loc), (loc, val)

            # 将 loc 的值赋给 res 数组中的第 i 个位置
            res[i] = loc

        # 返回结果数组 res
        return res
    # 定义一个方法，返回一个适用于从非唯一索引中获取索引器的函数
    # 返回的标签与目标顺序相同，并返回一个指向目标中缺失索引位置的索引器
    # Parameters 参数
    # ----------
    # targets : type(self.values)
    # 返回
    # -------
    # indexer : np.ndarray[np.intp]
    # missing : np.ndarray[np.intp]
    def get_indexer_non_unique(self, targets):
        # 使用Cython定义变量i和N，N为targets的长度
        cdef:
            Py_ssize_t i, N = len(targets)

        # 初始化空列表
        indexer = []
        missing = []

        # 查看 IntervalIndex.get_indexer_pointwise 了解更多信息
        for i in range(N):
            # 获取目标中第i个元素的值
            val = PySequence_GetItem(targets, i)

            try:
                # 尝试获取val在当前索引中的位置
                locs = self.get_loc(val)
            except KeyError:
                # 如果val不存在，则创建一个包含-1的数组，并将当前索引i加入missing列表
                locs = np.array([-1], dtype=np.intp)
                missing.append(i)
            else:
                if isinstance(locs, slice):
                    # 仅在get_indexer_non_unique中需要
                    # 如果locs是切片对象，则生成一个整数数组
                    locs = np.arange(locs.start, locs.stop, locs.step, dtype=np.intp)
                elif util.is_integer_object(locs):
                    # 如果locs是整数对象，则生成一个包含该整数的整数数组
                    locs = np.array([locs], dtype=np.intp)
                else:
                    # 否则locs应为布尔数组，获取其非零元素的索引位置
                    assert locs.dtype.kind == "b"
                    locs = locs.nonzero()[0]

            # 将处理后的locs添加到indexer列表中
            indexer.append(locs)

        try:
            # 尝试将indexer列表中的所有数组连接成一个大数组
            indexer = np.concatenate(indexer, dtype=np.intp)
        except TypeError:
            # 处理numpy<1.20版本不接受dtype关键字的情况
            indexer = np.concatenate(indexer).astype(np.intp, copy=False)
        
        # 将missing列表转换为numpy数组
        missing = np.array(missing, dtype=np.intp)

        # 返回生成的indexer数组和missing数组
        return indexer, missing
cdef class ExtensionEngine(SharedEngine):
    # ExtensionEngine 类的定义，继承自 SharedEngine
    def __init__(self, values: "ExtensionArray"):
        # ExtensionEngine 类的初始化方法，接受一个名为 values 的 ExtensionArray 对象参数
        self.values = values

        # 检查 values 的长度是否超过 _SIZE_CUTOFF，设置一个布尔值标志
        self.over_size_threshold = len(values) >= _SIZE_CUTOFF
        # 需要进行唯一性检查的标志，默认为 True
        self.need_unique_check = True
        # 需要进行单调性检查的标志，默认为 True
        self.need_monotonic_check = True
        # 再次设置需要进行唯一性检查的标志，确保唯一性检查会被执行
        self.need_unique_check = True

    cdef _do_monotonic_check(self):
        # ExtensionEngine 类的私有方法，执行单调性检查
        cdef:
            bint is_unique  # 声明一个布尔值变量 is_unique

        values = self.values  # 将 self.values 赋值给局部变量 values
        if values._hasna:
            # 如果 values 包含缺失值，则设置单调递增和单调递减为 0
            self.monotonic_inc = 0
            self.monotonic_dec = 0

            nunique = len(values.unique())  # 计算 values 中的唯一值数量
            self.unique = nunique == len(values)  # 检查 values 是否完全唯一，并设置标志
            self.need_unique_check = 0  # 不再需要进行唯一性检查
            return

        try:
            ranks = values._rank()  # 尝试计算 values 的排名

        except TypeError:
            # 如果排名计算出错，则设置单调递增和单调递减为 0，is_unique 为 0
            self.monotonic_inc = 0
            self.monotonic_dec = 0
            is_unique = 0
        else:
            # 如果排名计算成功，则调用 _call_monotonic 方法计算单调性信息
            self.monotonic_inc, self.monotonic_dec, is_unique = \
                self._call_monotonic(ranks)

        self.need_monotonic_check = 0  # 不再需要进行单调性检查

        # 如果 is_unique 为真，则可以确保唯一性
        if is_unique:
            self.unique = 1  # 设置唯一性标志为真
            self.need_unique_check = 0  # 不再需要进行唯一性检查

    cdef ndarray _get_bool_indexer(self, val):
        # ExtensionEngine 类的私有方法，根据给定的值返回布尔索引器
        if checknull(val):
            return self.values.isna()  # 如果 val 为 null，返回 values 的缺失值索引器

        try:
            return self.values == val  # 尝试比较 values 和 val，返回布尔值索引器
        except TypeError:
            # 如果比较出错，则尝试转换为布尔值索引器
            try:
                return (self.values == val).to_numpy(dtype=bool, na_value=False)
            except (TypeError, AttributeError) as err:
                # 如果转换失败，抛出 KeyError 异常
                # 可能的情况有：返回的是布尔数组而不是 ndarray[bool]，或者 val 没有长度等错误
                raise KeyError from err

    cdef _check_type(self, object val):
        # ExtensionEngine 类的私有方法，检查给定对象的类型并计算其哈希值
        hash(val)


cdef class MaskedIndexEngine(IndexEngine):
    # MaskedIndexEngine 类的定义，继承自 IndexEngine
    def __init__(self, object values):
        # MaskedIndexEngine 类的初始化方法，接受一个名为 values 的对象参数
        super().__init__(self._get_data(values))  # 调用父类 IndexEngine 的初始化方法，并传入数据

        # 获取 values 的掩码数据并保存在 self.mask 中
        self.mask = self._get_mask(values)

    def _get_data(self, object values) -> np.ndarray:
        # MaskedIndexEngine 类的私有方法，根据 values 返回数据的 numpy 数组
        if hasattr(values, "_mask"):
            return values._data  # 如果 values 包含 _mask 属性，返回其数据
        # 否则，假设为 ArrowExtensionArray，将其转换为 numpy 数组，设置缺失值为 1
        # TODO: 当 arrow 引擎实现后应移除该部分
        return values.to_numpy(na_value=1, dtype=values.dtype.numpy_dtype)

    def _get_mask(self, object values) -> np.ndarray:
        # MaskedIndexEngine 类的私有方法，根据 values 返回掩码数据的 numpy 数组
        if hasattr(values, "_mask"):
            return values._mask  # 如果 values 包含 _mask 属性，返回其掩码数据
        # 否则，假设为 ArrowExtensionArray，返回其缺失值的布尔掩码
        return values.isna()

    def get_indexer(self, object values) -> np.ndarray:
        # 获取 values 的索引器数组，并确保映射已填充
        self._ensure_mapping_populated()
        return self.mapping.lookup(self._get_data(values), self._get_mask(values))
```
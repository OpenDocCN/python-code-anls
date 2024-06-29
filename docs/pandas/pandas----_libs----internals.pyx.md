# `D:\src\scipysrc\pandas\pandas\_libs\internals.pyx`

```
# 导入必要的模块和类
from collections import defaultdict
import weakref

# 导入 Cython 相关模块和函数
cimport cython
from cpython.pyport cimport PY_SSIZE_T_MAX
from cpython.slice cimport PySlice_GetIndicesEx
from cython cimport Py_ssize_t

# 导入 NumPy 和相关类型
import numpy as np

cimport numpy as cnp
from numpy cimport (
    NPY_INTP,
    int64_t,
    intp_t,
    ndarray,
)

# 导入 NumPy C API 并初始化
cnp.import_array()

# 导入 pandas 内部算法函数
from pandas._libs.algos import ensure_int64

# 导入 pandas 内部工具函数和类型
from pandas._libs.util cimport (
    is_array,
    is_integer_object,
)


@cython.final
@cython.freelist(32)
# 定义 Cython 类 BlockPlacement
cdef class BlockPlacement:
    cdef:
        slice _as_slice  # 使用 Python 内置的 slice 类型
        ndarray _as_array  # 使用 NumPy 的 ndarray 类型；注意可能为 None，后续将被赋予 intp_t 类型
        bint _has_slice, _has_array, _is_known_slice_like

    # 构造函数，初始化对象
    def __cinit__(self, val):
        cdef:
            slice slc

        # 初始化成员变量
        self._as_slice = None
        self._as_array = None
        self._has_slice = False
        self._has_array = False

        # 根据输入值 val 的类型进行初始化
        if is_integer_object(val):  # 检查是否为整数对象
            slc = slice(val, val + 1, 1)  # 创建一个包含单个整数的 slice
            self._as_slice = slc
            self._has_slice = True
        elif isinstance(val, slice):  # 检查是否为 Python 内置的 slice 类型
            slc = slice_canonize(val)  # 规范化 slice
            if slc.start != slc.stop:
                self._as_slice = slc
                self._has_slice = True
            else:
                arr = np.empty(0, dtype=np.intp)  # 创建一个空的 intp_t 类型的 NumPy 数组
                self._as_array = arr
                self._has_array = True
        else:
            # 如果不是整数对象或 slice，则处理为 NumPy 数组
            # Cython 的内存视图接口要求 ndarray 必须是可写的
            if (
                not is_array(val)
                or not cnp.PyArray_ISWRITEABLE(val)
                or (<ndarray>val).descr.type_num != cnp.NPY_INTP
            ):
                arr = np.require(val, dtype=np.intp, requirements="W")  # 将输入值转换为 intp_t 类型的 NumPy 数组
            else:
                arr = val
            # 调用者需确保 arr.ndim == 1
            self._as_array = arr
            self._has_array = True

    # 返回对象的字符串表示形式
    def __str__(self) -> str:
        cdef:
            slice s = self._ensure_has_slice()  # 获取确保为 slice 的 slice 对象

        if s is not None:
            v = self._as_slice  # 返回 slice
        else:
            v = self._as_array  # 返回 ndarray

        return f"{type(self).__name__}({v})"

    # 返回对象的字符串表示形式（调用 __str__ 方法）
    def __repr__(self) -> str:
        return str(self)

    # 返回对象的长度
    def __len__(self) -> int:
        cdef:
            slice s = self._ensure_has_slice()  # 获取确保为 slice 的 slice 对象

        if s is not None:
            return slice_len(s)  # 返回 slice 的长度
        else:
            return len(self._as_array)  # 返回 ndarray 的长度

    # 返回对象的迭代器
    def __iter__(self):
        cdef:
            slice s = self._ensure_has_slice()  # 获取确保为 slice 的 slice 对象
            Py_ssize_t start, stop, step, _

        if s is not None:
            start, stop, step, _ = slice_get_indices_ex(s)  # 获取 slice 的起始、终止和步长
            return iter(range(start, stop, step))  # 返回 slice 的迭代器
        else:
            return iter(self._as_array)  # 返回 ndarray 的迭代器

    # 返回对象的 as_slice 属性
    @property
    def as_slice(self) -> slice:
        cdef:
            slice s = self._ensure_has_slice()  # 获取确保为 slice 的 slice 对象

        if s is not None:
            return s  # 返回 slice
        else:
            raise TypeError("Not slice-like")  # 如果不是 slice，则抛出类型错误

    # 确保对象具有 slice 的私有方法
    cdef slice _ensure_has_slice(self) -> slice:
        if self._has_slice:
            return self._as_slice
        else:
            return None
    # 返回当前对象的切片，确保已经初始化
    def indexer(self):
        cdef:
            slice s = self._ensure_has_slice()  # 获取确保已有的切片对象

        if s is not None:  # 如果切片对象不为 None，则返回该切片
            return s
        else:  # 否则返回作为数组的私有属性
            return self._as_array

    @property
    def as_array(self) -> np.ndarray:
        cdef:
            Py_ssize_t start, stop, _  # 定义 C 扩展类型的变量

        if not self._has_array:  # 如果没有数组缓存
            start, stop, step, _ = slice_get_indices_ex(self._as_slice)
            # 注意：这是相当于 C 优化版本的 `np.arange(start, stop, step, dtype=np.intp)`
            self._as_array = cnp.PyArray_Arange(start, stop, step, NPY_INTP)  # 使用 C 扩展库创建数组
            self._has_array = True  # 标记数组已经缓存

        return self._as_array  # 返回数组

    @property
    def is_slice_like(self) -> bool:
        cdef:
            slice s = self._ensure_has_slice()  # 获取确保已有的切片对象

        return s is not None  # 返回是否存在切片对象

    def __getitem__(self, loc):
        cdef:
            slice s = self._ensure_has_slice()  # 获取确保已有的切片对象

        if s is not None:  # 如果切片对象存在
            val = slice_getitem(s, loc)  # 使用切片获取元素
        else:
            val = self._as_array[loc]  # 否则直接从数组中获取元素

        if not isinstance(val, slice) and val.ndim == 0:  # 如果元素不是切片并且维度为 0
            return val  # 直接返回元素值

        return BlockPlacement(val)  # 否则返回封装后的对象

    def delete(self, loc) -> BlockPlacement:
        return BlockPlacement(np.delete(self.as_array, loc, axis=0))  # 删除数组中指定位置的元素，并返回新的封装对象

    def append(self, others) -> BlockPlacement:
        if not len(others):  # 如果输入为空列表
            return self  # 直接返回当前对象

        return BlockPlacement(
            np.concatenate([self.as_array] + [o.as_array for o in others])  # 连接当前数组和其他对象数组，返回新的封装对象
        )

    cdef BlockPlacement iadd(self, other):
        cdef:
            slice s = self._ensure_has_slice()  # 获取确保已有的切片对象
            Py_ssize_t other_int, start, stop, step  # 定义 C 扩展类型的变量

        if is_integer_object(other) and s is not None:  # 如果 other 是整数且存在切片对象
            other_int = <Py_ssize_t>other  # 将 other 转换为 Py_ssize_t 类型

            if other_int == 0:  # 如果 other_int 等于 0
                # BlockPlacement 被视为不可变对象，返回自身
                return self

            start, stop, step, _ = slice_get_indices_ex(s)  # 获取切片的起始、终止、步长和长度信息
            start += other_int  # 调整起始位置
            stop += other_int  # 调整终止位置

            if (step > 0 and start < 0) or (step < 0 and stop < step):  # 如果步长与调整后的范围不匹配
                raise ValueError("iadd causes length change")  # 抛出值错误异常，指示长度变化

            if stop < 0:  # 如果终止位置小于 0
                val = slice(start, None, step)  # 创建新的切片对象
            else:
                val = slice(start, stop, step)  # 创建新的切片对象

            return BlockPlacement(val)  # 返回封装后的对象
        else:
            newarr = self.as_array + other  # 将当前数组与 other 相加得到新的数组
            if (newarr < 0).any():  # 如果新数组中有负数元素
                raise ValueError("iadd causes length change")  # 抛出值错误异常，指示长度变化

            val = newarr  # 将新数组赋给 val
            return BlockPlacement(val)  # 返回封装后的对象

    def add(self, other) -> BlockPlacement:
        # 可以使用整数或 ndarray 进行操作
        return self.iadd(other)  # 调用 iadd 方法进行操作

    cdef slice _ensure_has_slice(self):
        if not self._has_slice:  # 如果没有切片缓存
            self._as_slice = indexer_as_slice(self._as_array)  # 将数组转换为切片对象
            self._has_slice = True  # 标记切片已经缓存

        return self._as_slice  # 返回切片对象
    # 定义一个 CPython 扩展函数，用于在 BlockPlacement 实例中将大于等于给定位置 loc 的所有索引值增加一
    cpdef BlockPlacement increment_above(self, Py_ssize_t loc):
        """
        Increment any entries of 'loc' or above by one.
        """
        cdef:
            slice nv, s = self._ensure_has_slice()  # 获取已确保为切片的 nv 变量，以及从 _ensure_has_slice 方法获取的 s 变量
            Py_ssize_t start, stop, step  # 定义用于存储切片索引的 start, stop, step 变量
            ndarray[intp_t, ndim=1] newarr  # 定义一个一维整数类型的 NumPy 数组 newarr

        if s is not None:
            # 查看是否完全在给定位置 loc 之上或之下，这两种情况下都有可用的快速路径。

            start, stop, step, _ = slice_get_indices_ex(s)  # 获取切片 s 的起始、终止、步长和长度信息

            if start < loc and stop <= loc:
                # 完全在 loc 之下，无需增加操作
                return self

            if start >= loc and stop >= loc:
                # 完全在 loc 之上，可以高效地增加切片范围
                nv = slice(start + 1, stop + 1, step)  # 增加切片的起始和终止位置
                return BlockPlacement(nv)

        if loc == 0:
            # 快速路径，其中我们知道所有值都大于等于 0
            newarr = self.as_array + 1  # 将 as_array 的所有元素加一
            return BlockPlacement(newarr)

        newarr = self.as_array.copy()  # 复制 as_array 数组
        newarr[newarr >= loc] += 1  # 将大于等于 loc 的元素加一
        return BlockPlacement(newarr)  # 返回一个新的 BlockPlacement 实例，包含增加后的数组

    def tile_for_unstack(self, factor: int) -> np.ndarray:
        """
        Find the new mgr_locs for the un-stacked version of a Block.
        """
        cdef:
            slice slc = self._ensure_has_slice()  # 获取已确保为切片的 slc 变量
            ndarray[intp_t, ndim=1] new_placement  # 定义一个一维整数类型的 NumPy 数组 new_placement

        if slc is not None and slc.step == 1:
            new_slc = slice(slc.start * factor, slc.stop * factor, 1)  # 创建一个新的切片，扩展了 factor 倍
            # 相当于使用 NumPy 创建等差数组：np.arange(new_slc.start, new_slc.stop, dtype=np.intp)
            new_placement = cnp.PyArray_Arange(new_slc.start, new_slc.stop, 1, NPY_INTP)
        else:
            # 注意：test_pivot_table_empty_aggfunc 在 slc 不为空的情况下会进入此分支
            mapped = [
                # 相当于使用 NumPy 创建等差数组：np.arange(x * factor, (x + 1) * factor, dtype=np.intp)
                cnp.PyArray_Arange(x * factor, (x + 1) * factor, 1, NPY_INTP)
                for x in self  # 对 self 中的每个元素进行迭代
            ]
            new_placement = np.concatenate(mapped)  # 将所有数组连接成一个新的数组

        return new_placement  # 返回新的位置数组
cdef slice slice_canonize(slice s):
    """
    Convert slice to canonical bounded form.
    """
    cdef:
        Py_ssize_t start = 0, stop = 0, step = 1  # 初始化起始、停止和步长变量

    if s.step is None:  # 如果切片对象的步长为空
        step = 1  # 将步长设为1（默认值）
    else:
        step = <Py_ssize_t>s.step  # 将步长转换为Py_ssize_t类型
        if step == 0:
            raise ValueError("slice step cannot be zero")  # 抛出异常，步长不能为零

    if step > 0:  # 如果步长大于0
        if s.stop is None:
            raise ValueError("unbounded slice")  # 如果停止值为空，抛出异常，切片未定义范围

        stop = <Py_ssize_t>s.stop  # 将停止值转换为Py_ssize_t类型
        if s.start is None:
            start = 0  # 如果起始值为空，则起始值设为0
        else:
            start = <Py_ssize_t>s.start  # 将起始值转换为Py_ssize_t类型
            if start > stop:
                start = stop  # 如果起始值大于停止值，则将起始值设为停止值
    elif step < 0:  # 如果步长小于0
        if s.start is None:
            raise ValueError("unbounded slice")  # 如果起始值为空，抛出异常，切片未定义范围

        start = <Py_ssize_t>s.start  # 将起始值转换为Py_ssize_t类型
        if s.stop is None:
            stop = -1  # 如果停止值为空，则停止值设为-1
        else:
            stop = <Py_ssize_t>s.stop  # 将停止值转换为Py_ssize_t类型
            if stop > start:
                stop = start  # 如果停止值大于起始值，则将停止值设为起始值

    if start < 0 or (stop < 0 and s.stop is not None and step > 0):
        raise ValueError("unbounded slice")  # 如果起始值或者（停止值小于0且不为空且步长大于0），抛出异常，切片未定义范围

    if stop < 0:
        return slice(start, None, step)  # 如果停止值小于0，返回包含起始值和步长的切片对象
    else:
        return slice(start, stop, step)  # 否则返回包含起始、停止和步长的切片对象


cpdef Py_ssize_t slice_len(slice slc, Py_ssize_t objlen=PY_SSIZE_T_MAX) except -1:
    """
    Get length of a bounded slice.

    The slice must not have any "open" bounds that would create dependency on
    container size, i.e.:
    - if ``s.step is None or s.step > 0``, ``s.stop`` is not ``None``
    - if ``s.step < 0``, ``s.start`` is not ``None``

    Otherwise, the result is unreliable.
    """
    cdef:
        Py_ssize_t start, stop, step, length  # 声明起始、停止、步长和长度变量

    if slc is None:
        raise TypeError("slc must be slice")  # 如果切片对象为空，抛出类型错误异常

    PySlice_GetIndicesEx(slc, objlen, &start, &stop, &step, &length)  # 调用PySlice_GetIndicesEx函数获取切片的起始、停止、步长和长度

    return length  # 返回切片的长度


cdef (Py_ssize_t, Py_ssize_t, Py_ssize_t, Py_ssize_t) slice_get_indices_ex(
    slice slc, Py_ssize_t objlen=PY_SSIZE_T_MAX
):
    """
    Get (start, stop, step, length) tuple for a slice.

    If `objlen` is not specified, slice must be bounded, otherwise the result
    will be wrong.
    """
    cdef:
        Py_ssize_t start, stop, step, length  # 声明起始、停止、步长和长度变量

    if slc is None:
        raise TypeError("slc should be a slice")  # 如果切片对象为空，抛出类型错误异常

    PySlice_GetIndicesEx(slc, objlen, &start, &stop, &step, &length)  # 调用PySlice_GetIndicesEx函数获取切片的起始、停止、步长和长度

    return start, stop, step, length  # 返回包含切片的起始、停止、步长和长度的元组


cdef slice_getitem(slice slc, ind):
    cdef:
        Py_ssize_t s_start, s_stop, s_step, s_len
        Py_ssize_t ind_start, ind_stop, ind_step, ind_len  # 声明切片起始、停止、步长和长度变量，以及索引起始、停止、步长和长度变量

    s_start, s_stop, s_step, s_len = slice_get_indices_ex(slc)  # 调用slice_get_indices_ex函数获取切片的起始、停止、步长和长度
    # 如果索引是一个切片对象
    if isinstance(ind, slice):
        # 使用函数 slice_get_indices_ex 获取切片的起始索引、终止索引、步长和长度
        ind_start, ind_stop, ind_step, ind_len = slice_get_indices_ex(ind, s_len)

        # 如果步长大于0并且切片长度等于序列长度，则直接返回原始切片
        if ind_step > 0 and ind_len == s_len:
            # 无操作切片的快捷方式
            if ind_len == s_len:
                return slc

        # 如果步长小于0，则调整起始位置
        if ind_step < 0:
            s_start = s_stop - s_step
            ind_step = -ind_step

        # 计算调整后的步长和起始、终止位置
        s_step *= ind_step
        s_stop = s_start + ind_stop * s_step
        s_start = s_start + ind_start * s_step

        # 如果步长小于0并且终止位置也小于0，则返回一个调整后的切片
        if s_step < 0 and s_stop < 0:
            return slice(s_start, None, s_step)
        else:
            return slice(s_start, s_stop, s_step)

    else:
        # 如果不是切片对象，返回一个说明性的注释
        # 这是对应于C优化的等效操作，用于实现 `np.arange(s_start, s_stop, s_step, dtype=np.intp)[ind]`
        return cnp.PyArray_Arange(s_start, s_stop, s_step, NPY_INTP)[ind]
# 定义一个 Cython 函数，用于将 intp_t 类型的数组 vals 转换为 slice 对象
@cython.boundscheck(False)
@cython.wraparound(False)
cdef slice indexer_as_slice(intp_t[:] vals):
    cdef:
        Py_ssize_t i, n, start, stop  # 定义变量 i, n, start, stop
        int64_t d  # 定义变量 d

    if vals is None:
        raise TypeError("vals must be ndarray")  # 如果 vals 是 None，则抛出类型错误异常，要求 vals 必须是 ndarray

    n = vals.shape[0]  # 获取 vals 数组的长度

    if n == 0 or vals[0] < 0:
        return None  # 如果 vals 的长度为 0 或第一个元素小于 0，则返回 None

    if n == 1:
        return slice(vals[0], vals[0] + 1, 1)  # 如果 vals 的长度为 1，则返回一个包含单个元素的 slice 对象

    if vals[1] < 0:
        return None  # 如果第二个元素小于 0，则返回 None

    # 当 n > 2 时
    d = vals[1] - vals[0]  # 计算第一个和第二个元素的差值作为步长 d

    if d == 0:
        return None  # 如果步长 d 等于 0，则返回 None

    for i in range(2, n):
        if vals[i] < 0 or vals[i] - vals[i - 1] != d:
            return None  # 如果出现负数或者当前元素与前一个元素的差值不等于 d，则返回 None

    start = vals[0]  # 获取序列的起始值
    stop = start + n * d  # 计算序列的结束值
    if stop < 0 and d < 0:
        return slice(start, None, d)  # 如果结束值小于 0 且步长 d 也小于 0，则返回一个 slice 对象
    else:
        return slice(start, stop, d)  # 否则返回包含起始值、结束值和步长的 slice 对象


# 定义一个 Cython 函数，用于生成多个 BlockManagers 的块编号索引器
@cython.boundscheck(False)
@cython.wraparound(False)
def get_concat_blkno_indexers(list blknos_list not None):
    """
    给定多个 BlockManagers 的 blknos，将 range(len(mgrs[0])) 拆分成若干段，
    每段内每个 i 对应的 blknos_list[i] 是恒定的。

    这是一个针对多个 Manager 的 get_blkno_indexers 的多 Manager 版本，设置 group=False。
    """
    # 我们有多个 BlockManagers 的 blknos
    # list[np.ndarray[int64_t]]
    cdef:
        Py_ssize_t i, j, k, start, ncols  # 定义变量 i, j, k, start, ncols
        cnp.npy_intp n_mgrs  # 定义 n_mgrs 变量
        ndarray[intp_t] blknos, cur_blknos, run_blknos  # 定义数组变量 blknos, cur_blknos, run_blknos
        BlockPlacement bp  # 定义 BlockPlacement 类型的变量 bp
        list result = []  # 定义空列表 result

    n_mgrs = len(blknos_list)  # 获取 blknos_list 的长度
    cur_blknos = cnp.PyArray_EMPTY(1, &n_mgrs, cnp.NPY_INTP, 0)  # 创建长度为 n_mgrs 的空数组 cur_blknos

    blknos = blknos_list[0]  # 获取第一个 BlockManager 的 blknos
    ncols = len(blknos)  # 获取 blknos 的列数
    if ncols == 0:
        return []  # 如果 ncols 为 0，则直接返回空列表

    start = 0  # 设置起始索引为 0
    for i in range(n_mgrs):
        blknos = blknos_list[i]  # 获取每个 BlockManager 的 blknos
        cur_blknos[i] = blknos[0]  # 将每个 BlockManager 的第一个 blknos 存入 cur_blknos
        assert len(blknos) == ncols  # 断言每个 BlockManager 的 blknos 长度与 ncols 相等

    for i in range(1, ncols):
        # 对于每一列，检查它是否属于相同的 Block（即 blknos[i] 等于 blknos[i-1]）
        # 并且这种情况对于 blknos_list 中的每个 blknos 都是一致的。如果不是，则开始一个新的 "run"。
        for k in range(n_mgrs):
            blknos = blknos_list[k]  # 获取当前 BlockManager 的 blknos
            # assert cur_blknos[k] == blknos[i - 1]

            if blknos[i] != blknos[i - 1]:  # 如果当前列的 blknos 不等于前一列的 blknos
                bp = BlockPlacement(slice(start, i))  # 创建一个新的 BlockPlacement 对象
                run_blknos = cnp.PyArray_Copy(cur_blknos)  # 复制 cur_blknos 到 run_blknos
                result.append((run_blknos, bp))  # 将 run_blknos 和 bp 添加到 result 列表中

                start = i  # 更新起始索引为当前列的索引
                for j in range(n_mgrs):
                    blknos = blknos_list[j]  # 获取每个 BlockManager 的 blknos
                    cur_blknos[j] = blknos[i]  # 更新 cur_blknos 中每个 BlockManager 的值为当前列的 blknos
                break  # 退出循环 `for k in range(n_mgrs)`

    if start != ncols:
        bp = BlockPlacement(slice(start, ncols))  # 创建最后一个 BlockPlacement 对象
        run_blknos = cnp.PyArray_Copy(cur_blknos)  # 复制 cur_blknos 到 run_blknos
        result.append((run_blknos, bp))  # 将 run_blknos 和 bp 添加到 result 列表中
    return result  # 返回 result 列表作为结果
    Iterate over elements of `blknos` yielding ``(blkno, slice(start, stop))``
    pairs for each contiguous run found.

    If `group` is True and there is more than one run for a certain blkno,
    ``(blkno, array)`` with an array containing positions of all elements equal
    to blkno.

    Returns
    -------
    list[tuple[int, slice | np.ndarray]]
    """
    # This function iterates through an array `blknos` to identify contiguous
    # blocks (`blkno`) and returns them as tuples of block number and slices or arrays.

    cdef:
        int64_t cur_blkno
        Py_ssize_t i, start, stop, n, diff
        cnp.npy_intp tot_len
        int64_t blkno
        object group_dict = defaultdict(list)
        ndarray[int64_t, ndim=1] arr

    # Get the number of elements in `blknos`
    n = blknos.shape[0]
    result = list()

    # If `blknos` is empty, return an empty list
    if n == 0:
        return result

    # Initialize starting point
    start = 0
    cur_blkno = blknos[start]

    # If `group` is False, collect contiguous runs of the same blkno
    if group is False:
        for i in range(1, n):
            if blknos[i] != cur_blkno:
                result.append((cur_blkno, slice(start, i)))

                start = i
                cur_blkno = blknos[i]

        # Append the last contiguous run
        result.append((cur_blkno, slice(start, n)))
    else:
        # If `group` is True, group contiguous runs of the same blkno together
        for i in range(1, n):
            if blknos[i] != cur_blkno:
                group_dict[cur_blkno].append((start, i))

                start = i
                cur_blkno = blknos[i]

        # Append the last group
        group_dict[cur_blkno].append((start, n))

        # Process each group in `group_dict`
        for blkno, slices in group_dict.items():
            if len(slices) == 1:
                # If there is only one slice, append it as a tuple with slice
                result.append((blkno, slice(slices[0][0], slices[0][1])))
            else:
                # If there are multiple slices, calculate total length
                tot_len = sum(stop - start for start, stop in slices)
                # Create a new numpy array of int64 with `tot_len` size
                arr = cnp.PyArray_EMPTY(1, &tot_len, cnp.NPY_INT64, 0)

                i = 0
                # Fill the array with positions
                for start, stop in slices:
                    for diff in range(start, stop):
                        arr[i] = diff
                        i += 1

                # Append the tuple of blkno and the created array
                result.append((blkno, arr))

    # Return the final result list
    return result
def get_blkno_placements(blknos, group: bool = True):
    """
    Parameters
    ----------
    blknos : np.ndarray[int64]
        一个包含块号的NumPy数组，类型为int64。
    group : bool, default True
        一个布尔值，指示是否要对块号进行分组。

    Returns
    -------
    iterator
        返回一个迭代器，每次生成一个元组(blkno, BlockPlacement)，
        其中blkno为块号，BlockPlacement为块位置对象。
    """
    blknos = ensure_int64(blknos)

    for blkno, indexer in get_blkno_indexers(blknos, group):
        yield blkno, BlockPlacement(indexer)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef update_blklocs_and_blknos(
    ndarray[intp_t, ndim=1] blklocs,
    ndarray[intp_t, ndim=1] blknos,
    Py_ssize_t loc,
    intp_t nblocks,
):
    """
    Update blklocs and blknos when a new column is inserted at 'loc'.
    
    Parameters
    ----------
    blklocs : ndarray[intp_t, ndim=1]
        一个intp_t类型的一维NumPy数组，表示块位置。
    blknos : ndarray[intp_t, ndim=1]
        一个intp_t类型的一维NumPy数组，表示块号。
    loc : Py_ssize_t
        要插入新列的位置。
    nblocks : intp_t
        新块的数量。
    
    Returns
    -------
    tuple
        返回更新后的blklocs和blknos，均为一维NumPy数组。
    """
    cdef:
        Py_ssize_t i
        cnp.npy_intp length = blklocs.shape[0] + 1
        ndarray[intp_t, ndim=1] new_blklocs, new_blknos

    # equiv: new_blklocs = np.empty(length, dtype=np.intp)
    new_blklocs = cnp.PyArray_EMPTY(1, &length, cnp.NPY_INTP, 0)
    new_blknos = cnp.PyArray_EMPTY(1, &length, cnp.NPY_INTP, 0)

    for i in range(loc):
        new_blklocs[i] = blklocs[i]
        new_blknos[i] = blknos[i]

    new_blklocs[loc] = 0
    new_blknos[loc] = nblocks

    for i in range(loc, length - 1):
        new_blklocs[i + 1] = blklocs[i]
        new_blknos[i + 1] = blknos[i]

    return new_blklocs, new_blknos


def _unpickle_block(values, placement, ndim):
    """
    Deserialize a block from pickled format.
    
    Parameters
    ----------
    values : object
        要解序列化的值。
    placement : BlockPlacement
        块的位置。
    ndim : int
        块的维度。
    
    Returns
    -------
    Block
        返回一个新的块对象。
    """
    # We have to do some gymnastics b/c "ndim" is keyword-only

    from pandas.core.internals.blocks import (
        maybe_coerce_values,
        new_block,
    )
    values = maybe_coerce_values(values)

    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)
    return new_block(values, placement, ndim=ndim)


@cython.freelist(64)
cdef class Block:
    """
    Defining __init__ in a cython class significantly improves performance.
    """
    cdef:
        public BlockPlacement _mgr_locs
        public BlockValuesRefs refs
        readonly int ndim
        # 2023-08-15 no apparent performance improvement from declaring values
        #  as ndarray in a type-special subclass (similar for NDArrayBacked).
        #  This might change if slice_block_rows can be optimized with something
        #  like https://github.com/numpy/numpy/issues/23934
        public object values

    def __cinit__(
        self,
        values,
        placement: BlockPlacement,
        ndim: int,
        refs: BlockValuesRefs | None = None,
    ):
        """
        Initialize Block object in Cython.
        
        Parameters
        ----------
        values : object
            块的值。
        placement : BlockPlacement
            块的位置。
        ndim : int
            块的维度。
        refs : BlockValuesRefs, optional
            块的值引用，可选参数，默认为None。
        """
    ):
        """
        Parameters
        ----------
        values : np.ndarray or ExtensionArray
            We assume maybe_coerce_values has already been called.
            # 设置 Block 对象的值为传入的 values 参数
        placement : BlockPlacement
            # 设置 Block 对象的位置信息为传入的 placement 参数
        ndim : int
            # 设置 Block 对象的维度信息为传入的 ndim 参数
            1 for SingleBlockManager/Series, 2 for BlockManager/DataFrame
        refs: BlockValuesRefs, optional
            Ref tracking object or None if block does not have any refs.
            # 如果 refs 参数为 None，创建一个新的 BlockValuesRefs 对象并将其赋值给 self.refs
            # 如果 refs 参数不为 None，则将当前 Block 添加到已有的 refs 对象中
        """
        self.values = values

        self._mgr_locs = placement
        self.ndim = ndim
        if refs is None:
            # if no refs are passed, that means we are creating a Block from
            # new values that it uniquely owns -> start a new BlockValuesRefs
            # object that only references this block
            # 如果没有传入 refs，意味着我们正在从唯一拥有的新值创建一个 Block ->
            # 开始一个新的 BlockValuesRefs 对象，它只引用这个块
            self.refs = BlockValuesRefs(self)
        else:
            # if refs are passed, this is the BlockValuesRefs object that is shared
            # with the parent blocks which share the values, and a reference to this
            # new block is added
            # 如果传入了 refs，这是与共享值的父块共享的 BlockValuesRefs 对象，
            # 并且添加了对这个新块的引用
            refs.add_reference(self)
            self.refs = refs

    cpdef __reduce__(self):
        """
        Method to support pickling by providing necessary information.

        Returns
        -------
        tuple
            Tuple containing function to unpickle and arguments for unpickling.
        """
        args = (self.values, self.mgr_locs.indexer, self.ndim)
        return _unpickle_block, args

    cpdef __setstate__(self, state):
        """
        Method to set the state of the object during unpickling.

        Parameters
        ----------
        state : tuple
            Tuple containing the state information to restore the object.
        """
        from pandas.core.construction import extract_array

        # Restore BlockPlacement object from state
        self.mgr_locs = BlockPlacement(state[0])
        # Extract array data from state and assign to self.values
        self.values = extract_array(state[1], extract_numpy=True)
        if len(state) > 2:
            # we stored ndim
            # If ndim information is available in the state, restore it
            self.ndim = state[2]
        else:
            # older pickle
            # For older pickles, infer ndim from values and mgr_locs
            from pandas.core.internals.api import maybe_infer_ndim

            ndim = maybe_infer_ndim(self.values, self.mgr_locs)
            self.ndim = ndim

    cpdef Block slice_block_rows(self, slice slicer):
        """
        Method to slice the block along rows based on the given slicer.

        Parameters
        ----------
        slicer : slice
            Slice object representing the rows to slice.

        Returns
        -------
        Block
            A new Block object sliced according to the slicer.
        """
        # Slice the values array along rows using the given slicer
        new_values = self.values[..., slicer]
        # Return a new instance of the same Block type with sliced values
        return type(self)(new_values, self._mgr_locs, ndim=self.ndim, refs=self.refs)
@cython.freelist(64)
cdef class BlockManager:
    cdef:
        public tuple blocks  # 块数据的元组
        public list axes  # 轴列表
        public bint _known_consolidated, _is_consolidated  # 布尔值变量，表示已知和是否已合并
        public ndarray _blknos, _blklocs  # 块编号和块位置的 ndarray 数组

    def __cinit__(
        self,
        blocks=None,
        axes=None,
        verify_integrity=True,
    ):
        # None as defaults for unpickling GH#42345
        if blocks is None:
            # 这里增加了 1-2 微秒用于 DataFrame(np.array([])) 的反序列化
            return

        if isinstance(blocks, list):
            # 兼容旧版本，例如 pyarrow
            blocks = tuple(blocks)

        self.blocks = blocks  # 设置块数据
        self.axes = axes.copy()  # 复制轴列表以确保不可远程修改

        # 懒加载已知的合并信息、块编号和块位置
        self._known_consolidated = False
        self._is_consolidated = False
        self._blknos = None
        self._blklocs = None

    # -------------------------------------------------------------------
    # Block Placement

    def _rebuild_blknos_and_blklocs(self) -> None:
        """
        Update mgr._blknos / mgr._blklocs.
        """
        cdef:
            intp_t blkno, i, j
            cnp.npy_intp length = self.shape[0]
            Block blk
            BlockPlacement bp
            ndarray[intp_t, ndim=1] new_blknos, new_blklocs

        # 等效于 np.empty(length, dtype=np.intp)
        new_blknos = cnp.PyArray_EMPTY(1, &length, cnp.NPY_INTP, 0)
        new_blklocs = cnp.PyArray_EMPTY(1, &length, cnp.NPY_INTP, 0)
        # 等效于 new_blknos.fill(-1)
        cnp.PyArray_FILLWBYTE(new_blknos, -1)
        cnp.PyArray_FILLWBYTE(new_blklocs, -1)

        for blkno, blk in enumerate(self.blocks):
            bp = blk._mgr_locs
            # 遍历 bp 的速度更快，相当于
            # new_blknos[bp.indexer] = blkno
            # new_blklocs[bp.indexer] = np.arange(len(bp))
            for i, j in enumerate(bp):
                new_blknos[j] = blkno
                new_blklocs[j] = i

        for i in range(length):
            # 检查是否有任何 -1，表示 mgr_locs 无效
            blkno = new_blknos[i]
            if blkno == -1:
                raise AssertionError("Gaps in blk ref_locs")

        self._blknos = new_blknos  # 更新块编号
        self._blklocs = new_blklocs  # 更新块位置

    # -------------------------------------------------------------------
    # Pickle

    cpdef __reduce__(self):
        if len(self.axes) == 1:
            # 对于 SingleBlockManager，__init__ 期望 Block 和 axis 参数
            args = (self.blocks[0], self.axes[0])
        else:
            args = (self.blocks, self.axes)
        return type(self), args
    # 定义一个特殊方法，用于从序列化状态恢复对象
    cpdef __setstate__(self, state):
        # 导入需要的函数和类
        from pandas.core.construction import extract_array
        from pandas.core.internals.blocks import (
            ensure_block_shape,
            maybe_coerce_values,
            new_block,
        )
        from pandas.core.internals.managers import ensure_index

        # 检查状态是否为元组且长度至少为4，并且包含特定版本信息
        if isinstance(state, tuple) and len(state) >= 4 and "0.14.1" in state[3]:
            # 从状态中提取特定版本信息的数据
            state = state[3]["0.14.1"]
            # 确保索引是正确的类型
            axes = [ensure_index(ax) for ax in state["axes"]]
            ndim = len(axes)

            # 遍历所有数据块
            for blk in state["blocks"]:
                vals = blk["values"]
                # 提取数据数组，可能将其转换为NumPy数组
                vals = extract_array(vals, extract_numpy=True)
                # 确保数据块的形状和维度一致，并且可能强制转换值的类型
                blk["values"] = maybe_coerce_values(ensure_block_shape(vals, ndim=ndim))

                # 如果数据块位置信息不是BlockPlacement对象，则创建一个新的BlockPlacement对象
                if not isinstance(blk["mgr_locs"], BlockPlacement):
                    blk["mgr_locs"] = BlockPlacement(blk["mgr_locs"])

            # 创建新的数据块列表
            nbs = [
                new_block(blk["values"], blk["mgr_locs"], ndim=ndim)
                for blk in state["blocks"]
            ]
            blocks = tuple(nbs)
            # 设置对象的数据块和索引
            self.blocks = blocks
            self.axes = axes

        else:  # pragma: no cover
            # 如果版本较旧，则抛出NotImplementedError
            raise NotImplementedError("pre-0.14.1 pickles are no longer supported")

        # 在设置状态后执行后处理
        self._post_setstate()

    # 执行设置状态后的后处理
    def _post_setstate(self) -> None:
        self._is_consolidated = False
        self._known_consolidated = False
        self._rebuild_blknos_and_blklocs()

    # -------------------------------------------------------------------
    # 索引操作

    # 定义一个Cython函数，用于按行对数据块进行切片操作
    cdef BlockManager _slice_mgr_rows(self, slice slobj):
        cdef:
            Block blk, nb
            BlockManager mgr
            ndarray blknos, blklocs

        # 创建新的数据块列表
        nbs = []
        for blk in self.blocks:
            # 对每个数据块按行进行切片
            nb = blk.slice_block_rows(slobj)
            nbs.append(nb)

        # 创建新的轴，并使用切片后的数据块列表创建新的BlockManager对象
        new_axes = [self.axes[0], self.axes[1]._getitem_slice(slobj)]
        mgr = type(self)(tuple(nbs), new_axes, verify_integrity=False)

        # 复制当前对象的块位置和块编号，以避免重建
        blklocs = self._blklocs
        blknos = self._blknos
        if blknos is not None:
            mgr._blknos = blknos.copy()
            mgr._blklocs = blklocs.copy()
        return mgr

    # 获取指定切片和轴的BlockManager对象
    def get_slice(self, slobj: slice, axis: int = 0) -> BlockManager:

        # 根据轴的不同选择不同的切片操作
        if axis == 0:
            new_blocks = self._slice_take_blocks_ax0(slobj)
        elif axis == 1:
            # 对行进行切片操作
            return self._slice_mgr_rows(slobj)
        else:
            # 如果请求的轴不在管理器中，则抛出IndexError
            raise IndexError("Requested axis not found in manager")

        # 创建新的轴，并使用新的数据块列表创建新的BlockManager对象
        new_axes = list(self.axes)
        new_axes[axis] = new_axes[axis]._getitem_slice(slobj)

        return type(self)(tuple(new_blocks), new_axes, verify_integrity=False)
cdef class BlockValuesRefs:
    """Tracks all references to a given array.

    Keeps track of all blocks (through weak references) that reference the same
    data.
    """
    cdef:
        public list referenced_blocks  # 保存所有对给定数组的引用的列表
        public int clear_counter  # 清理引用的计数器

    def __cinit__(self, blk: Block | None = None) -> None:
        if blk is not None:
            self.referenced_blocks = [weakref.ref(blk)]  # 如果提供了块对象，创建一个弱引用列表
        else:
            self.referenced_blocks = []  # 否则，创建一个空的引用列表
        self.clear_counter = 500  # 设置清理引用的阈值为500，初始设定较高以减少清理次数

    def _clear_dead_references(self, force=False) -> None:
        # Use exponential backoff to decide when we want to clear references
        # if force=False. Clearing for every insertion causes slowdowns if
        # all these objects stay alive, e.g. df.items() for wide DataFrames
        # see GH#55245 and GH#55008
        if force or len(self.referenced_blocks) > self.clear_counter:
            # 清理已经失效的引用对象
            self.referenced_blocks = [
                ref for ref in self.referenced_blocks if ref() is not None
            ]
            nr_of_refs = len(self.referenced_blocks)
            # 根据引用数量动态调整清理阈值
            if nr_of_refs < self.clear_counter // 2:
                self.clear_counter = max(self.clear_counter // 2, 500)
            elif nr_of_refs > self.clear_counter:
                self.clear_counter = max(self.clear_counter * 2, nr_of_refs)

    def add_reference(self, blk: Block) -> None:
        """Adds a new reference to our reference collection.

        Parameters
        ----------
        blk : Block
            The block that the new references should point to.
        """
        self._clear_dead_references()  # 添加新引用之前先清理失效引用
        self.referenced_blocks.append(weakref.ref(blk))  # 添加一个新的块对象引用

    def add_index_reference(self, index: object) -> None:
        """Adds a new reference to our reference collection when creating an index.

        Parameters
        ----------
        index : Index
            The index that the new reference should point to.
        """
        self._clear_dead_references()  # 添加新引用之前先清理失效引用
        self.referenced_blocks.append(weakref.ref(index))  # 添加一个新的索引对象引用

    def has_reference(self) -> bool:
        """Checks if block has foreign references.

        A reference is only relevant if it is still alive. The reference to
        ourselves does not count.

        Returns
        -------
        bool
        """
        self._clear_dead_references(force=True)  # 强制清理所有失效引用
        # 检查是否有多于自身引用的其他引用存在
        return len(self.referenced_blocks) > 1
```
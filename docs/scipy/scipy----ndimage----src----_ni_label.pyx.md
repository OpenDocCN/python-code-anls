# `D:\src\scipysrc\scipy\scipy\ndimage\src\_ni_label.pyx`

```
######################################################################
# Cython version of scipy.ndimage.measurements.label().
# Requires Cython version 0.17 or greater due to type templating.
######################################################################

# 导入 NumPy 库，并使用 Cython 特定语法导入 NumPy 库
import numpy as np
cimport numpy as np

# 调用 NumPy 库的 import_array() 函数
np.import_array()

# 定义 C 扩展类型 Py_intptr_t
cdef extern from *:
   ctypedef int Py_intptr_t

# 定义枚举类型 BACKGROUND 和 FOREGROUND
cdef enum:
    BACKGROUND = 0
    FOREGROUND = 1

# 在 numpy/arrayobject.h 文件中导入必要的 C 结构和函数声明
cdef extern from "numpy/arrayobject.h" nogil:
    ctypedef struct PyArrayIterObject:
        np.npy_intp *coordinates

    void PyArray_ITER_NEXT(PyArrayIterObject *it)
    int PyArray_ITER_NOTDONE(PyArrayIterObject *it)
    void PyArray_ITER_RESET(PyArrayIterObject *it)
    void *PyArray_ITER_DATA(PyArrayIterObject *it)

    void *PyDataMem_NEW(size_t)
    void PyDataMem_FREE(void *)
    void *PyDataMem_RENEW(void *, size_t)

# 定义自定义异常类 NeedMoreBits
class NeedMoreBits(Exception):
    pass

######################################################################
# Use Cython's type templates for type specialization
######################################################################

# 使用 Cython 的类型模板定义 fused 数据类型，支持多种数据类型
ctypedef fused data_t:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t

######################################################################
# Load a line from a fused data array, setting the line to FOREGROUND wherever
# the fused data is nonzero, BACKGROUND elsewhere
######################################################################

# 定义函数 fused_nonzero_line，从 fused 数据数组中加载一行数据到 np.uintp_t 数组
cdef void fused_nonzero_line(data_t *p, np.intp_t stride,
                             np.uintp_t *line, np.intp_t L) noexcept nogil:
    cdef np.intp_t i
    for i in range(L):
        # 如果 fused 数据数组中的值非零，则设置为 FOREGROUND，否则为 BACKGROUND
        line[i] = FOREGROUND if \
            (<data_t *> ((<char *> p) + i * stride))[0] \
            else BACKGROUND

######################################################################
# Load a line from a fused data array to a np.uintp_t array
######################################################################

# 定义函数 fused_read_line，从 fused 数据数组中加载一行数据到 np.uintp_t 数组
cdef void fused_read_line(data_t *p, np.intp_t stride,
                          np.uintp_t *line, np.intp_t L) noexcept nogil:
    cdef np.intp_t i
    for i in range(L):
        line[i] = <np.uintp_t> (<data_t *> ((<char *> p) + i * stride))[0]

######################################################################
# Store a line from a np.uintp_t array to a fused data array if possible,
# returning True if overflowed
######################################################################

# 定义函数 fused_write_line，将 np.uintp_t 数组中的一行数据存储到 fused 数据数组中（如果可能），并返回是否溢出
cdef bint fused_write_line(data_t *p, np.intp_t stride,
                           np.uintp_t *line, np.intp_t L) noexcept nogil:
    cdef np.intp_t i
    # 对于给定的范围 L 内的每一个索引 i，执行以下操作：
    for i in range(L):
        # 检查是否需要覆盖写入数据，以避免意外将 0 写入前景中，这样可以在就地操作时重新尝试。
        # 检查当前行 line 的第 i 个元素是否与指定类型 <np.uintp_t> <data_t> 相等，
        # 如果不相等，则返回 True 表示需要覆盖写入。
        if line[i] != <np.uintp_t> <data_t> line[i]:
            return True
        # 将 line 的第 i 个元素赋值给位于指针 p 偏移 i * stride 处的 <data_t> 类型数据。
        (<data_t *> ((<char *> p) + i * stride))[0] = <data_t> line[i]
    # 若没有需要覆盖写入的情况，则返回 False。
    return False
######################################################################
# Function specializers
######################################################################

# 获取非零行的特化函数
def get_nonzero_line(np.ndarray[data_t] a):
    return <Py_intptr_t> fused_nonzero_line[data_t]

# 获取读取行的特化函数
def get_read_line(np.ndarray[data_t] a):
    return <Py_intptr_t> fused_read_line[data_t]

# 获取写入行的特化函数
def get_write_line(np.ndarray[data_t] a):
    return <Py_intptr_t> fused_write_line[data_t]


######################################################################
# Typedefs for referring to specialized instances of fused functions
######################################################################

# 用于引用非零行函数特化实例的类型定义
ctypedef void (*nonzero_line_func_t)(void *p, np.intp_t stride,
                                     np.uintp_t *line, np.intp_t L) noexcept nogil
# 用于引用读取行函数特化实例的类型定义
ctypedef void (*read_line_func_t)(void *p, np.intp_t stride,
                                  np.uintp_t *line, np.intp_t L) noexcept nogil
# 用于引用写入行函数特化实例的类型定义
ctypedef bint (*write_line_func_t)(void *p, np.intp_t stride,
                                   np.uintp_t *line, np.intp_t L) noexcept nogil


######################################################################
# Mark two labels to be merged
######################################################################

# 将两个标签标记为要合并的函数
cdef inline np.uintp_t mark_for_merge(np.uintp_t a,
                                      np.uintp_t b,
                                      np.uintp_t *mergetable) noexcept nogil:

    cdef:
        np.uintp_t orig_a, orig_b, minlabel

    orig_a = a
    orig_b = b

    # 找到 a 和 b 的最小根
    while a != mergetable[a]:
        a = mergetable[a]
    while b != mergetable[b]:
        b = mergetable[b]
    minlabel = a if (a < b) else b

    # 合并根节点
    mergetable[a] = mergetable[b] = minlabel

    # 将所有步骤合并到 minlabel
    a = orig_a
    b = orig_b
    while a != minlabel:
        a, mergetable[a] = mergetable[a], minlabel
    while b != minlabel:
        b, mergetable[b] = mergetable[b], minlabel

    return minlabel


######################################################################
# Take the label of a neighbor, or mark them for merging
######################################################################

# 获取邻居的标签或将它们标记为合并的函数
cdef inline np.uintp_t take_label_or_merge(np.uintp_t cur_label,
                                           np.uintp_t neighbor_label,
                                           np.uintp_t *mergetable) noexcept nogil:
    if neighbor_label == BACKGROUND:
        return cur_label
    if cur_label == FOREGROUND:
        return neighbor_label  # 邻居不是背景
    if neighbor_label:
        if cur_label != neighbor_label:
            cur_label = mark_for_merge(neighbor_label, cur_label, mergetable)
    return cur_label


######################################################################
# Label one line of input, using a neighbor line that has already been labeled.
######################################################################

# 标记输入的一行，使用已经标记的邻居行的函数
cdef np.uintp_t label_line_with_neighbor(np.uintp_t *line,
                                         np.uintp_t *neighbor,
                                         int neighbor_use_previous,
                                         int neighbor_use_adjacent,
                                         int neighbor_use_next,
                                         np.intp_t L,
                                         bint label_unlabeled,
                                         bint use_previous,
                                         np.uintp_t next_region,
                                         np.uintp_t *mergetable) noexcept nogil:
    cdef:
        np.intp_t i  # 定义循环变量 i

    for i in range(L):  # 遍历长度为 L 的数组 line
        if line[i] != BACKGROUND:  # 检查当前元素是否为背景标签
            # 如果需要使用前一个邻居标签，则将当前元素与前一个邻居标签进行合并或选择
            if neighbor_use_previous:
                line[i] = take_label_or_merge(line[i], neighbor[i - 1], mergetable)
            # 如果需要使用相邻邻居标签，则将当前元素与相邻邻居标签进行合并或选择
            if neighbor_use_adjacent:
                line[i] = take_label_or_merge(line[i], neighbor[i], mergetable)
            # 如果需要使用下一个邻居标签，则将当前元素与下一个邻居标签进行合并或选择
            if neighbor_use_next:
                line[i] = take_label_or_merge(line[i], neighbor[i + 1], mergetable)
            # 如果需要标记未标记的元素
            if label_unlabeled:
                # 如果允许使用前一个元素的标签，则将当前元素与前一个元素的标签进行合并或选择
                if use_previous:
                    line[i] = take_label_or_merge(line[i], line[i - 1], mergetable)
                # 如果当前元素仍然是前景标签，则为其分配一个新的区域标签
                if line[i] == FOREGROUND:
                    line[i] = next_region
                    mergetable[next_region] = next_region
                    next_region += 1  # 更新下一个可用的区域标签值
    return next_region  # 返回更新后的下一个可用的区域标签值

######################################################################
# Label regions
######################################################################
cpdef _label(np.ndarray input,
             np.ndarray structure,
             np.ndarray output):
    # 检查输入和输出数组的维度是否一致
    assert (<object> input).shape == (<object> output).shape, \
        ("Shapes must match for input and output,"
         "{} != {}".format((<object> input).shape, (<object> output).shape))

    structure = np.asanyarray(structure, dtype=np.bool_).copy()  # 将结构元素转换为布尔数组的副本
    # 检查输入数组和结构元素的维度是否相同
    assert input.ndim == structure.ndim, \
        ("Structuring element must have same "
         "# of dimensions as input, "
         "{:d} != {:d}".format(input.ndim, structure.ndim))

    # 检查结构元素在每个维度上是否都是尺寸为 3
    assert set((<object> structure).shape) <= set([3]), \
        ("Structuring element must be size 3 in every dimension, "
         "was {}".format((<object> structure).shape))

    # 检查结构元素是否对称
    assert np.all(structure == structure[(np.s_[::-1],) * structure.ndim]), \
        "Structuring element is not symmetric"

    # 确保处理的是非空且非标量的数组
    assert input.ndim > 0 and input.size > 0, "Cannot label scalars or empty arrays"
    # 如果输入数据类型为布尔型，将其视为无符号8位整数处理
    if input.dtype == np.bool_:
        input = input.view(dtype=np.uint8)
    if output.dtype == np.bool_:
        # 如果输出数据类型为布尔型，触发特殊的位深度检查？
        output = output.view(dtype=np.uint8)

    cdef:
        # 定义非零行函数指针类型
        nonzero_line_func_t nonzero_line = \
            <nonzero_line_func_t> <void *> <Py_intptr_t> get_nonzero_line(input.take([0]))
        # 定义读取行函数指针类型
        read_line_func_t read_line = \
            <read_line_func_t> <void *> <Py_intptr_t> get_read_line(output.take([0]))
        # 定义写入行函数指针类型
        write_line_func_t write_line = \
            <write_line_func_t> <void *> <Py_intptr_t> get_write_line(output.take([0]))
        # 定义迭代器和迭代对象
        np.flatiter _iti, _ito, _itstruct
        PyArrayIterObject *iti
        PyArrayIterObject *ito
        PyArrayIterObject *itstruct
        # 定义整数类型变量
        int axis, idim, num_neighbors, ni
        np.intp_t L, delta, i
        np.intp_t si, so, ss
        np.intp_t total_offset
        # 定义维度变量
        np.intp_t output_ndim, structure_ndim
        # 定义布尔类型变量
        bint needs_self_labeling, valid, use_previous, overflowed
        # 定义数组和缓冲区
        np.ndarray _line_buffer, _neighbor_buffer
        np.uintp_t *line_buffer
        np.uintp_t *neighbor_buffer
        np.uintp_t *tmp
        # 定义标签和合并表变量
        np.uintp_t next_region, src_label, dest_label
        np.uintp_t mergetable_size
        np.uintp_t *mergetable

    axis = -1  # 根据输出选择最佳的轴
    _ito = np.PyArray_IterAllButAxis(output, &axis)
    _iti = np.PyArray_IterAllButAxis(input, &axis)
    _itstruct = np.PyArray_IterAllButAxis(structure, &axis)

    ito = <PyArrayIterObject *> _ito
    iti = <PyArrayIterObject *> _iti
    itstruct = <PyArrayIterObject *> _itstruct

    # 在 itstruct 迭代器中心之前，仅处理这么多邻居
    num_neighbors = structure.size // (3 * 2)

    # 创建两个用于读取/写入标签的缓冲区数组
    # 在末尾和开头添加一个条目以简化某些边界检查
    L = input.shape[axis]
    _line_buffer = np.empty(L + 2, dtype=np.uintp)
    _neighbor_buffer = np.empty(L + 2, dtype=np.uintp)
    line_buffer = <np.uintp_t *> _line_buffer.data
    neighbor_buffer = <np.uintp_t *> _neighbor_buffer.data

    # 使用背景值添加栅栏
    line_buffer[0] = neighbor_buffer[0] = BACKGROUND
    line_buffer[L + 1] = neighbor_buffer[L + 1] = BACKGROUND
    line_buffer = line_buffer + 1
    neighbor_buffer = neighbor_buffer + 1

    # 分配合并表的内存空间
    mergetable_size = 2 * output.shape[axis]
    mergetable = <np.uintp_t *> PyDataMem_NEW(mergetable_size * sizeof(np.uintp_t))
    if mergetable == NULL:
        raise MemoryError()

    except:  # noqa: E722
        # 清理并重新引发异常
        PyDataMem_FREE(<void *> mergetable)
        raise

    # 释放合并表的内存空间
    PyDataMem_FREE(<void *> mergetable)
    return dest_label - 1
```
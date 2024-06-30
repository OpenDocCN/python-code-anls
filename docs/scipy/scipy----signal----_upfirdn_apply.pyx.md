# `D:\src\scipysrc\scipy\scipy\signal\_upfirdn_apply.pyx`

```
# -*- coding: utf-8 -*-

# Code adapted from "upfirdn" python library with permission:
#
# Copyright (c) 2009, Motorola, Inc
#
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# * Neither the name of Motorola nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# 导入 Cython 相关库
cimport cython
# 导入 NumPy 库
cimport numpy as np
import numpy as np
# 导入布尔整型类型
from cython import bint  # boolean integer type
# 导入 C 标准库中的内存分配和释放函数
from libc.stdlib cimport malloc, free
# 导入 C 标准库中的内存设置函数
from libc.string cimport memset

# 导入 NumPy 数组处理函数
np.import_array()

# 定义复数数据类型
ctypedef double complex double_complex
ctypedef float complex float_complex

# 定义融合类型 DTYPE_t，包含 float、float_complex、double、double_complex
ctypedef fused DTYPE_t:
    float
    float_complex
    double
    double_complex

# 定义结构体 ArrayInfo，包含数组的形状、步长和维度
cdef struct ArrayInfo:
    np.intp_t * shape
    np.intp_t * strides
    np.intp_t ndim

# 定义函数 _output_len，计算给定输入的输出长度
def _output_len(np.int64_t len_h,
                np.int64_t in_len,
                np.int64_t up,
                np.int64_t down):
    """The output length that results from a given input"""
    # ceil(((in_len - 1) * up + len_h) / down), but using integer arithmetic
    return (((in_len - 1) * up + len_h) - 1) // down + 1

# 信号扩展模式枚举
ctypedef enum MODE:
    MODE_CONSTANT = 0
    MODE_SYMMETRIC = 1       #  3 2 1 | 1 2 3 | 3 2 1
    MODE_CONSTANT_EDGE = 2
    MODE_SMOOTH = 3
    MODE_PERIODIC = 4
    MODE_REFLECT = 5         #  3 2 | 1 2 3 | 2 1
    MODE_ANTISYMMETRIC = 6
    MODE_ANTIREFLECT = 7
    MODE_LINE = 8  # slope determined by first and last entries of the array

# 定义函数 mode_enum，将字符串模式转换为枚举类型 MODE
cpdef MODE mode_enum(mode):
    if mode == 'constant':
        return MODE_CONSTANT
    # 如果模式是对称模式，返回对应的常量 MODE_SYMMETRIC
    elif mode == 'symmetric':
        return MODE_SYMMETRIC
    # 如果模式是边缘模式，返回对应的常量 MODE_CONSTANT_EDGE
    elif mode == 'edge':
        return MODE_CONSTANT_EDGE
    # 如果模式是平滑模式，返回对应的常量 MODE_SMOOTH
    elif mode == 'smooth':
        return MODE_SMOOTH
    # 如果模式是环绕模式，返回对应的常量 MODE_PERIODIC
    elif mode == 'wrap':
        return MODE_PERIODIC
    # 如果模式是反射模式，返回对应的常量 MODE_REFLECT
    elif mode == 'reflect':
        return MODE_REFLECT
    # 如果模式是反对称模式，返回对应的常量 MODE_ANTISYMMETRIC
    elif mode == 'antisymmetric':
        return MODE_ANTISYMMETRIC
    # 如果模式是反反射模式，返回对应的常量 MODE_ANTIREFLECT
    elif mode == 'antireflect':
        return MODE_ANTIREFLECT
    # 如果模式是线性模式，返回对应的常量 MODE_LINE
    elif mode == 'line':
        return MODE_LINE
    # 如果模式不在预定义的模式列表中，则抛出值错误异常
    else:
        raise ValueError("Unknown mode: {}".format(mode))
# 使用 Cython 的优化指令，加速模块运算
@cython.cdivision(True)  # faster modulo
@cython.boundscheck(False)  # designed to stay within bounds
@cython.wraparound(False)  # we don't use negative indexing
# 定义一个 Cython 编写的函数 _extend_left，用于处理边界扩展操作
cdef DTYPE_t _extend_left(DTYPE_t *x, np.intp_t idx, np.intp_t len_x,
                          MODE mode, DTYPE_t cval) noexcept nogil:
    # 左边界扩展函数，初始化左边界值和线性斜率为零
    cdef DTYPE_t le = 0.
    cdef DTYPE_t lin_slope = 0.

    # 如果 idx < 0，即需要左边界扩展
    # 根据不同的扩展模式进行处理
    if mode == MODE_SYMMETRIC:
        if (-idx) < len_x:
            return x[-idx - 1]
        else:
            # 对于多重反射的一般情况：
            # 模式以 2*len_x 周期重复
            idx = (-idx - 1) % (2 * len_x)
            if idx < len_x:
                return x[idx]
            else:
                return x[len_x - 1 - (idx - len_x)]
    elif mode == MODE_REFLECT:
        if (-idx) < (len_x - 1):
            return x[-idx]
        else:
            # 对于多重反射的一般情况：
            # 模式以 2*(len_x - 1) 周期重复
            idx = (-idx - 1) % (2 * (len_x - 1))
            if idx < (len_x - 1):
                return x[idx + 1]
            else:
                return x[len_x - 2 - (idx - (len_x - 1))]
    elif mode == MODE_PERIODIC:
        idx = (-idx - 1) % len_x
        return x[len_x - idx - 1]
    elif mode == MODE_SMOOTH:
        return x[0] + idx * (x[1] - x[0])
    elif mode == MODE_LINE:
        lin_slope = (x[len_x - 1] - x[0]) / (len_x - 1)
        return x[0] + idx * lin_slope
    elif mode == MODE_ANTISYMMETRIC:
        if (-idx) < len_x:
            return -x[-idx - 1]
        else:
            idx = (-idx - 1) % (2 * len_x)
            if idx < len_x:
                return -x[idx]
            else:
                return x[len_x - 1 - (idx - len_x)]
    elif mode == MODE_ANTIREFLECT:
        if (-idx) < len_x:
            return x[0] - (x[-idx] - x[0])
        else:
            le = x[0] + (x[0] - x[len_x - 1]) * ((-(idx) - 1) // (len_x - 1))
            idx = (-idx - 1) % (2 * (len_x - 1))
            if idx < (len_x - 1):
                return le - (x[idx + 1] - x[0])
            else:
                return le - (x[len_x - 1] - x[len_x - 2 - (idx - (len_x - 1))])
    elif mode == MODE_CONSTANT_EDGE:
        return x[0]
    elif mode == MODE_CONSTANT:
        return cval
    else:
        return -1.


# 使用 Cython 的优化指令，加速模块运算
@cython.cdivision(True)  # faster modulo
@cython.boundscheck(False)  # designed to stay within bounds
@cython.wraparound(False)  # we don't use negative indexing
# 定义一个 Cython 编写的函数 _extend_right，用于处理边界扩展操作
cdef DTYPE_t _extend_right(DTYPE_t *x, np.intp_t idx, np.intp_t len_x,
                           MODE mode, DTYPE_t cval) noexcept nogil:
    # 右边界扩展函数，初始化右边界值和线性斜率为零
    cdef DTYPE_t re = 0.
    cdef DTYPE_t lin_slope = 0.
    # 如果模式为对称模式
    if mode == MODE_SYMMETRIC:
        # 如果索引小于2倍长度x，返回对称位置的元素
        if idx < (2 * len_x):
            return x[len_x - 1 - (idx - len_x)]
        else:
            # 对索引进行模运算，确保在有效范围内
            idx = idx % (2 * len_x)
            if idx < len_x:
                return x[idx]
            else:
                return x[len_x - 1 - (idx - len_x)]
    
    # 如果模式为反射模式
    elif mode == MODE_REFLECT:
        # 如果索引小于2倍长度x减1，返回反射位置的元素
        if idx < (2 * len_x - 1):
            return x[len_x - 2 - (idx - len_x)]
        else:
            # 对索引进行模运算，确保在有效范围内
            idx = idx % (2 * (len_x - 1))
            if idx < (len_x - 1):
                return x[idx]
            else:
                return x[len_x - 1 - (idx - (len_x - 1))]
    
    # 如果模式为周期模式
    elif mode == MODE_PERIODIC:
        # 返回索引对长度x取模后的元素
        return x[idx % len_x]
    
    # 如果模式为平滑模式
    elif mode == MODE_SMOOTH:
        # 使用线性插值计算平滑模式下的元素值
        return x[len_x - 1] + (idx - len_x + 1) * (x[len_x - 1] - x[len_x - 2])
    
    # 如果模式为线性模式
    elif mode == MODE_LINE:
        # 计算线性斜率
        lin_slope = (x[len_x - 1] - x[0]) / (len_x - 1)
        # 返回线性模式下的元素值
        return x[len_x - 1] + (idx - len_x + 1) * lin_slope
    
    # 如果模式为常数边缘模式
    elif mode == MODE_CONSTANT_EDGE:
        # 返回常数边缘模式下的元素值（最后一个元素）
        return x[len_x - 1]
    
    # 如果模式为反对称模式
    elif mode == MODE_ANTISYMMETRIC:
        # 如果索引小于2倍长度x，返回反对称位置的元素的负值
        if idx < (2 * len_x):
            return -x[len_x - 1 - (idx - len_x)]
        else:
            # 对索引进行模运算，确保在有效范围内
            idx = idx % (2 * len_x)
            if idx < len_x:
                return x[idx]
            else:
                return -x[len_x - 1 - (idx - len_x)]
    
    # 如果模式为反反射模式
    elif mode == MODE_ANTIREFLECT:
        # 如果索引小于2倍长度x减1，返回反反射位置的元素
        if idx < (2 * len_x - 1):
            return x[len_x - 1] - (x[len_x - 2 - (idx - len_x)] - x[len_x - 1])
        else:
            # 使用复杂的插值计算反反射模式下的元素值
            re = x[len_x - 1] + (x[len_x - 1] - x[0]) * (idx // (len_x - 1) - 1)
            idx = idx % (2 * (len_x - 1))
            if idx < (len_x - 1):
                return re + (x[idx] - x[0])
            else:
                return re + (x[len_x - 1] - x[len_x - 1 - (idx - (len_x - 1))])
    
    # 如果模式为常数模式
    elif mode == MODE_CONSTANT:
        # 返回常数模式下的指定常数值
        return cval
    
    # 如果模式未知或不支持的情况，返回-1
    else:
        return -1
# 使用 Cython 的编译指令：禁用数组边界检查
@cython.boundscheck(False)
# 使用 Cython 的编译指令：禁用负数索引包装
@cython.wraparound(False)
# 定义一个 Cython 编译的函数，接受以下参数：
# - data: 一个一维 NumPy 数组，数据类型为 DTYPE_t
# - npre: 整数，表示在数组前端添加的数据点数，默认为 0
# - npost: 整数，表示在数组尾端添加的数据点数，默认为 0
# - mode: 一个对象，表示信号扩展的模式，默认为 0
cpdef _pad_test(np.ndarray[DTYPE_t] data, np.intp_t npre=0, np.intp_t npost=0,
                object mode=0):
    """1D test function for signal extension modes.

    Returns ``data extended by ``npre``, ``npost`` at the beginning, end.
    """
    # 声明 Cython 的整型变量 idx 和 cnt
    cdef np.intp_t idx
    cdef np.intp_t cnt = 0
    # 获取输入数组 data 的长度
    cdef np.intp_t len_x = data.size
    # 计算输出数组 out 的长度
    cdef np.intp_t len_out = npre + len_x + npost
    # 声明 Cython 的数据类型变量 xval 和 out
    cdef DTYPE_t xval
    cdef DTYPE_t [::1] out
    # 声明指向数据的指针 data_ptr 和 _mode 变量
    cdef DTYPE_t* data_ptr
    cdef MODE _mode
    # 将 mode 转换为枚举类型 MODE
    _mode = mode_enum(mode)

    # 根据数据类型 DTYPE_t 初始化输出数组 out
    if DTYPE_t is float:
        out = np.zeros((len_out,), dtype=np.float32)
    elif DTYPE_t is float_complex:
        out = np.zeros((len_out,), dtype=np.complex64)
    elif DTYPE_t is double:
        out = np.zeros((len_out,), dtype=np.float64)
    elif DTYPE_t is double_complex:
        out = np.zeros((len_out,), dtype=np.complex128)
    else:
        raise ValueError("unsupported dtype")

    # 获取输入数组 data 的数据指针
    data_ptr = <DTYPE_t*> data.data
    # 使用 nogil 上下文，实现无全局解锁的并行计算
    with nogil:
        # 遍历扩展后的数据区域，填充输出数组 out
        for idx in range(-npre, len_x + npost, 1):
            if idx < 0:
                xval = _extend_left(data_ptr, idx, len_x, _mode, 0.0)
            elif idx >= len_x:
                xval = _extend_right(data_ptr, idx, len_x, _mode, 0.0)
            else:
                xval = data_ptr[idx]
            out[cnt] = xval
            cnt += 1
    # 将 Cython 数组 out 转换为 NumPy 数组，并返回
    return np.asarray(out)


# 定义一个 Cython 编译的函数，接受以下参数：
# - data: 输入的 NumPy 数组
# - h_trans_flip: 输入的常量一维数组，数据类型为 DTYPE_t
# - out: 输出的 NumPy 数组
# - up, down: 整数，表示数据处理时的上采样和下采样倍数
# - axis: 整数，表示在哪个轴上应用滤波器
# - mode: 整数，表示信号扩展的模式
# - cval: 数据类型为 DTYPE_t 的常量值
def _apply(np.ndarray data, const DTYPE_t [::1] h_trans_flip, np.ndarray out,
           np.intp_t up, np.intp_t down, np.intp_t axis, np.intp_t mode,
           DTYPE_t cval):
    # 声明 Cython 的结构体变量 data_info 和 output_info
    cdef ArrayInfo data_info, output_info
    # 声明 Cython 的整型变量 len_h 和 retval
    cdef np.intp_t len_h = h_trans_flip.size
    cdef int retval
    # 获取输出数组 out 在指定轴上的长度
    cdef np.intp_t len_out = out.shape[axis]

    # 设置输入数据信息结构体 data_info 的属性
    data_info.ndim = data.ndim
    data_info.strides = <np.intp_t *> data.strides
    data_info.shape = <np.intp_t *> data.shape

    # 设置输出数据信息结构体 output_info 的属性
    output_info.ndim = out.ndim
    output_info.strides = <np.intp_t *> out.strides
    output_info.shape = <np.intp_t *> out.shape

    # 获取输入数组 data 和 h_trans_flip 的数据指针
    data_ptr = <DTYPE_t*> data.data
    filter_ptr = <DTYPE_t*> &h_trans_flip[0]
    out_ptr = <DTYPE_t*> out.data

    # 使用 nogil 上下文，实现无全局解锁的并行计算
    with nogil:
        # 调用内部函数 _apply_axis_inner 处理数据
        retval = _apply_axis_inner(data_ptr, data_info,
                                   filter_ptr, len_h,
                                   out_ptr, output_info,
                                   up, down, axis, <MODE>mode, cval, len_out)
    # 根据返回值判断函数调用是否成功，若失败则抛出异常
    if retval == 1:
        raise ValueError("failure in _apply_axis_inner: data and output arrays"
                         " must have the same number of dimensions.")
    elif retval == 2:
        raise ValueError(
            ("failure in _apply_axis_inner: axis = {}, ".format(axis) +
             "but data_info.ndim is only {}.".format(data_info.ndim)))
    elif retval == 3 or retval == 4:
        raise MemoryError()
    # 返回处理后的输出数组
    return out


# 使用 Cython 的编译指令：启用 C 的除法规则
@cython.cdivision(True)
# 使用 Cython 的编译指令：禁用数组边界检查
@cython.boundscheck(False)
# 使用 Cython 的编译指令：禁用负数索引包装
@cython.wraparound(False)
# 定义一个 C 语言扩展函数 _apply_axis_inner，计算某轴上的操作结果
cdef int _apply_axis_inner(DTYPE_t* data, ArrayInfo data_info,
                           DTYPE_t* h_trans_flip, np.intp_t len_h,
                           DTYPE_t* output, ArrayInfo output_info,
                           np.intp_t up, np.intp_t down,
                           np.intp_t axis, MODE mode, DTYPE_t cval,
                           np.intp_t len_out) noexcept nogil:
    # 定义循环索引变量 i
    cdef np.intp_t i
    # 初始化循环次数为 1
    cdef np.intp_t num_loops = 1
    # 定义标志变量，用于判断是否需要创建临时数据和输出数组
    cdef bint make_temp_data, make_temp_output
    # 初始化临时数据和输出数组的指针为空
    cdef DTYPE_t* temp_data = NULL
    cdef DTYPE_t* temp_output = NULL
    # 初始化行大小（字节）
    cdef size_t row_size_bytes = 0

    # 检查输入和输出数组的维度是否一致，若不一致则返回错误码 1
    if data_info.ndim != output_info.ndim:
        return 1
    # 检查轴的索引是否超出数据维度范围，若超出则返回错误码 2
    if axis >= data_info.ndim:
        return 2

    # 根据数据和输出数组在轴上的步幅是否为数据类型大小判断是否需要创建临时数据数组
    make_temp_data = data_info.strides[axis] != sizeof(DTYPE_t);
    # 根据输出数组在轴上的步幅是否为数据类型大小判断是否需要创建临时输出数组
    make_temp_output = output_info.strides[axis] != sizeof(DTYPE_t);
    
    # 若需要创建临时数据数组
    if make_temp_data:
        # 分配内存给临时数据数组，大小为该轴上的元素个数乘以数据类型大小
        temp_data = <DTYPE_t*>malloc(data_info.shape[axis] * sizeof(DTYPE_t))
        # 若内存分配失败，则释放已分配的临时数据数组内存并返回错误码 3
        if not temp_data:
            free(temp_data)
            return 3
    # 若需要创建临时输出数组
    if make_temp_output:
        # 计算输出数组在轴上的总字节数
        row_size_bytes = output_info.shape[axis] * sizeof(DTYPE_t)
        # 分配内存给临时输出数组，大小为输出数组在轴上的字节数
        temp_output = <DTYPE_t*>malloc(row_size_bytes)
        # 若内存分配失败，则释放已分配的临时数据数组和临时输出数组内存并返回错误码 4
        if not temp_output:
            free(temp_data)
            free(temp_output)
            return 4

    # 计算除了指定轴以外的其他维度的总循环次数
    for i in range(output_info.ndim):
        if i != axis:
            num_loops *= output_info.shape[i]

    # 计算数据数组在指定轴上的步幅（元素个数）
    # 注意：这里的步幅是以元素个数而不是字节数表示的
    cdef np.intp_t idx_stride = data_info.strides[axis] / sizeof(DTYPE_t)
    # 计算输出数组在指定轴上的步幅（元素个数）
    cdef np.intp_t idx_stride_out = output_info.strides[axis] / sizeof(DTYPE_t)

    # 定义循环索引变量 j 和数据偏移量、输出偏移量
    cdef np.intp_t j
    cdef np.intp_t data_offset
    cdef np.intp_t output_offset
    # 定义数据数组和输出数组当前行的指针
    cdef DTYPE_t* data_row
    cdef DTYPE_t* output_row
    # 定义减少的索引值、反向索引值、轴上的索引
    cdef np.intp_t reduced_idx
    cdef np.intp_t j_rev
    cdef np.intp_t axis_idx
    # 初始化临时指针为空
    cdef DTYPE_t* tmp_ptr = NULL
    # 循环执行指定次数
    for i in range(num_loops):
        # 初始化数据偏移量和输出偏移量
        data_offset = 0
        output_offset = 0

        # 计算线性缓冲区的偏移量
        reduced_idx = i
        for j in range(output_info.ndim):
            j_rev = output_info.ndim - 1 - j
            # 如果 j_rev 不等于 axis，则计算当前轴的索引
            if j_rev != axis:
                axis_idx = reduced_idx % output_info.shape[j_rev]
                reduced_idx /= output_info.shape[j_rev]
                data_offset += (axis_idx * data_info.strides[j_rev])
                output_offset += (axis_idx * output_info.strides[j_rev])

        # 如果需要创建临时数据，则复制到临时数据
        if make_temp_data:
            # 偏移量是字节偏移量，需要进行 char 类型的转换然后再转回
            tmp_ptr = <DTYPE_t *>((<char *> data) + data_offset)
            for j in range(data_info.shape[axis]):
                temp_data[j] = tmp_ptr[idx_stride*j]

        # 根据需求选择临时输出或直接输出以及数据
        if make_temp_data:
            data_row = temp_data
        else:
            data_row = <DTYPE_t *>((<char *>data) + data_offset)
        if make_temp_output:
            output_row = temp_output
            # 将输出行清零
            memset(output_row, 0, row_size_bytes)
        else:
            output_row = <DTYPE_t *>((<char *>output) + output_offset)

        # 调用 1D upfirdn 函数进行处理
        _apply_impl(data_row, data_info.shape[axis],
                    h_trans_flip, len_h, output_row, up, down, mode, cval,
                    len_out)

        # 如果需要创建临时输出，则从临时输出复制数据
        if make_temp_output:
            tmp_ptr = <DTYPE_t *>((<char *>output) + output_offset)
            for j in range(output_info.shape[axis]):
                tmp_ptr[idx_stride_out*j] = output_row[j]

    # 清理临时数据和临时输出
    free(temp_data)
    free(temp_output)
    # 返回 0 表示成功完成
    return 0
# 设置 Cython 优化选项：启用 C 语言风格的整除运算，加快速度
@cython.cdivision(True)  # faster modulo
# 设置 Cython 优化选项：禁用边界检查，假设数组索引永远在范围内
@cython.boundscheck(False)  # designed to stay within bounds
# 设置 Cython 优化选项：禁用负索引访问，不使用负数索引
@cython.wraparound(False)  # we don't use negative indexing
# 定义内部函数 _apply_impl，用于执行卷积操作
cdef void _apply_impl(DTYPE_t *x, np.intp_t len_x, DTYPE_t *h_trans_flip,
                      np.intp_t len_h, DTYPE_t *out,
                      np.intp_t up, np.intp_t down, MODE mode,
                      DTYPE_t cval, np.intp_t len_out) noexcept nogil:
    # 计算每个卷积阶段的滤波器长度
    cdef np.intp_t h_per_phase = len_h // up
    # 计算填充后的输入长度
    cdef np.intp_t padded_len = len_x + h_per_phase - 1
    # 初始化索引和计数变量
    cdef np.intp_t x_idx = 0
    cdef np.intp_t y_idx = 0
    cdef np.intp_t h_idx = 0
    cdef np.intp_t t = 0
    cdef np.intp_t x_conv_idx = 0
    cdef DTYPE_t xval
    cdef bint zpad

    # 根据模式和常数值设定是否进行零填充
    zpad = (mode == MODE_CONSTANT and cval == 0)
    # 如果输出长度为0，则直接返回
    if len_out == 0:
        return

    # 主循环，对输入信号进行卷积操作
    while x_idx < len_x:
        # 计算当前滤波器索引
        h_idx = t * h_per_phase
        # 计算卷积索引，处理边界情况
        x_conv_idx = x_idx - h_per_phase + 1
        if x_conv_idx < 0:
            # 处理左边界情况
            if zpad:
                h_idx -= x_conv_idx
            else:
                for x_conv_idx in range(x_conv_idx, 0):
                    # 执行左边界扩展操作
                    xval = _extend_left(x, x_conv_idx, len_x, mode, cval)
                    out[y_idx] += xval * h_trans_flip[h_idx]
                    h_idx += 1
            x_conv_idx = 0
        # 执行卷积计算
        for x_conv_idx in range(x_conv_idx, x_idx + 1):
            out[y_idx] = out[y_idx] + x[x_conv_idx] * h_trans_flip[h_idx]
            h_idx += 1
        # 存储结果并递增索引
        y_idx += 1
        # 如果输出索引超过长度，则返回
        if y_idx >= len_out:
            return
        # 更新时间步长
        t += down
        x_idx += t // up
        # 更新滤波器阶段
        t = t % up

    # 使用简化循环处理剩余部分的卷积计算
    while x_idx < padded_len:
        h_idx = t * h_per_phase
        x_conv_idx = x_idx - h_per_phase + 1
        for x_conv_idx in range(x_conv_idx, x_idx + 1):
            if x_conv_idx >= len_x:
                # 处理右边界情况
                xval = _extend_right(x, x_conv_idx, len_x, mode, cval)
            elif x_conv_idx < 0:
                # 处理左边界情况
                xval = _extend_left(x, x_conv_idx, len_x, mode, cval)
            else:
                xval = x[x_conv_idx]
            out[y_idx] += xval * h_trans_flip[h_idx]
            h_idx += 1
        y_idx += 1
        if y_idx >= len_out:
            return
        t += down
        x_idx += t // up
        t = t % up
```
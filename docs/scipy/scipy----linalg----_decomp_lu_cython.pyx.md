# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_lu_cython.pyx`

```
# 设置 Cython 的语言级别为3（Python 3），指示Cython编译器使用Python 3的语言特性
# 导入 Cython 模块
# 从 CPython 的内存管理中导入 PyMem_Malloc 和 PyMem_Free 函数
# 从 scipy.linalg.cython_lapack 中导入 sgetrf, dgetrf, cgetrf, zgetrf 函数
# 从 scipy.linalg._cythonized_array_utils 中导入 swap_c_and_f_layout 函数
# 导入 NumPy C-API 的 cnp 模块，并调用 import_array() 函数进行初始化
# 定义一个类型融合（fused）的 lapack_t，包括 cnp.float32_t, cnp.float64_t, cnp.complex64_t, cnp.complex128_t

@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
# 使用 Cython 的装饰器设置函数的 nonecheck, wraparound, boundscheck, initializedcheck 属性
# cdef 声明一个 C 函数 lu_decompose，接受以下参数：
# - a: lapack_t 类型的二维 NumPy 数组，用于 LU 分解
# - lu: lapack_t 类型的二维 NumPy 数组，用于存储 LU 分解结果
# - perm: int 类型的一维 NumPy 数组，用于存储排列信息
# - permute_l: 布尔值，控制是否进行 L 的置换
# 函数是无异常抛出的（noexcept），即声明不会抛出异常

"""LU decomposition and copy operations using ?getrf routines

This function overwrites inputs. For interfacing LAPACK,
it creates a memory buffer and copies into with F-order
then swaps back to C order hence no need for dealing with
Fortran arrays which are inconvenient.

After the LU factorization, to minimize the amount of data
copied, for rectangle arrays, and depending on the size,
the smaller portion is copied out to U and the rest becomes
either U or L. The logic is handled by the caller.

        ┌     ┐
        │ \ U │
        │  \  │       ┌                ┐
        │   \ │       │  \             │
     a= │     │  or   │   \      U     │
        │ L   │       │ L  \           │
        │     │       └                ┘
        └     ┘
         tall               wide
     (extract U)         (extract L)

"""
# 函数文档字符串，解释了 LU 分解的操作以及数据拷贝的方式和优化逻辑

    cdef int m = a.shape[0], n = a.shape[1], mn = min(m, n)
    # 声明整型变量 m, n, mn，分别为数组 a 的行数、列数以及 m 和 n 中的较小值
    cdef cnp.npy_intp dims[2]
    # 声明一个 NumPy 数组 dims，用于存储数组的维度信息
    cdef int info = 0, ind1, ind2, tmp_int
    # 声明整型变量 info, ind1, ind2, tmp_int
    cdef lapack_t *aa = <lapack_t *>cnp.PyArray_DATA(a)
    # 声明 lapack_t 类型的指针 aa，指向数组 a 的数据区域
    cdef lapack_t *bb
    # 声明 lapack_t 类型的指针 bb
    cdef int *ipiv = <int*>PyMem_Malloc(m * sizeof(int))
    # 使用 PyMem_Malloc 分配存储整型数据的内存，大小为 m * sizeof(int)，并将指针赋给 ipiv
    if not ipiv:
        raise MemoryError('scipy.linalg.lu failed to allocate '
                          'required memory.')
    # 如果内存分配失败，则抛出 MemoryError 异常

    dims[0] = m
    dims[1] = n
    # 设置 dims 数组的值为 m 和 n

    if lapack_t is cnp.float32_t:
        b = cnp.PyArray_SimpleNew(2, dims, cnp.NPY_FLOAT32)
        bb = <cnp.float32_t *>cnp.PyArray_DATA(b)
        swap_c_and_f_layout(aa, bb, m, n)
        sgetrf(&m, &n, bb, &m, ipiv, &info)
    elif lapack_t is cnp.float64_t:
        b = cnp.PyArray_SimpleNew(2, dims, cnp.NPY_FLOAT64)
        bb = <cnp.float64_t *>cnp.PyArray_DATA(b)
        swap_c_and_f_layout(aa, bb, m, n)
        dgetrf(&m, &n, bb, &m, ipiv, &info)
    elif lapack_t is cnp.complex64_t:
        b = cnp.PyArray_SimpleNew(2, dims, cnp.NPY_COMPLEX64)
        bb = <cnp.complex64_t *>cnp.PyArray_DATA(b)
        swap_c_and_f_layout(aa, bb, m, n)
        cgetrf(&m, &n, bb, &m, ipiv, &info)
    else:
        b = cnp.PyArray_SimpleNew(2, dims, cnp.NPY_COMPLEX128)
        bb = <cnp.complex128_t *>cnp.PyArray_DATA(b)
        swap_c_and_f_layout(aa, bb, m, n)
        zgetrf(&m, &n, bb, &m, ipiv, &info)
# 根据 lapack_t 类型选择相应的 NumPy 数组 b，并调用相应的 ?getrf 函数进行 LU 分解
    # 如果 info 小于 0，说明 lu 分解过程中遇到错误，抛出 ValueError 异常
    if info < 0:
        raise ValueError('scipy.linalg.lu has encountered an internal'
                         ' error in ?getrf routine with invalid value'
                         f' at {-info}-th parameter.')

    # 将结果转换为 C 风格的内存布局并进行清理
    swap_c_and_f_layout(bb, aa, n, m)

    # 将 A 上的置换转换为 L 的排列，因为 A = P @ L @ U
    try:
        # 主要是按照 ipiv 中的循环进行操作，并用逆排列交换 "np.arange" 数组
        # 初始化 perm
        for ind1 in range(m): perm[ind1] = ind1
        for ind1 in range(mn):
            tmp_int = perm[ipiv[ind1]-1]
            perm[ipiv[ind1]-1] = perm[ind1]
            perm[ind1] = tmp_int

        # 将 iperm 转换为 perm，并存储回 ipiv 中作为最终解。无需 argsort：ipiv[perm] = np.arange(m)
        for ind1 in range(m):
            ipiv[perm[ind1]] = ind1
        for ind1 in range(m):
            perm[ind1] = ipiv[ind1]

    finally:
        PyMem_Free(ipiv)

    # 分离 L 和 U 部分

    if m > n:  # 高瘦型，"a" 包含较大的 L
        # 提取右上角矩形区域到 lu
        for ind1 in range(mn):  # 行
            lu[ind1, ind1:mn] = a[ind1, ind1:mn]

        for ind1 in range(mn):
            a[ind1, ind1] = 1
            a[ind1, ind1+1:mn] = 0

    else:  # 方阵或矮胖型，"a" 包含较大的 U

        lu[0, 0] = 1
        for ind1 in range(1, mn):  # 行
            lu[ind1, :ind1] = a[ind1, :ind1]
            lu[ind1, ind1] = 1

        for ind2 in range(mn - 1):  # 列
            for ind1 in range(ind2+1, m):  # 行
                a[ind1, ind2] = 0

    if permute_l:
        # b 仍然存在 -> 将其用作临时数组
        # 将所有内容复制到 b，并按照 perm 指示的行重新从 b 中选择

        if m > n:
            b[:, :] = a[:, :]
            # memcpy(bb, &a[0, 0], m*mn*sizeof(lapack_t))
            for ind1 in range(m):
                if perm[ind1] == ind1:
                    continue
                else:
                    a[ind1, :] = b[perm[ind1], :]

        else:  # 对 lu 数组做相同操作
            b[:mn, :mn] = lu[:, :]
            # memcpy(bb, &lu[0, 0], mn*n*sizeof(lapack_t))
            for ind1 in range(mn):
                if perm[ind1] == ind1:
                    continue
                else:
                    lu[ind1, :] = b[perm[ind1], :mn]
# 设置 Cython 的 nonecheck 和 initializedcheck 标志为 False，用于关闭空指针和未初始化变量的检查
@cython.nonecheck(False)
@cython.initializedcheck(False)
# 定义一个 lu_dispatcher 函数，根据输入数组的数据类型选择合适的 LU 分解函数
def lu_dispatcher(a, u, piv, permute_l):
    # 如果输入数组的数据类型为单精度浮点数（'f'）
    if a.dtype.char == 'f':
        # 调用 lu_decompose 函数，传入单精度浮点数类型标识符 cnp.float32_t
        lu_decompose[cnp.float32_t](a, u, piv, permute_l)
    # 如果输入数组的数据类型为双精度浮点数（'d'）
    elif a.dtype.char == 'd':
        # 调用 lu_decompose 函数，传入双精度浮点数类型标识符 cnp.float64_t
        lu_decompose[cnp.float64_t](a, u, piv, permute_l)
    # 如果输入数组的数据类型为单精度复数（'F'）
    elif a.dtype.char == 'F':
        # 调用 lu_decompose 函数，传入单精度复数类型标识符 cnp.complex64_t
        lu_decompose[cnp.complex64_t](a, u, piv, permute_l)
    # 如果输入数组的数据类型为双精度复数（'D'）
    elif a.dtype.char == 'D':
        # 调用 lu_decompose 函数，传入双精度复数类型标识符 cnp.complex128_t
        lu_decompose[cnp.complex128_t](a, u, piv, permute_l)
    else:
        # 如果输入数组的数据类型不在支持的范围内，则抛出 TypeError 异常
        raise TypeError("Unsupported type given to lu_dispatcher")
```
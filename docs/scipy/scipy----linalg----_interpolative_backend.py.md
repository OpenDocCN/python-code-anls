# `D:\src\scipysrc\scipy\scipy\linalg\_interpolative_backend.py`

```
"""
Direct wrappers for Fortran `id_dist` backend.
"""

import scipy.linalg._interpolative as _id  # 导入 scipy 中的 `_interpolative` 模块，并将其命名为 `_id`
import numpy as np  # 导入 numpy 库，并将其命名为 np

_RETCODE_ERROR = RuntimeError("nonzero return code")  # 定义一个 RuntimeError 类型的异常对象，用于表示非零返回码错误


def _asfortranarray_copy(A):
    """
    Same as np.asfortranarray, but ensure a copy
    
    将输入数组 A 转换为 Fortran 风格的数组，并确保返回一个拷贝
    """
    A = np.asarray(A)  # 将输入 A 转换为 numpy 数组
    if A.flags.f_contiguous:  # 检查数组是否是 Fortran 连续的
        A = A.copy(order="F")  # 如果是 Fortran 连续的，则进行深拷贝并设置为 Fortran 风格
    else:
        A = np.asfortranarray(A)  # 否则，使用 np.asfortranarray 将 A 转换为 Fortran 风格的数组
    return A


#------------------------------------------------------------------------------
# id_rand.f
#------------------------------------------------------------------------------

def id_srand(n):
    """
    Generate standard uniform pseudorandom numbers via a very efficient lagged
    Fibonacci method.

    :param n:
        Number of pseudorandom numbers to generate.
    :type n: int

    :return:
        Pseudorandom numbers.
    :rtype: :class:`numpy.ndarray`
    
    使用高效的滞后 Fibonacci 方法生成标准均匀分布的伪随机数数组。
    """
    return _id.id_srand(n)  # 调用 `_id` 模块中的 `id_srand` 函数并返回结果


def id_srandi(t):
    """
    Initialize seed values for :func:`id_srand` (any appropriately random
    numbers will do).

    :param t:
        Array of 55 seed values.
    :type t: :class:`numpy.ndarray`
        
    为 `id_srand` 函数初始化种子值数组 `t`（可以是任意合适的随机数）。
    """
    t = np.asfortranarray(t)  # 将输入的数组 `t` 转换为 Fortran 风格的数组
    _id.id_srandi(t)  # 调用 `_id` 模块中的 `id_srandi` 函数，对种子值进行初始化


def id_srando():
    """
    Reset seed values to their original values.
    
    将种子值重置为其原始值。
    """
    _id.id_srando()  # 调用 `_id` 模块中的 `id_srando` 函数，将种子值重置为原始值
#------------------------------------------------------------------------------
# idd_frm.f
#------------------------------------------------------------------------------

# 定义函数 idd_frm，用于对实数向量进行变换，包括 Rokhlin 随机变换、随机子选择和 FFT
def idd_frm(n, w, x):
    """
    Transform real vector via a composition of Rokhlin's random transform,
    random subselection, and an FFT.

    In contrast to :func:`idd_sfrm`, this routine works best when the length of
    the transformed vector is the power-of-two integer output by
    :func:`idd_frmi`, or when the length is not specified but instead
    determined a posteriori from the output. The returned transformed vector is
    randomly permuted.

    :param n:
        Greatest power-of-two integer satisfying ``n <= x.size`` as obtained from
        :func:`idd_frmi`; `n` is also the length of the output vector.
    :type n: int
    :param w:
        Initialization array constructed by :func:`idd_frmi`.
    :type w: :class:`numpy.ndarray`
    :param x:
        Vector to be transformed.
    :type x: :class:`numpy.ndarray`

    :return:
        Transformed vector.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用 _id 模块中的 idd_frm 函数进行实际处理
    return _id.idd_frm(n, w, x)


# 定义函数 idd_sfrm，用于对实数向量进行变换，包括 Rokhlin 随机变换、随机子选择和 FFT
def idd_sfrm(l, n, w, x):
    """
    Transform real vector via a composition of Rokhlin's random transform,
    random subselection, and an FFT.

    In contrast to :func:`idd_frm`, this routine works best when the length of
    the transformed vector is known a priori.

    :param l:
        Length of transformed vector, satisfying ``l <= n``.
    :type l: int
    :param n:
        Greatest power-of-two integer satisfying ``n <= x.size`` as obtained from
        :func:`idd_sfrmi`.
    :type n: int
    :param w:
        Initialization array constructed by :func:`idd_sfrmi`.
    :type w: :class:`numpy.ndarray`
    :param x:
        Vector to be transformed.
    :type x: :class:`numpy.ndarray`

    :return:
        Transformed vector.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用 _id 模块中的 idd_sfrm 函数进行实际处理
    return _id.idd_sfrm(l, n, w, x)


# 定义函数 idd_frmi，用于初始化 idd_frm 函数所需的数据
def idd_frmi(m):
    """
    Initialize data for :func:`idd_frm`.

    :param m:
        Length of vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying ``n <= m``.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idd_frm`.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用 _id 模块中的 idd_frmi 函数进行实际处理
    return _id.idd_frmi(m)


# 定义函数 idd_sfrmi，用于初始化 idd_sfrm 函数所需的数据
def idd_sfrmi(l, m):
    """
    Initialize data for :func:`idd_sfrm`.

    :param l:
        Length of output transformed vector.
    :type l: int
    :param m:
        Length of the vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying ``n <= m``.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idd_sfrm`.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用 _id 模块中的 idd_sfrmi 函数进行实际处理
    return _id.idd_sfrmi(l, m)


#------------------------------------------------------------------------------
# idd_id.f
#------------------------------------------------------------------------------
def idd_id2svd(B, idx, proj):
    """
    Convert real ID to SVD.

    :param B:
        Skeleton matrix.
    :type B: :class:`numpy.ndarray`
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`
    """
    # 将输入的骨架矩阵 B 转换为 Fortran 风格的数组
    B = np.asfortranarray(B)
    # 调用 _id 模块中的 idd_id2svd 函数，将实际ID转换为SVD
    return _id.idd_id2svd(B, idx, proj)
    # 将输入数组 B 转换为列优先（Fortran）顺序的数组
    B = np.asfortranarray(B)
    
    # 调用 _id.idd_id2svd 函数进行奇异值分解计算
    # 返回值 U 是左奇异向量，V 是右奇异向量，S 是奇异值，ier 表示操作是否成功
    U, V, S, ier = _id.idd_id2svd(B, idx, proj)
    
    # 如果 ier 非零，表示奇异值分解出现错误，抛出 _RETCODE_ERROR 异常
    if ier:
        raise _RETCODE_ERROR
    
    # 返回计算得到的左奇异向量 U，右奇异向量 V，和奇异值 S
    return U, V, S
#------------------------------------------------------------------------------
# idd_snorm.f
#------------------------------------------------------------------------------

# 估算实数矩阵的谱范数，使用随机幂方法
def idd_snorm(m, n, matvect, matvec, its=20):
    """
    Estimate spectral norm of a real matrix by the randomized power method.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param its:
        Number of power method iterations.
    :type its: int

    :return:
        Spectral norm estimate.
    :rtype: float
    """
    # 调用底层函数 _id.idd_snorm 进行实际计算
    snorm, v = _id.idd_snorm(m, n, matvect, matvec, its)
    # 返回谱范数估计结果
    return snorm


# 估算两个实数矩阵差的谱范数，使用随机幂方法
def idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its=20):
    """
    Estimate spectral norm of the difference of two real matrices by the
    randomized power method.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the transpose of the first matrix to a vector, with
        call signature `y = matvect(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matvect: function
    :param matvect2:
        Function to apply the transpose of the second matrix to a vector, with
        call signature `y = matvect2(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matvect2: function
    :param matvec:
        Function to apply the first matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param matvec2:
        Function to apply the second matrix to a vector, with call signature
        `y = matvec2(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec2: function
    :param its:
        Number of power method iterations.
    :type its: int

    :return:
        Spectral norm estimate of matrix difference.
    :rtype: float
    """
    # 调用底层函数 _id.idd_diffsnorm 进行实际计算
    return _id.idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its)


#------------------------------------------------------------------------------
# idd_svd.f
#------------------------------------------------------------------------------

# 计算实矩阵的奇异值分解到指定秩
def iddr_svd(A, k):
    """
    Compute SVD of a real matrix to a specified rank.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of SVD.
    :type k: int
    """
    # 将输入的矩阵 A 转换为 Fortran（列主序）存储顺序的数组
    A = np.asfortranarray(A)
    # 调用某个函数 _id.iddr_svd 对 A 进行奇异值分解，返回左奇异向量 U、右奇异向量 V、奇异值 S 和错误码 ier
    U, V, S, ier = _id.iddr_svd(A, k)
    # 如果错误码 ier 不为零，抛出一个错误 _RETCODE_ERROR
    if ier:
        raise _RETCODE_ERROR
    # 返回计算得到的左奇异向量 U、右奇异向量 V 和奇异值 S
    return U, V, S
def iddp_asvd(eps, A):
    """
    Compute SVD of a real matrix to a specified relative precision using random
    sampling.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入的矩阵 A 转换为 Fortran 格式的数组
    A = np.asfortranarray(A)
    # 获取矩阵 A 的行数 m 和列数 n
    m, n = A.shape
    # 调用外部库函数 _id.idd_frmi，获取 n2 和 winit
    n2, winit = _id.idd_frmi(m)
    # 创建一个空的 NumPy 数组 w，用于存储计算中的中间结果
    w = np.empty(
        # 计算数组 w 的长度，使用两个表达式中的最大值作为数组长度
        max((min(m, n) + 1)*(3*m + 5*n + 1) + 25*min(m, n)**2,
            (2*n + 1)*(n2 + 1)),
        # 指定数组的存储顺序为 Fortran 风格
        order='F')
    
    # 调用 _id 模块中的 iddp_asvd 函数进行奇异值分解
    k, iU, iV, iS, w, ier = _id.iddp_asvd(eps, A, winit, w)
    
    # 如果返回值 ier 非零，抛出一个自定义的异常 _RETCODE_ERROR
    if ier:
        raise _RETCODE_ERROR
    
    # 根据返回的索引和尺寸信息从数组 w 中提取 U, V, S 矩阵
    U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')  # 提取并重塑矩阵 U
    V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')  # 提取并重塑矩阵 V
    S = w[iS-1:iS+k-1]  # 提取奇异值向量 S
    
    # 返回分解后的矩阵 U, V 和奇异值向量 S
    return U, V, S
#------------------------------------------------------------------------------
# iddp_rid.f
#------------------------------------------------------------------------------

def iddp_rid(eps, m, n, matvect):
    """
    Compute ID of a real matrix to a specified relative precision using random
    matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    # Allocate memory for projection array
    proj = np.empty(m + 1 + 2*n*(min(m, n) + 1), order='F')
    # Call external C function to compute ID with random matrix-vector multiplication
    k, idx, proj, ier = _id.iddp_rid(eps, m, n, matvect, proj)
    # Check if error occurred during computation
    if ier != 0:
        raise _RETCODE_ERROR
    # Reshape projection array to the correct dimensions
    proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
    # Return computed values
    return k, idx, proj


def idd_findrank(eps, m, n, matvect):
    """
    Estimate rank of a real matrix to a specified relative precision using
    random matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function

    :return:
        Rank estimate.
    :rtype: int
    """
    # Call external C function to estimate rank using random matrix-vector multiplication
    k, ra, ier = _id.idd_findrank(eps, m, n, matvect)
    # Check if error occurred during computation
    if ier:
        raise _RETCODE_ERROR
    # Return estimated rank
    return k


#------------------------------------------------------------------------------
# iddp_rsvd.f
#------------------------------------------------------------------------------

def iddp_rsvd(eps, m, n, matvect, matvec):
    """
    Compute SVD of a real matrix to a specified relative precision using random
    matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function

    :return:
        Left singular vectors.
    """
    # This function is intended to compute SVD using randomized matrix techniques,
    # but detailed implementation specifics are not provided in the current context.
    pass
    # 调用 _id 模块的 iddp_rsvd 函数进行随机化奇异值分解（RSVD），返回多个结果
    k, iU, iV, iS, w, ier = _id.iddp_rsvd(eps, m, n, matvect, matvec)
    # 如果返回值 ier 不为 0，表示出现错误，抛出 _RETCODE_ERROR 异常
    if ier:
        raise _RETCODE_ERROR
    # 根据返回的结果 w，从中提取右奇异向量 U，按列主序列出，reshape 成 (m, k) 形状的 numpy.ndarray
    U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
    # 根据返回的结果 w，从中提取右奇异向量 V，按列主序列出，reshape 成 (n, k) 形状的 numpy.ndarray
    V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
    # 根据返回的结果 w，从中提取奇异值 S，作为形状为 (k,) 的 numpy.ndarray 返回
    S = w[iS-1:iS+k-1]
    # 返回计算得到的 U（右奇异向量）、V（右奇异向量）、S（奇异值）作为元组
    return U, V, S
#------------------------------------------------------------------------------
# iddr_aid.f
#------------------------------------------------------------------------------

def iddr_aid(A, k):
    """
    Compute ID of a real matrix to a specified rank using random sampling.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入的矩阵 A 转换为 Fortran 风格的数组
    A = np.asfortranarray(A)
    # 获取矩阵 A 的维度信息
    m, n = A.shape
    # 调用 iddr_aidi 函数生成权重数组 w
    w = iddr_aidi(m, n, k)
    # 调用 _id 模块中的 iddr_aid 函数，计算 ID，并返回列索引数组和插值系数
    idx, proj = _id.iddr_aid(A, k, w)
    # 如果 k 等于 n，创建一个空的插值系数数组
    if k == n:
        proj = np.empty((k, n-k), dtype='float64', order='F')
    else:
        proj = proj.reshape((k, n-k), order='F')
    # 返回列索引数组和插值系数数组
    return idx, proj


def iddr_aidi(m, n, k):
    """
    Initialize array for :func:`iddr_aid`.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Initialization array to be used by :func:`iddr_aid`.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用 _id 模块中的 iddr_aidi 函数，初始化用于 iddr_aid 的数组
    return _id.iddr_aidi(m, n, k)


#------------------------------------------------------------------------------
# iddr_asvd.f
#------------------------------------------------------------------------------

def iddr_asvd(A, k):
    """
    Compute SVD of a real matrix to a specified rank using random sampling.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入的矩阵 A 转换为 Fortran 风格的数组
    A = np.asfortranarray(A)
    # 获取矩阵 A 的维度信息
    m, n = A.shape
    # 创建用于 SVD 计算的工作数组 w，设置为 Fortran 风格存储
    w = np.empty((2*k + 28)*m + (6*k + 21)*n + 25*k**2 + 100, order='F')
    # 调用 iddr_aidi 函数生成权重数组 w_
    w_ = iddr_aidi(m, n, k)
    # 将 w_ 复制到 w 中
    w[:w_.size] = w_
    # 调用 _id 模块中的 iddr_asvd 函数，计算 SVD，并返回左奇异向量、右奇异向量和奇异值
    U, V, S, ier = _id.iddr_asvd(A, k, w)
    # 如果返回错误码 ier 不为 0，则抛出异常 _RETCODE_ERROR
    if ier != 0:
        raise _RETCODE_ERROR
    # 返回左奇异向量、右奇异向量和奇异值
    return U, V, S


#------------------------------------------------------------------------------
# iddr_rid.f
#------------------------------------------------------------------------------

def iddr_rid(m, n, matvect, k):
    """
    Compute ID of a real matrix to a specified rank using random matrix-vector
    multiplication.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    """
    # 此函数实现通过随机矩阵向量乘法计算矩阵到指定秩的 ID
    # 返回列索引数组
    return _id.iddr_rid(m, n, matvect, k)
    # 调用 _id 模块中的 iddr_rid 函数，获取插值系数的索引和投影矩阵
    idx, proj = _id.iddr_rid(m, n, matvect, k)
    
    # 将 proj 数组切片至前 k*(n-k) 个元素，并按列优先（Fortran）顺序重新整形为 k 行 n-k 列的二维数组
    proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
    
    # 返回插值系数的索引和投影矩阵
    return idx, proj
#------------------------------------------------------------------------------
# iddr_rsvd.f
#------------------------------------------------------------------------------

def iddr_rsvd(m, n, matvect, matvec, k):
    """
    Compute SVD of a real matrix to a specified rank using random matrix-vector
    multiplication.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用底层 C 函数计算 SVD，返回左奇异向量 U，右奇异向量 V，奇异值 S 和错误标志 ier
    U, V, S, ier = _id.iddr_rsvd(m, n, matvect, matvec, k)
    # 如果错误标志不为零，则抛出错误代码
    if ier != 0:
        raise _RETCODE_ERROR
    # 返回计算得到的左奇异向量 U，右奇异向量 V，奇异值 S
    return U, V, S


#------------------------------------------------------------------------------
# idz_frm.f
#------------------------------------------------------------------------------

def idz_frm(n, w, x):
    """
    Transform complex vector via a composition of Rokhlin's random transform,
    random subselection, and an FFT.

    In contrast to :func:`idz_sfrm`, this routine works best when the length of
    the transformed vector is the power-of-two integer output by
    :func:`idz_frmi`, or when the length is not specified but instead
    determined a posteriori from the output. The returned transformed vector is
    randomly permuted.

    :param n:
        Greatest power-of-two integer satisfying ``n <= x.size`` as obtained from
        :func:`idz_frmi`; `n` is also the length of the output vector.
    :type n: int
    :param w:
        Initialization array constructed by :func:`idz_frmi`.
    :type w: :class:`numpy.ndarray`
    :param x:
        Vector to be transformed.
    :type x: :class:`numpy.ndarray`

    :return:
        Transformed vector.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用底层 C 函数执行复杂向量的转换，返回转换后的结果向量
    return _id.idz_frm(n, w, x)


def idz_sfrm(l, n, w, x):
    """
    Transform complex vector via a composition of Rokhlin's random transform,
    random subselection, and an FFT.

    In contrast to :func:`idz_frm`, this routine works best when the length of
    the transformed vector is known a priori.

    :param l:
        Length of transformed vector, satisfying ``l <= n``.
    :type l: int
    :param n:
        Greatest power-of-two integer satisfying ``n <= x.size`` as obtained from
        :func:`idz_sfrmi`.
    :type n: int
    """
    # 直接调用底层 C 函数执行复杂向量的转换，并返回转换后的结果向量
    return _id.idz_sfrm(l, n, w, x)
    # 使用 _id 模块中的 idz_sfrm 函数进行向量变换
    return _id.idz_sfrm(l, n, w, x)
def idz_frmi(m):
    """
    Initialize data for :func:`idz_frm`.

    :param m:
        Length of vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying ``n <= m``.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idz_frm`.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用 _id 模块的 idz_frmi 函数，返回初始化数据
    return _id.idz_frmi(m)


def idz_sfrmi(l, m):
    """
    Initialize data for :func:`idz_sfrm`.

    :param l:
        Length of output transformed vector.
    :type l: int
    :param m:
        Length of the vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying ``n <= m``.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idz_sfrm`.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用 _id 模块的 idz_sfrmi 函数，返回初始化数据
    return _id.idz_sfrmi(l, m)


#------------------------------------------------------------------------------
# idz_id.f
#------------------------------------------------------------------------------

def idzp_id(eps, A):
    """
    Compute ID of a complex matrix to a specified relative precision.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    # 将 A 转换为 Fortran 数组，然后调用 _id 模块的 idzp_id 函数计算 ID
    A = _asfortranarray_copy(A)
    k, idx, rnorms = _id.idzp_id(eps, A)
    n = A.shape[1]
    # 从 A 的转置中提取部分投影矩阵
    proj = A.T.ravel()[:k*(n-k)].reshape((k, n-k), order='F')
    return k, idx, proj


def idzr_id(A, k):
    """
    Compute ID of a complex matrix to a specified rank.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    # 将 A 转换为 Fortran 数组，然后调用 _id 模块的 idzr_id 函数计算指定秩的 ID
    A = _asfortranarray_copy(A)
    idx, rnorms = _id.idzr_id(A, k)
    n = A.shape[1]
    # 从 A 的转置中提取部分投影矩阵
    proj = A.T.ravel()[:k*(n-k)].reshape((k, n-k), order='F')
    return idx, proj


def idz_reconid(B, idx, proj):
    """
    Reconstruct matrix from complex ID.

    :param B:
        Skeleton matrix.
    :type B: :class:`numpy.ndarray`
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

    :return:
        Reconstructed matrix.
    :rtype: :class:`numpy.ndarray`
    """
    # 将 B 转换为 Fortran 数组，然后根据 idx 和 proj 重建复杂矩阵
    B = np.asfortranarray(B)
    if proj.size > 0:
        return _id.idz_reconid(B, idx, proj)
    else:
        # 如果 proj 数组为空，则按 idx 排序 B 的列
        return B[:, np.argsort(idx)]


def idz_reconint(idx, proj):
    """
    Reconstruct interpolation matrix from complex ID.

    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`
    """
    # 如果需要，可以在这里添加具体的重建插值矩阵的说明，但是该函数体尚未提供实际代码
    pass
    # 指定参数 proj 的类型为 numpy.ndarray
    :type proj: :class:`numpy.ndarray`

    # 返回值为插值矩阵
    :return:
        Interpolation matrix.
    # 返回值的类型为 numpy.ndarray
    :rtype: :class:`numpy.ndarray`
    """
    # 调用 _id.idz_reconint 函数并返回结果
    return _id.idz_reconint(idx, proj)
def idz_copycols(A, k, idx):
    """
    Reconstruct skeleton matrix from complex ID.

    :param A:
        Original matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`

    :return:
        Skeleton matrix.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入的原始矩阵 A 转换为 Fortran 风格（列优先）存储的数组
    A = np.asfortranarray(A)
    # 调用底层模块中的函数 _id.idz_copycols 进行复制列操作
    return _id.idz_copycols(A, k, idx)


#------------------------------------------------------------------------------
# idz_id2svd.f
#------------------------------------------------------------------------------

def idz_id2svd(B, idx, proj):
    """
    Convert complex ID to SVD.

    :param B:
        Skeleton matrix.
    :type B: :class:`numpy.ndarray`
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入的骨架矩阵 B 转换为 Fortran 风格（列优先）存储的数组
    B = np.asfortranarray(B)
    # 调用底层模块中的函数 _id.idz_id2svd 进行 ID 到 SVD 的转换
    U, V, S, ier = _id.idz_id2svd(B, idx, proj)
    # 如果返回的错误标志 ier 非零，则抛出异常 _RETCODE_ERROR
    if ier:
        raise _RETCODE_ERROR
    # 返回左奇异向量 U、右奇异向量 V 和奇异值 S
    return U, V, S


#------------------------------------------------------------------------------
# idz_snorm.f
#------------------------------------------------------------------------------

def idz_snorm(m, n, matveca, matvec, its=20):
    """
    Estimate spectral norm of a complex matrix by the randomized power method.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param its:
        Number of power method iterations.
    :type its: int

    :return:
        Spectral norm estimate.
    :rtype: float
    """
    # 调用底层模块中的函数 _id.idz_snorm 进行矩阵的谱范数估计
    snorm, v = _id.idz_snorm(m, n, matveca, matvec, its)
    # 返回估计的谱范数
    return snorm


def idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its=20):
    """
    Estimate spectral norm of the difference of two complex matrices by the
    randomized power method.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the adjoint of the first matrix to a vector, with
        call signature `y = matveca(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matveca: function
    """
    # 调用底层模块中的函数 _id.idz_diffsnorm 进行两个矩阵差的谱范数估计
    # matveca 和 matveca2 分别对应两个矩阵的转置与向量的乘积
    :param matveca2:
        第二个矩阵的伴随作用于向量的函数，调用签名为 `y = matveca2(x)`，其中 `x` 和 `y` 分别是输入和输出向量。
    :type matveca2: function
    :param matvec:
        应用第一个矩阵于向量的函数，调用签名为 `y = matvec(x)`，其中 `x` 和 `y` 分别是输入和输出向量。
    :type matvec: function
    :param matvec2:
        应用第二个矩阵于向量的函数，调用签名为 `y = matvec2(x)`，其中 `x` 和 `y` 分别是输入和输出向量。
    :type matvec2: function
    :param its:
        幂法迭代的次数。
    :type its: int

    :return:
        矩阵差的谱范数估计。
    :rtype: float
    """
    返回 _id.idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its)
#------------------------------------------------------------------------------
# idz_svd.f
#------------------------------------------------------------------------------

# 计算复矩阵的SVD到指定秩
def idzr_svd(A, k):
    """
    Compute SVD of a complex matrix to a specified rank.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入矩阵A转换为Fortran风格数组
    A = np.asfortranarray(A)
    # 调用底层函数_id.idzr_svd计算SVD，返回左奇异向量U、右奇异向量V、奇异值S和错误标志ier
    U, V, S, ier = _id.idzr_svd(A, k)
    # 如果计算过程中出现错误标志ier，抛出_RETCODE_ERROR异常
    if ier:
        raise _RETCODE_ERROR
    # 返回计算得到的左奇异向量U、右奇异向量V、奇异值S
    return U, V, S


# 计算复矩阵的SVD到指定相对精度
def idzp_svd(eps, A):
    """
    Compute SVD of a complex matrix to a specified relative precision.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入矩阵A转换为Fortran风格数组
    A = np.asfortranarray(A)
    # 获取矩阵A的行数m和列数n
    m, n = A.shape
    # 调用底层函数_id.idzp_svd计算SVD，返回相关结果
    k, iU, iV, iS, w, ier = _id.idzp_svd(eps, A)
    # 如果计算过程中出现错误标志ier，抛出_RETCODE_ERROR异常
    if ier:
        raise _RETCODE_ERROR
    # 根据返回的索引和数据，构造左奇异向量U、右奇异向量V、奇异值S
    U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
    V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
    S = w[iS-1:iS+k-1]
    # 返回计算得到的左奇异向量U、右奇异向量V、奇异值S
    return U, V, S


#------------------------------------------------------------------------------
# idzp_aid.f
#------------------------------------------------------------------------------

# 使用随机采样计算复矩阵到指定相对精度的ID
def idzp_aid(eps, A):
    """
    Compute ID of a complex matrix to a specified relative precision using
    random sampling.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入矩阵A转换为Fortran风格数组
    A = np.asfortranarray(A)
    # 获取矩阵A的行数m和列数n
    m, n = A.shape
    # 调用底层函数_id.idzp_aid计算ID，返回相关结果
    n2, w = idz_frmi(m)
    proj = np.empty(n*(2*n2 + 1) + n2 + 1, dtype='complex128', order='F')
    k, idx, proj = _id.idzp_aid(eps, A, w, proj)
    # 根据返回的数据，构造ID的秩、列索引数组idx和插值系数proj
    proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
    # 返回计算得到的ID的秩、列索引数组idx和插值系数proj
    return k, idx, proj


# 使用随机采样估计复矩阵的秩到指定相对精度
def idz_estrank(eps, A):
    """
    Estimate rank of a complex matrix to a specified relative precision using
    random sampling.

    The output rank is typically about 8 higher than the actual rank.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank estimate.
    :rtype: int
    """
    # 将输入矩阵A转换为Fortran风格数组
    A = np.asfortranarray(A)
    # 获取矩阵A的行数m和列数n
    m, n = A.shape
    # 调用idz_frmi函数获取相关数据
    n2, w = idz_frmi(m)
    ra = np.empty(n*n2 + (n + 1)*(n2 + 1), dtype='complex128', order='F')
    # 调用 _id 模块中的 idz_estrank 函数，传入参数 eps, A, w, ra，并获取返回的 k 和更新后的 ra
    k, ra = _id.idz_estrank(eps, A, w, ra)
    # 返回 k 的值作为函数的结果
    return k
#------------------------------------------------------------------------------
# idzp_asvd.f
#------------------------------------------------------------------------------

def idzp_asvd(eps, A):
    """
    Compute SVD of a complex matrix to a specified relative precision using
    random sampling.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入矩阵 A 转换为 Fortran 风格的数组
    A = np.asfortranarray(A)
    # 获取矩阵 A 的行数 m 和列数 n
    m, n = A.shape
    # 调用 _id.idz_frmi 函数获取返回的两个值：n2 和 winit
    n2, winit = _id.idz_frmi(m)
    # 计算所需的数组 w 的长度，并初始化为复数类型的空数组，使用 Fortran 顺序存储
    w = np.empty(
        max((min(m, n) + 1)*(3*m + 5*n + 11) + 8*min(m, n)**2,
            (2*n + 1)*(n2 + 1)),
        dtype=np.complex128, order='F')
    # 调用 _id.idzp_asvd 函数执行奇异值分解，返回的结果包括：k, iU, iV, iS, w, ier
    k, iU, iV, iS, w, ier = _id.idzp_asvd(eps, A, winit, w)
    # 如果 ier 非零，抛出 _RETCODE_ERROR 异常
    if ier:
        raise _RETCODE_ERROR
    # 根据返回的索引和计算出的 k 值从 w 中提取左奇异向量 U、右奇异向量 V 和奇异值 S
    U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
    V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
    S = w[iS-1:iS+k-1]
    # 返回计算得到的奇异值分解结果
    return U, V, S


#------------------------------------------------------------------------------
# idzp_rid.f
#------------------------------------------------------------------------------

def idzp_rid(eps, m, n, matveca):
    """
    Compute ID of a complex matrix to a specified relative precision using
    random matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    # 创建一个复数类型的空数组 proj，用于存储计算过程中的中间结果
    proj = np.empty(
        m + 1 + 2*n*(min(m, n) + 1),
        dtype=np.complex128, order='F')
    # 调用 _id.idzp_rid 函数执行交互奇异值分解，返回的结果包括：k, idx, proj, ier
    k, idx, proj, ier = _id.idzp_rid(eps, m, n, matveca, proj)
    # 如果 ier 非零，抛出 _RETCODE_ERROR 异常
    if ier:
        raise _RETCODE_ERROR
    # 从 proj 数组中提取出用于计算 ID 的数据，并返回
    proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
    return k, idx, proj


def idz_findrank(eps, m, n, matveca):
    """
    Estimate rank of a complex matrix to a specified relative precision using
    random matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function
    :return:
        返回一个秩的估计值。
    :rtype: int
    """
    # 调用 _id.idz_findrank 函数来计算矩阵的秩
    k, ra, ier = _id.idz_findrank(eps, m, n, matveca)
    # 如果计算秩时返回了错误标志 ier，则触发异常 _RETCODE_ERROR
    if ier:
        raise _RETCODE_ERROR
    # 返回计算得到的秩 k
    return k
#------------------------------------------------------------------------------
# idzp_rsvd.f
#------------------------------------------------------------------------------

def idzp_rsvd(eps, m, n, matveca, matvec):
    """
    Compute SVD of a complex matrix to a specified relative precision using
    random matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用底层实现的函数 _id.idzp_rsvd 计算 SVD
    k, iU, iV, iS, w, ier = _id.idzp_rsvd(eps, m, n, matveca, matvec)
    # 如果计算出错，抛出异常 _RETCODE_ERROR
    if ier:
        raise _RETCODE_ERROR
    # 根据返回的数据 w，从中提取左奇异向量 U、右奇异向量 V 和奇异值 S
    U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
    V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
    S = w[iS-1:iS+k-1]
    return U, V, S


#------------------------------------------------------------------------------
# idzr_aid.f
#------------------------------------------------------------------------------

def idzr_aid(A, k):
    """
    Compute ID of a complex matrix to a specified rank using random sampling.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入矩阵 A 转换为 Fortran 风格的数组
    A = np.asfortranarray(A)
    # 获取矩阵的行数 m 和列数 n
    m, n = A.shape
    # 调用 idzr_aidi 函数生成随机采样的权重 w
    w = idzr_aidi(m, n, k)
    # 调用底层实现的 _id.idzr_aid 计算 ID，返回列索引数组 idx 和投影矩阵 proj
    idx, proj = _id.idzr_aid(A, k, w)
    # 如果 k 等于 n，则初始化一个空的投影矩阵 proj
    if k == n:
        proj = np.empty((k, n-k), dtype='complex128', order='F')
    else:
        proj = proj.reshape((k, n-k), order='F')
    return idx, proj


def idzr_aidi(m, n, k):
    """
    Initialize array for :func:`idzr_aid`.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Initialization array to be used by :func:`idzr_aid`.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用底层实现的 _id.idzr_aidi 函数生成初始化数组
    return _id.idzr_aidi(m, n, k)


#------------------------------------------------------------------------------
# idzr_asvd.f
#------------------------------------------------------------------------------

def idzr_asvd(A, k):
    """
    Compute Approximate SVD of a complex matrix to a specified rank using
    randomized sampling.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of the approximate SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    Compute SVD of a complex matrix to a specified rank using random sampling.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    # 将输入矩阵 A 转换为 Fortran 风格的数组，以便与 Fortran 源代码兼容
    A = np.asfortranarray(A)
    # 获取矩阵 A 的维度
    m, n = A.shape
    # 计算所需的工作空间大小并创建复数数组 w 用于存储工作区数据
    w = np.empty(
        (2*k + 22)*m + (6*k + 21)*n + 8*k**2 + 10*k + 90,
        dtype='complex128', order='F')
    # 调用 idzr_aidi 函数，返回用于工作空间的数据并将其复制到 w 数组中
    w_ = idzr_aidi(m, n, k)
    w[:w_.size] = w_
    # 调用底层函数 _id.idzr_asvd 进行 SVD 计算，返回左奇异向量 U、右奇异向量 V、奇异值 S 和错误码 ier
    U, V, S, ier = _id.idzr_asvd(A, k, w)
    # 如果计算返回错误码 ier 非零，则引发 _RETCODE_ERROR 异常
    if ier:
        raise _RETCODE_ERROR
    # 返回计算得到的左奇异向量 U、右奇异向量 V 和奇异值 S
    return U, V, S
#------------------------------------------------------------------------------
# idzr_rid.f
#------------------------------------------------------------------------------

# 计算复杂矩阵到指定秩的随机矩阵-向量乘法的ID
def idzr_rid(m, n, matveca, k):
    """
    Compute ID of a complex matrix to a specified rank using random
    matrix-vector multiplication.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用底层函数计算 ID
    idx, proj = _id.idzr_rid(m, n, matveca, k)
    # 重新整理投影矩阵，以便符合指定的形状
    proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
    return idx, proj


#------------------------------------------------------------------------------
# idzr_rsvd.f
#------------------------------------------------------------------------------

# 计算复杂矩阵到指定秩的随机矩阵-向量乘法的SVD
def idzr_rsvd(m, n, matveca, matvec, k):
    """
    Compute SVD of a complex matrix to a specified rank using random
    matrix-vector multiplication.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    # 调用底层函数计算 SVD
    U, V, S, ier = _id.idzr_rsvd(m, n, matveca, matvec, k)
    # 如果返回错误码，抛出异常
    if ier:
        raise _RETCODE_ERROR
    return U, V, S
```
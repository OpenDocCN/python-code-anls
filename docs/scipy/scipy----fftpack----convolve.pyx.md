# `D:\src\scipysrc\scipy\scipy\fftpack\convolve.pyx`

```
# 从 scipy.fft._pocketfft.pypocketfft 模块中导入 r2r_fftpack 函数
from scipy.fft._pocketfft.pypocketfft import r2r_fftpack
# 导入 numpy 库并使用 np 别名
import numpy as np
# 使用 cimport 导入 numpy 库，主要用于 Cython 的声明
cimport numpy as np
# 使用 cimport 导入 Cython 库
cimport cython

# 调用 np.import_array() 函数，用于导入数组对象接口
np.import_array()

# 定义模块的公开接口列表
__all__ = ['destroy_convolve_cache', 'convolve', 'convolve_z',
           'init_convolution_kernel']

# 定义函数 destroy_convolve_cache，用于销毁卷积缓存，但实际上什么也不做，仅为兼容性需求
def destroy_convolve_cache():
    pass  # We don't cache anything, needed for compatibility

# 使用 Cython 的装饰器，关闭边界检查和数组溢出检查
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义函数 convolve，用于一维数组的卷积操作
def convolve(inout, omega, swap_real_imag=False, overwrite_x=False):
    """y = convolve(x,omega,[swap_real_imag,overwrite_x])

    Wrapper for ``convolve``.

    Parameters
    ----------
    x : input rank-1 array('d') with bounds (n)
        输入的一维数组 x，数据类型为双精度浮点数，长度为 n
    omega : input rank-1 array('d') with bounds (n)
        输入的一维数组 omega，数据类型为双精度浮点数，长度为 n

    Other Parameters
    ----------------
    overwrite_x : input int, optional
        Default: 0
        是否覆盖输入数组 x，默认为 False，即不覆盖

    swap_real_imag : input int, optional
         Default: 0
         是否交换实部和虚部，默认为 False

    Returns
    -------
    y : rank-1 array('d') with bounds (n) and x storage
        返回的一维数组 y，数据类型为双精度浮点数，长度为 n，与输入数组 x 使用相同的存储空间
    """
    cdef:
        np.ndarray[np.float64_t, ndim=1] X_arr, w_arr
        double [:] w, X
        double c
        size_t n, i

    # 将输入数组 inout 转换为 numpy 数组 X_arr，并根据 overwrite_x 参数选择是否复制数据
    X = X_arr = np.array(inout, np.float64, copy=not overwrite_x)
    # 将输入数组 omega 转换为 numpy 数组 w_arr
    w = w_arr = np.asarray(omega, np.float64)
    # 获取数组的长度 n
    n = X_arr.shape[0]
    # 检查 inout 和 omega 是否为一维数组且长度相同，否则抛出 ValueError 异常
    if X_arr.ndim != 1 or w.ndim != 1 or w.shape[0] != n:
        raise ValueError(
            "inout and omega must be 1-dimensional arrays of the same length")

    # 使用 r2r_fftpack 函数进行快速傅里叶变换（实数到实数），将结果存储在 X_arr 中
    r2r_fftpack(X_arr, None, True, True, out=X_arr)

    # 如果 swap_real_imag 为 True，则交换实部和虚部
    if swap_real_imag:
        # 将 X_arr 和 w_arr 赋值给 X 和 w
        X, w = X_arr, w_arr

        # 计算首个元素的乘积
        X[0] *= w[0];

        # 遍历处理奇数索引处的元素对
        for i in range(1, n - 1, 2):
            c = X[i] * w[i]
            X[i] = X[i + 1] * w[i + 1]
            X[i + 1] = c

        # 处理数组长度为偶数的情况
        if (n % 2) == 0:
            X[n - 1] *= w[n - 1]
    else:
        # 否则直接进行元素级乘法
        X_arr *= w_arr

    # 再次使用 r2r_fftpack 函数进行反向变换，将结果存储在 X_arr 中
    r2r_fftpack(X_arr, None, False, False, out=X_arr)
    # 返回计算结果数组 X_arr
    return X_arr

# 使用 Cython 的装饰器，关闭边界检查和数组溢出检查
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义函数 convolve_z，用于复数数组的卷积操作
def convolve_z(inout, omega_real, omega_imag, overwrite_x=False):
    """y = convolve_z(x,omega_real,omega_imag,[overwrite_x])

    Wrapper for ``convolve_z``.

    Parameters
    ----------
    x : input rank-1 array('d') with bounds (n)
        输入的一维数组 x，数据类型为双精度浮点数，长度为 n
    omega_real : input rank-1 array('d') with bounds (n)
        输入的一维数组 omega_real，数据类型为双精度浮点数，长度为 n
    omega_imag : input rank-1 array('d') with bounds (n)
        输入的一维数组 omega_imag，数据类型为双精度浮点数，长度为 n

    Other Parameters
    ----------------
    overwrite_x : input int, optional
        Default: 0
        是否覆盖输入数组 x，默认为 False，即不覆盖

    Returns
    -------
    y : rank-1 array('d') with bounds (n) and x storage
        返回的一维数组 y，数据类型为双精度浮点数，长度为 n，与输入数组 x 使用相同的存储空间
    """
    cdef:
        np.ndarray[np.float64_t, ndim=1] X_arr
        double [:] wr, wi, X
        size_t n, i
        double c

    # 将输入数组 inout 转换为 numpy 数组 X_arr，并根据 overwrite_x 参数选择是否复制数据
    X = X_arr = np.array(inout, np.float64, copy=not overwrite_x)
    # 将输入数组 omega_real 和 omega_imag 转换为 numpy 数组 wr 和 wi
    wr = np.asarray(omega_real, np.float64)
    wi = np.asarray(omega_imag, np.float64)

    # 获取数组的长度 n
    n = X_arr.shape[0]
    # 检查 inout、omega_real 和 omega_imag 是否为一维数组且长度相同，否则抛出 ValueError 异常
    if (X_arr.ndim != 1 or wr.ndim != 1 or wr.shape[0] != n
        or wi.ndim != 1 or wi.shape[0] != n):
        raise ValueError(
            "inout and omega must be 1-dimensional arrays of the same length")

    # 使用 r2r_fftpack 函数进行快速傅里叶变换（实数到实数），将结果存储在 X_arr 中
    r2r_fftpack(X_arr, None, True, True, out=X_arr)
    # 将 X[0] 乘以 wr[0] + wi[0]
    X[0] *= wr[0] + wi[0]
    # 如果 n 是偶数，则将 X[n-1] 乘以 wr[n-1] + wi[n-1]
    if (n % 2) == 0:
        X[n - 1] *= wr[n - 1] + wi[n - 1]

    # 对于 i 从 1 到 n-2，步长为 2
    for i in range(1, n - 1, 2):
        # 计算交换的复数乘积
        c = X[i + 1] * wr[i + 1] + X[i] * wi[i]
        # 更新 X[i] 为其乘以 wr[i] + wi[i+1]
        X[i] = X[i] * wr[i] + X[i + 1] * wi[i + 1]
        # 更新 X[i+1] 为之前计算的 c
        X[i + 1] = c

    # 使用 r2r_fftpack 函数进行实数到实数的 FFT 变换，将结果存储在 X_arr 中
    r2r_fftpack(X_arr, None, False, False, out=X_arr)
    # 返回变换后的 X_arr
    return X_arr
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个 Cython 函数，用于初始化卷积核数组 omega
def init_convolution_kernel(size_t n, object kernel_func,
                            ssize_t d=0, zero_nyquist=None,
                            tuple kernel_func_extra_args=()):
    """omega = init_convolution_kernel(n,kernel_func,[d,zero_nyquist,kernel_func_extra_args])

    Wrapper for ``init_convolution_kernel``.

    Parameters
    ----------
    n : input int
        数组长度
    kernel_func : call-back function
        回调函数，返回浮点数

    Other Parameters
    ----------------
    d : input int, optional
        默认值: 0
        控制系数，可以影响输出值
    kernel_func_extra_args : input tuple, optional
        默认值: ()
        回调函数的额外参数
    zero_nyquist : input int, optional
        默认值: d%2
        控制 Nyquist 频率是否为零点

    Returns
    -------
    omega : rank-1 array('d') with bounds (n)
        包含卷积核结果的数组

    Notes
    -----
    Call-back functions::

      def kernel_func(k): return kernel_func
      Required arguments:
        k : input int
      Return objects:
        kernel_func : float
    """
    cdef:
        np.ndarray[np.float64_t, ndim=1] omega_arr  # 声明一个双精度浮点型的一维数组 omega_arr
        double [::1] omega  # 定义一个双精度浮点型的一维数组 omega
        size_t j, k, l  # 声明无符号整数类型的变量 j, k, l
        double scale_real, scale_imag, x  # 声明双精度浮点型变量 scale_real, scale_imag, x

    if zero_nyquist is None:
        zero_nyquist = (d % 2 != 0)  # 如果未指定 zero_nyquist，则根据 d 的奇偶性确定其值

    omega = omega_arr = np.empty(n, np.float64)  # 分配大小为 n 的空数组给 omega_arr，并赋值给 omega
    l = n if n % 2 != 0 else n - 1  # 如果 n 是奇数，则 l = n，否则 l = n - 1

    # 初始化 omega[0]，即卷积核的第一个值
    x = kernel_func(0, *kernel_func_extra_args)
    omega[0] = x / n

    d %= 4  # 将 d 取模 4
    scale_real = 1./n if d == 0 or d == 1 else -1./n  # 根据 d 的值选择 scale_real
    scale_imag = 1./n if d == 0 or d == 3 else -1./n  # 根据 d 的值选择 scale_imag

    k = 1
    # 填充 omega 数组，每次迭代填充两个位置
    for j in range(1, l, 2):
        x = kernel_func(k, *kernel_func_extra_args)
        omega[j] = scale_real * x
        omega[j + 1] = scale_imag * x
        k += 1

    # 处理 n 为偶数时的情况
    if (n % 2) == 0:
        if zero_nyquist:
            omega[n - 1] = 0.
        else:
            x = kernel_func(k, *kernel_func_extra_args)
            omega[n - 1] = scale_real * x

    return omega_arr  # 返回填充好的 omega_arr 数组作为输出
```
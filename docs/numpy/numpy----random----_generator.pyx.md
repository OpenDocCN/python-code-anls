# `D:\src\scipysrc\numpy\numpy\random\_generator.pyx`

```py
# 导入必要的模块和函数
#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, binding=True
import operator  # 导入operator模块，用于操作符函数
import warnings  # 导入warnings模块，用于警告处理
from collections.abc import Sequence  # 从collections.abc模块导入Sequence抽象基类

# 导入Cython模块和函数
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from cpython cimport (Py_INCREF, PyFloat_AsDouble)
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cimport cython  # 导入Cython编译器指令
import numpy as np  # 导入NumPy库
cimport numpy as np  # 在Cython中导入NumPy库，用于高效访问NumPy数组的C语言API
from numpy.lib.array_utils import normalize_axis_index  # 从NumPy库导入数组工具模块中的函数

# 导入自定义Cython扩展模块
from .c_distributions cimport *
from libc cimport string  # 从C标准库libc中导入string模块，用于字符串处理
from libc.math cimport sqrt  # 从C标准库libc中导入math模块中的sqrt函数，用于计算平方根
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int32_t, int64_t, INT64_MAX, SIZE_MAX)  # 从C标准库libc中导入整数类型和常量

# 导入自定义Cython扩展模块中的函数和类
from ._bounded_integers cimport (_rand_bool, _rand_int32, _rand_int64,
         _rand_int16, _rand_int8, _rand_uint64, _rand_uint32, _rand_uint16,
         _rand_uint8, _gen_mask)  # 导入各种整数生成函数和掩码生成函数
from ._pcg64 import PCG64  # 导入PCG64伪随机数生成器类
from numpy.random cimport bitgen_t  # 在Cython中导入NumPy库中定义的位生成器类型
from ._common cimport (POISSON_LAM_MAX, CONS_POSITIVE, CONS_NONE,
            CONS_NON_NEGATIVE, CONS_BOUNDED_0_1, CONS_BOUNDED_GT_0_1,
            CONS_BOUNDED_LT_0_1, CONS_GT_1, CONS_POSITIVE_NOT_NAN, CONS_POISSON,
            double_fill, cont, kahan_sum, cont_broadcast_3, float_fill, cont_f,
            check_array_constraint, check_constraint, disc, discrete_broadcast_iii,
            validate_output_shape
        )  # 导入共享的常量和函数，用于约束检查和数值处理

# 定义与NumPy数组对象相关的C语言API接口
cdef extern from "numpy/arrayobject.h":
    int PyArray_ResolveWritebackIfCopy(np.ndarray)  # 解析数组对象的写回行为（若存在）
    int PyArray_FailUnlessWriteable(np.PyArrayObject *obj,
                                    const char *name) except -1  # 检查数组对象是否可写
    object PyArray_FromArray(np.PyArrayObject *, np.PyArray_Descr *, int)  # 根据现有数组对象创建新的NumPy数组对象

    enum:
        NPY_ARRAY_WRITEBACKIFCOPY  # NumPy数组写回标志

np.import_array()  # 导入NumPy的数组接口

# 定义一个Cython函数，安全地对非负整数数组求和
cdef int64_t _safe_sum_nonneg_int64(size_t num_colors, int64_t *colors):
    """
    Sum the values in the array `colors`.

    Return -1 if an overflow occurs.
    The values in *colors are assumed to be nonnegative.
    """
    cdef size_t i
    cdef int64_t sum

    sum = 0  # 初始化总和为0
    for i in range(num_colors):
        if colors[i] > INT64_MAX - sum:
            return -1  # 若加法溢出则返回-1
        sum += colors[i]  # 累加数组中的值
    return sum  # 返回总和结果

# 定义一个内联函数，用于封装对原始数据的重新排序操作
cdef inline void _shuffle_raw_wrap(bitgen_t *bitgen, np.npy_intp n,
                                   np.npy_intp first, np.npy_intp itemsize,
                                   np.npy_intp stride,
                                   char* data, char* buf) noexcept nogil:
    # 通过调用_cythonized的_shuffle_raw函数实现具体的原始数据重排操作
    # 这里通过优化特定情况（itemsize等于sizeof(np.npy_intp)）提高性能约33%
    if itemsize == sizeof(np.npy_intp):
        _shuffle_raw(bitgen, n, first, sizeof(np.npy_intp), stride, data, buf)
    else:
        _shuffle_raw(bitgen, n, first, itemsize, stride, data, buf)
cdef inline void _shuffle_raw(bitgen_t *bitgen, np.npy_intp n,
                              np.npy_intp first, np.npy_intp itemsize,
                              np.npy_intp stride,
                              char* data, char* buf) noexcept nogil:
    """
    Parameters
    ----------
    bitgen
        指向 bitgen_t 实例的指针。
    n
        data 中的元素数量。
    first
        首个要进行洗牌的观察结果。洗牌 n-1, n-2, ..., first，当 first=1 时整个数组被洗牌。
    itemsize
        每个项的字节大小。
    stride
        数组的步长。
    data
        数据的位置。
    buf
        缓冲区的位置 (itemsize)。
    """
    cdef np.npy_intp i, j

    for i in reversed(range(first, n)):
        j = random_interval(bitgen, i)
        string.memcpy(buf, data + j * stride, itemsize)
        string.memcpy(data + j * stride, data + i * stride, itemsize)
        string.memcpy(data + i * stride, buf, itemsize)


cdef inline void _shuffle_int(bitgen_t *bitgen, np.npy_intp n,
                              np.npy_intp first, int64_t* data) noexcept nogil:
    """
    Parameters
    ----------
    bitgen
        指向 bitgen_t 实例的指针。
    n
        data 中的元素数量。
    first
        首个要进行洗牌的观察结果。洗牌 n-1, n-2, ..., first，当 first=1 时整个数组被洗牌。
    data
        数据的位置。
    """
    cdef np.npy_intp i, j
    cdef int64_t temp
    for i in reversed(range(first, n)):
        j = random_bounded_uint64(bitgen, 0, i, 0, 0)
        temp = data[j]
        data[j] = data[i]
        data[i] = temp


cdef bint _check_bit_generator(object bitgen):
    """检查对象是否符合 BitGenerator 接口。"""
    if not hasattr(bitgen, "capsule"):
        return False
    cdef const char *name = "BitGenerator"
    return PyCapsule_IsValid(bitgen.capsule, name)


cdef class Generator:
    """
    Generator(bit_generator)

    `Generator` 是 BitGenerators 的容器。

    `Generator` 提供了从各种概率分布生成随机数的多种方法。除了特定于分布的参数外，每个方法还接受一个关键字参数 `size`，默认为 `None`。如果 `size` 是 `None`，则生成并返回单个值。如果 `size` 是整数，则返回填充有生成值的一维数组。如果 `size` 是元组，则返回填充有相应形状的数组。

    函数 :func:`numpy.random.default_rng` 将使用 numpy 默认的 `BitGenerator` 实例化一个 `Generator`。

    **不保证兼容性**

    `Generator` 不提供版本兼容性的保证。特别是随着更好的算法的发展，位流可能会发生变化。

    Parameters
    ----------
    bit_generator : BitGenerator
        作为核心生成器使用的 BitGenerator 实例。

    Notes
    -----
    """
    """
    The Python stdlib module `random` contains pseudo-random number generator
    with a number of methods that are similar to the ones available in
    `Generator`. It uses Mersenne Twister, and this bit generator can
    be accessed using `MT19937`. `Generator`, besides being
    NumPy-aware, has the advantage that it provides a much larger number
    of probability distributions to choose from.

    Examples
    --------
    >>> from numpy.random import Generator, PCG64
    >>> rng = Generator(PCG64())
    >>> rng.standard_normal()
    -0.203  # random

    See Also
    --------
    default_rng : Recommended constructor for `Generator`.
    """
    
    # 定义Cython公共属性_bit_generator，用于存储位生成器对象
    cdef public object _bit_generator
    
    # 定义Cython变量_bitgen，用于存储位生成器结构体
    cdef bitgen_t _bitgen
    
    # 定义Cython变量_binomial，用于存储二项式分布结构体
    cdef binomial_t _binomial
    
    # 定义Cython属性lock，用于存储锁对象
    cdef object lock
    
    # 初始化类，接受一个bit_generator参数作为输入
    _poisson_lam_max = POISSON_LAM_MAX
    
    def __init__(self, bit_generator):
        # 将输入的bit_generator存储到实例的_bit_generator属性中
        self._bit_generator = bit_generator
        
        # 获取bit_generator的capsule
        capsule = bit_generator.capsule
        
        # 定义Cython变量name为"BitGenerator"
        cdef const char *name = "BitGenerator"
        
        # 如果capsule无效，则抛出值错误异常
        if not PyCapsule_IsValid(capsule, name):
            raise ValueError("Invalid bit generator. The bit generator must "
                             "be instantiated.")
        
        # 将capsule中的指针解析为bitgen_t类型，并存储到_bitgen中
        self._bitgen = (<bitgen_t *> PyCapsule_GetPointer(capsule, name))[0]
        
        # 将bit_generator的锁对象存储到实例的lock属性中
        self.lock = bit_generator.lock

    def __repr__(self):
        # 返回实例的字符串表示形式，包含内存地址信息
        return self.__str__() + ' at 0x{:X}'.format(id(self))

    def __str__(self):
        # 返回实例的字符串表示形式，包含类名和位生成器的类名
        _str = self.__class__.__name__
        _str += '(' + self.bit_generator.__class__.__name__ + ')'
        return _str

    # Pickling support:
    def __getstate__(self):
        # 返回None，表示此类不能被序列化
        return None

    def __setstate__(self, bit_gen):
        # 如果bit_gen是字典类型，则为兼容旧版本路径
        if isinstance(bit_gen, dict):
            # 以旧版本方式设置bit_generator的状态
            # 在2.0.x之前，仅保留底层位生成器的状态，任何种子序列信息将丢失
            self.bit_generator.state = bit_gen

    def __reduce__(self):
        # 导入内部函数__generator_ctor
        from ._pickle import __generator_ctor
        
        # 返回元组(__generator_ctor, (self._bit_generator, ), None)，用于序列化实例
        # __generator_ctor要求参数为(bit_generator, )
        return __generator_ctor, (self._bit_generator, ), None

    @property
    def bit_generator(self):
        """
        Gets the bit generator instance used by the generator

        Returns
        -------
        bit_generator : BitGenerator
            The bit generator instance used by the generator
        """
        # 返回实例的_bit_generator属性，即位生成器对象
        return self._bit_generator
    def spawn(self, int n_children):
        """
        spawn(n_children)

        Create new independent child generators.

        See :ref:`seedsequence-spawn` for additional notes on spawning
        children.

        .. versionadded:: 1.25.0

        Parameters
        ----------
        n_children : int
            Number of child generators to create.

        Returns
        -------
        child_generators : list of Generators
            List containing newly spawned child generator objects.

        Raises
        ------
        TypeError
            Raised when the underlying SeedSequence does not support spawning.

        See Also
        --------
        random.BitGenerator.spawn, random.SeedSequence.spawn :
            Equivalent method on the bit generator and seed sequence.
        bit_generator :
            The bit generator instance used by the generator.

        Examples
        --------
        Starting from a seeded default generator:

        >>> # High quality entropy created with: f"0x{secrets.randbits(128):x}"
        >>> entropy = 0x3034c61a9ae04ff8cb62ab8ec2c4b501
        >>> rng = np.random.default_rng(entropy)

        Create two new generators for example for parallel execution:

        >>> child_rng1, child_rng2 = rng.spawn(2)

        Drawn numbers from each are independent but derived from the initial
        seeding entropy:

        >>> rng.uniform(), child_rng1.uniform(), child_rng2.uniform()
        (0.19029263503854454, 0.9475673279178444, 0.4702687338396767)

        It is safe to spawn additional children from the original ``rng`` or
        the children:

        >>> more_child_rngs = rng.spawn(20)
        >>> nested_spawn = child_rng1.spawn(20)

        """
        # 返回一个列表，其中包含从当前生成器的位生成器上生成的n_children个子生成器实例
        return [type(self)(g) for g in self._bit_generator.spawn(n_children)]
    def random(self, size=None, dtype=np.float64, out=None):
        """
        random(size=None, dtype=np.float64, out=None)

        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` use `uniform`
        or multiply the output of `random` by ``(b - a)`` and add ``a``::

            (b - a) * random() + a

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result, only `float64` and `float32` are supported.
            Byteorder must be native. The default value is np.float64.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is not None,
            it must have the same shape as the provided size and must match the type of
            the output values.

        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned).

        See Also
        --------
        uniform : Draw samples from the parameterized uniform distribution.

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.random()
        0.47108547995356098 # random
        >>> type(rng.random())
        <class 'float'>
        >>> rng.random((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428]) # random

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * rng.random((3, 2)) - 5
        array([[-3.99149989, -0.52338984], # random
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])

        """
        # 定义临时变量 temp
        cdef double temp
        # 将 dtype 转换为 numpy 的 dtype 对象
        _dtype = np.dtype(dtype)
        # 根据 dtype 类型选择不同的填充函数，并返回结果
        if _dtype == np.float64:
            # 使用双精度填充函数处理随机数生成
            return double_fill(&random_standard_uniform_fill, &self._bitgen, size, self.lock, out)
        elif _dtype == np.float32:
            # 使用单精度填充函数处理随机数生成
            return float_fill(&random_standard_uniform_fill_f, &self._bitgen, size, self.lock, out)
        else:
            # 抛出类型错误，如果不支持指定的 dtype
            raise TypeError('Unsupported dtype %r for random' % _dtype)
    # 定义一个方法用于生成标准指数分布的随机数，可以指定输出的数据类型和方法
    def standard_exponential(self, size=None, dtype=np.float64, method='zig', out=None):
        """
        standard_exponential(size=None, dtype=np.float64, method='zig', out=None)

        Draw samples from the standard exponential distribution.

        `standard_exponential` is identical to the exponential distribution
        with a scale parameter of 1.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result, only `float64` and `float32` are supported.
            Byteorder must be native. The default value is np.float64.
        method : str, optional
            Either 'inv' or 'zig'. 'inv' uses the default inverse CDF method.
            'zig' uses the much faster Ziggurat method of Marsaglia and Tsang.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is not None,
            it must have the same shape as the provided size and must match the type of
            the output values.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        Examples
        --------
        Output a 3x8000 array:

        >>> rng = np.random.default_rng()
        >>> n = rng.standard_exponential((3, 8000))

        """
        # 获取指定dtype的dtype对象
        _dtype = np.dtype(dtype)
        # 根据dtype选择不同的生成方法
        if _dtype == np.float64:
            if method == 'zig':
                # 使用double_fill函数生成标准指数分布的双精度浮点数样本
                return double_fill(&random_standard_exponential_fill, &self._bitgen, size, self.lock, out)
            else:
                # 使用double_fill函数生成标准指数分布的双精度浮点数样本（使用逆CDF方法）
                return double_fill(&random_standard_exponential_inv_fill, &self._bitgen, size, self.lock, out)
        elif _dtype == np.float32:
            if method == 'zig':
                # 使用float_fill函数生成标准指数分布的单精度浮点数样本
                return float_fill(&random_standard_exponential_fill_f, &self._bitgen, size, self.lock, out)
            else:
                # 使用float_fill函数生成标准指数分布的单精度浮点数样本（使用逆CDF方法）
                return float_fill(&random_standard_exponential_inv_fill_f, &self._bitgen, size, self.lock, out)
        else:
            # 抛出错误，不支持的dtype类型
            raise TypeError('Unsupported dtype %r for standard_exponential'
                            % _dtype)
    def bytes(self, np.npy_intp length):
        """
        bytes(length)

        Return random bytes.

        Parameters
        ----------
        length : int
            Number of random bytes.

        Returns
        -------
        out : bytes
            String of length `length`.

        Notes
        -----
        This function generates random bytes from a discrete uniform 
        distribution. The generated bytes are independent from the CPU's 
        native endianness.
        
        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.bytes(10)
        b'\\xfeC\\x9b\\x86\\x17\\xf2\\xa1\\xafcp'  # random

        """
        # Calculate the number of 32-bit unsigned integers needed to cover 'length' bytes
        cdef Py_ssize_t n_uint32 = ((length - 1) // 4 + 1)
        
        # Generate random integers from 0 to 4294967295 (2^32 - 1), size adjusted for 'length'
        # Convert these integers to little-endian byte order, then convert to bytes and truncate to 'length'
        return self.integers(0, 4294967296, size=n_uint32,
                             dtype=np.uint32).astype('<u4').tobytes()[:length]

    @cython.wraparound(True)
    # Enable wraparound indexing for Cython (handle negative indices as Python does):
    def standard_normal(self, size=None, dtype=np.float64, out=None):
        """
        standard_normal(size=None, dtype=np.float64, out=None)

        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result, only `float64` and `float32` are supported.
            Byteorder must be native. The default value is np.float64.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is not None,
            it must have the same shape as the provided size and must match the type of
            the output values.

        Returns
        -------
        out : float or ndarray
            A floating-point array of shape ``size`` of drawn samples, or a
            single sample if ``size`` was not specified.

        See Also
        --------
        normal :
            Equivalent function with additional ``loc`` and ``scale`` arguments
            for setting the mean and standard deviation.

        Notes
        -----
        For random samples from the normal distribution with mean ``mu`` and
        standard deviation ``sigma``, use one of::

            mu + sigma * rng.standard_normal(size=...)
            rng.normal(mu, sigma, size=...)

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.standard_normal()
        2.1923875335537315 # random

        >>> s = rng.standard_normal(8000)
        >>> s
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311,  # random
               -0.38672696, -0.4685006 ])                                # random
        >>> s.shape
        (8000,)
        >>> s = rng.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        Two-by-four array of samples from the normal distribution with
        mean 3 and standard deviation 2.5:

        >>> 3 + 2.5 * rng.standard_normal(size=(2, 4))
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random

        """
        # 获取指定的数据类型
        _dtype = np.dtype(dtype)
        # 检查数据类型是否为 np.float64
        if _dtype == np.float64:
            # 如果是 np.float64，则调用 double_fill 函数生成标准正态分布的样本
            return double_fill(&random_standard_normal_fill, &self._bitgen, size, self.lock, out)
        # 检查数据类型是否为 np.float32
        elif _dtype == np.float32:
            # 如果是 np.float32，则调用 float_fill 函数生成标准正态分布的样本
            return float_fill(&random_standard_normal_fill_f, &self._bitgen, size, self.lock, out)
        else:
            # 抛出错误，不支持的数据类型
            raise TypeError('Unsupported dtype %r for standard_normal' % _dtype)
    def chisquare(self, df, size=None):
        """
        chisquare(df, size=None)

        Draw samples from a chi-square distribution.

        When `df` independent random variables, each with standard normal
        distributions (mean 0, variance 1), are squared and summed, the
        resulting distribution is chi-square (see Notes).  This distribution
        is often used in hypothesis testing.

        Parameters
        ----------
        df : float or array_like of floats
             Number of degrees of freedom, must be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``df`` is a scalar.  Otherwise,
            ``np.array(df).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized chi-square distribution.

        Raises
        ------
        ValueError
            When `df` <= 0 or when an inappropriate `size` (e.g. ``size=-1``)
            is given.

        Notes
        -----
        The variable obtained by summing the squares of `df` independent,
        standard normally distributed random variables:

        .. math:: Q = \\sum_{i=0}^{\\mathtt{df}} X^2_i

        is chi-square distributed, denoted

        .. math:: Q \\sim \\chi^2_k.

        The probability density function of the chi-squared distribution is

        .. math:: p(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}
                         x^{k/2 - 1} e^{-x/2},

        where :math:`\\Gamma` is the gamma function,

        .. math:: \\Gamma(x) = \\int_0^{-\\infty} t^{x - 1} e^{-t} dt.

        References
        ----------
        .. [1] NIST "Engineering Statistics Handbook"
               https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.chisquare(2,4)
        array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272]) # random

        The distribution of a chi-square random variable
        with 20 degrees of freedom looks as follows:
        
        >>> import matplotlib.pyplot as plt
        >>> import scipy.stats as stats
        >>> s = rng.chisquare(20, 10000)
        >>> count, bins, _ = plt.hist(s, 30, density=True)
        >>> x = np.linspace(0, 60, 1000)
        >>> plt.plot(x, stats.chi2.pdf(x, df=20))
        >>> plt.xlim([0, 60])
        >>> plt.show()

        """
        # 调用底层函数以生成卡方分布的随机样本
        return cont(&random_chisquare, &self._bitgen, size, self.lock, 1,
                    df, 'df', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)
    # 定义一个方法用于生成标准 Cauchy 分布的样本数据
    def standard_cauchy(self, size=None):
        """
        standard_cauchy(size=None)

        Draw samples from a standard Cauchy distribution with mode = 0.

        Also known as the Lorentz distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The drawn samples.

        Notes
        -----
        The probability density function for the full Cauchy distribution is

        .. math:: P(x; x_0, \\gamma) = \\frac{1}{\\pi \\gamma \\bigl[ 1+
                  (\\frac{x-x_0}{\\gamma})^2 \\bigr] }

        and the Standard Cauchy distribution just sets :math:`x_0=0` and
        :math:`\\gamma=1`

        The Cauchy distribution arises in the solution to the driven harmonic
        oscillator problem, and also describes spectral line broadening. It
        also describes the distribution of values at which a line tilted at
        a random angle will cut the x axis.

        When studying hypothesis tests that assume normality, seeing how the
        tests perform on data from a Cauchy distribution is a good indicator of
        their sensitivity to a heavy-tailed distribution, since the Cauchy looks
        very much like a Gaussian distribution, but with heavier tails.

        References
        ----------
        .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "Cauchy
              Distribution",
              https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
        .. [2] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A
              Wolfram Web Resource.
              https://mathworld.wolfram.com/CauchyDistribution.html
        .. [3] Wikipedia, "Cauchy distribution"
              https://en.wikipedia.org/wiki/Cauchy_distribution

        Examples
        --------
        Draw samples and plot the distribution:

        >>> import matplotlib.pyplot as plt
        >>> rng = np.random.default_rng()
        >>> s = rng.standard_cauchy(1000000)
        >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
        >>> plt.hist(s, bins=100)
        >>> plt.show()

        """
        # 调用 C 库函数以生成标准 Cauchy 分布的样本数据
        return cont(&random_standard_cauchy, &self._bitgen, size, self.lock, 0,
                    0.0, '', CONS_NONE, 0.0, '', CONS_NONE, 0.0, '', CONS_NONE, None)
    def pareto(self, a, size=None):
        """
        pareto(a, size=None)

        Draw samples from a Pareto II (AKA Lomax) distribution with
        specified shape.

        Parameters
        ----------
        a : float or array_like of floats
            Shape of the distribution. Must be positive.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``a`` is a scalar.  Otherwise,
            ``np.array(a).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the Pareto II distribution.

        See Also
        --------
        scipy.stats.pareto : Pareto I distribution
        scipy.stats.lomax : Lomax (Pareto II) distribution
        scipy.stats.genpareto : Generalized Pareto distribution

        Notes
        -----
        The probability density for the Pareto II distribution is

        .. math:: p(x) = \\frac{a}{{x+1}^{a+1}} , x \ge 0

        where :math:`a > 0` is the shape.

        The Pareto II distribution is a shifted and scaled version of the
        Pareto I distribution, which can be found in `scipy.stats.pareto`.

        References
        ----------
        .. [1] Francis Hunt and Paul Johnson, On the Pareto Distribution of
               Sourceforge projects.
        .. [2] Pareto, V. (1896). Course of Political Economy. Lausanne.
        .. [3] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme
               Values, Birkhauser Verlag, Basel, pp 23-30.
        .. [4] Wikipedia, "Pareto distribution",
               https://en.wikipedia.org/wiki/Pareto_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 3.
        >>> rng = np.random.default_rng()
        >>> s = rng.pareto(a, 10000)

        Display the histogram of the samples, along with the probability
        density function:

        >>> import matplotlib.pyplot as plt
        >>> x = np.linspace(0, 3, 50)
        >>> pdf = a / (x+1)**(a+1)
        >>> plt.hist(s, bins=x, density=True, label='histogram')
        >>> plt.plot(x, pdf, linewidth=2, color='r', label='pdf')
        >>> plt.xlim(x.min(), x.max())
        >>> plt.legend()
        >>> plt.show()

        """
        # 调用 C 函数生成 Pareto II 分布的随机样本
        return cont(&random_pareto, &self._bitgen, size, self.lock, 1,
                    # 参数 a：分布的形状参数
                    a, 'a', CONS_POSITIVE,
                    # 默认值 0.0
                    0.0, '', CONS_NONE,
                    # 默认值 0.0
                    0.0, '', CONS_NONE, None)
    def rayleigh(self, scale=1.0, size=None):
        """
        rayleigh(scale=1.0, size=None)

        从 Rayleigh 分布中抽取样本。

        :math:`\\chi` 和 Weibull 分布是 Rayleigh 分布的推广。

        Parameters
        ----------
        scale : float or array_like of floats, optional
            尺度参数，也等于众数。必须为非负数。默认为 1。
        size : int or tuple of ints, optional
            输出的形状。如果给定形状为 ``(m, n, k)``，则抽取 ``m * n * k`` 个样本。
            如果 `size` 为 `None`（默认），且 `scale` 为标量，则返回单个值。
            否则，抽取 `np.array(scale).size` 个样本。

        Returns
        -------
        out : ndarray or scalar
            从参数化的 Rayleigh 分布中抽取的样本。

        Notes
        -----
        Rayleigh 分布的概率密度函数为：

        .. math:: P(x;scale) = \\frac{x}{scale^2}e^{\\frac{-x^2}{2 \\cdotp scale^2}}

        例如，如果风速的东向分量和北向分量都服从相同的均值为零的高斯分布，
        那么风速将服从 Rayleigh 分布。

        References
        ----------
        .. [1] Brighton Webs Ltd., "Rayleigh Distribution,"
               https://web.archive.org/web/20090514091424/http://brighton-webs.co.uk:80/distributions/rayleigh.asp
        .. [2] Wikipedia, "Rayleigh distribution"
               https://en.wikipedia.org/wiki/Rayleigh_distribution

        Examples
        --------
        从分布中抽取值并绘制直方图：

        >>> from matplotlib.pyplot import hist
        >>> rng = np.random.default_rng()
        >>> values = hist(rng.rayleigh(3, 100000), bins=200, density=True)

        浪高往往遵循 Rayleigh 分布。如果平均浪高为 1 米，有多少浪的高度可能大于 3 米？

        >>> meanvalue = 1
        >>> modevalue = np.sqrt(2 / np.pi) * meanvalue
        >>> s = rng.rayleigh(modevalue, 1000000)

        高于 3 米的浪的百分比为：

        >>> 100.*sum(s>3)/1000000.
        0.087300000000000003 # 随机

        """
        # 调用 `random_rayleigh` 函数以从 Rayleigh 分布中抽取随机样本
        return cont(&random_rayleigh, &self._bitgen, size, self.lock, 1,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)
    # 定义 Wald 分布的抽样方法，返回符合参数化 Wald 分布的样本
    def wald(self, mean, scale, size=None):
        """
        wald(mean, scale, size=None)

        Draw samples from a Wald, or inverse Gaussian, distribution.

        As the scale approaches infinity, the distribution becomes more like a
        Gaussian. Some references claim that the Wald is an inverse Gaussian
        with mean equal to 1, but this is by no means universal.

        The inverse Gaussian distribution was first studied in relationship to
        Brownian motion. In 1956 M.C.K. Tweedie used the name inverse Gaussian
        because there is an inverse relationship between the time to cover a
        unit distance and distance covered in unit time.

        Parameters
        ----------
        mean : float or array_like of floats
            Distribution mean, must be > 0.
        scale : float or array_like of floats
            Scale parameter, must be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``mean`` and ``scale`` are both scalars.
            Otherwise, ``np.broadcast(mean, scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Wald distribution.

        Notes
        -----
        The probability density function for the Wald distribution is

        .. math:: P(x;mean,scale) = \\sqrt{\\frac{scale}{2\\pi x^3}}e^
                                    \\frac{-scale(x-mean)^2}{2\\cdotp mean^2x}

        As noted above the inverse Gaussian distribution first arise
        from attempts to model Brownian motion. It is also a
        competitor to the Weibull for use in reliability modeling and
        modeling stock returns and interest rate processes.

        References
        ----------
        .. [1] Brighton Webs Ltd., Wald Distribution,
               https://web.archive.org/web/20090423014010/http://www.brighton-webs.co.uk:80/distributions/wald.asp
        .. [2] Chhikara, Raj S., and Folks, J. Leroy, "The Inverse Gaussian
               Distribution: Theory : Methodology, and Applications", CRC Press,
               1988.
        .. [3] Wikipedia, "Inverse Gaussian distribution"
               https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram:

        >>> import matplotlib.pyplot as plt
        >>> rng = np.random.default_rng()
        >>> h = plt.hist(rng.wald(3, 2, 100000), bins=200, density=True)
        >>> plt.show()

        """
        # 调用底层 C 函数 cont，实现复杂的 Wald 分布抽样
        return cont(&random_wald, &self._bitgen, size, self.lock, 2,
                    mean, 'mean', CONS_POSITIVE,
                    scale, 'scale', CONS_POSITIVE,
                    0.0, '', CONS_NONE, None)

    # 复杂的离散分布:
    def poisson(self, lam=1.0, size=None):
        """
        poisson(lam=1.0, size=None)

        Draw samples from a Poisson distribution.

        The Poisson distribution is the limit of the binomial distribution
        for large N.

        Parameters
        ----------
        lam : float or array_like of floats
            Expected number of events occurring in a fixed-time interval,
            must be >= 0. A sequence must be broadcastable over the requested
            size.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``lam`` is a scalar. Otherwise,
            ``np.array(lam).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Poisson distribution.

        Notes
        -----
        The Poisson distribution

        .. math:: f(k; \\lambda)=\\frac{\\lambda^k e^{-\\lambda}}{k!}

        For events with an expected separation :math:`\\lambda` the Poisson
        distribution :math:`f(k; \\lambda)` describes the probability of
        :math:`k` events occurring within the observed
        interval :math:`\\lambda`.

        Because the output is limited to the range of the C int64 type, a
        ValueError is raised when `lam` is within 10 sigma of the maximum
        representable value.

        References
        ----------
        .. [1] Weisstein, Eric W. "Poisson Distribution."
               From MathWorld--A Wolfram Web Resource.
               https://mathworld.wolfram.com/PoissonDistribution.html
        .. [2] Wikipedia, "Poisson distribution",
               https://en.wikipedia.org/wiki/Poisson_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> rng = np.random.default_rng()
        >>> lam, size = 5, 10000
        >>> s = rng.poisson(lam=lam, size=size)

        Verify the mean and variance, which should be approximately ``lam``:
        
        >>> s.mean(), s.var()
        (4.9917 5.1088311)  # may vary

        Display the histogram and probability mass function:

        >>> import matplotlib.pyplot as plt
        >>> from scipy import stats
        >>> x = np.arange(0, 21)
        >>> pmf = stats.poisson.pmf(x, mu=lam)
        >>> plt.hist(s, bins=x, density=True, width=0.5)
        >>> plt.stem(x, pmf, 'C1-')
        >>> plt.show()

        Draw each 100 values for lambda 100 and 500:

        >>> s = rng.poisson(lam=(100., 500.), size=(100, 2))

        """
        使用指定的参数调用底层的随机泊松函数，返回随机数样本
        return disc(&random_poisson, &self._bitgen, size, self.lock, 1, 0,
                    lam, 'lam', CONS_POISSON,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)
    # 定义 geometric 方法，用于从几何分布中抽取样本
    def geometric(self, p, size=None):
        """
        geometric(p, size=None)

        Draw samples from the geometric distribution.

        Bernoulli trials are experiments with one of two outcomes:
        success or failure (an example of such an experiment is flipping
        a coin).  The geometric distribution models the number of trials
        that must be run in order to achieve success.  It is therefore
        supported on the positive integers, ``k = 1, 2, ...``.

        The probability mass function of the geometric distribution is

        .. math:: f(k) = (1 - p)^{k - 1} p

        where `p` is the probability of success of an individual trial.

        Parameters
        ----------
        p : float or array_like of floats
            The probability of success of an individual trial.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``p`` is a scalar.  Otherwise,
            ``np.array(p).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized geometric distribution.

        References
        ----------

        .. [1] Wikipedia, "Geometric distribution",
               https://en.wikipedia.org/wiki/Geometric_distribution

        Examples
        --------
        Draw 10,000 values from the geometric distribution, with the 
        probability of an individual success equal to ``p = 0.35``:

        >>> p, size = 0.35, 10000
        >>> rng = np.random.default_rng()
        >>> sample = rng.geometric(p=p, size=size)

        What proportion of trials succeeded after a single run?

        >>> (sample == 1).sum()/size
        0.34889999999999999  # may vary

        The geometric distribution with ``p=0.35`` looks as follows:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(sample, bins=30, density=True)
        >>> plt.plot(bins, (1-p)**(bins-1)*p)
        >>> plt.xlim([0, 25])
        >>> plt.show()
        
        """
        # 调用底层的离散分布函数 disc，抽取几何分布的样本
        return disc(&random_geometric, &self._bitgen, size, self.lock, 1, 0,
                    p, 'p', CONS_BOUNDED_GT_0_1,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    # Multivariate distributions:
    def permutation(self, object x, axis=0):
        """
        permutation(x, axis=0)

        Randomly permute a sequence, or return a permuted range.

        Parameters
        ----------
        x : int or array_like
            If `x` is an integer, randomly permute ``np.arange(x)``.
            If `x` is an array, make a copy and shuffle the elements
            randomly.
        axis : int, optional
            The axis which `x` is shuffled along. Default is 0.

        Returns
        -------
        out : ndarray
            Permuted sequence or array range.

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.permutation(10)
        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random

        >>> rng.permutation([1, 4, 9, 12, 15])
        array([15,  1,  9,  4, 12]) # random

        >>> arr = np.arange(9).reshape((3, 3))
        >>> rng.permutation(arr)
        array([[6, 7, 8], # random
               [0, 1, 2],
               [3, 4, 5]])

        >>> rng.permutation("abc")
        Traceback (most recent call last):
            ...
        numpy.exceptions.AxisError: axis 0 is out of bounds for array of dimension 0

        >>> arr = np.arange(9).reshape((3, 3))
        >>> rng.permutation(arr, axis=1)
        array([[0, 2, 1], # random
               [3, 5, 4],
               [6, 8, 7]])

        """
        # 如果 `x` 是整数，则创建一个包含 0 到 x-1 的整数数组，并进行随机排列
        if isinstance(x, (int, np.integer)):
            arr = np.arange(x)
            self.shuffle(arr)
            return arr

        # 将 `x` 转换为 NumPy 数组
        arr = np.asarray(x)

        # 根据轴的索引规范化轴的索引值
        axis = normalize_axis_index(axis, arr.ndim)

        # 如果数组是一维的，进行快速的 shuffle 操作
        if arr.ndim == 1:
            # 如果 `arr` 和 `x` 使用相同的内存，返回一个复制的数组
            if np.may_share_memory(arr, x):
                arr = np.array(arr)
            self.shuffle(arr)
            return arr

        # 对索引数组进行 shuffle，使用 intp 类型以确保快速路径
        idx = np.arange(arr.shape[axis], dtype=np.intp)
        self.shuffle(idx)

        # 构造切片对象以进行索引数组的应用
        slices = [slice(None)] * arr.ndim
        slices[axis] = idx
        return arr[tuple(slices)]
# 在 Cython 中嵌入函数签名，并设置为 True
@cython.embedsignature(True)
# 定义一个函数 default_rng，用于构造一个新的 Generator，使用默认的 BitGenerator (PCG64)
def default_rng(seed=None):
    """Construct a new Generator with the default BitGenerator (PCG64).

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then all values must be non-negative and will be
        passed to `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.
        Additionally, when passed a `BitGenerator`, it will be wrapped by
        `Generator`. If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    Generator
        The initialized generator object.

    Notes
    -----
    If ``seed`` is not a `BitGenerator` or a `Generator`, a new `BitGenerator`
    is instantiated. This function does not manage a default global instance.

    See :ref:`seeding_and_entropy` for more information about seeding.
    
    Examples
    --------
    `default_rng` is the recommended constructor for the random number class
    `Generator`. Here are several ways we can construct a random 
    number generator using `default_rng` and the `Generator` class. 
    
    Here we use `default_rng` to generate a random float:
 
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> print(rng)
    Generator(PCG64)
    >>> rfloat = rng.random()
    >>> rfloat
    0.22733602246716966
    >>> type(rfloat)
    <class 'float'>
     
    Here we use `default_rng` to generate 3 random integers between 0 
    (inclusive) and 10 (exclusive):
        
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> rints = rng.integers(low=0, high=10, size=3)
    >>> rints
    array([6, 2, 7])
    >>> type(rints[0])
    <class 'numpy.int64'>
    
    Here we specify a seed so that we have reproducible results:
    
    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> print(rng)
    Generator(PCG64)
    >>> arr1 = rng.random((3, 3))
    >>> arr1
    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235],
           [0.7611397 , 0.78606431, 0.12811363]])

    If we exit and restart our Python interpreter, we'll see that we
    generate the same random numbers again:

    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> arr2 = rng.random((3, 3))
    >>> arr2
    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235],
           [0.7611397 , 0.78606431, 0.12811363]])

    """
    # 检查传入的 seed 是否是一个 BitGenerator
    if _check_bit_generator(seed):
        # 如果是 BitGenerator，则直接将其包装为 Generator 并返回
        return Generator(seed)
    # 如果传入的 seed 是 Generator 的实例，则直接返回该实例
    elif isinstance(seed, Generator):
        return seed
   `
    # 否则，我们需要像正常一样实例化一个新的 BitGenerator 和 Generator。
    return Generator(PCG64(seed))
```
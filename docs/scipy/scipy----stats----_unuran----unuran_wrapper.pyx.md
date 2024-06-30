# `D:\src\scipysrc\scipy\scipy\stats\_unuran\unuran_wrapper.pyx`

```
# cython: language_level=3

# 导入 Cython 扩展和必要的 CPython 对象
cimport cython
from cpython.object cimport PyObject
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t

# 导入 SciPy 相关模块和 UNU.RAN 中的必要头文件
from scipy._lib.ccallback cimport ccallback_t
from scipy._lib.messagestream cimport MessageStream
from .unuran cimport *
import warnings
import threading
import functools
from collections import namedtuple
import numpy as np
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce, rv_frozen
from scipy._lib._util import check_random_state
import warnings

# 导入 NumPy C 库
np.import_array()

# 定义公开的 API 列表
__all__ = ['UNURANError', 'TransformedDensityRejection', 'DiscreteAliasUrn',
           'NumericalInversePolynomial']

# 定义 Python.h 头文件中的错误处理相关函数
cdef extern from "Python.h":
    PyObject *PyErr_Occurred()
    void PyErr_Fetch(PyObject **ptype, PyObject **pvalue, PyObject **ptraceback)
    void PyErr_Restore(PyObject *type, PyObject *value, PyObject *traceback)

# 定义用于处理 Python 回调的内部 API
cdef extern from "unuran_callback.h":
    int init_unuran_callback(ccallback_t *callback, fcn) except -1
    int release_unuran_callback(ccallback_t *callback) except -1

    double pdf_thunk(double x, const unur_distr *distr) nogil
    double dpdf_thunk(double x, const unur_distr *distr) nogil
    double logpdf_thunk(double x, const unur_distr *distr) nogil
    double cont_cdf_thunk(double x, const unur_distr *distr) nogil
    double pmf_thunk(int x, const unur_distr *distr) nogil
    double discr_cdf_thunk(int x, const unur_distr *distr) nogil

    void error_handler(const char *objid, const char *file,
                       int line, const char *errortype,
                       int unur_errno, const char *reason) nogil

# 外部链接的 UNU.RAN 头文件中的全局常量 UNUR_INFINITY
cdef extern from "unuran.h":
    cdef double UNUR_INFINITY

# 定义一个自定义异常类，用于表示 UNU.RAN 库中的运行时错误
class UNURANError(RuntimeError):
    """Raised when an error occurs in the UNU.RAN library."""
    pass

# 定义一个 C 语言类型别名 URNG_FUNCT，用于表示返回类型为 double，参数为 void* 的函数指针类型
ctypedef double (*URNG_FUNCT)(void *) noexcept nogil

# 定义一个 Python 函数，根据输入种子创建一个 NumPy Generator 对象
cdef object get_numpy_rng(object seed = None):
    """
    Create a NumPy Generator object from a given seed.

    Parameters
    ----------
    seed : object, optional
        Seed for the generator. If None, no seed is set. The seed can be
        an integer, Generator, or RandomState.

    Returns
    -------
    numpy_rng : object
        An instance of NumPy's Generator class.
    """
    seed = check_random_state(seed)
    if isinstance(seed, np.random.RandomState):
        return np.random.default_rng(seed._bit_generator)
    return seed

# 定义一个 Cython 类 _URNG，用于封装一个从 NumPy 随机数生成器创建的 UNU.RAN 的均匀分布随机数生成器
@cython.final
cdef class _URNG:
    """
    Build a UNU.RAN's uniform random number generator from a NumPy random
    number generator.

    Parameters
    ----------
    numpy_rng : object
        An instance of NumPy's Generator or RandomState class. i.e. a NumPy
        random number generator.
    """
    # 声明一个 C 类型的对象 numpy_rng
    cdef object numpy_rng
    # 声明一个 C 类型的一维双精度浮点数组 qrvs_array
    cdef double[::1] qrvs_array
    # 声明一个 C 类型的 size_t 类型变量 i

    # 初始化方法，接收一个 numpy_rng 参数
    def __init__(self, numpy_rng):
        self.numpy_rng = numpy_rng

    # 使用 Cython 的装饰器关闭边界检查
    @cython.boundscheck(False)
    # 使用 Cython 的装饰器关闭负数索引包装
    @cython.wraparound(False)
    # 定义一个 C 类型函数 _next_qdouble，无异常抛出，无 GIL
    cdef double _next_qdouble(self) noexcept nogil:
        # 自增实例变量 i
        self.i += 1
        # 返回数组 qrvs_array 中索引为 self.i-1 的元素
        return self.qrvs_array[self.i-1]

    # 定义一个方法 get_urng，返回一个 unur_urng 指针对象，可能抛出异常
    cdef unur_urng * get_urng(self) except *:
        """
        Get a ``unur_urng`` object from given ``numpy_rng``.

        Returns
        -------
        unuran_urng : unur_urng *
            A UNU.RAN uniform random number generator.
        """
        # 声明变量
        cdef unur_urng *unuran_urng
        cdef:
            bitgen_t *numpy_urng
            const char *capsule_name = "BitGenerator"

        # 从 numpy_rng 中获取 bit_generator 的 capsule
        capsule = self.numpy_rng.bit_generator.capsule

        # 检查 capsule 是否有效，如果无效则抛出 ValueError
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state.")

        # 从 capsule 中获取指针，并将其转换为 bitgen_t 类型指针
        numpy_urng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

        # 使用 bitgen_t 对象创建 unur_urng 对象
        unuran_urng = unur_urng_new(numpy_urng.next_double,
                                    <void *>(numpy_urng.state))

        # 返回 unur_urng 指针对象
        return unuran_urng

    # 定义一个方法 get_qurng，返回一个 unur_urng 指针对象，可能抛出异常
    cdef unur_urng *get_qurng(self, size, qmc_engine) except *:
        # 声明变量
        cdef unur_urng *unuran_urng

        # 重置 self.i 为 0
        self.i = 0

        # 使用 qmc_engine 生成随机数并转换为连续内存的 float64 数组，然后扁平化为一维数组
        self.qrvs_array = np.ascontiguousarray(
            qmc_engine.random(size).ravel().astype(np.float64)
        )

        # 使用 URNG_FUNCT 函数指针和当前对象创建 unur_urng 对象
        unuran_urng = unur_urng_new(<URNG_FUNCT>self._next_qdouble,
                                    <void *>self)

        # 返回 unur_urng 指针对象
        return unuran_urng
# 模块级锁。用于提供线程安全的错误报告。
# UNU.RAN 具有线程不安全的全局 FILE 流用于记录错误。
# 为了实现线程安全，可以在调用 `unur_set_stream` 前获取锁，并在流不再需要时释放。
cdef object _lock = threading.RLock()

cdef:
    unur_urng *default_urng           # UNU.RAN 默认的均匀随机数生成器指针
    object default_numpy_rng          # 默认的 NumPy 随机数生成器对象
    _URNG _urng_builder              # URNG 构建器对象


cdef object _setup_unuran():
    """
    设置默认的 UNU.RAN 均匀随机数生成器和错误处理器。
    """
    global default_urng
    global default_numpy_rng
    global _urng_builder

    default_numpy_rng = get_numpy_rng()   # 获取默认的 NumPy 随机数生成器对象

    cdef MessageStream _messages = MessageStream()   # 创建消息流对象 _messages

    _lock.acquire()   # 获取锁，确保线程安全
    try:
        unur_set_stream(_messages.handle)   # 设置消息流到 UNU.RAN
        # 尝试设置默认的 URNG。
        try:
            _urng_builder = _URNG(default_numpy_rng)   # 使用 NumPy 随机数生成器创建 URNG 构建器对象
            default_urng = _urng_builder.get_urng()    # 获取 URNG 对象
            if default_urng == NULL:
                raise UNURANError(_messages.get())    # 如果获取的 URNG 为空指针，则引发 UNURANError
        except Exception as e:
            msg = "Failed to initialize the default URNG."
            raise RuntimeError(msg) from e   # 在初始化默认 URNG 失败时引发 RuntimeError
    finally:
        _lock.release()   # 释放锁，结束对临界区的保护

    unur_set_default_urng(default_urng)   # 设置默认的 UNU.RAN URNG
    unur_set_error_handler(error_handler)   # 设置错误处理器


_setup_unuran()


cdef dict _unpack_dist(object dist, str dist_type, list meths = None,
                       list optional_meths = None):
    """
    从 Python 类或对象中获取所需的方法/属性。

    Parameters
    ----------
    dist : object
        Python 类的实例或具有所需方法的对象。
    dist_type : str
        分布的类型。"cont" 表示连续分布，"discr" 表示离散分布。
    meths : list
        要从 `dist` 中获取的方法列表。
    optional_meths : list, optional
        可选的方法列表。如果找到则返回，未找到的方法不会引发错误。

    Returns
    -------
    callbacks : dict
        回调函数的字典（找到的方法）。

    Raises
    ------
    ValueError
        如果在 `meths` 列表中找不到某些方法，则引发 ValueError。
    """
    cdef dict callbacks = {}
    # 检查 `dist` 是否为 `rv_frozen` 类型的对象
    if isinstance(dist, rv_frozen):
        # 检查 `dist.dist` 是否为 `rv_continuous` 类型的分布
        if isinstance(dist.dist, stats.rv_continuous):
            # 定义一个包装分布的类
            class wrap_dist:
                # 初始化方法，接受一个分布对象 `dist`
                def __init__(self, dist):
                    self.dist = dist
                    # 解析分布参数并存储到类属性中
                    (self.args, self.loc,
                     self.scale) = dist.dist._parse_args(*dist.args,
                                                         **dist.kwds)
                    self.support = dist.support
                # 概率密度函数，计算给定值 `x` 的概率密度
                def pdf(self, x):
                    # 一些分布要求输入为数组
                    x = np.asarray((x-self.loc)/self.scale)
                    # 返回概率密度函数值，确保不小于0
                    return max(0, self.dist.dist._pdf(x, *self.args)/self.scale)
                # 对数概率密度函数，计算给定值 `x` 的对数概率密度
                def logpdf(self, x):
                    # 一些分布要求输入为数组
                    x = np.asarray((x-self.loc)/self.scale)
                    # 如果概率密度大于0，则返回对数概率密度值，否则返回负无穷
                    if self.pdf(x) > 0:
                        return self.dist.dist._logpdf(x, *self.args) - np.log(self.scale)
                    return -np.inf
                # 累积分布函数，计算给定值 `x` 的累积概率
                def cdf(self, x):
                    # 将输入值转换为数组
                    x = np.asarray((x-self.loc)/self.scale)
                    # 计算累积分布函数值，并确保在 [0, 1] 区间内
                    res = self.dist.dist._cdf(x, *self.args)
                    if res < 0:
                        return 0
                    elif res > 1:
                        return 1
                    return res
        # 如果 `dist.dist` 是 `rv_discrete` 类型的分布
        elif isinstance(dist.dist, stats.rv_discrete):
            # 定义一个包装分布的类
            class wrap_dist:
                # 初始化方法，接受一个分布对象 `dist`
                def __init__(self, dist):
                    self.dist = dist
                    # 解析分布参数并存储到类属性中
                    (self.args, self.loc,
                     _) = dist.dist._parse_args(*dist.args,
                                                **dist.kwds)
                    self.support = dist.support
                # 概率质量函数，计算给定值 `x` 的概率质量
                def pmf(self, x):
                    # 一些分布要求输入为数组
                    x = np.asarray(x-self.loc)
                    # 返回概率质量函数值，确保不小于0
                    return max(0, self.dist.dist._pmf(x, *self.args))
                # 累积分布函数，计算给定值 `x` 的累积概率
                def cdf(self, x):
                    # 将输入值转换为数组
                    x = np.asarray(x-self.loc)
                    # 计算累积分布函数值，并确保在 [0, 1] 区间内
                    res = self.dist.dist._cdf(x, *self.args)
                    if res < 0:
                        return 0
                    elif res > 1:
                        return 1
                    return res
        # 将 `dist` 包装为 `wrap_dist` 类的实例
        dist = wrap_dist(dist)
    # 如果 `meths` 参数不为空
    if meths is not None:
        # 遍历 `meths` 中的每个方法名
        for meth in meths:
            # 检查 `dist` 是否具有该方法
            if hasattr(dist, meth):
                # 将该方法添加到 `callbacks` 字典中
                callbacks[meth] = getattr(dist, meth)
            else:
                # 如果方法不存在，则抛出异常
                msg = f"`{meth}` required but not found."
                raise ValueError(msg)
    # 如果 `optional_meths` 参数不为空
    if optional_meths is not None:
        # 遍历 `optional_meths` 中的每个方法名
        for meth in optional_meths:
            # 检查 `dist` 是否具有该方法
            if hasattr(dist, meth):
                # 将该方法添加到 `callbacks` 字典中
                callbacks[meth] = getattr(dist, meth)
    # 返回包含回调函数的 `callbacks` 字典
    return callbacks
# 设置一个连续或离散分布对象的方法，使用回调函数字典。
# 参数：
# distr : unur_distr *
#     连续或离散分布对象。
# callbacks : dict
#     回调函数的字典。
cdef void _pack_distr(unur_distr *distr, dict callbacks) except *:
    if unur_distr_is_cont(distr):
        # 如果是连续分布，设置 PDF 回调函数
        if "pdf" in callbacks:
            unur_distr_cont_set_pdf(distr, pdf_thunk)
        # 设置导数 PDF 回调函数
        if "dpdf" in callbacks:
            unur_distr_cont_set_dpdf(distr, dpdf_thunk)
        # 设置累积分布函数 CDF 回调函数
        if "cdf" in callbacks:
            unur_distr_cont_set_cdf(distr, cont_cdf_thunk)
        # 设置对数 PDF 回调函数
        if "logpdf" in callbacks:
            unur_distr_cont_set_logpdf(distr, logpdf_thunk)
    else:
        # 如果是离散分布，设置 PMF 回调函数
        if "pmf" in callbacks:
            unur_distr_discr_set_pmf(distr, pmf_thunk)
        # 设置累积分布函数 CDF 回调函数
        if "cdf" in callbacks:
            unur_distr_discr_set_cdf(distr, discr_cdf_thunk)


# 验证域 domain 是否合法，并返回处理后的 domain。
def _validate_domain(domain, dist):
    if domain is None and hasattr(dist, 'support'):
        # 如果分布对象具有 support 方法，则使用该方法获取 domain
        domain = dist.support()
    if domain is not None:
        # UNU.RAN 不识别概率向量中的 NaN 值，会抛出 "unknown error" 错误，因此在这里检查 NaN 值
        if np.isnan(domain).any():
            raise ValueError("`domain` must contain only non-nan values.")
        # domain 的长度必须为 2
        if len(domain) != 2:
            raise ValueError("`domain` must be a length 2 tuple.")
        # 如果无法转换成元组，则抛出错误
        domain = tuple(domain)
    return domain


# 验证概率向量 pv 是否合法，并返回其连续内存视图。
cdef double[::1] _validate_pv(pv) except *:
    cdef double[::1] pv_view = None
    if pv is not None:
        # 确保 pv 是连续的双精度数组
        pv = pv_view = np.ascontiguousarray(pv, dtype=np.float64)
        # 禁止空数组
        if pv.size == 0:
            raise ValueError("probability vector must contain at least "
                             "one element.")
        # UNU.RAN 不识别 NaN 和无穷值，因此在这里检查是否包含非有限值
        if not np.isfinite(pv).all():
            raise ValueError("probability vector must contain only "
                             "finite / non-nan values.")
        # UNU.RAN 不处理全为零的特殊情况，会抛出 "unknown error" 错误
        if (pv == 0).all():
            raise ValueError("probability vector must contain at least "
                             "one non-zero value.")
    # 返回 pv 的连续内存视图
    return pv_view


# 验证 QMC 引擎和维度 d 的输入是否合法。
def _validate_qmc_input(qmc_engine, d):
    # 对 `qmc_engine` 和 `d` 进行输入验证
    # 无效的 `d` 将由 QMCEngine 抛出错误信息
    # 或许我们可以使用 stats.qmc.check_qrandom_state 进行验证
    # 检查 `qmc_engine` 是否为 `stats.qmc.QMCEngine` 的实例
    if isinstance(qmc_engine, stats.qmc.QMCEngine):
        # 如果指定了 `d` 并且 `qmc_engine` 的维度与 `d` 不一致，则引发错误
        if d is not None and qmc_engine.d != d:
            message = "`d` must be consistent with dimension of `qmc_engine`."
            raise ValueError(message)
        # 如果未指定 `d`，则将 `d` 设置为 `qmc_engine` 的维度
        d = qmc_engine.d if d is None else d
    # 如果 `qmc_engine` 为 None
    elif qmc_engine is None:
        # 如果未指定 `d`，则将 `d` 设置为 1
        d = 1 if d is None else d
        # 创建一个 Halton 序列的 QMC 引擎，使用指定的维度 `d`
        qmc_engine = stats.qmc.Halton(d)
    else:
        # 如果 `qmc_engine` 不是预期的类型，抛出错误
        message = ("`qmc_engine` must be an instance of "
                    "`scipy.stats.qmc.QMCEngine` or `None`.")
        raise ValueError(message)

    # 返回 QMC 引擎对象和确定的维度 `d`
    return qmc_engine, d
    """
    A base class for all the wrapped generators.

    There are 6 basic functions of this base class:

    * It provides a `_set_rng` method to initialize and set a `unur_gen`
      object. It should be called during the setup stage in the `__cinit__`
      method. As it uses MessageStream, the call must be protected under
      the module-level lock.
    * `_check_errorcode` must be called after calling a UNU.RAN function
      that returns an error code. It raises an error if an error has
      occurred in UNU.RAN.
    * It implements the `rvs` public method for sampling. No child class
      should override this method.
    * Provides a `set_random_state` method to change the seed.
    * Implements the __dealloc__ method. The child class must not override
      this method.
    * Implements __reduce__ method to allow pickling.

    """
    # Distribution object pointer in UNU.RAN
    cdef unur_distr *distr
    # Parameter object pointer in UNU.RAN
    cdef unur_par *par
    # Generator object pointer in UNU.RAN for random number generation
    cdef unur_gen *rng
    # Uniform random number generator object pointer in UNU.RAN
    cdef unur_urng *urng
    # Object to hold numpy random number generator
    cdef object numpy_rng
    # Builder for uniform random number generator
    cdef _URNG _urng_builder
    # Callbacks associated with the generator
    cdef object callbacks
    # Wrapper for callback functions
    cdef object _callback_wrapper
    # Message stream for handling messages and errors
    cdef MessageStream _messages
    # Dictionary to save all arguments for pickling
    # Object to save all keyword arguments for pickling
    cdef object _kwargs

    # Method to check for error code returned by UNU.RAN functions
    cdef inline void _check_errorcode(self, int errorcode) except *:
        # check for non-zero errorcode
        if errorcode != UNUR_SUCCESS:
            # Retrieve the error message from the message stream
            msg = self._messages.get()
            # Raise an error if the message is not empty
            # A non-empty message indicates an error condition in UNU.RAN
            if msg:
                raise UNURANError(msg)
    cdef inline void _set_rng(self, object random_state) except *:
        """
        Create a UNU.RAN random number generator.

        Parameters
        ----------
        random_state : object
            Seed for the uniform random number generator. Can be a integer,
            Generator, or RandomState.
        """
        # 定义回调函数对象
        cdef ccallback_t callback
        # 获取 NumPy 随机数生成器对象
        self.numpy_rng = get_numpy_rng(random_state)
        # 使用 NumPy 随机数生成器创建 UNU.RAN 的 URNG 对象
        self._urng_builder = _URNG(self.numpy_rng)
        # 获取 URNG 对象
        self.urng = self._urng_builder.get_urng()
        # 如果获取的 URNG 对象为空，则抛出 UNURANError 异常
        if self.urng == NULL:
            raise UNURANError(self._messages.get())
        # 将 URNG 对象设置为当前 UNU.RAN 参数对象的随机数生成器
        self._check_errorcode(unur_set_urng(self.par, self.urng))
        # 检查是否定义了回调函数包装器
        has_callback_wrapper = (self._callback_wrapper is not None)
        try:
            # 如果定义了回调函数包装器，则初始化回调函数
            if has_callback_wrapper:
                init_unuran_callback(&callback, self._callback_wrapper)
            # 初始化 UNU.RAN 的随机数生成器对象
            self.rng = unur_init(self.par)
            # 设置 self.par = NULL，因为调用 `unur_init` 会销毁参数对象
            # 参见 UNU.RAN 文档中 "Creating a generator object"
            self.par = NULL
            # 如果初始化的随机数生成器对象为空，则抛出 UNURANError 异常
            if self.rng == NULL:
                if PyErr_Occurred():
                    return
                raise UNURANError(self._messages.get())
            # 释放分布对象的内存
            unur_distr_free(self.distr)
            self.distr = NULL
        finally:
            # 如果定义了回调函数包装器，则释放回调函数
            if has_callback_wrapper:
                release_unuran_callback(&callback)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void _rvs_cont(self, double[::1] out) except *:
        """
        Sample random variates from a continuous distribution.

        Parameters
        ----------
        out : double[::1]
            A memory view of size ``size`` to store the result.
        """
        # 定义回调函数对象
        cdef:
            ccallback_t callback
            # 获取当前的 UNU.RAN 随机数生成器对象
            unur_gen *rng = self.rng
            # 获取输出数组的大小
            size_t i
            size_t size = len(out)
            PyObject *type
            PyObject *value
            PyObject *traceback

        # 检查是否定义了回调函数包装器
        has_callback_wrapper = (self._callback_wrapper is not None)
        # 错误标志
        error = 0

        # 获取全局锁，保证线程安全
        _lock.acquire()
        try:
            # 清空消息列表
            self._messages.clear()
            # 设置 UNU.RAN 的消息处理句柄
            unur_set_stream(self._messages.handle)

            # 如果定义了回调函数包装器，则初始化回调函数
            if has_callback_wrapper:
                init_unuran_callback(&callback, self._callback_wrapper)
            # 循环生成指定数量的随机数样本
            for i in range(size):
                out[i] = unur_sample_cont(rng)
                # 检查是否有 Python 异常发生
                if PyErr_Occurred():
                    error = 1
                    return
            # 获取消息列表中的错误消息
            msg = self._messages.get()
            # 如果存在错误消息，则抛出 UNURANError 异常
            if msg:
                raise UNURANError(msg)
        finally:
            # 如果发生错误，则恢复 Python 的异常状态
            if error:
                PyErr_Fetch(&type, &value, &traceback)
            # 释放全局锁
            _lock.release()
            # 如果发生错误，则恢复 Python 的异常状态
            if error:
                PyErr_Restore(type, value, traceback)
            # 如果定义了回调函数包装器，则释放回调函数
            if has_callback_wrapper:
                release_unuran_callback(&callback)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void _rvs_discr(self, int[::1] out) except *:
        """
        Sample random variates from a discrete distribution.

        Parameters
        ----------
        out : int[::1]
            A memory view of size ``size`` to store the result.
        """
        cdef:
            ccallback_t callback  # 声明一个回调函数变量
            unur_gen *rng = self.rng  # 使用 self.rng 初始化一个随机数生成器
            size_t i  # 声明一个循环计数变量 i
            size_t size = len(out)  # 计算输出数组 out 的长度并赋值给 size
            PyObject *type  # 异常类型的指针
            PyObject *value  # 异常值的指针
            PyObject *traceback  # 异常回溯信息的指针

        has_callback_wrapper = (self._callback_wrapper is not None)  # 检查是否存在回调包装函数
        error = 0  # 初始化错误标志为 0

        _lock.acquire()  # 获取全局锁
        try:
            self._messages.clear()  # 清空 self._messages
            unur_set_stream(self._messages.handle)  # 设置消息流的处理句柄

            if has_callback_wrapper:
                init_unuran_callback(&callback, self._callback_wrapper)  # 初始化回调函数

            # 遍历输出数组 out
            for i in range(size):
                out[i] = unur_sample_discr(rng)  # 从离散分布中抽取随机变量并存储到 out[i]
                if PyErr_Occurred():  # 检查是否发生了 Python 异常
                    error = 1  # 设置错误标志为 1
                    return  # 返回

            msg = self._messages.get()  # 获取消息
            if msg:
                raise UNURANError(msg)  # 如果有消息则抛出 UNURANError 异常
        finally:
            if error:
                PyErr_Fetch(&type, &value, &traceback)  # 捕获异常信息
            _lock.release()  # 释放全局锁
            if error:
                PyErr_Restore(type, value, traceback)  # 恢复异常信息
            if has_callback_wrapper:
                release_unuran_callback(&callback)  # 释放回调函数资源
    def rvs(self, size=None, random_state=None):
        """
        rvs(size=None, random_state=None)

        Sample from the distribution.

        Parameters
        ----------
        size : int or tuple, optional
            The shape of samples. Default is ``None`` in which case a scalar
            sample is returned.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            A NumPy random number generator or seed for the underlying NumPy random
            number generator used to generate the stream of uniform random numbers.
            If `random_state` is None (or `np.random`), `random_state` provided during
            initialization is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState`` instance then
            that instance is used.

        Returns
        -------
        rvs : array_like
            A NumPy array of random variates.
        """
        # 定义两个 C 语言风格的数组指针变量
        cdef double[::1] out_cont
        cdef int[::1] out_discr
        # 计算需要生成的随机变量数量 N
        N = 1 if size is None else np.prod(size)
        # 保存当前的随机数生成器状态
        prev_random_state = self.numpy_rng
        # 如果指定了 random_state，设置新的随机数生成器状态
        if random_state is not None:
            self.set_random_state(random_state)
        # 检查分布是否是连续分布
        if unur_distr_is_cont(unur_get_distr(self.rng)):
            # 分配内存给连续变量数组
            out_cont = np.empty(N, dtype=np.float64)
            # 调用连续变量生成函数
            self._rvs_cont(out_cont)
            # 如果设置了 random_state，则恢复之前的随机数生成器状态
            if random_state is not None:
                self.set_random_state(prev_random_state)
            # 如果 size 为 None，则返回单个值
            if size is None:
                return out_cont[0]
            # 返回数组，并根据 size 调整形状
            return np.asarray(out_cont).reshape(size)
        # 检查分布是否是离散分布
        elif unur_distr_is_discr(unur_get_distr(self.rng)):
            # 分配内存给离散变量数组
            out_discr = np.empty(N, dtype=np.int32)
            # 调用离散变量生成函数
            self._rvs_discr(out_discr)
            # 如果设置了 random_state，则恢复之前的随机数生成器状态
            if random_state is not None:
                self.set_random_state(prev_random_state)
            # 如果 size 为 None，则返回单个值
            if size is None:
                return out_discr[0]
            # 返回数组，并根据 size 调整形状
            return np.asarray(out_discr).reshape(size)
        else:
            # 如果分布既不是连续也不是离散，则抛出异常
            raise NotImplementedError("only univariate continuous and "
                                      "discrete distributions supported")
    # 设置随机状态的方法，用于设定底层的均匀随机数生成器。
    def set_random_state(self, random_state=None):
        # 调用函数获取适当的 NumPy 随机数生成器对象
        self.numpy_rng = get_numpy_rng(random_state)
        # 获取锁，确保线程安全操作
        _lock.acquire()
        try:
            # 清空消息队列
            self._messages.clear()
            # 设置 UNURAN 库的随机数流
            unur_set_stream(self._messages.handle)
            # 释放当前的 URNG 对象
            unur_urng_free(self.urng)
            # 创建新的 URNG 对象，基于 NumPy 随机数生成器
            self._urng_builder = _URNG(self.numpy_rng)
            self.urng = self._urng_builder.get_urng()
            # 如果 URNG 为 NULL，则抛出 UNURAN 错误
            if self.urng == NULL:
                raise UNURANError(self._messages.get())
            # 将 URNG 对象应用到 RNG 对象中
            unur_chg_urng(self.rng, self.urng)
        finally:
            # 释放锁，结束线程安全操作
            _lock.release()

    # 对象销毁方法，在 Cython 中定义
    @cython.final
    def __dealloc__(self):
        # 如果分布对象不为空，则释放其内存
        if self.distr != NULL:
            unur_distr_free(self.distr)
            self.distr = NULL
        # 如果参数对象不为空，则释放其内存
        if self.par != NULL:
            unur_par_free(self.par)
            self.par = NULL
        # 如果 RNG 对象不为空，则释放其内存
        if self.rng != NULL:
            unur_free(self.rng)
            self.rng = NULL
        # 如果 URNG 对象不为空，则释放其内存
        if self.urng != NULL:
            unur_urng_free(self.urng)
            self.urng = NULL

    # 支持 Pickling（序列化）的方法，在 Cython 中定义
    @cython.final
    def __reduce__(self):
        # 使用 functools.partial 函数创建类的实例
        klass = functools.partial(self.__class__, **self._kwargs)
        # 返回用于 Pickling 的元组，这里没有需要传递的参数
        return (klass, ())
cdef class TransformedDensityRejection(Method):
    r"""
    TransformedDensityRejection(dist, *, mode=None, center=None, domain=None, c=-0.5, construction_points=30, use_dars=True, max_squeeze_hat_ratio=0.99, random_state=None)

    Transformed Density Rejection (TDR) Method.

    TDR is an acceptance/rejection method that uses the concavity of a
    transformed density to construct hat function and squeezes automatically.
    Most universal algorithms are very slow compared to algorithms that are
    specialized to that distribution. Algorithms that are fast have a slow
    setup and require large tables. The aim of this universal method is to
    provide an algorithm that is not too slow and needs only a short setup.
    This method can be applied to univariate and unimodal continuous
    distributions with T-concave density function. See [1]_ and [2]_ for
    more details.

    Parameters
    ----------
    dist : object
        An instance of a class with ``pdf`` and ``dpdf`` methods.

        * ``pdf``: PDF of the distribution. The signature of the PDF is
          expected to be: ``def pdf(self, x: float) -> float``. i.e.
          the PDF should accept a Python float and
          return a Python float. It doesn't need to integrate to 1 i.e.
          the PDF doesn't need to be normalized.
        * ``dpdf``: Derivative of the PDF w.r.t x (i.e. the variate). Must
          have the same signature as the PDF.

    mode : float, optional
        (Exact) Mode of the distribution. Default is ``None``.
    center : float, optional
        Approximate location of the mode or the mean of the distribution.
        This location provides some information about the main part of the
        PDF and is used to avoid numerical problems. Default is ``None``.
    domain : list or tuple of length 2, optional
        The support of the distribution.
        Default is ``None``. When ``None``:

        * If a ``support`` method is provided by the distribution object
          `dist`, it is used to set the domain of the distribution.
        * Otherwise the support is assumed to be :math:`(-\infty, \infty)`.

    c : {-0.5, 0.}, optional
        Set parameter ``c`` for the transformation function ``T``. The
        default is -0.5. The transformation of the PDF must be concave in
        order to construct the hat function. Such a PDF is called T-concave.
        Currently the following transformations are supported:

        .. math::

            c = 0.: T(x) &= \log(x)\\
            c = -0.5: T(x) &= \frac{1}{\sqrt{x}} \text{ (Default)}

    construction_points : int or array_like, optional
        If an integer, it defines the number of construction points. If it
        is array-like, the elements of the array are used as construction
        points. Default is 30.
"""
    # 是否使用“derandomized adaptive rejection sampling” (DARS) 算法的标志。如果为 True，将在设置中使用DARS算法。
    # 参见引用 [1]_ 获取关于DARS算法的详细信息。默认为 True。
    use_dars : bool, optional

    # 设置挤压区域下方面积与帽子函数下方面积比值的上限。必须是介于 0 和 1 之间的数值。默认为 0.99。
    max_squeeze_hat_ratio : float, optional

    # 用于生成均匀随机数流的基础 NumPy 随机数生成器或其种子。
    # 如果 `random_state` 是 None（或 `np.random`），则使用 `numpy.random.RandomState` 单例对象。
    # 如果 `random_state` 是一个整数，则使用该整数种子生成一个新的 `RandomState` 实例。
    # 如果 `random_state` 已经是 `Generator` 或 `RandomState` 实例，则直接使用该实例。
    random_state : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
    # 定义 Cython 类型扩展中的 `support` 方法，返回固定的元组 (-1, 1)
    def support(self):
        return (-1, 1)

    # 创建 MyDist 类的实例 dist
    dist = MyDist()
    # 使用 TransformedDensityRejection 类初始化 rng 对象，使用指定的随机种子 urng
    rng = TransformedDensityRejection(dist, random_state=urng)

    # 现在可以使用 `rvs` 方法从分布中生成样本
    rvs = rng.rvs(1000)

    # 可以通过绘制直方图来验证生成的样本是否符合给定分布

    # 导入 matplotlib.pyplot 库
    import matplotlib.pyplot as plt

    # 创建一个包含 1000 个点的均匀间隔的数组 x，范围从 -1 到 1
    x = np.linspace(-1, 1, 1000)
    # 计算真实分布的概率密度函数值 fx，3/4 是归一化常数
    fx = 3/4 * dist.pdf(x)
    # 绘制真实分布曲线
    plt.plot(x, fx, 'r-', lw=2, label='true distribution')
    # 绘制生成样本的直方图
    plt.hist(rvs, bins=20, density=True, alpha=0.8, label='random variates')
    # 设置 x 轴标签
    plt.xlabel('x')
    # 设置 y 轴标签
    plt.ylabel('PDF(x)')
    # 设置图表标题
    plt.title('Transformed Density Rejection Samples')
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()
    """
    # 声明一个 Cython 双精度数组 construction_points_array

    # 定义 _validate_args 方法，验证参数 dist、domain、c、construction_points，并返回验证后的 domain、c、construction_points

        # 调用 _validate_domain 函数验证 domain，确保在给定的分布 dist 中有效
        domain = _validate_domain(domain, dist)

        # 检查 c 是否在 {-0.5, 0.} 中，若不是则引发 ValueError 异常
        if c not in {-0.5, 0.}:
            raise ValueError("`c` must either be -0.5 or 0.")

        # 如果 construction_points 不是标量，将其转换为连续的 float64 类型的数组，存储在 construction_points_array 中
        if not np.isscalar(construction_points):
            self.construction_points_array = np.ascontiguousarray(construction_points,
                                                                  dtype=np.float64)
            # 如果 construction_points_array 长度为 0，则引发 ValueError 异常
            if len(self.construction_points_array) == 0:
                raise ValueError("`construction_points` must either be a scalar or a "
                                 "non-empty array.")
        else:
            # 否则将 construction_points_array 设为 None
            self.construction_points_array = None
            # 如果 construction_points 不是正整数或小于等于 0，则引发 ValueError 异常
            if (construction_points <= 0 or
                construction_points != int(construction_points)):
                raise ValueError("`construction_points` must be a positive integer.")

        # 返回经验证后的 domain、c、construction_points

    # 定义 squeeze_hat_ratio 属性，返回当前生成器的挤压下方面积与帽子下方面积的比率
    @property
    def squeeze_hat_ratio(self):
        return unur_tdr_get_sqhratio(self.rng)

    # 定义 hat_area 属性，返回生成器帽子下方的面积
    @property
    def hat_area(self):
        return unur_tdr_get_hatarea(self.rng)

    # 定义 squeeze_area 属性，返回生成器挤压下方的面积
    @property
    def squeeze_area(self):
        return unur_tdr_get_squeezearea(self.rng)

    # 使用 Cython 的 boundscheck(False) 和 wraparound(False) 修饰符定义 _ppf_hat 内联函数
    cdef inline void _ppf_hat(self, const double *u, double *out, size_t N) except *:
        cdef:
            size_t i
        # 对于给定的 N，使用 unur_tdr_eval_invcdfhat 函数计算逆累积分布函数的帽子部分
        for i in range(N):
            out[i] = unur_tdr_eval_invcdfhat(self.rng, u[i], NULL, NULL, NULL)
    def ppf_hat(self, u):
        """
        ppf_hat(u)

        Evaluate the inverse of the CDF of the hat distribution at `u`.

        Parameters
        ----------
        u : array_like
            An array of percentiles

        Returns
        -------
        ppf_hat : array_like
            Array of quantiles corresponding to the given percentiles.

        Examples
        --------
        >>> from scipy.stats.sampling import TransformedDensityRejection
        >>> from scipy.stats import norm
        >>> import numpy as np
        >>> from math import exp
        >>>
        >>> class MyDist:
        ...     def pdf(self, x):
        ...         return exp(-0.5 * x**2)
        ...     def dpdf(self, x):
        ...         return -x * exp(-0.5 * x**2)
        ...
        >>> dist = MyDist()
        >>> rng = TransformedDensityRejection(dist)
        >>>
        >>> rng.ppf_hat(0.5)
        -0.00018050266342393984
        >>> norm.ppf(0.5)
        0.0
        >>> u = np.linspace(0, 1, num=1000)
        >>> ppf_hat = rng.ppf_hat(u)
        """
        # 将输入参数 u 转换为 NumPy 数组，并指定数据类型为双精度浮点数
        u = np.asarray(u, dtype='d')
        # 记录原始 u 的形状
        oshape = u.shape
        # 将 u 展平为一维数组
        u = u.ravel()
        
        # 以下代码段处理 u 的有效性范围，确保在 [0, 1] 之间的值有效，超出范围的值填充为 NaN
        # UNU.RAN 填充 u < 0 或 u > 1 的支持端点，而 SciPy 使用 NaN。这里选择了 SciPy 的行为。
        
        # 创建条件以筛选有效的 u 值
        cond0 = 0 <= u
        cond1 = u <= 1
        cond2 = cond0 & cond1
        # 使用 argsreduce 函数获取有效的 u 值
        goodu = argsreduce(cond2, u)[0]
        
        # 初始化输出数组，形状与 u 相同
        out = np.empty_like(u)
        
        # 将 goodu 转换为连续的双精度浮点数视图
        cdef double[::1] u_view = np.ascontiguousarray(goodu)
        cdef double[::1] goodout = np.empty_like(u_view)
        
        # 如果存在有效的 u 值，调用 _ppf_hat 方法计算对应的输出
        if cond2.any():
            self._ppf_hat(&u_view[0], &goodout[0], len(goodu))
        
        # 根据条件 cond2，将计算得到的 goodout 放入 out 数组
        np.place(out, cond2, goodout)
        # 将超出范围的 u 值位置填充为 NaN
        np.place(out, ~cond2, np.nan)
        
        # 将 out 数组重新形状为 oshape，并返回一个标量（由于 oshape 是 ()，所以用 [()] 来访问）
        return np.asarray(out).reshape(oshape)[()]
# 定义一个简单的比例均匀分布方法的类，继承自 Method 类
cdef class SimpleRatioUniforms(Method):
    r"""
    SimpleRatioUniforms(dist, *, mode=None, pdf_area=1, domain=None, cdf_at_mode=None, random_state=None)
    
    Simple Ratio-of-Uniforms (SROU) Method.
    
    SROU is based on the ratio-of-uniforms method that uses universal inequalities for
    constructing a (universal) bounding rectangle. It works for T-concave distributions
    with ``T(x) = -1/sqrt(x)``. The main advantage of the method is a fast setup. This
    can be beneficial if one repeatedly needs to generate small to moderate samples of
    a distribution with different shape parameters. In such a situation, the setup step of
    `NumericalInverseHermite` or `NumericalInversePolynomial` will lead to poor performance.
    
    Parameters
    ----------
    dist : object
        An instance of a class with ``pdf`` method.
        
        * ``pdf``: PDF of the distribution. The signature of the PDF is
          expected to be: ``def pdf(self, x: float) -> float``. i.e.
          the PDF should accept a Python float and
          return a Python float. It doesn't need to integrate to 1 i.e.
          the PDF doesn't need to be normalized. If not normalized, `pdf_area`
          should be set to the area under the PDF.
    
    mode : float, optional
        (Exact) Mode of the distribution. When the mode is ``None``, a slow
        numerical routine is used to approximate it. Default is ``None``.
    pdf_area : float, optional
        Area under the PDF. Optionally, an upper bound to the area under
        the PDF can be passed at the cost of increased rejection constant.
        Default is 1.
    domain : list or tuple of length 2, optional
        The support of the distribution.
        Default is ``None``. When ``None``:
        
        * If a ``support`` method is provided by the distribution object
          `dist`, it is used to set the domain of the distribution.
        * Otherwise the support is assumed to be :math:`(-\infty, \infty)`.
    
    cdf_at_mode : float, optional
        CDF at the mode. It can be given to increase the performance of the
        algorithm. The rejection constant is halfed when CDF at mode is given.
        Default is ``None``.
    random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional
    
        A NumPy random number generator or seed for the underlying NumPy random
        number generator used to generate the stream of uniform random numbers.
        If `random_state` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    
    References
    ----------
    ```
    # 引用文献 [1] UNU.RAN 参考手册，第 5.3.16 节，“SROU - 简单比例均匀方法”，详见：http://statmath.wu.ac.at/software/unuran/doc/unuran.html#SROU
    # 引用文献 [2] Leydold, Josef. "A simple universal generator for continuous and
    #           discrete univariate T-concave distributions." ACM Transactions on
    #           Mathematical Software (TOMS) 27.1 (2001): 66-82
    # 引用文献 [3] Leydold, Josef. "Short universal generators via generalized ratio-of-uniforms
    #           method." Mathematics of Computation 72.243 (2003): 1453-1471
    
    Examples
    --------
    >>> from scipy.stats.sampling import SimpleRatioUniforms
    >>> import numpy as np
    
    假设我们有一个正态分布：
    
    >>> class StdNorm:
    ...     def pdf(self, x):
    ...         return np.exp(-0.5 * x**2)
    
    注意，PDF 不一定要积分为1。在生成器初始化时，可以传递精确的PDF下的区域或PDF下的上界。同时，建议传递分布的模式来加快设置过程：
    
    >>> urng = np.random.default_rng()
    >>> dist = StdNorm()
    >>> rng = SimpleRatioUniforms(dist, mode=0,
    ...                           pdf_area=np.sqrt(2*np.pi),
    ...                           random_state=urng)
    
    现在，我们可以使用 `rvs` 方法从分布中生成样本：
    
    >>> rvs = rng.rvs(10)
    
    如果模式下的CDF可用，可以设置它以提高 `rvs` 的性能：
    
    >>> from scipy.stats import norm
    >>> rng = SimpleRatioUniforms(dist, mode=0,
    ...                           pdf_area=np.sqrt(2*np.pi),
    ...                           cdf_at_mode=norm.cdf(0),
    ...                           random_state=urng)
    >>> rvs = rng.rvs(1000)
    
    我们可以通过可视化其直方图来检查样本是否来自给定的分布：
    
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(rvs.min()-0.1, rvs.max()+0.1, 1000)
    >>> fx = 1/np.sqrt(2*np.pi) * dist.pdf(x)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, fx, 'r-', lw=2, label='true distribution')
    >>> ax.hist(rvs, bins=10, density=True, alpha=0.8, label='random variates')
    >>> ax.set_xlabel('x')
    >>> ax.set_ylabel('PDF(x)')
    >>> ax.set_title('Simple Ratio-of-Uniforms Samples')
    >>> ax.legend()
    >>> plt.show()
    # 初始化函数，用于设置连续分布的参数
    def __cinit__(self,
                  dist,
                  *,
                  mode=None,
                  pdf_area=1,
                  domain=None,
                  cdf_at_mode=None,
                  random_state=None):
        # 调用_validate_args方法验证参数，并返回有效的domain和pdf_area
        (domain, pdf_area) = self._validate_args(dist, domain, pdf_area)

        # 将所有参数保存到_kwargs字典，以支持pickle序列化
        self._kwargs = {
            'dist': dist,
            'mode': mode,
            'pdf_area': pdf_area,
            'domain': domain,
            'cdf_at_mode': cdf_at_mode,
            'random_state': random_state
        }

        # 解析分布对象，获取对应的回调函数，存储到callbacks属性中
        self.callbacks = _unpack_dist(dist, "cont", meths=["pdf"])

        # 定义一个回调函数的包装器，用于调用存储在self.callbacks中的具体函数
        def _callback_wrapper(x, name):
            return self.callbacks[name](x)
        
        self._callback_wrapper = _callback_wrapper

        # 初始化消息流对象，用于记录和处理消息
        self._messages = MessageStream()

        # 获取全局锁，确保线程安全操作
        _lock.acquire()
        try:
            # 设置消息处理器，将消息流的句柄传递给UNUR库
            unur_set_stream(self._messages.handle)

            # 创建连续分布对象
            self.distr = unur_distr_cont_new()
            if self.distr == NULL:
                # 如果创建失败，抛出UNURANError异常，并传递错误消息
                raise UNURANError(self._messages.get())

            # 将分布对象打包，设置其回调函数
            _pack_distr(self.distr, self.callbacks)

            # 如果指定了domain，设置分布的定义域
            if domain is not None:
                self._check_errorcode(unur_distr_cont_set_domain(self.distr, domain[0],
                                                                 domain[1]))

            # 如果指定了mode，设置分布的模式
            if mode is not None:
                self._check_errorcode(unur_distr_cont_set_mode(self.distr, mode))

            # 设置分布的概率密度函数区域
            self._check_errorcode(unur_distr_cont_set_pdfarea(self.distr, pdf_area))

            # 根据分布对象创建随机数生成器对象
            self.par = unur_srou_new(self.distr)
            if self.par == NULL:
                # 如果创建失败，抛出UNURANError异常，并传递错误消息
                raise UNURANError(self._messages.get())

            # 如果指定了cdf_at_mode，设置生成器对象的模式处累积分布函数值
            if cdf_at_mode is not None:
                self._check_errorcode(unur_srou_set_cdfatmode(self.par, cdf_at_mode))
                # 当给定cdf_at_mode时，始终使用squeeze以提高性能
                self._check_errorcode(unur_srou_set_usesqueeze(self.par, True))

            # 设置随机数生成器的种子
            self._set_rng(random_state)
        finally:
            # 释放全局锁
            _lock.release()

    # 定义私有方法_validate_args，用于验证参数合法性并返回有效的domain和pdf_area
    cdef object _validate_args(self, dist, domain, pdf_area):
        # 调用_validate_domain方法验证domain是否有效
        domain = _validate_domain(domain, dist)
        # 如果pdf_area小于0，则引发ValueError异常
        if pdf_area < 0:
            raise ValueError("`pdf_area` must be > 0")
        # 返回验证后的domain和pdf_area
        return domain, pdf_area
UError = namedtuple('UError', ['max_error', 'mean_absolute_error'])
# 定义命名元组 UError，包含两个字段：max_error 和 mean_absolute_error

cdef class NumericalInversePolynomial(Method):
    """
    NumericalInversePolynomial(dist, *, mode=None, center=None, domain=None, order=5, u_resolution=1e-10, random_state=None)

    Polynomial interpolation based INVersion of CDF (PINV).

    PINV is a variant of numerical inversion, where the inverse CDF is approximated
    using Newton's interpolating formula. The interval ``[0,1]`` is split into several
    subintervals. In each of these, the inverse CDF is constructed at nodes ``(CDF(x),x)``
    for some points ``x`` in this subinterval. If the PDF is given, then the CDF is
    computed numerically from the given PDF using adaptive Gauss-Lobatto integration with
    5 points. Subintervals are split until the requested accuracy goal is reached.

    The method is not exact, as it only produces random variates of the approximated
    distribution. Nevertheless, the maximal tolerated approximation error can be set to
    be the resolution (but, of course, is bounded by the machine precision). We use the
    u-error ``|U - CDF(X)|`` to measure the error where ``X`` is the approximate
    percentile corresponding to the quantile ``U`` i.e. ``X = approx_ppf(U)``. We call
    the maximal tolerated u-error the u-resolution of the algorithm.

    Both the order of the interpolating polynomial and the u-resolution can be selected.
    Note that very small values of the u-resolution are possible but increase the cost
    for the setup step.

    The interpolating polynomials have to be computed in a setup step. However, it only
    works for distributions with bounded domain; for distributions with unbounded domain
    the tails are cut off such that the probability for the tail regions is small compared
    to the given u-resolution.

    The construction of the interpolation polynomial only works when the PDF is unimodal
    or when the PDF does not vanish between two modes.

    There are some restrictions for the given distribution:

    * The support of the distribution (i.e., the region where the PDF is strictly
      positive) must be connected. In practice this means, that the region where PDF
      is "not too small" must be connected. Unimodal densities satisfy this condition.
      If this condition is violated then the domain of the distribution might be
      truncated.
    * When the PDF is integrated numerically, then the given PDF must be continuous
      and should be smooth.
    * The PDF must be bounded.
    * The algorithm has problems when the distribution has heavy tails (as then the
      inverse CDF becomes very steep at 0 or 1) and the requested u-resolution is
      very small. E.g., the Cauchy distribution is likely to show this problem when
      the requested u-resolution is less then 1.e-12.


    Parameters
    ----------

    """
    # 定义一个 Cython 类 NumericalInversePolynomial，继承自 Method

    def __init__(self, dist, *, mode=None, center=None, domain=None, order=5, u_resolution=1e-10, random_state=None):
        # 类的初始化方法，接受多个参数来配置数值逆多项式插值

        # 父类的初始化方法
        super().__init__()

        # 存储各种配置参数
        self.dist = dist
        self.mode = mode
        self.center = center
        self.domain = domain
        self.order = order
        self.u_resolution = u_resolution
        self.random_state = random_state
    # dist : object
    #     An instance of a class with a ``pdf`` or ``logpdf`` method,
    #     optionally a ``cdf`` method.
    #
    #     * ``pdf``: PDF of the distribution. The signature of the PDF is expected to be:
    #       ``def pdf(self, x: float) -> float``, i.e., the PDF should accept a Python
    #       float and return a Python float. It doesn't need to integrate to 1,
    #       i.e., the PDF doesn't need to be normalized. This method is optional,
    #       but either ``pdf`` or ``logpdf`` need to be specified. If both are given,
    #       ``logpdf`` is used.
    #     * ``logpdf``: The log of the PDF of the distribution. The signature is
    #       the same as for ``pdf``. Similarly, log of the normalization constant
    #       of the PDF can be ignored. This method is optional, but either ``pdf`` or
    #       ``logpdf`` need to be specified. If both are given, ``logpdf`` is used.
    #     * ``cdf``: CDF of the distribution. This method is optional. If provided, it
    #       enables the calculation of "u-error". See `u_error`. Must have the same
    #       signature as the PDF.
    #
    # mode : float, optional
    #     (Exact) Mode of the distribution. Default is ``None``.
    # center : float, optional
    #     Approximate location of the mode or the mean of the distribution. This location
    #     provides some information about the main part of the PDF and is used to avoid
    #     numerical problems. Default is ``None``.
    # domain : list or tuple of length 2, optional
    #     The support of the distribution.
    #     Default is ``None``. When ``None``:
    #     * If a ``support`` method is provided by the distribution object
    #       `dist`, it is used to set the domain of the distribution.
    #     * Otherwise the support is assumed to be :math:`(-\infty, \infty)`.
    #
    # order : int, optional
    #     Order of the interpolating polynomial. Valid orders are between 3 and 17.
    #     Higher orders result in fewer intervals for the approximations. Default
    #     is 5.
    # u_resolution : float, optional
    #     Set maximal tolerated u-error. Values of u_resolution must at least 1.e-15 and
    #     1.e-5 at most. Notice that the resolution of most uniform random number sources
    #     is 2-32= 2.3e-10. Thus a value of 1.e-10 leads to an inversion algorithm that
    #     could be called exact. For most simulations slightly bigger values for the
    #     maximal error are enough as well. Default is 1e-10.
    random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional
    # 参数 `random_state`：随机数生成器的种子或者 `numpy.random.Generator` 或 `numpy.random.RandomState` 的实例
        A NumPy random number generator or seed for the underlying NumPy random
        number generator used to generate the stream of uniform random numbers.
        如果 `random_state` 为 None（或 `np.random`），则使用 `numpy.random.RandomState` 单例。
        如果 `random_state` 是一个整数，则使用一个新的 `RandomState` 实例，并使用该整数作为种子。
        如果 `random_state` 已经是 `Generator` 或 `RandomState` 的实例，则直接使用该实例。

    References
    ----------
    .. [1] Derflinger, Gerhard, Wolfgang Hörmann, and Josef Leydold. "Random variate
           generation by numerical inversion when only the density is known." ACM
           Transactions on Modeling and Computer Simulation (TOMACS) 20.4 (2010): 1-25.
    .. [2] UNU.RAN reference manual, Section 5.3.12,
           "PINV - Polynomial interpolation based INVersion of CDF",
           https://statmath.wu.ac.at/software/unuran/doc/unuran.html#PINV

    Examples
    --------
    >>> from scipy.stats.sampling import NumericalInversePolynomial
    >>> from scipy.stats import norm
    >>> import numpy as np

    To create a generator to sample from the standard normal distribution, do:

    >>> class StandardNormal:
    ...    def pdf(self, x):
    ...        return np.exp(-0.5 * x*x)
    ...
    >>> dist = StandardNormal()
    >>> urng = np.random.default_rng()
    >>> rng = NumericalInversePolynomial(dist, random_state=urng)

    Once a generator is created, samples can be drawn from the distribution by calling
    the `rvs` method:

    >>> rng.rvs()
    -1.5244996276336318

    To check that the random variates closely follow the given distribution, we can
    look at it's histogram:

    >>> import matplotlib.pyplot as plt
    >>> rvs = rng.rvs(10000)
    >>> x = np.linspace(rvs.min()-0.1, rvs.max()+0.1, 1000)
    >>> fx = norm.pdf(x)
    >>> plt.plot(x, fx, 'r-', lw=2, label='true distribution')
    >>> plt.hist(rvs, bins=20, density=True, alpha=0.8, label='random variates')
    >>> plt.xlabel('x')
    >>> plt.ylabel('PDF(x)')
    >>> plt.title('Numerical Inverse Polynomial Samples')
    >>> plt.legend()
    >>> plt.show()

    It is possible to estimate the u-error of the approximated PPF if the exact
    CDF is available during setup. To do so, pass a `dist` object with exact CDF of
    the distribution during initialization:

    >>> from scipy.special import ndtr
    >>> class StandardNormal:
    ...    def pdf(self, x):
    ...        return np.exp(-0.5 * x*x)
    ...    def cdf(self, x):
    ...        return ndtr(x)
    ...
    >>> dist = StandardNormal()
    >>> urng = np.random.default_rng()
    >>> rng = NumericalInversePolynomial(dist, random_state=urng)

    Now, the u-error can be estimated by calling the `u_error` method. It runs a
    Monte-Carlo simulation to estimate the u-error. By default, 100000 samples are
    used. To change this, you can pass the number of samples as an argument:

    >>> rng.u_error(sample_size=1000000)  # uses one million samples
    UError(max_error=8.785994154436594e-11, mean_absolute_error=2.930890027826552e-11)

    This returns a namedtuple which contains the maximum u-error and the mean
    absolute u-error.

    The u-error can be reduced by decreasing the u-resolution (maximum allowed u-error):

    >>> urng = np.random.default_rng()
    >>> rng = NumericalInversePolynomial(dist, u_resolution=1.e-12, random_state=urng)
    >>> rng.u_error(sample_size=1000000)
    UError(max_error=9.07496300328603e-13, mean_absolute_error=3.5255644517257716e-13)

    Note that this comes at the cost of increased setup time.

    The approximated PPF can be evaluated by calling the `ppf` method:

    >>> rng.ppf(0.975)
    1.9599639857012559
    >>> norm.ppf(0.975)
    1.959963984540054

    Since the PPF of the normal distribution is available as a special function, we
    can also check the x-error, i.e. the difference between the approximated PPF and
    exact PPF::

    >>> import matplotlib.pyplot as plt
    >>> u = np.linspace(0.01, 0.99, 1000)
    >>> approxppf = rng.ppf(u)
    >>> exactppf = norm.ppf(u)
    >>> error = np.abs(exactppf - approxppf)
    >>> plt.plot(u, error)
    >>> plt.xlabel('u')
    >>> plt.ylabel('error')
    >>> plt.title('Error between exact and approximated PPF (x-error)')
    >>> plt.show()
    # 初始化方法，设置分布参数和选项
    def __cinit__(self,
                  dist,
                  *,
                  mode=None,
                  center=None,
                  domain=None,
                  order=5,
                  u_resolution=1e-10,
                  random_state=None):
        # 调用私有方法验证参数，并返回验证后的 domain, order, u_resolution
        (domain, order, u_resolution) = self._validate_args(
            dist, domain, order, u_resolution
        )

        # 保存所有参数以支持序列化
        self._kwargs = {
            'dist': dist,
            'center': center,
            'domain': domain,
            'order': order,
            'u_resolution': u_resolution,
            'random_state': random_state
        }

        # 解包分布对象的回调函数，必须包含 'pdf' 或 'logpdf' 方法
        self.callbacks = _unpack_dist(dist, "cont", meths=None, optional_meths=["cdf", "pdf", "logpdf"])
        if not ("pdf" in self.callbacks or "logpdf" in self.callbacks):
            # 如果没有找到 'pdf' 或 'logpdf' 方法，则抛出 ValueError 异常
            msg = ("Either of the methods `pdf` or `logpdf` must be specified "
                   "for the distribution object `dist`.")
            raise ValueError(msg)

        # 定义回调函数的包装器，用于调用具体的分布方法
        def _callback_wrapper(x, name):
            return self.callbacks[name](x)
        self._callback_wrapper = _callback_wrapper

        # 初始化消息流对象
        self._messages = MessageStream()

        # 获取全局锁对象，确保线程安全操作
        _lock.acquire()
        try:
            # 将消息流对象设置给 UNU.RAN 库处理
            unur_set_stream(self._messages.handle)

            # 创建连续分布对象
            self.distr = unur_distr_cont_new()
            if self.distr == NULL:
                # 如果分布对象创建失败，抛出 UNURANError 异常并传递错误消息
                raise UNURANError(self._messages.get())

            # 将回调函数打包并设置给 UNU.RAN 分布对象
            _pack_distr(self.distr, self.callbacks)

            # 如果提供了 domain 参数，则设置分布的定义域
            if domain is not None:
                self._check_errorcode(unur_distr_cont_set_domain(self.distr, domain[0],
                                                                 domain[1]))

            # 如果提供了 mode 参数，则设置分布的模式
            if mode is not None:
                self._check_errorcode(unur_distr_cont_set_mode(self.distr, mode))

            # 如果提供了 center 参数，则设置分布的中心
            if center is not None:
                self._check_errorcode(unur_distr_cont_set_center(self.distr, center))

            # 创建逆变换对象
            self.par = unur_pinv_new(self.distr)
            if self.par == NULL:
                # 如果逆变换对象创建失败，抛出 UNURANError 异常并传递错误消息
                raise UNURANError(self._messages.get())

            # 设置逆变换对象的阶数
            self._check_errorcode(unur_pinv_set_order(self.par, order))
            # 设置逆变换对象的分辨率
            self._check_errorcode(unur_pinv_set_u_resolution(self.par, u_resolution))
            # 设置逆变换对象的最大区间数，这里设置为 UNU.RAN 允许的最大值 1,000,000
            self._check_errorcode(unur_pinv_set_max_intervals(self.par, 1000000))
            # 在 SciPy 中始终保持 CDF，而 UNU.RAN 的默认值为 False
            self._check_errorcode(unur_pinv_set_keepcdf(self.par, 1))

            # 设置随机数生成器的状态
            self._set_rng(random_state)
        finally:
            # 释放全局锁对象
            _lock.release()
    # 定义一个Cython函数来验证参数，并返回有效的域、阶数和u分辨率
    cdef object _validate_args(self, dist, domain, order, u_resolution):
        # 使用_validate_domain函数验证域的有效性
        domain = _validate_domain(domain, dist)
        
        # 检查阶数order是否在 [3, 17] 的范围内且为整数，否则引发错误
        if not (3 <= order <= 17 and int(order) == order):
            raise ValueError("`order` must be an integer in the range [3, 17].")
        
        # 检查u分辨率u_resolution是否在 [1e-15, 1e-5] 的范围内，否则引发错误
        if not (1e-15 <= u_resolution <= 1e-5):
            raise ValueError("`u_resolution` must be between 1e-15 and 1e-5.")
        
        # 返回验证后的域、阶数和u分辨率
        return (domain, order, u_resolution)

    @property
    def intervals(self):
        """
        获取在计算中使用的区间数。
        """
        return unur_pinv_get_n_intervals(self.rng)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void _cdf(self, const double *x, double *out, size_t N) except *:
        cdef:
            size_t i  # 定义循环计数器
            ccallback_t callback  # 回调函数类型
            PyObject *type  # 异常类型
            PyObject *value  # 异常值
            PyObject *traceback  # 异常追踪信息

        error = 0  # 初始化错误标志

        _lock.acquire()  # 获取全局锁
        try:
            self._messages.clear()  # 清空消息列表
            unur_set_stream(self._messages.handle)  # 设置UNU.RAN的消息流
            init_unuran_callback(&callback, self._callback_wrapper)  # 初始化回调函数
            for i in range(N):
                out[i] = unur_pinv_eval_approxcdf(self.rng, x[i])  # 计算逼近的累积分布函数值
                if PyErr_Occurred():  # 检查是否发生了Python异常
                    error = 1  # 设置错误标志
                    return  # 返回
                if out[i] == UNUR_INFINITY or out[i] == -UNUR_INFINITY:
                    raise UNURANError(self._messages.get())  # 如果出现无穷大值，引发UNU.RAN错误
        finally:
            if error:
                PyErr_Fetch(&type, &value, &traceback)  # 捕获Python异常信息
            _lock.release()  # 释放全局锁
            if error:
                PyErr_Restore(type, value, traceback)  # 恢复Python异常状态
            release_unuran_callback(&callback)  # 释放UNU.RAN回调函数资源

    def cdf(self, x):
        """
        计算给定分布的逼近累积分布函数。

        Parameters
        ----------
        x : array_like
            分位数，其中`x`的最后一个轴表示组件。

        Returns
        -------
        cdf : array_like
            在`x`处评估的逼近累积分布函数。
        """
        x = np.asarray(x, dtype='d')  # 将输入转换为双精度数组
        oshape = x.shape  # 记录原始形状
        x = x.ravel()  # 展平输入数组
        cdef double[::1] x_view = np.ascontiguousarray(x)  # 创建Cython视图以优化内存布局
        cdef double[::1] out = np.empty_like(x)  # 创建输出数组
        if x.size == 0:
            return np.asarray(out).reshape(oshape)  # 如果输入为空，返回空数组
        self._cdf(&x_view[0], &out[0], len(x_view))  # 调用Cython函数计算累积分布函数
        return np.asarray(out).reshape(oshape)[()]  # 将结果转换为原始形状并返回

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void _ppf(self, const double *u, double *out, size_t N) noexcept:
        cdef:
            size_t i  # 定义循环计数器
        for i in range(N):
            out[i] = unur_pinv_eval_approxinvcdf(self.rng, u[i])  # 计算逼近的反累积分布函数值
    def ppf(self, u):
        """
        ppf(u)

        Approximated PPF of the given distribution.

        Parameters
        ----------
        u : array_like
            Quantiles.

        Returns
        -------
        ppf : array_like
            Percentiles corresponding to given quantiles `u`.
        """
        # 将输入的 `u` 转换为双精度浮点数的 NumPy 数组
        u = np.asarray(u, dtype='d')
        # 保存原始 `u` 的形状
        oshape = u.shape
        # 将 `u` 展平为一维数组
        u = u.ravel()
        
        # UNU.RAN 填充支持的端点当 `u < 0` 或 `u > 1`，而 SciPy 使用 NaN。优先选择 SciPy 的行为。
        cond0 = 0 <= u
        cond1 = u <= 1
        cond2 = cond0 & cond1
        
        # 调用 argsreduce 函数来获取满足条件 `cond2` 的有效 `u` 值
        goodu = argsreduce(cond2, u)[0]
        
        # 创建一个与 `u` 形状相同的空数组 `out`
        out = np.empty_like(u)
        
        # 使用 Cython 的方式创建连续内存视图
        cdef double[::1] u_view = np.ascontiguousarray(goodu)
        cdef double[::1] goodout = np.empty_like(u_view)
        
        # 如果至少有一个元素满足条件 `cond2`
        if cond2.any():
            # 调用 `_ppf` 方法进行计算
            self._ppf(&u_view[0], &goodout[0], len(goodu))
        
        # 将满足条件 `cond2` 的结果放入 `out` 数组
        np.place(out, cond2, goodout)
        # 将不满足条件 `cond2` 的位置填充为 NaN
        np.place(out, ~cond2, np.nan)
        
        # 将 `out` 数组重新调整为原始 `u` 的形状，并返回其数据
        return np.asarray(out).reshape(oshape)[()]

    def u_error(self, sample_size=100000):
        """
        u_error(sample_size=100000)

        Estimate the u-error of the approximation using Monte Carlo simulation.
        This is only available if the generator was initialized with a `dist`
        object containing the implementation of the exact CDF under `cdf` method.

        Parameters
        ----------
        sample_size : int, optional
            Number of samples to use for the estimation. It must be greater than
            or equal to 1000.

        Returns
        -------
        max_error : float
            Maximum u-error.
        mean_absolute_error : float
            Mean absolute u-error.
        """
        # 如果 `sample_size` 小于 1000，则抛出 ValueError 异常
        if sample_size < 1000:
            raise ValueError("`sample_size` must be greater than or equal to 1000.")
        
        # 如果回调函数列表中不包含 'cdf'，则抛出 ValueError 异常
        if 'cdf' not in self.callbacks:
            raise ValueError("Exact CDF required but not found. Reinitialize the generator "
                             " with a `dist` object that contains a `cdf` method to enable "
                             " the estimation of u-error.")
        
        # 使用 Cython 的方式定义变量 `max_error` 和 `mae`
        cdef double max_error, mae
        # 定义 Cython 回调函数 `callback`
        cdef ccallback_t callback
        
        # 获取全局锁对象并尝试执行以下代码块
        _lock.acquire()
        try:
            # 清空 `_messages` 列表
            self._messages.clear()
            # 设置 UNU.RAN 的消息处理句柄
            unur_set_stream(self._messages.handle)
            # 初始化 Cython 回调函数 `callback`，使用 `_callback_wrapper` 函数包装
            init_unuran_callback(&callback, self._callback_wrapper)
            # 调用 UNU.RAN 中的 `unur_pinv_estimate_error` 函数，并检查返回的错误码
            self._check_errorcode(unur_pinv_estimate_error(self.rng, sample_size,
                                                           &max_error, &mae))
        finally:
            # 释放全局锁对象
            _lock.release()
            # 释放 Cython 回调函数 `callback`
            release_unuran_callback(&callback)
        
        # 返回 `UError` 类的实例，包含 `max_error` 和 `mae` 属性值
        return UError(max_error, mae)
# 定义一个 Cython 类 NumericalInverseHermite，继承自 Method 类
cdef class NumericalInverseHermite(Method):
    """
    NumericalInverseHermite(dist, *, domain=None, order=3, u_resolution=1e-12, construction_points=None, random_state=None)

    Hermite interpolation based INVersion of CDF (HINV).

    HINV is a variant of numerical inversion, where the inverse CDF is approximated using
    Hermite interpolation, i.e., the interval [0,1] is split into several intervals and
    in each interval the inverse CDF is approximated by polynomials constructed by means
    of values of the CDF and PDF at interval boundaries. This makes it possible to improve
    the accuracy by splitting a particular interval without recomputations in unaffected
    intervals. Three types of splines are implemented: linear, cubic, and quintic
    interpolation. For linear interpolation only the CDF is required. Cubic interpolation
    also requires PDF and quintic interpolation PDF and its derivative.

    These splines have to be computed in a setup step. However, it only works for
    distributions with bounded domain; for distributions with unbounded domain the tails
    are chopped off such that the probability for the tail regions is small compared to
    the given u-resolution.

    The method is not exact, as it only produces random variates of the approximated
    distribution. Nevertheless, the maximal numerical error in "u-direction" (i.e.
    ``|U - CDF(X)|`` where ``X`` is the approximate percentile corresponding to the
    quantile ``U`` i.e. ``X = approx_ppf(U)``) can be set to the
    required resolution (within machine precision). Notice that very small values of
    the u-resolution are possible but may increase the cost for the setup step.

    Parameters
    ----------
    dist : object
        An instance of a class with a ``cdf`` and optionally a ``pdf`` and ``dpdf`` method.

        * ``cdf``: CDF of the distribution. The signature of the CDF is expected to be:
          ``def cdf(self, x: float) -> float``. i.e. the CDF should accept a Python
          float and return a Python float.
        * ``pdf``: PDF of the distribution. This method is optional when ``order=1``.
          Must have the same signature as the PDF.
        * ``dpdf``: Derivative of the PDF w.r.t the variate (i.e. ``x``). This method is
          optional with ``order=1`` or ``order=3``. Must have the same signature as the CDF.

    domain : list or tuple of length 2, optional
        The support of the distribution.
        Default is ``None``. When ``None``:

        * If a ``support`` method is provided by the distribution object
          `dist`, it is used to set the domain of the distribution.
        * Otherwise the support is assumed to be :math:`(-\infty, \infty)`.

    """
    # order: int, default: ``3``
    # 设置 Hermite 插值的阶数。有效的阶数为 1、3 和 5。
    # 注意，大于 1 的阶数需要分布的密度，大于 3 的阶数甚至需要密度的导数。
    # 使用阶数 1 会导致大量区间，因此不建议使用。如果 u 方向的最大误差非常小（小于 1.e-10），
    # 推荐使用阶数 5，因为这样可以减少设计点的数量，前提是没有极点或重尾部分。

    # u_resolution: float, default: ``1e-12``
    # 设置最大允许的 u 方向误差。注意，大多数均匀随机数源的分辨率为 2^-32 = 2.3e-10。
    # 因此，1.e-10 的值会导致一个几乎可以称为精确的反演算法。
    # 对于大多数模拟而言，稍大一些的最大误差值也足够了。默认值为 1e-12。

    # construction_points: array_like, optional
    # 设置 Hermite 插值的起始构造点（节点）。由于设置中仅估计可能的最大误差，
    # 可能需要设置一些特殊的设计点来计算 Hermite 插值，以确保最大的 u 方向误差不大于期望值。
    # 这些点是密度不可微分或具有局部极值的点。

    # random_state: {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
    # 用于生成均匀随机数流的 NumPy 随机数生成器或种子。
    # 如果 `random_state` 为 None（或 `np.random`），则使用 `numpy.random.RandomState` 单例。
    # 如果 `random_state` 是一个整数，则使用一个新的 `RandomState` 实例，并以 `random_state` 为种子。
    # 如果 `random_state` 已经是 `Generator` 或 `RandomState` 实例，则直接使用该实例。

    # Notes
    # -----
    # `NumericalInverseHermite` 使用 Hermite 样条近似连续统计分布的累积分布函数（CDF）的逆函数。
    # 可以通过传递 `order` 参数来指定 Hermite 样条的阶数。

    # 如 [1]_ 中描述的，它首先在分布的支持区间内评估分布的概率密度函数（PDF）和累积分布函数（CDF）的网格量化点 `x`。
    # 使用这些结果来拟合 Hermite 样条 `H`，使得 `H(p) == x`，其中 `p` 是与量化点 `x` 对应的百分位数组。
    # 因此，样条近似分布的累积分布函数的逆函数在百分位 `p` 处达到机器精度，但通常在百分位点之间的中点处，
    # 样条的精度不会那么高。

        # p_mid = (p[:-1] + p[1:])/2

    # 因此，根据需要调整量化点的网格，以减少最大误差。
    "u-error"::

        # 计算误差 `u_error`，其定义为 PPF 和 CDF 之间的最大绝对差值
        u_error = np.max(np.abs(dist.cdf(H(p_mid)) - p_mid))

    below the specified tolerance `u_resolution`. Refinement stops when the required
    tolerance is achieved or when the number of mesh intervals after the next
    refinement could exceed the maximum allowed number of intervals, which is
    100000.

    References
    ----------
    .. [1] Hörmann, Wolfgang, and Josef Leydold. "Continuous random variate
           generation by fast numerical inversion." ACM Transactions on
           Modeling and Computer Simulation (TOMACS) 13.4 (2003): 347-362.
    .. [2] UNU.RAN reference manual, Section 5.3.5,
           "HINV - Hermite interpolation based INVersion of CDF",
           https://statmath.wu.ac.at/software/unuran/doc/unuran.html#HINV

    Examples
    --------
    >>> from scipy.stats.sampling import NumericalInverseHermite
    >>> from scipy.stats import norm, genexpon
    >>> from scipy.special import ndtr
    >>> import numpy as np

    To create a generator to sample from the standard normal distribution, do:

    >>> class StandardNormal:
    ...     def pdf(self, x):
    ...        return 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)
    ...     def cdf(self, x):
    ...        return ndtr(x)
    ...
    >>> dist = StandardNormal()
    >>> urng = np.random.default_rng()
    >>> rng = NumericalInverseHermite(dist, random_state=urng)

    The `NumericalInverseHermite` has a method that approximates the PPF of the
    distribution.

    >>> rng = NumericalInverseHermite(dist)
    >>> p = np.linspace(0.01, 0.99, 99) # percentiles from 1% to 99%
    >>> np.allclose(rng.ppf(p), norm.ppf(p))
    True

    Depending on the implementation of the distribution's random sampling
    method, the random variates generated may be nearly identical, given
    the same random state.

    >>> dist = genexpon(9, 16, 3)
    >>> rng = NumericalInverseHermite(dist)
    >>> # `seed` ensures identical random streams are used by each `rvs` method
    >>> seed = 500072020
    >>> rvs1 = dist.rvs(size=100, random_state=np.random.default_rng(seed))
    >>> rvs2 = rng.rvs(size=100, random_state=np.random.default_rng(seed))
    >>> np.allclose(rvs1, rvs2)
    True

    To check that the random variates closely follow the given distribution, we can
    look at its histogram:

    >>> import matplotlib.pyplot as plt
    >>> dist = StandardNormal()
    >>> rng = NumericalInverseHermite(dist)
    >>> rvs = rng.rvs(10000)
    >>> x = np.linspace(rvs.min()-0.1, rvs.max()+0.1, 1000)
    >>> fx = norm.pdf(x)
    >>> plt.plot(x, fx, 'r-', lw=2, label='true distribution')
    >>> plt.hist(rvs, bins=20, density=True, alpha=0.8, label='random variates')
    >>> plt.xlabel('x')
    >>> plt.ylabel('PDF(x)')
    >>> plt.title('Numerical Inverse Hermite Samples')
    >>> plt.legend()
    >>> plt.show()

    Given the derivative of the PDF w.r.t the variate (i.e. ``x``), we can use
    quintic Hermite interpolation to approximate the PPF by passing the `order`
    # 定义一个 Python 类 StandardNormal，用于描述标准正态分布的概率密度函数、导数和累积分布函数
    >>> class StandardNormal:
    ...     # 概率密度函数的实现
    ...     def pdf(self, x):
    ...        return 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)
    ...     # 概率密度函数导数的实现
    ...     def dpdf(self, x):
    ...        return -1/np.sqrt(2*np.pi) * x * np.exp(-x**2 / 2)
    ...     # 累积分布函数的实现
    ...     def cdf(self, x):
    ...        return ndtr(x)
    ...

    # 创建一个 StandardNormal 类的实例 dist，用于描述标准正态分布
    >>> dist = StandardNormal()
    # 使用 NumPy 提供的默认随机数生成器创建一个随机数生成器实例 urng
    >>> urng = np.random.default_rng()
    # 使用 NumericalInverseHermite 类构造函数，传入 dist 实例、阶数为 5 和随机数生成器 urng 创建一个 rng 实例
    >>> rng = NumericalInverseHermite(dist, order=5, random_state=urng)

    # 阶数越高，区间数目越少的结果展示：
    >>> rng3 = NumericalInverseHermite(dist, order=3)
    >>> rng5 = NumericalInverseHermite(dist, order=5)
    >>> rng3.intervals, rng5.intervals
    (3000, 522)

    # 通过调用 u_error 方法可以估计 u 误差。它运行一个小型蒙特卡洛模拟来估计 u 误差，默认使用 100,000 个样本，可以通过传递 sample_size 参数进行更改：
    >>> rng1 = NumericalInverseHermite(dist, u_resolution=1e-10)
    >>> rng1.u_error(sample_size=1000000)  # 使用一百万个样本
    UError(max_error=9.53167544892608e-11, mean_absolute_error=2.2450136432146864e-11)

    # 返回一个命名元组，其中包含最大 u 误差和平均绝对 u 误差。

    # 通过减少 u 分辨率（允许的最大 u 误差）可以减小 u 误差：
    >>> rng2 = NumericalInverseHermite(dist, u_resolution=1e-13)
    >>> rng2.u_error(sample_size=1000000)
    UError(max_error=9.32027892364129e-14, mean_absolute_error=1.5194172675685075e-14)

    # 注意，这会增加设置时间和区间数目的成本。
    >>> rng1.intervals
    1022
    >>> rng2.intervals
    5687

    # 导入 timeit 库，测量构造 NumericalInverseHermite 实例的时间开销：
    >>> from timeit import timeit
    >>> f = lambda: NumericalInverseHermite(dist, u_resolution=1e-10)
    >>> timeit(f, number=1)
    0.017409582000254886  # 实际结果可能有所变化
    >>> f = lambda: NumericalInverseHermite(dist, u_resolution=1e-13)
    >>> timeit(f, number=1)
    0.08671202100003939  # 实际结果可能有所变化

    # 由于正态分布的 PPF 函数作为一个特殊函数可用，我们也可以检查 x 误差，即近似 PPF 与精确 PPF 之间的差异：
    >>> import matplotlib.pyplot as plt
    >>> u = np.linspace(0.01, 0.99, 1000)
    >>> approxppf = rng.ppf(u)
    >>> exactppf = norm.ppf(u)
    >>> error = np.abs(exactppf - approxppf)
    >>> plt.plot(u, error)
    >>> plt.xlabel('u')
    >>> plt.ylabel('error')
    >>> plt.title('Error between exact and approximated PPF (x-error)')
    >>> plt.show()

    """
    # 定义一个 C 数组 construction_points_array，用于后续的构造
    cdef double[::1] construction_points_array
    def __cinit__(self,
                  dist,
                  *,
                  domain=None,
                  order=3,
                  u_resolution=1e-12,
                  construction_points=None,
                  random_state=None):
        # 验证参数，获取有效的域、阶数和u分辨率
        domain, order, u_resolution = self._validate_args(dist, domain, order,
                                                          u_resolution, construction_points)

        # 保存所有参数以支持 pickle
        self._kwargs = {
            'dist': dist,
            'domain': domain,
            'order': order,
            'u_resolution': u_resolution,
            'construction_points': construction_points,
            'random_state': random_state
        }

        # 解析分布并设置回调函数
        self.callbacks = _unpack_dist(dist, "cont", meths=["cdf"], optional_meths=["pdf", "dpdf"])

        # 定义回调函数的包装器
        def _callback_wrapper(x, name):
            return self.callbacks[name](x)
        self._callback_wrapper = _callback_wrapper

        # 初始化消息流
        self._messages = MessageStream()

        # 获取全局锁
        _lock.acquire()
        try:
            # 将消息流设置给 UNU.RAN 库
            unur_set_stream(self._messages.handle)

            # 创建连续分布对象
            self.distr = unur_distr_cont_new()
            if self.distr == NULL:
                # 如果创建失败，抛出 UNURANError 异常
                raise UNURANError(self._messages.get())

            # 将回调函数打包到分布对象中
            _pack_distr(self.distr, self.callbacks)

            # 如果定义了域，设置分布的定义域
            if domain is not None:
                self._check_errorcode(unur_distr_cont_set_domain(self.distr, domain[0],
                                                                 domain[1]))

            # 创建反变换对象
            self.par = unur_hinv_new(self.distr)
            if self.par == NULL:
                # 如果创建失败，抛出 UNURANError 异常
                raise UNURANError(self._messages.get())

            # 设置反变换对象的阶数
            self._check_errorcode(unur_hinv_set_order(self.par, order))

            # 设置反变换对象的 u 分辨率
            self._check_errorcode(unur_hinv_set_u_resolution(self.par, u_resolution))

            # 设置反变换对象的构造点
            self._check_errorcode(unur_hinv_set_cpoints(self.par, &self.construction_points_array[0],
                                                        len(self.construction_points_array)))

            # 设置随机数生成器状态
            self._set_rng(random_state)
        finally:
            # 释放全局锁
            _lock.release()
    cdef inline void _ppf(self, const double *u, double *out, size_t N) noexcept:
        cdef:
            size_t i
        for i in range(N):
            out[i] = unur_hinv_eval_approxinvcdf(self.rng, u[i])



    def ppf(self, u):
        """
        ppf(u)

        Approximated PPF of the given distribution.

        Parameters
        ----------
        u : array_like
            Quantiles.

        Returns
        -------
        ppf : array_like
            Percentiles corresponding to given quantiles `u`.
        """
        # 将输入转换为双精度浮点数的 NumPy 数组
        u = np.asarray(u, dtype='d')
        # 保存原始形状
        oshape = u.shape
        # 展平数组
        u = u.ravel()
        # 确定 u 在 [0, 1] 范围内的条件
        cond0 = 0 <= u
        cond1 = u <= 1
        cond2 = cond0 & cond1
        # 使用 argsreduce 函数处理符合条件的 u
        goodu = argsreduce(cond2, u)[0]
        # 创建与 u 相同形状的空数组
        out = np.empty_like(u)
        # 将 goodu 转换为连续的双精度浮点数数组
        cdef double[::1] u_view = np.ascontiguousarray(goodu)
        # 创建与 u_view 相同形状的空数组 goodout
        cdef double[::1] goodout = np.empty_like(u_view)
        # 如果 cond2 中有任何 True 值，调用内部函数 _ppf 处理
        if cond2.any():
            self._ppf(&u_view[0], &goodout[0], len(goodu))
        # 将 goodout 的值根据 cond2 复制到 out 中，未符合条件的位置填充为 NaN
        np.place(out, cond2, goodout)
        np.place(out, ~cond2, np.nan)
        # 将结果重新调整为原始形状并返回
        return np.asarray(out).reshape(oshape)[()]



    def u_error(self, sample_size=100000):
        """
        u_error(sample_size=100000)

        Estimate the u-error of the approximation using Monte Carlo simulation.
        This is only available if the generator was initialized with a `dist`
        object containing the implementation of the exact CDF under `cdf` method.

        Parameters
        ----------
        sample_size : int, optional
            Number of samples to use for the estimation. It must be greater than
            or equal to 1000.

        Returns
        -------
        max_error : float
            Maximum u-error.
        mean_absolute_error : float
            Mean absolute u-error.
        """
        # 如果 sample_size 小于 1000，抛出数值错误异常
        if sample_size < 1000:
            raise ValueError("`sample_size` must be greater than or equal to 1000.")
        cdef double max_error, mae
        cdef ccallback_t callback
        # 获取全局锁对象
        _lock.acquire()
        try:
            # 清空消息列表
            self._messages.clear()
            # 设置消息处理流
            unur_set_stream(self._messages.handle)
            # 初始化回调函数
            init_unuran_callback(&callback, self._callback_wrapper)
            # 调用内部函数计算 u-error 的估计值
            self._check_errorcode(unur_hinv_estimate_error(self.rng, sample_size,
                                                           &max_error, &mae))
        finally:
            # 释放全局锁对象
            _lock.release()
            # 释放回调函数资源
            release_unuran_callback(&callback)
        # 返回 UError 对象，包含最大 u-error 和平均绝对 u-error
        return UError(max_error, mae)



    @property
    def intervals(self):
        """
        Get number of nodes (design points) used for Hermite interpolation in the
        generator object. The number of intervals is the number of nodes minus 1.
        """
        # 返回 Hermite 插值中使用的节点数（设计点数）
        return unur_hinv_get_n_intervals(self.rng)
    # 定义一个方法 `midpoint_error`，用于计算某种误差的中点值
    def midpoint_error(self):
        # 调用对象的 `u_error` 方法，并返回其结果的第一个元素
        return self.u_error()[0]
cdef class DiscreteAliasUrn(Method):
    r"""
    DiscreteAliasUrn(dist, *, domain=None, urn_factor=1, random_state=None)

    Discrete Alias-Urn Method.

    This method is used to sample from univariate discrete distributions with
    a finite domain. It uses the probability vector of size :math:`N` or a
    probability mass function with a finite support to generate random
    numbers from the distribution.

    Parameters
    ----------
    dist : array_like or object, optional
        Probability vector (PV) of the distribution. If PV isn't available,
        an instance of a class with a ``pmf`` method is expected. The signature
        of the PMF is expected to be: ``def pmf(self, k: int) -> float``. i.e. it
        should accept a Python integer and return a Python float.
    domain : int, optional
        Support of the PMF. If a probability vector (``pv``) is not available, a
        finite domain must be given. i.e. the PMF must have a finite support.
        Default is ``None``. When ``None``:

        * If a ``support`` method is provided by the distribution object
          `dist`, it is used to set the domain of the distribution.
        * Otherwise, the support is assumed to be ``(0, len(pv))``. When this
          parameter is passed in combination with a probability vector, ``domain[0]``
          is used to relocate the distribution from ``(0, len(pv))`` to
          ``(domain[0], domain[0]+len(pv))`` and ``domain[1]`` is ignored. See Notes
          and tutorial for a more detailed explanation.

    urn_factor : float, optional
        Size of the urn table *relative* to the size of the probability
        vector. It must not be less than 1. Larger tables result in faster
        generation times but require a more expensive setup. Default is 1.
    random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

        A NumPy random number generator or seed for the underlying NumPy random
        number generator used to generate the stream of uniform random numbers.
        If `random_state` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Notes
    -----
    This method works when either a finite probability vector is available or
    the PMF of the distribution is available. In case a PMF is only available,
    the *finite* support (domain) of the PMF must also be given. It is
    recommended to first obtain the probability vector by evaluating the PMF
    at each point in the support and then using it instead.

    If a probability vector is given, it must be a 1-dimensional array of
    non-negative floats without any ``inf`` or ``nan`` values. Also, there
    """
    # 定义一个类 DiscreteAliasUrn，继承自 Method 类

    def __init__(self, dist, *, domain=None, urn_factor=1, random_state=None):
        # 初始化方法，用于设置对象的初始属性和状态

        # 调用父类的初始化方法
        super().__init__()

        # 将参数存储为对象的属性
        self.dist = dist
        self.domain = domain
        self.urn_factor = urn_factor
        self.random_state = random_state

    def __call__(self, *args, **kwargs):
        # 定义对象可调用时的行为，这里未给出具体实现，留待后续填充
        pass
    must be at least one non-zero entry otherwise an exception is raised.



    By default, the probability vector is indexed starting at 0. However, this
    can be changed by passing a ``domain`` parameter. When ``domain`` is given
    in combination with the PV, it has the effect of relocating the
    distribution from ``(0, len(pv))`` to ``(domain[0]``, ``domain[0] + len(pv))``.
    ``domain[1]`` is ignored in this case.



    The parameter ``urn_factor`` can be increased for faster generation at the
    cost of increased setup time. This method uses a table for random
    variate generation. ``urn_factor`` controls the size of this table
    relative to the size of the probability vector (or width of the support,
    in case a PV is not available). As this table is computed during setup
    time, increasing this parameter linearly increases the time required to
    setup. It is recommended to keep this parameter under 2.



    References
    ----------
    .. [1] UNU.RAN reference manual, Section 5.8.2,
           "DAU - (Discrete) Alias-Urn method",
           http://statmath.wu.ac.at/software/unuran/doc/unuran.html#DAU
    .. [2] A.J. Walker (1977). An efficient method for generating discrete
           random variables with general distributions, ACM Trans. Math.
           Software 3, pp. 253-256.



    Examples
    --------
    >>> from scipy.stats.sampling import DiscreteAliasUrn
    >>> import numpy as np



    To create a random number generator using a probability vector, use:



    >>> pv = [0.1, 0.3, 0.6]
    >>> urng = np.random.default_rng()
    >>> rng = DiscreteAliasUrn(pv, random_state=urng)



    The RNG has been setup. Now, we can now use the `rvs` method to
    generate samples from the distribution:



    >>> rvs = rng.rvs(size=1000)



    To verify that the random variates follow the given distribution, we can
    use the chi-squared test (as a measure of goodness-of-fit):



    >>> from scipy.stats import chisquare
    >>> _, freqs = np.unique(rvs, return_counts=True)
    >>> freqs = freqs / np.sum(freqs)
    >>> freqs
    array([0.092, 0.292, 0.616])
    >>> chisquare(freqs, pv).pvalue
    0.9993602047563164



    As the p-value is very high, we fail to reject the null hypothesis that
    the observed frequencies are the same as the expected frequencies. Hence,
    we can safely assume that the variates have been generated from the given
    distribution. Note that this just gives the correctness of the algorithm
    and not the quality of the samples.



    If a PV is not available, an instance of a class with a PMF method and a
    finite domain can also be passed.



    >>> urng = np.random.default_rng()
    >>> class Binomial:
    ...     def __init__(self, n, p):
    ...         self.n = n
    ...         self.p = p
    ...     def pmf(self, x):
    ...         # note that the pmf doesn't need to be normalized.
    ...         return self.p**x * (1-self.p)**(self.n-x)
    ...     def support(self):
    # 返回一个元组，包含值0和self.n，表示一个范围
    return (0, self.n)
    def __cinit__(self,
                  dist,
                  *,
                  domain=None,
                  urn_factor=1,
                  random_state=None):
        cdef double[::1] pv_view
        # 调用 _validate_args 方法验证参数，并获取验证后的 pv_view 和 domain
        (pv_view, domain) = self._validate_args(dist, domain)
        
        # 将 pv_view 赋值给实例变量 pv_view，防止被垃圾回收器回收
        self.pv_view = pv_view
        
        # 保存所有参数以支持 pickle
        self._kwargs = {'dist': dist, 'domain': domain, 'urn_factor': urn_factor, 'random_state': random_state}

        # 初始化消息流
        self._messages = MessageStream()
        
        # 获取全局锁，确保线程安全
        _lock.acquire()
        try:
            # 设置 UNUR 库使用 self._messages.handle 作为消息输出流
            unur_set_stream(self._messages.handle)

            # 创建离散分布对象
            self.distr = unur_distr_discr_new()
            if self.distr == NULL:
                # 如果创建失败，抛出 UNURANError 异常
                raise UNURANError(self._messages.get())

            # 获取 pv_view 的长度，并将其设置为分布对象的概率向量
            n_pv = len(pv_view)
            self._check_errorcode(unur_distr_discr_set_pv(self.distr, &pv_view[0], n_pv))

            # 如果 domain 不为 None，则设置分布对象的定义域
            if domain is not None:
                self._check_errorcode(unur_distr_discr_set_domain(self.distr, domain[0],
                                                                  domain[1]))

            # 创建 DAU 参数对象
            self.par = unur_dau_new(self.distr)
            if self.par == NULL:
                # 如果创建失败，抛出 UNURANError 异常
                raise UNURANError(self._messages.get())
            
            # 设置 DAU 参数对象的重抽样因子
            self._check_errorcode(unur_dau_set_urnfactor(self.par, urn_factor))

            # 设置随机数生成器
            self._set_rng(random_state)
        finally:
            # 释放全局锁
            _lock.release()

    # 定义 _validate_args 方法用于验证参数
    cdef object _validate_args(self, dist, domain):
        cdef double[::1] pv_view

        # 调用 _validate_domain 方法验证 domain，并返回验证后的 domain
        domain = _validate_domain(domain, dist)
        
        # 如果 domain 不为 None，则检查其所有元素是否有限
        if domain is not None:
            if not np.isfinite(domain).all():
                raise ValueError("`domain` must be finite.")
        else:
            # 如果 dist 具有 'pmf' 属性但未提供 domain，则抛出 ValueError
            if hasattr(dist, 'pmf'):
                raise ValueError("`domain` must be provided when the "
                                 "probability vector is not available.")
        
        # 如果 dist 具有 'pmf' 属性，则假定其接受和返回浮点数，需要向量化以用于数组调用
        if hasattr(dist, 'pmf'):
            pmf = np.vectorize(dist.pmf)
            k = np.arange(domain[0], domain[1]+1)
            pv = pmf(k)
            try:
                # 验证概率向量 pv，获取验证后的 pv_view
                pv_view = _validate_pv(pv)
            except ValueError as err:
                msg = "PMF returned invalid values: " + err.args[0]
                raise ValueError(msg) from None
        else:
            # 如果 dist 没有 'pmf' 属性，则直接验证 dist，获取验证后的 pv_view
            pv_view = _validate_pv(dist)

        # 返回验证后的 pv_view 和 domain
        return pv_view, domain
# 定义一个 Cython 类 `DiscreteGuideTable`，继承自 `Method` 类
cdef class DiscreteGuideTable(Method):
    # 构造函数，描述了 Discrete Guide Table 方法的初始化和功能
    r"""
    DiscreteGuideTable(dist, *, domain=None, guide_factor=1, random_state=None)

    Discrete Guide Table method.

    The Discrete Guide Table method  samples from arbitrary, but finite,
    probability vectors. It uses the probability vector of size :math:`N` or a
    probability mass function with a finite support to generate random
    numbers from the distribution. Discrete Guide Table has a very slow set up
    (linear with the vector length) but provides very fast sampling.

    Parameters
    ----------
    dist : array_like or object, optional
        Probability vector (PV) of the distribution. If PV isn't available,
        an instance of a class with a ``pmf`` method is expected. The signature
        of the PMF is expected to be: ``def pmf(self, k: int) -> float``. i.e. it
        should accept a Python integer and return a Python float.
    domain : int, optional
        Support of the PMF. If a probability vector (``pv``) is not available, a
        finite domain must be given. i.e. the PMF must have a finite support.
        Default is ``None``. When ``None``:

        * If a ``support`` method is provided by the distribution object
          `dist`, it is used to set the domain of the distribution.
        * Otherwise, the support is assumed to be ``(0, len(pv))``. When this
          parameter is passed in combination with a probability vector, ``domain[0]``
          is used to relocate the distribution from ``(0, len(pv))`` to
          ``(domain[0], domain[0]+len(pv))`` and ``domain[1]`` is ignored. See Notes
          and tutorial for a more detailed explanation.
    guide_factor: int, optional
        Size of the guide table relative to length of PV. Larger guide tables
        result in faster generation time but require a more expensive setup.
        Sizes larger than 3 are not recommended. If the relative size is set to
        0, sequential search is used. Default is 1.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        A NumPy random number generator or seed for the underlying NumPy random
        number generator used to generate the stream of uniform random numbers.
        If `random_state` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Notes
    -----
    This method works when either a finite probability vector is available or
    the PMF of the distribution is available. In case a PMF is only available,
    the *finite* support (domain) of the PMF must also be given. It is
    recommended to first obtain the probability vector by evaluating the PMF
    at each point in the support and then using it instead.
    ```
    # DGT samples from arbitrary but finite probability vectors. Random numbers
    # are generated by the inversion method, i.e.
    
    # 1. Generate a random number U ~ U(0,1).
    # 2. Find smallest integer I such that F(I) = P(X<=I) >= U.
    
    # Step (2) is the crucial step. Using sequential search requires O(E(X))
    # comparisons, where E(X) is the expectation of the distribution. Indexed
    # search, however, uses a guide table to jump to some I' <= I near I to find
    # X in constant time. Indeed the expected number of comparisons is reduced to
    # 2, when the guide table has the same size as the probability vector
    # (this is the default). For larger guide tables this number becomes smaller
    # (but is always larger than 1), for smaller tables it becomes larger. For the
    # limit case of table size 1 the algorithm simply does sequential search.
    
    # On the other hand the setup time for guide table is O(N), where N denotes
    # the length of the probability vector (for size 1 no preprocessing is
    # required). Moreover, for very large guide tables memory effects might even
    # reduce the speed of the algorithm. So we do not recommend to use guide
    # tables that are more than three times larger than the given probability
    # vector. If only a few random numbers have to be generated, (much) smaller
    # table sizes are better. The size of the guide table relative to the length
    # of the given probability vector can be set by the ``guide_factor`` parameter.
    
    # If a probability vector is given, it must be a 1-dimensional array of
    # non-negative floats without any ``inf`` or ``nan`` values. Also, there
    # must be at least one non-zero entry otherwise an exception is raised.
    
    # By default, the probability vector is indexed starting at 0. However, this
    # can be changed by passing a ``domain`` parameter. When ``domain`` is given
    # in combination with the PV, it has the effect of relocating the
    # distribution from ``(0, len(pv))`` to ``(domain[0], domain[0] + len(pv))``.
    # ``domain[1]`` is ignored in this case.
    
    # References
    # ----------
    # .. [1] UNU.RAN reference manual, Section 5.8.4,
    #        "DGT - (Discrete) Guide Table method (indexed search)"
    #        https://statmath.wu.ac.at/unuran/doc/unuran.html#DGT
    # .. [2] H.C. Chen and Y. Asau (1974). On generating random variates from an
    #        empirical distribution, AIIE Trans. 6, pp. 163-166.
    
    # Examples
    # --------
    # >>> from scipy.stats.sampling import DiscreteGuideTable
    # >>> import numpy as np
    
    # To create a random number generator using a probability vector, use:
    
    # >>> pv = [0.1, 0.3, 0.6]
    # >>> urng = np.random.default_rng()
    # >>> rng = DiscreteGuideTable(pv, random_state=urng)
    
    # The RNG has been setup. Now, we can now use the `rvs` method to
    # generate samples from the distribution:
    
    # >>> rvs = rng.rvs(size=1000)
    
    # To verify that the random variates follow the given distribution, we can
    # 使用卡方检验作为拟合优度的度量：

    # 导入需要的库函数
    >>> from scipy.stats import chisquare
    # 统计数组rvs中各元素的频数
    >>> _, freqs = np.unique(rvs, return_counts=True)
    # 将频数转换为频率（归一化）
    >>> freqs = freqs / np.sum(freqs)
    >>> freqs
    array([0.092, 0.355, 0.553])
    # 计算卡方检验的p值
    >>> chisquare(freqs, pv).pvalue
    0.9987382966178464

    # 由于p值很高，我们无法拒绝零假设，即观察频率与期望频率相同。
    # 因此，我们可以安全地假设变量是从给定分布中生成的。注意，这仅仅说明了算法的正确性，而非样本质量。

    # 如果没有pv可用，也可以传递具有PMF方法和有限域的类的实例。

    # 使用numpy的默认随机数生成器创建实例urng
    >>> urng = np.random.default_rng()
    # 导入二项分布
    >>> from scipy.stats import binom
    >>> n, p = 10, 0.2
    # 创建binom分布实例dist
    >>> dist = binom(n, p)
    # 使用DiscreteGuideTable类创建随机数生成器rng，其中包含指定的分布和随机数种子urng

    # 现在，我们可以使用rvs方法从分布中抽样，并且还可以测量样本的拟合优度：

    # 从rng中抽取1000个随机样本
    >>> rvs = rng.rvs(1000)
    # 计算随机样本中各值的频数
    >>> _, freqs = np.unique(rvs, return_counts=True)
    # 将频数转换为频率（归一化）
    >>> freqs = freqs / np.sum(freqs)
    # 创建用于观察频率的数组，长度为11，初始化为0
    >>> obs_freqs = np.zeros(11)  # some frequencies may be zero.
    # 将计算得到的频率值存入数组obs_freqs
    >>> obs_freqs[:freqs.size] = freqs
    # 计算期望频率pv，对应分布的每个可能取值0到10
    >>> pv = [dist.pmf(i) for i in range(0, 11)]
    # 将pv转换为概率分布（归一化）
    >>> pv = np.asarray(pv) / np.sum(pv)
    # 计算观察频率obs_freqs与期望频率pv之间的卡方检验的p值
    >>> chisquare(obs_freqs, pv).pvalue
    0.9999999999999989

    # 可以通过可视化样本的直方图来检查样本是否来自正确的分布：

    # 导入matplotlib库
    >>> import matplotlib.pyplot as plt
    # 从rng中再次抽取1000个随机样本
    >>> rvs = rng.rvs(1000)
    # 创建一个新的图形窗口
    >>> fig = plt.figure()
    # 添加一个子图到图形窗口，编号为111
    >>> ax = fig.add_subplot(111)
    # 创建x轴上的取值范围x，对应分布的所有可能取值
    >>> x = np.arange(0, n+1)
    # 计算真实分布的概率质量函数PMF
    >>> fx = dist.pmf(x)
    # 将fx归一化
    >>> fx = fx / fx.sum()
    # 绘制真实分布的散点图和折线图
    >>> ax.plot(x, fx, 'bo', label='true distribution')
    >>> ax.vlines(x, 0, fx, lw=2)
    # 绘制样本的直方图
    >>> ax.hist(rvs, bins=np.r_[x, n+1]-0.5, density=True, alpha=0.5,
    ...         color='r', label='samples')
    # 设置x轴标签
    >>> ax.set_xlabel('x')
    # 设置y轴标签
    >>> ax.set_ylabel('PMF(x)')
    # 设置图的标题
    >>> ax.set_title('Discrete Guide Table Samples')
    # 添加图例
    >>> plt.legend()
    # 显示图形
    >>> plt.show()

    # 可以使用guide_factor关键字参数设置guide table的大小。
    # 它将guide table的大小相对于概率向量进行设置

    # 使用概率向量pv和指定的随机数生成器urng创建DiscreteGuideTable实例rng。

    # 计算n=4和p=0.1的二项分布的百分位点函数PPF：

    # 定义n和p的值
    >>> n, p = 4, 0.1
    # 创建binom分布实例dist
    >>> dist = binom(n, p)
    # 使用DiscreteGuideTable类创建随机数生成器rng，其中包含指定的分布和随机数种子42
    >>> rng = DiscreteGuideTable(dist, random_state=42)
    # 计算分布的50%百分位点
    >>> rng.ppf(0.5)
    0.0
    """
    # 声明一个双精度浮点数的视图数组pv_view
    cdef double[::1] pv_view
    # 声明一个域对象domain
    cdef object domain
    # 初始化函数，用于设置分布对象的参数
    def __cinit__(self,
                  dist,
                  *,
                  domain=None,
                  guide_factor=1,
                  random_state=None):

        # 声明C语言风格的连续内存视图，用于存储分布的概率密度函数
        cdef double[::1] pv_view

        # 调用私有方法验证参数，并返回验证后的参数及其处理后的域
        (pv_view, domain) = self._validate_args(dist, domain, guide_factor)
        # 将处理后的域保存在实例变量中
        self.domain = domain

        # 将pv_view的引用计数增加，确保其不会被垃圾回收
        self.pv_view = pv_view

        # 保存所有参数以支持pickle序列化
        self._kwargs = {
            'dist': dist,
            'domain': domain,
            'guide_factor': guide_factor,
            'random_state': random_state
        }

        # 初始化消息流对象用于记录消息
        self._messages = MessageStream()
        # 获取全局锁，确保线程安全
        _lock.acquire()

        try:
            # 将消息流对象设置为UNUR库的消息处理流
            unur_set_stream(self._messages.handle)

            # 创建UNUR库的离散分布对象
            self.distr = unur_distr_discr_new()

            # 如果分布对象创建失败，抛出UNURANError异常并捕获消息
            if self.distr == NULL:
                raise UNURANError(self._messages.get())

            # 获取pv_view的长度作为参数个数
            n_pv = len(pv_view)
            # 设置分布对象的概率密度函数
            self._check_errorcode(unur_distr_discr_set_pv(self.distr, &pv_view[0], n_pv))

            # 如果指定了域，设置分布对象的定义域
            if domain is not None:
                self._check_errorcode(unur_distr_discr_set_domain(self.distr, domain[0], domain[1]))

            # 创建UNUR库的分布生成器对象
            self.par = unur_dgt_new(self.distr)
            # 如果生成器对象创建失败，抛出UNURANError异常并捕获消息
            if self.par == NULL:
                raise UNURANError(self._messages.get())

            # 设置分布生成器对象的引导因子
            self._check_errorcode(unur_dgt_set_guidefactor(self.par, guide_factor))
            # 设置随机数生成器
            self._set_rng(random_state)
        finally:
            # 释放全局锁
            _lock.release()
    # 定义 Cython 函数 _validate_args，用于验证参数和设置域
    cdef object _validate_args(self, dist, domain, guide_factor):
        # 声明一个双精度一维数组 pv_view
        cdef double[::1] pv_view

        # 调用 _validate_domain 函数验证并返回 domain
        domain = _validate_domain(domain, dist)
        # 如果 domain 不为 None，则检查其所有值是否有限
        if domain is not None:
            if not np.isfinite(domain).all():
                raise ValueError("`domain` must be finite.")
        else:
            # 如果 dist 具有 'pmf' 属性，则要求必须提供 domain
            if hasattr(dist, 'pmf'):
                raise ValueError("`domain` must be provided when the "
                                 "probability vector is not available.")

        # 如果 guide_factor 大于 3，则发出运行时警告
        if guide_factor > 3:
            msg = "guide_factor sizes larger than 3 are not recommended."
            warnings.warn(msg, RuntimeWarning)

        # 如果 guide_factor 等于 0，则发出运行时警告，说明使用顺序搜索的情况
        if guide_factor == 0:
            msg = ("If the relative size (guide_factor) is set to 0, "
                   "sequential search is used. However, this is not "
                   "recommended, except in exceptional cases, since the "
                   "discrete sequential search method has almost no setup and "
                   "is thus faster.")
            warnings.warn(msg, RuntimeWarning)

        # 如果 dist 具有 'pmf' 属性，则向量化其 pmf 方法以处理域中的点阵列
        if hasattr(dist, 'pmf'):
            pmf = np.vectorize(dist.pmf)
            # 创建整数数组 k，范围从 domain 的第一个元素到第二个元素加一
            k = np.arange(domain[0], domain[1]+1)
            # 计算 pmf 在 k 上的值，并存储在 pv 中
            pv = pmf(k)
            try:
                # 调用 _validate_pv 函数验证 pv，并返回 pv_view
                pv_view = _validate_pv(pv)
            except ValueError as err:
                # 如果 _validate_pv 函数返回错误，则生成带有错误消息的 ValueError
                msg = "PMF returned invalid values: " + err.args[0]
                raise ValueError(msg) from None
        else:
            # 如果 dist 没有 'pmf' 属性，则直接验证 dist 并返回 pv_view
            pv_view = _validate_pv(dist)

        # 返回验证后的 pv_view 和 domain
        return pv_view, domain

    # 设置 Cython 函数 _ppf，用于计算逆累积分布函数
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void _ppf(self, const double *u, double *out, size_t N) noexcept:
        cdef:
            size_t i
        # 对于给定的 N，计算 u[i] 的逆累积分布函数并存储在 out[i] 中
        for i in range(N):
            out[i] = unur_dgt_eval_invcdf(self.rng, u[i])
    def ppf(self, u):
        """
        ppf(u)

        PPF of the given distribution.

        Parameters
        ----------
        u : array_like
            Quantiles.

        Returns
        -------
        ppf : array_like
            Percentiles corresponding to given quantiles `u`.
        """
        # 将输入的 u 转换为双精度浮点型的 NumPy 数组
        u = np.asarray(u, dtype='d')
        # 保存原始 u 的形状
        oshape = u.shape
        # 将 u 展平为一维数组
        u = u.ravel()

        # UNU.RAN 在 u < 0 或 u > 1 时填充支持范围的端点，而 SciPy 使用 NaN 填充。我们选择使用 SciPy 的行为。
        cond0 = 0 <= u
        cond1 = u <= 1
        cond2 = cond0 & cond1
        # 使用 argsreduce 函数处理 cond2，获取有效的 u 值
        goodu = argsreduce(cond2, u)[0]
        # 创建一个和 u 大小相同的空数组 out
        out = np.empty_like(u)

        # 将 goodu 视图转换为连续的双精度浮点型数组
        cdef double[::1] u_view = np.ascontiguousarray(goodu)
        # 创建一个和 u_view 大小相同的空数组 goodout
        cdef double[::1] goodout = np.empty_like(u_view)

        # 如果 cond2 中有任何值为 True
        if cond2.any():
            # 调用 C 扩展函数 _ppf 处理 u_view 和 goodout
            self._ppf(&u_view[0], &goodout[0], len(goodu))
        # 将 goodout 的值放回 out 中对应的位置
        np.place(out, cond2, goodout)
        # 将 cond2 中为 False 的位置填充为 NaN
        np.place(out, ~cond2, np.nan)

        # 如果指定了 domain 属性
        if self.domain is not None:
            # 将 u 等于 0 的位置填充为 domain[0] - 1
            np.place(out, u == 0, self.domain[0] - 1)
        else:
            # 如果未指定 domain 属性，则将 u 等于 0 的位置填充为 -1
            np.place(out, u == 0, -1)
        # 将 out 重新调整为原始形状并返回
        return np.asarray(out).reshape(oshape)[()]
```
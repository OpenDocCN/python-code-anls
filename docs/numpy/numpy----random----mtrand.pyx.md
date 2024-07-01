# `.\numpy\numpy\random\mtrand.pyx`

```py
# 导入运算符模块，提供了各种运算符的函数实现
import operator
# 导入警告模块，用于发出警告信息
import warnings
# 从collections.abc模块中导入Sequence类，用于定义抽象基类
from collections.abc import Sequence

# 导入NumPy库，并使用np作为别名
import numpy as np

# 从cpython.pycapsule中的cimport部分导入PyCapsule_IsValid和PyCapsule_GetPointer函数
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
# 从cpython中的cimport部分导入Py_INCREF和PyFloat_AsDouble函数
from cpython cimport (Py_INCREF, PyFloat_AsDouble)

# 使用cython声明cimport导入NumPy库
cimport cython
cimport numpy as np

# 从libc中的cimport部分导入string和stdint模块
from libc cimport string
from libc.stdint cimport int64_t, uint64_t

# 从._bounded_integers模块的cimport部分导入各种随机整数生成函数
from ._bounded_integers cimport (_rand_bool, _rand_int32, _rand_int64,
         _rand_int16, _rand_int8, _rand_uint64, _rand_uint32, _rand_uint16,
         _rand_uint8,)
# 从._mt19937模块导入MT19937类，并使用_MT19937作为别名
from ._mt19937 import MT19937 as _MT19937
# 从numpy.random中的cimport部分导入bitgen_t结构体
from numpy.random cimport bitgen_t
# 从._common模块的cimport部分导入各种常量和函数
from ._common cimport (POISSON_LAM_MAX, CONS_POSITIVE, CONS_NONE,
            CONS_NON_NEGATIVE, CONS_BOUNDED_0_1, CONS_BOUNDED_GT_0_1,
            CONS_BOUNDED_LT_0_1, CONS_GTE_1, CONS_GT_1, LEGACY_CONS_POISSON,
            LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG,
            double_fill, cont, kahan_sum, cont_broadcast_3,
            check_array_constraint, check_constraint, disc, discrete_broadcast_iii,
            validate_output_shape
        )

# 从NumPy随机分布头文件中的cdef extern部分声明s_binomial_t结构体
cdef extern from "numpy/random/distributions.h":
    struct s_binomial_t:
        int has_binomial
        double psave
        int64_t nsave
        double r
        double q
        double fm
        int64_t m
        double p1
        double xm
        double xl
        double xr
        double c
        double laml
        double lamr
        double p2
        double p3
        double p4

    ctypedef s_binomial_t binomial_t

    # 声明多个随机数生成函数，使用nogil关键字表示函数是无GIL（全局解释器锁）的
    void random_standard_uniform_fill(bitgen_t* bitgen_state, np.npy_intp cnt, double *out) nogil
    int64_t random_positive_int(bitgen_t *bitgen_state) nogil
    double random_uniform(bitgen_t *bitgen_state, double lower, double range) nogil
    double random_laplace(bitgen_t *bitgen_state, double loc, double scale) nogil
    double random_gumbel(bitgen_t *bitgen_state, double loc, double scale) nogil
    double random_logistic(bitgen_t *bitgen_state, double loc, double scale) nogil
    double random_rayleigh(bitgen_t *bitgen_state, double mode) nogil
    double random_triangular(bitgen_t *bitgen_state, double left, double mode,
                                 double right) nogil
    uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max) nogil

# 从包含遗留分布的头文件中的cdef extern部分声明aug_bitgen结构体
cdef extern from "include/legacy-distributions.h":
    struct aug_bitgen:
        bitgen_t *bit_generator
        int has_gauss
        double gauss

    ctypedef aug_bitgen aug_bitgen_t

    # 声明多个遗留分布函数，使用nogil关键字表示函数是无GIL的
    double legacy_gauss(aug_bitgen_t *aug_state) nogil
    double legacy_pareto(aug_bitgen_t *aug_state, double a) nogil
    double legacy_weibull(aug_bitgen_t *aug_state, double a) nogil
    double legacy_standard_gamma(aug_bitgen_t *aug_state, double shape) nogil
    double legacy_normal(aug_bitgen_t *aug_state, double loc, double scale) nogil
    double legacy_standard_t(aug_bitgen_t *aug_state, double df) nogil
    // 计算服从标准指数分布的随机数，使用给定的随机数生成器状态
    double legacy_standard_exponential(aug_bitgen_t *aug_state) nogil
    
    // 计算服从幂律分布的随机数，使用给定的随机数生成器状态和指数参数
    double legacy_power(aug_bitgen_t *aug_state, double a) nogil
    
    // 计算服从伽马分布的随机数，使用给定的随机数生成器状态、形状参数和尺度参数
    double legacy_gamma(aug_bitgen_t *aug_state, double shape, double scale) nogil
    
    // 计算服从卡方分布的随机数，使用给定的随机数生成器状态和自由度
    double legacy_chisquare(aug_bitgen_t *aug_state, double df) nogil
    
    // 计算服从瑞利分布的随机数，使用给定的随机数生成器状态和模式参数
    double legacy_rayleigh(aug_bitgen_t *aug_state, double mode) nogil
    
    // 计算服从非中心卡方分布的随机数，使用给定的随机数生成器状态、自由度和非中心参数
    double legacy_noncentral_chisquare(aug_bitgen_t *aug_state, double df, double nonc) nogil
    
    // 计算服从非中心 F 分布的随机数，使用给定的随机数生成器状态、分子自由度、分母自由度和非中心参数
    double legacy_noncentral_f(aug_bitgen_t *aug_state, double dfnum, double dfden, double nonc) nogil
    
    // 计算服从瓦尔德分布的随机数，使用给定的随机数生成器状态、均值和尺度参数
    double legacy_wald(aug_bitgen_t *aug_state, double mean, double scale) nogil
    
    // 计算服从对数正态分布的随机数，使用给定的随机数生成器状态、均值和标准差参数
    double legacy_lognormal(aug_bitgen_t *aug_state, double mean, double sigma) nogil
    
    // 生成服从二项分布的随机数，使用给定的随机数生成器状态、成功概率和试验次数
    int64_t legacy_random_binomial(bitgen_t *bitgen_state, double p, int64_t n, binomial_t *binomial) nogil
    
    // 生成服从负二项分布的随机数，使用给定的随机数生成器状态、失败次数和成功概率
    int64_t legacy_negative_binomial(aug_bitgen_t *aug_state, double n, double p) nogil
    
    // 生成服从超几何分布的随机数，使用给定的随机数生成器状态、成功数、失败数和样本数
    int64_t legacy_random_hypergeometric(bitgen_t *bitgen_state, int64_t good, int64_t bad, int64_t sample) nogil
    
    // 生成服从对数级数分布的随机数，使用给定的随机数生成器状态和概率参数
    int64_t legacy_logseries(bitgen_t *bitgen_state, double p) nogil
    
    // 生成服从泊松分布的随机数，使用给定的随机数生成器状态和均值参数
    int64_t legacy_random_poisson(bitgen_t *bitgen_state, double lam) nogil
    
    // 生成服从 Zipf 分布的随机数，使用给定的随机数生成器状态和参数 a
    int64_t legacy_random_zipf(bitgen_t *bitgen_state, double a) nogil
    
    // 生成服从几何分布的随机数，使用给定的随机数生成器状态和成功概率参数
    int64_t legacy_random_geometric(bitgen_t *bitgen_state, double p) nogil
    
    // 生成多项式分布的随机数，使用给定的随机数生成器状态、总数、各类别数量和各类别概率
    void legacy_random_multinomial(bitgen_t *bitgen_state, long n, long *mnix, double *pix, np.npy_intp d, binomial_t *binomial) nogil
    
    // 计算服从标准柯西分布的随机数，使用给定的随机数生成器状态
    double legacy_standard_cauchy(aug_bitgen_t *state) nogil
    
    // 计算服从 Beta 分布的随机数，使用给定的随机数生成器状态、参数 a 和 b
    double legacy_beta(aug_bitgen_t *aug_state, double a, double b) nogil
    
    // 计算服从 F 分布的随机数，使用给定的随机数生成器状态、分子自由度和分母自由度
    double legacy_f(aug_bitgen_t *aug_state, double dfnum, double dfden) nogil
    
    // 计算服从指数分布的随机数，使用给定的随机数生成器状态和尺度参数
    double legacy_exponential(aug_bitgen_t *aug_state, double scale) nogil
    
    // 计算服从幂律分布的随机数，使用给定的随机数生成器状态和指数参数
    double legacy_power(aug_bitgen_t *state, double a) nogil
    
    // 计算服从 von Mises 分布的随机数，使用给定的随机数生成器状态、均值和集中度参数
    double legacy_vonmises(bitgen_t *bitgen_state, double mu, double kappa) nogil
# 导入 NumPy 库，并确保其内部 C 函数可以正常使用
np.import_array()

# 定义一个 C 函数 int64_to_long，用于将 int64 类型转换为 long 类型以保持向后兼容性
cdef object int64_to_long(object x):
    """
    Convert int64 to long for legacy compatibility, which used long for integer
    distributions
    """
    cdef int64_t x64
    
    # 如果 x 是标量，则直接转换为 int64，并返回其 long 类型值
    if np.isscalar(x):
        x64 = x
        return <long>x64
    
    # 否则，将 x 转换为 long 类型并返回
    return x.astype('l', casting='unsafe')


# 定义一个 Cython 类 RandomState，用于包装 Mersenne Twister 伪随机数生成器
cdef class RandomState:
    """
    RandomState(seed=None)

    Container for the slow Mersenne Twister pseudo-random number generator.
    Consider using a different BitGenerator with the Generator container
    instead.

    `RandomState` and `Generator` expose a number of methods for generating
    random numbers drawn from a variety of probability distributions. In
    addition to the distribution-specific arguments, each method takes a
    keyword argument `size` that defaults to ``None``. If `size` is ``None``,
    then a single value is generated and returned. If `size` is an integer,
    then a 1-D array filled with generated values is returned. If `size` is a
    tuple, then an array with that shape is filled and returned.

    **Compatibility Guarantee**

    A fixed bit generator using a fixed seed and a fixed series of calls to
    'RandomState' methods using the same parameters will always produce the
    same results up to roundoff error except when the values were incorrect.
    `RandomState` is effectively frozen and will only receive updates that
    are required by changes in the internals of Numpy. More substantial
    changes, including algorithmic improvements, are reserved for
    `Generator`.

    Parameters
    ----------
    seed : {None, int, array_like, BitGenerator}, optional
        Random seed used to initialize the pseudo-random number generator or
        an instantized BitGenerator.  If an integer or array, used as a seed for
        the MT19937 BitGenerator. Values can be any integer between 0 and
        2**32 - 1 inclusive, an array (or other sequence) of such integers,
        or ``None`` (the default).  If `seed` is ``None``, then the `MT19937`
        BitGenerator is initialized by reading data from ``/dev/urandom``
        (or the Windows analogue) if available or seed from the clock
        otherwise.

    Notes
    -----
    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator with a number of methods that are similar
    to the ones available in `RandomState`. `RandomState`, besides being
    NumPy-aware, has the advantage that it provides a much larger number
    of probability distributions to choose from.

    See Also
    --------
    Generator
    MT19937
    numpy.random.BitGenerator

    """
    cdef public object _bit_generator
    cdef bitgen_t _bitgen
    cdef aug_bitgen_t _aug_state
    cdef binomial_t _binomial
    cdef object lock
    _poisson_lam_max = POISSON_LAM_MAX
    def __init__(self, seed=None):
        # 如果未提供种子，则使用默认的 MT19937 生成器
        if seed is None:
            bit_generator = _MT19937()
        # 如果种子不是一个带有 'capsule' 属性的对象，则使用默认的 MT19937 生成器并进行传统种子生成
        elif not hasattr(seed, 'capsule'):
            bit_generator = _MT19937()
            bit_generator._legacy_seeding(seed)
        # 否则，使用给定的种子作为生成器
        else:
            bit_generator = seed

        # 初始化 BitGenerator
        self._initialize_bit_generator(bit_generator)

    def __repr__(self):
        # 返回对象的字符串表示形式，包括对象的类名和位生成器的类名
        return self.__str__() + ' at 0x{:X}'.format(id(self))

    def __str__(self):
        # 返回对象的字符串表示形式，包括对象的类名和位生成器的类名
        _str = self.__class__.__name__
        _str += '(' + self._bit_generator.__class__.__name__ + ')'
        return _str

    # Pickling support:
    def __getstate__(self):
        # 获取对象的状态信息，不使用遗留模式
        return self.get_state(legacy=False)

    def __setstate__(self, state):
        # 设置对象的状态信息
        self.set_state(state)

    def __reduce__(self):
        from ._pickle import __randomstate_ctor
        # 为了支持序列化，返回一个元组，包含用于重建对象的构造函数和状态信息
        return __randomstate_ctor, (self._bit_generator, ), self.get_state(legacy=False)

    cdef _initialize_bit_generator(self, bit_generator):
        # 初始化位生成器，并检查其有效性
        self._bit_generator = bit_generator
        capsule = bit_generator.capsule
        cdef const char *name = "BitGenerator"
        if not PyCapsule_IsValid(capsule, name):
            raise ValueError("Invalid bit generator. The bit generator must "
                             "be instantized.")
        # 获取位生成器的指针并设置对象状态
        self._bitgen = (<bitgen_t *> PyCapsule_GetPointer(capsule, name))[0]
        self._aug_state.bit_generator = &self._bitgen
        # 重置高斯随机数状态
        self._reset_gauss()
        self.lock = bit_generator.lock

    cdef _reset_gauss(self):
        # 重置高斯随机数状态
        self._aug_state.has_gauss = 0
        self._aug_state.gauss = 0.0

    def seed(self, seed=None):
        """
        seed(seed=None)

        Reseed a legacy MT19937 BitGenerator

        Notes
        -----
        This is a convenience, legacy function.

        The best practice is to **not** reseed a BitGenerator, rather to
        recreate a new one. This method is here for legacy reasons.
        This example demonstrates best practice.

        >>> from numpy.random import MT19937
        >>> from numpy.random import RandomState, SeedSequence
        >>> rs = RandomState(MT19937(SeedSequence(123456789)))
        # Later, you want to restart the stream
        >>> rs = RandomState(MT19937(SeedSequence(987654321)))
        """
        # 如果不是 MT19937 类型的生成器，则抛出类型错误
        if not isinstance(self._bit_generator, _MT19937):
            raise TypeError('can only re-seed a MT19937 BitGenerator')
        # 使用给定的种子重新生成生成器的状态
        self._bit_generator._legacy_seeding(seed)
        # 重置高斯随机数状态
        self._reset_gauss()
    # 定义一个方法用于获取生成器的内部状态
    def get_state(self, legacy=True):
        """
        get_state(legacy=True)

        返回表示生成器内部状态的元组或字典。

        更多细节请参见 `set_state`。

        Parameters
        ----------
        legacy : bool, optional
            标志，指示是否返回 MT19937 生成器的传统元组状态，而不是字典。
            如果底层的位生成器不是 MT19937 的实例，则抛出 ValueError。

        Returns
        -------
        out : {tuple(str, ndarray of 624 uints, int, int, float), dict}
            如果 legacy 为 True，则返回的元组包含以下项：

            1. 字符串 'MT19937'。
            2. 624 个无符号整数组成的一维数组。
            3. 整数 `pos`。
            4. 整数 `has_gauss`。
            5. 浮点数 `cached_gaussian`。

            如果 `legacy` 为 False，或者位生成器不是 MT19937，则作为字典返回状态。

        See Also
        --------
        set_state

        Notes
        -----
        `set_state` 和 `get_state` 对于使用 NumPy 中的任何随机分布都是不必要的。
        如果手动更改内部状态，则用户必须清楚自己在做什么。
        """
        # 获取位生成器的当前状态
        st = self._bit_generator.state
        # 如果位生成器不是 MT19937，并且 legacy 为 True，则发出警告并将 legacy 置为 False
        if st['bit_generator'] != 'MT19937' and legacy:
            warnings.warn('get_state 和 legacy 仅可与 MT19937 位生成器一起使用。'
                          '要消除此警告，请将 `legacy` 设置为 False。',
                          RuntimeWarning)
            legacy = False
        # 将增强状态中的高斯值添加到状态字典中
        st['has_gauss'] = self._aug_state.has_gauss
        st['gauss'] = self._aug_state.gauss
        # 如果 legacy 为 True 并且位生成器不是 _MT19937 的实例，则引发 ValueError
        if legacy and not isinstance(self._bit_generator, _MT19937):
            raise ValueError(
                "legacy 仅在底层位生成器是 MT19937 的实例时才能为 True。"
            )
        # 如果 legacy 为 True，则返回传统元组状态
        if legacy:
            return (st['bit_generator'], st['state']['key'], st['state']['pos'],
                    st['has_gauss'], st['gauss'])
        # 否则返回状态字典
        return st
    def set_state(self, state):
        """
        set_state(state)

        Set the internal state of the generator from a tuple.

        For use if one has reason to manually (re-)set the internal state of
        the bit generator used by the RandomState instance. By default,
        RandomState uses the "Mersenne Twister"[1]_ pseudo-random number
        generating algorithm.

        Parameters
        ----------
        state : {tuple(str, ndarray of 624 uints, int, int, float), dict}
            The `state` tuple has the following items:

            1. the string 'MT19937', specifying the Mersenne Twister algorithm.
            2. a 1-D array of 624 unsigned integers ``keys``.
            3. an integer ``pos``.
            4. an integer ``has_gauss``.
            5. a float ``cached_gaussian``.

            If state is a dictionary, it is directly set using the BitGenerators
            `state` property.

        Returns
        -------
        out : None
            Returns 'None' on success.

        See Also
        --------
        get_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in NumPy. If the internal state is manually altered,
        the user should know exactly what he/she is doing.

        For backwards compatibility, the form (str, array of 624 uints, int) is
        also accepted although it is missing some information about the cached
        Gaussian value: ``state = ('MT19937', keys, pos)``.

        References
        ----------
        .. [1] M. Matsumoto and T. Nishimura, "Mersenne Twister: A
           623-dimensionally equidistributed uniform pseudorandom number
           generator," *ACM Trans. on Modeling and Computer Simulation*,
           Vol. 8, No. 1, pp. 3-30, Jan. 1998.

        """

        # 如果 state 是字典类型
        if isinstance(state, dict):
            # 检查字典中是否包含必要的 'bit_generator' 和 'state' 键
            if 'bit_generator' not in state or 'state' not in state:
                raise ValueError('state dictionary is not valid.')
            # 直接使用给定的字典作为状态 st
            st = state
        else:
            # 如果 state 不是字典，必须是元组或列表类型
            if not isinstance(state, (tuple, list)):
                raise TypeError('state must be a dict or a tuple.')
            # 使用 Cython 检查边界
            with cython.boundscheck(True):
                # 检查是否使用了正确的 Mersenne Twister 状态实例
                if state[0] != 'MT19937':
                    raise ValueError('set_state can only be used with legacy '
                                     'MT19937 state instances.')
                # 构建状态字典 st，包括键、位置等信息
                st = {'bit_generator': state[0],
                      'state': {'key': state[1], 'pos': state[2]}}
                # 如果状态元组长度大于3，也包括额外的高斯信息
                if len(state) > 3:
                    st['has_gauss'] = state[3]
                    st['gauss'] = state[4]

        # 设置增强状态中的高斯值和高斯标志
        self._aug_state.gauss = st.get('gauss', 0.0)
        self._aug_state.has_gauss = st.get('has_gauss', 0)
        # 设置位生成器的状态
        self._bit_generator.state = st
    def random_sample(self, size=None):
        """
        random_sample(size=None)

        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random_sample` by `(b-a)` and add `a`::

          (b - a) * random_sample() + a

        .. note::
            New code should use the `~numpy.random.Generator.random`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned).

        See Also
        --------
        random.Generator.random: which should be used for new code.

        Examples
        --------
        >>> np.random.random_sample()
        0.47108547995356098 # random
        >>> type(np.random.random_sample())
        <class 'float'>
        >>> np.random.random_sample((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428]) # random

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * np.random.random_sample((3, 2)) - 5
        array([[-3.99149989, -0.52338984], # random
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])

        """
        cdef double temp
        return double_fill(&random_standard_uniform_fill, &self._bitgen, size, self.lock, None)
    def beta(self, a, b, size=None):
        """
        beta(a, b, size=None)

        Draw samples from a Beta distribution.

        The Beta distribution is a special case of the Dirichlet distribution,
        and is related to the Gamma distribution.  It has the probability
        distribution function

        .. math:: f(x; a,b) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}
                                                         (1 - x)^{\\beta - 1},

        where the normalization, B, is the beta function,

        .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}
                                     (1 - t)^{\\beta - 1} dt.

        It is often seen in Bayesian inference and order statistics.

        .. note::
            New code should use the `~numpy.random.Generator.beta`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        a : float or array_like of floats
            Alpha, positive (>0).
        b : float or array_like of floats
            Beta, positive (>0).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``a`` and ``b`` are both scalars.
            Otherwise, ``np.broadcast(a, b).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized beta distribution.

        See Also
        --------
        random.Generator.beta: which should be used for new code.
        """
        # 调用Cython实现的底层函数，用于生成 Beta 分布的样本
        return cont(&legacy_beta, &self._aug_state, size, self.lock, 2,
                    a, 'a', CONS_POSITIVE,
                    b, 'b', CONS_POSITIVE,
                    0.0, '', CONS_NONE, None)
    def standard_exponential(self, size=None):
        """
        standard_exponential(size=None)

        Draw samples from the standard exponential distribution.

        `standard_exponential` is identical to the exponential distribution
        with a scale parameter of 1.

        .. note::
            New code should use the
            `~numpy.random.Generator.standard_exponential`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        See Also
        --------
        random.Generator.standard_exponential: which should be used for new code.

        Examples
        --------
        Output a 3x8000 array:

        >>> n = np.random.standard_exponential((3, 8000))

        """
        # 调用 legacy_standard_exponential 函数从指定的随机状态中生成标准指数分布的样本
        return cont(&legacy_standard_exponential, &self._aug_state, size, self.lock, 0,
                    None, None, CONS_NONE,
                    None, None, CONS_NONE,
                    None, None, CONS_NONE,
                    None)
    def tomaxint(self, size=None):
        """
        tomaxint(size=None)

        Return a sample of uniformly distributed random integers in the interval
        [0, ``np.iinfo("long").max``].

        .. warning::
           This function uses the C-long dtype, which is 32bit on windows
           and otherwise 64bit on 64bit platforms (and 32bit on 32bit ones).
           Since NumPy 2.0, NumPy's default integer is 32bit on 32bit platforms
           and 64bit on 64bit platforms.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Uniform sampling over a given half-open interval of integers.
        random_integers : Uniform sampling over a given closed interval of
            integers.

        Examples
        --------
        >>> rs = np.random.RandomState() # need a RandomState object
        >>> rs.tomaxint((2,2,2))
        array([[[1170048599, 1600360186], # random
                [ 739731006, 1947757578]],
               [[1871712945,  752307660],
                [1601631370, 1479324245]]])
        >>> rs.tomaxint((2,2,2)) < np.iinfo(np.int_).max
        array([[[ True,  True],
                [ True,  True]],
               [[ True,  True],
                [ True,  True]]])

        """

        # 声明变量和数组以存储随机数
        cdef np.npy_intp n
        cdef np.ndarray randoms
        cdef int64_t *randoms_data

        # 如果 size 为 None，则返回单个随机整数值
        if size is None:
            # 使用锁保证线程安全，返回从位生成器中生成的随机正整数
            with self.lock:
                return random_positive_int(&self._bitgen)

        # 创建一个空的 ndarray，用于存储随机数，指定数据类型为 np.int64
        randoms = <np.ndarray>np.empty(size, dtype=np.int64)
        # 获取 randoms 数组的数据指针
        randoms_data = <int64_t*>np.PyArray_DATA(randoms)
        # 获取 randoms 数组的大小
        n = np.PyArray_SIZE(randoms)

        # 遍历数组中的每个位置
        for i in range(n):
            # 使用锁保证线程安全，并且不使用全局解释器锁（GIL）
            with self.lock, nogil:
                # 调用 random_positive_int 函数生成随机正整数，并将其存储在数组中
                randoms_data[i] = random_positive_int(&self._bitgen)
        
        # 返回生成的随机数数组
        return randoms
    # 定义一个方法 bytes，接受一个整数参数 length
    def bytes(self, np.npy_intp length):
        """
        bytes(length)

        Return random bytes.

        .. note::
            New code should use the `~numpy.random.Generator.bytes`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        length : int
            Number of random bytes.

        Returns
        -------
        out : bytes
            String of length `length`.

        See Also
        --------
        random.Generator.bytes: which should be used for new code.

        Examples
        --------
        >>> np.random.bytes(10)
        b' eh\\x85\\x022SZ\\xbf\\xa4' #random
        """
        # 计算需要的 uint32 数组的长度
        cdef Py_ssize_t n_uint32 = ((length - 1) // 4 + 1)
        # 使用 self.randint 生成指定范围内的随机整数数组，并转换为 bytes 类型
        # 截取所需长度的字节数据并返回
        return self.randint(0, 4294967296, size=n_uint32,
                            dtype=np.uint32).astype('<u4').tobytes()[:length]

    # 定义一个装饰器为 True 的方法 rand，接受可变数量的参数
    @cython.wraparound(True)
    def rand(self, *args):
        """
        rand(d0, d1, ..., dn)

        Random values in a given shape.

        .. note::
            This is a convenience function for users porting code from Matlab,
            and wraps `random_sample`. That function takes a
            tuple to specify the size of the output, which is consistent with
            other NumPy functions like `numpy.zeros` and `numpy.ones`.

        Create an array of the given shape and populate it with
        random samples from a uniform distribution
        over ``[0, 1)``.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, must be non-negative.
            If no argument is given a single Python float is returned.

        Returns
        -------
        out : ndarray, shape ``(d0, d1, ..., dn)``
            Random values.

        See Also
        --------
        random

        Examples
        --------
        >>> np.random.rand(3,2)
        array([[ 0.14022471,  0.96360618],  #random
               [ 0.37601032,  0.25528411],  #random
               [ 0.49313049,  0.94909878]]) #random

        """
        # 如果没有传入参数，则调用 self.random_sample() 返回随机样本
        if len(args) == 0:
            return self.random_sample()
        else:
            # 否则，根据传入的参数 size，调用 self.random_sample(size=args) 返回随机样本
            return self.random_sample(size=args)
    # 定义一个方法 randn，用于生成服从标准正态分布的随机数或数组

    """
    randn(d0, d1, ..., dn)

    从标准正态分布中返回一个样本（或多个样本）。

    .. note::
        这是一个方便的函数，用于从 Matlab 移植代码，
        它封装了 `standard_normal`。该函数接受一个元组来指定输出的大小，
        这与 `numpy.zeros` 和 `numpy.ones` 等其他 NumPy 函数保持一致。

    .. note::
        新代码应该使用 `~numpy.random.Generator.standard_normal`
        方法的方式来生成随机数；
        请参阅 :ref:`random-quick-start`。

    如果提供了正整数参数，`randn` 将生成一个形状为 ``(d0, d1, ..., dn)`` 的数组，
    其中填充有从均值为 0、方差为 1 的单变量“正态”（高斯）分布中随机抽取的浮点数。
    如果没有提供参数，则返回从分布中随机抽取的单个浮点数。

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        返回数组的维度，必须是非负整数。
        如果没有给出参数，则返回一个 Python 浮点数。

    Returns
    -------
    Z : ndarray or float
        形状为 ``(d0, d1, ..., dn)`` 的浮点数样本数组，
        或者如果未提供参数，则返回单个这样的浮点数。

    See Also
    --------
    standard_normal : 类似，但接受一个元组作为其参数。
    normal : 还接受 mu 和 sigma 参数。
    random.Generator.standard_normal: 新代码应使用此方法。

    Notes
    -----
    对于均值为 ``mu``，标准差为 ``sigma`` 的正态分布的随机样本，请使用::

        sigma * np.random.randn(...) + mu

    Examples
    --------
    >>> np.random.randn()
    2.1923875335537315  # 随机数

    从均值为 3、标准差为 2.5 的正态分布中抽取的 2x4 数组样本：

    >>> 3 + 2.5 * np.random.randn(2, 4)
    array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # 随机数
           [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # 随机数

    """

    # 如果参数长度为 0，调用 self.standard_normal() 返回一个随机数
    if len(args) == 0:
        return self.standard_normal()
    # 否则，调用 self.standard_normal(size=args) 返回一个随机数组
    else:
        return self.standard_normal(size=args)

# 复杂、连续分布的生成函数：
    def standard_normal(self, size=None):
        """
        standard_normal(size=None)

        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        .. note::
            New code should use the
            `~numpy.random.Generator.standard_normal`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

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
        random.Generator.standard_normal: which should be used for new code.

        Notes
        -----
        For random samples from the normal distribution with mean ``mu`` and
        standard deviation ``sigma``, use one of::

            mu + sigma * np.random.standard_normal(size=...)
            np.random.normal(mu, sigma, size=...)

        Examples
        --------
        >>> np.random.standard_normal()
        2.1923875335537315 #random

        >>> s = np.random.standard_normal(8000)
        >>> s
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311,  # random
               -0.38672696, -0.4685006 ])                                # random
        >>> s.shape
        (8000,)
        >>> s = np.random.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        Two-by-four array of samples from the normal distribution with
        mean 3 and standard deviation 2.5:

        >>> 3 + 2.5 * np.random.standard_normal(size=(2, 4))
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random

        """
        使用C语言扩展函数cont()来生成标准正态分布的随机数样本，此函数接受多个参数用于控制生成的随机数属性和状态。
        返回生成的随机数数组或单个随机数值。
        """
        return cont(&legacy_gauss, &self._aug_state, size, self.lock, 0,
                    None, None, CONS_NONE,
                    None, None, CONS_NONE,
                    None, None, CONS_NONE,
                    None)
    def chisquare(self, df, size=None):
        """
        chisquare(df, size=None)

        Draw samples from a chi-square distribution.

        When `df` independent random variables, each with standard normal
        distributions (mean 0, variance 1), are squared and summed, the
        resulting distribution is chi-square (see Notes).  This distribution
        is often used in hypothesis testing.

        .. note::
            New code should use the `~numpy.random.Generator.chisquare`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.chisquare: which should be used for new code.

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
        >>> np.random.chisquare(2,4)
        array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272]) # random
        """
        # 调用 legacy_chisquare 函数生成卡方分布的样本，并返回结果
        return cont(&legacy_chisquare, &self._aug_state, size, self.lock, 1,
                    df, 'df', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)
    def standard_cauchy(self, size=None):
        """
        standard_cauchy(size=None)

        Draw samples from a standard Cauchy distribution with mode = 0.

        Also known as the Lorentz distribution.

        .. note::
            New code should use the
            `~numpy.random.Generator.standard_cauchy`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.standard_cauchy: which should be used for new code.

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
        >>> s = np.random.standard_cauchy(1000000)
        >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
        >>> plt.hist(s, bins=100)
        >>> plt.show()

        """
        # 返回一个从标准 Cauchy 分布中抽取的样本
        return cont(&legacy_standard_cauchy, &self._aug_state, size, self.lock, 0,
                    0.0, '', CONS_NONE, 0.0, '', CONS_NONE, 0.0, '', CONS_NONE, None)
    # 定义一个 rayleigh 方法，用于生成 Rayleigh 分布的随机样本
    def rayleigh(self, scale=1.0, size=None):
        """
        rayleigh(scale=1.0, size=None)

        Draw samples from a Rayleigh distribution.

        The :math:`\\chi` and Weibull distributions are generalizations of the
        Rayleigh.

        .. note::
            New code should use the `~numpy.random.Generator.rayleigh`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        scale : float or array_like of floats, optional
            Scale, also equals the mode. Must be non-negative. Default is 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``scale`` is a scalar.  Otherwise,
            ``np.array(scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Rayleigh distribution.

        See Also
        --------
        random.Generator.rayleigh: which should be used for new code.

        Notes
        -----
        The probability density function for the Rayleigh distribution is

        .. math:: P(x;scale) = \\frac{x}{scale^2}e^{\\frac{-x^2}{2 \\cdotp scale^2}}

        The Rayleigh distribution would arise, for example, if the East
        and North components of the wind velocity had identical zero-mean
        Gaussian distributions.  Then the wind speed would have a Rayleigh
        distribution.

        References
        ----------
        .. [1] Brighton Webs Ltd., "Rayleigh Distribution,"
               https://web.archive.org/web/20090514091424/http://brighton-webs.co.uk:80/distributions/rayleigh.asp
        .. [2] Wikipedia, "Rayleigh distribution"
               https://en.wikipedia.org/wiki/Rayleigh_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram

        >>> from matplotlib.pyplot import hist
        >>> values = hist(np.random.rayleigh(3, 100000), bins=200, density=True)

        Wave heights tend to follow a Rayleigh distribution. If the mean wave
        height is 1 meter, what fraction of waves are likely to be larger than 3
        meters?

        >>> meanvalue = 1
        >>> modevalue = np.sqrt(2 / np.pi) * meanvalue
        >>> s = np.random.rayleigh(modevalue, 1000000)

        The percentage of waves larger than 3 meters is:

        >>> 100.*sum(s>3)/1000000.
        0.087300000000000003 # random

        """
        # 调用底层 C 函数生成 Rayleigh 分布的样本
        return cont(&legacy_rayleigh, &self._bitgen, size, self.lock, 1,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)

    # Complicated, discrete distributions:
    def poisson(self, lam=1.0, size=None):
        """
        poisson(lam=1.0, size=None)

        Draw samples from a Poisson distribution.

        The Poisson distribution is the limit of the binomial distribution
        for large N.

        .. note::
            New code should use the `~numpy.random.Generator.poisson`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.poisson: which should be used for new code.

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

        >>> import numpy as np
        >>> s = np.random.poisson(5, 10000)

        Display histogram of the sample:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 14, density=True)
        >>> plt.show()

        Draw each 100 values for lambda 100 and 500:

        >>> s = np.random.poisson(lam=(100., 500.), size=(100, 2))

        """

        # 使用底层函数 disc() 生成泊松分布的随机样本
        out = disc(&legacy_random_poisson, &self._bitgen, size, self.lock, 1, 0,
                   lam, 'lam', LEGACY_CONS_POISSON,
                   0.0, '', CONS_NONE,
                   0.0, '', CONS_NONE)
        # 将输出从 int64 转换为 Python long 类型，以匹配历史输出类型
        return int64_to_long(out)
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

        .. note::
            New code should use the `~numpy.random.Generator.geometric`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.geometric: which should be used for new code.

        Examples
        --------
        Draw ten thousand values from the geometric distribution,
        with the probability of an individual success equal to 0.35:

        >>> z = np.random.geometric(p=0.35, size=10000)

        How many trials succeeded after a single run?

        >>> (z == 1).sum() / 10000.
        0.34889999999999999 #random

        """
        # 调用底层函数 disc，实现几何分布的采样
        out = disc(&legacy_random_geometric, &self._bitgen, size, self.lock, 1, 0,
                   p, 'p', CONS_BOUNDED_GT_0_1,
                   0.0, '', CONS_NONE,
                   0.0, '', CONS_NONE)
        # 将输出转换为 long 类型并返回，以匹配历史输出类型
        return int64_to_long(out)

    # Multivariate distributions:
    # Shuffling and permutations:
    # 在原地对数据进行混洗操作，实现洗牌和排列
    cdef inline _shuffle_raw(self, np.npy_intp n, np.npy_intp itemsize,
                             np.npy_intp stride, char* data, char* buf):
        cdef np.npy_intp i, j
        # 从 n-1 到 1 的范围内反向遍历
        for i in reversed(range(1, n)):
            # 使用随机数生成器选择一个随机位置 j
            j = random_interval(&self._bitgen, i)
            # 将位置 j 的数据复制到缓冲区
            string.memcpy(buf, data + j * stride, itemsize)
            # 将位置 i 的数据复制到位置 j
            string.memcpy(data + j * stride, data + i * stride, itemsize)
            # 将缓冲区的数据复制到位置 i
            string.memcpy(data + i * stride, buf, itemsize)
    def permutation(self, object x):
        """
        permutation(x)

        Randomly permute a sequence, or return a permuted range.

        If `x` is a multi-dimensional array, it is only shuffled along its
        first index.

        .. note::
            New code should use the
            `~numpy.random.Generator.permutation`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        x : int or array_like
            If `x` is an integer, randomly permute ``np.arange(x)``.
            If `x` is an array, make a copy and shuffle the elements
            randomly.

        Returns
        -------
        out : ndarray
            Permuted sequence or array range.

        See Also
        --------
        random.Generator.permutation: which should be used for new code.

        Examples
        --------
        >>> np.random.permutation(10)
        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random

        >>> np.random.permutation([1, 4, 9, 12, 15])
        array([15,  1,  9,  4, 12]) # random

        >>> arr = np.arange(9).reshape((3, 3))
        >>> np.random.permutation(arr)
        array([[6, 7, 8], # random
               [0, 1, 2],
               [3, 4, 5]])

        """

        if isinstance(x, (int, np.integer)):
            # 创建一个整数范围的数组 arr，使用 long 作为默认类型（主 numpy 已切换到 intp）
            arr = np.arange(x, dtype=np.result_type(x, np.long))
            # 对数组 arr 进行洗牌操作
            self.shuffle(arr)
            # 返回洗牌后的数组 arr
            return arr

        arr = np.asarray(x)
        if arr.ndim < 1:
            # 如果 x 不是整数且不是至少一维的数组，则引发索引错误
            raise IndexError("x must be an integer or at least 1-dimensional")

        # 对于一维数组，shuffle 方法有快速路径
        if arr.ndim == 1:
            # 如果 arr 和 x 共享内存，则返回 arr 的拷贝
            if np.may_share_memory(arr, x):
                arr = np.array(arr)
            # 对 arr 进行洗牌操作
            self.shuffle(arr)
            # 返回洗牌后的数组 arr
            return arr

        # 洗牌索引数组 idx，指定 dtype 以确保快速路径
        idx = np.arange(arr.shape[0], dtype=np.intp)
        self.shuffle(idx)
        # 根据洗牌后的 idx 对 arr 进行索引，返回重新排列后的数组
        return arr[idx]
# 创建一个名为 _rand 的 RandomState 的实例，用于生成随机数
_rand = RandomState()

# 以下是一系列变量，它们分别是 _rand 实例的不同随机变量生成方法
beta = _rand.beta                # 生成 Beta 分布的随机数
binomial = _rand.binomial        # 生成二项分布的随机数
bytes = _rand.bytes              # 生成随机字节串
chisquare = _rand.chisquare      # 生成卡方分布的随机数
choice = _rand.choice            # 从给定序列中随机选择元素
dirichlet = _rand.dirichlet      # 生成 Dirichlet 分布的随机数
exponential = _rand.exponential  # 生成指数分布的随机数
f = _rand.f                      # 生成 F 分布的随机数
gamma = _rand.gamma              # 生成 Gamma 分布的随机数
get_state = _rand.get_state      # 获取 RandomState 实例的状态
geometric = _rand.geometric      # 生成几何分布的随机数
gumbel = _rand.gumbel            # 生成 Gumbel 分布的随机数
hypergeometric = _rand.hypergeometric  # 生成超几何分布的随机数
laplace = _rand.laplace          # 生成拉普拉斯分布的随机数
logistic = _rand.logistic        # 生成 Logistic 分布的随机数
lognormal = _rand.lognormal      # 生成对数正态分布的随机数
logseries = _rand.logseries      # 生成对数级数分布的随机数
multinomial = _rand.multinomial  # 生成多项分布的随机数
multivariate_normal = _rand.multivariate_normal  # 生成多元正态分布的随机数
negative_binomial = _rand.negative_binomial  # 生成负二项分布的随机数
noncentral_chisquare = _rand.noncentral_chisquare  # 生成非中心卡方分布的随机数
noncentral_f = _rand.noncentral_f  # 生成非中心 F 分布的随机数
normal = _rand.normal            # 生成正态分布的随机数
pareto = _rand.pareto            # 生成帕累托分布的随机数
permutation = _rand.permutation  # 随机排列给定序列
poisson = _rand.poisson          # 生成泊松分布的随机数
power = _rand.power              # 生成功率分布的随机数
rand = _rand.rand                # 生成均匀分布的随机数
randint = _rand.randint          # 生成指定范围内的随机整数
randn = _rand.randn              # 生成标准正态分布的随机数
random = _rand.random            # 生成 [0, 1) 范围内的随机数
random_integers = _rand.random_integers  # 生成指定范围内的随机整数
random_sample = _rand.random_sample  # 生成 [0, 1) 范围内的随机数
rayleigh = _rand.rayleigh        # 生成瑞利分布的随机数
set_state = _rand.set_state      # 设置 RandomState 实例的状态
shuffle = _rand.shuffle          # 将序列随机排序
standard_cauchy = _rand.standard_cauchy  # 生成标准 Cauchy 分布的随机数
standard_exponential = _rand.standard_exponential  # 生成标准指数分布的随机数
standard_gamma = _rand.standard_gamma  # 生成标准 Gamma 分布的随机数
standard_normal = _rand.standard_normal  # 生成标准正态分布的随机数
standard_t = _rand.standard_t    # 生成标准 t 分布的随机数
triangular = _rand.triangular    # 生成三角分布的随机数
uniform = _rand.uniform          # 生成均匀分布的随机数
vonmises = _rand.vonmises        # 生成 Von Mises 分布的随机数
wald = _rand.wald                # 生成 Wald 分布的随机数
weibull = _rand.weibull          # 生成 Weibull 分布的随机数
zipf = _rand.zipf                # 生成 Zipf 分布的随机数

def seed(seed=None):
    """
    设置 RandomState 单例实例的随机数种子

    Parameters
    ----------
    seed : int or None, optional
        随机数种子，若为 None，则使用系统时间

    Notes
    -----
    这是一个便利的、遗留的函数，用于支持旧代码中使用的单例 RandomState。
    最佳实践是使用专用的 ``Generator`` 实例，而不是直接在 random 模块中暴露的随机变量生成方法。

    See Also
    --------
    numpy.random.Generator
    """
    if isinstance(_rand._bit_generator, _MT19937):
        return _rand.seed(seed)
    else:
        bg_type = type(_rand._bit_generator)
        _rand._bit_generator.state = bg_type(seed).state

def get_bit_generator():
    """
    返回 RandomState 单例实例的比特生成器

    Returns
    -------
    BitGenerator
        单例 RandomState 实例下面的比特生成器

    Notes
    -----
    单例 RandomState 提供了 ``numpy.random`` 命名空间中的随机变量生成器。
    这个函数及其对应的设置方法提供了一条路径，可以用用户提供的替代方案
    替换默认的 MT19937 比特生成器。这些函数旨在提供一个连续的路径，
    单个底层比特生成器可以同时用于 ``Generator`` 实例和单例 RandomState 实例。

    See Also
    --------
    set_bit_generator
    numpy.random.Generator
    """
    return _rand._bit_generator

def set_bit_generator(bitgen):
    """
    设置 RandomState 单例实例的比特生成器

    Parameters
    ----------
    bitgen
        比特生成器实例

    Notes
    -----
    这个函数用于设置单例 RandomState 的比特生成器。

    """
    # 创建一个变量 singleton，用于存储 _rand 的 RandomState 实例
    cdef RandomState singleton
    # 将 _rand 赋值给 singleton，使其成为 RandomState 的单例实例
    singleton = _rand
    # 使用给定的 bitgen 参数初始化 singleton 的位生成器
    singleton._initialize_bit_generator(bitgen)
# 定义函数 `sample`，作为 `random_sample` 的别名，详细文档请参考 `random_sample`
def sample(*args, **kwargs):
    # 调用 `_rand` 模块的 `random_sample` 函数，并将参数和关键字参数传递给它
    return _rand.random_sample(*args, **kwargs)

# 定义函数 `ranf`，作为 `random_sample` 的别名，详细文档请参考 `random_sample`
def ranf(*args, **kwargs):
    # 调用 `_rand` 模块的 `random_sample` 函数，并将参数和关键字参数传递给它
    return _rand.random_sample(*args, **kwargs)

# 列出该模块中公开的所有函数和类的名称，以便通过 `from module import *` 导入
__all__ = [
    'beta',
    'binomial',
    'bytes',
    'chisquare',
    'choice',
    'dirichlet',
    'exponential',
    'f',
    'gamma',
    'geometric',
    'get_bit_generator',
    'get_state',
    'gumbel',
    'hypergeometric',
    'laplace',
    'logistic',
    'lognormal',
    'logseries',
    'multinomial',
    'multivariate_normal',
    'negative_binomial',
    'noncentral_chisquare',
    'noncentral_f',
    'normal',
    'pareto',
    'permutation',
    'poisson',
    'power',
    'rand',
    'randint',
    'randn',
    'random',
    'random_integers',
    'random_sample',
    'ranf',
    'rayleigh',
    'sample',
    'seed',
    'set_bit_generator',
    'set_state',
    'shuffle',
    'standard_cauchy',
    'standard_exponential',
    'standard_gamma',
    'standard_normal',
    'standard_t',
    'triangular',
    'uniform',
    'vonmises',
    'wald',
    'weibull',
    'zipf',
    'RandomState',
]
```
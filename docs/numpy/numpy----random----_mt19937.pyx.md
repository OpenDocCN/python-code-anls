# `.\numpy\numpy\random\_mt19937.pyx`

```py
#cython: binding=True
# 引入operator模块，用于操作符函数的标准操作
import operator

# 引入numpy库，并且使用Cython语法进行导入
import numpy as np
cimport numpy as np

# 从C标准库libc中导入stdint头文件中的uint32_t和uint64_t类型
from libc.stdint cimport uint32_t, uint64_t
# 从numpy.random中导入BitGenerator和SeedSequence类
from numpy.random cimport BitGenerator, SeedSequence

# 定义模块的公开接口，仅包含MT19937
__all__ = ['MT19937']

# 调用numpy的C API导入数组对象
np.import_array()

# 从"src/mt19937/mt19937.h"头文件中导入以下内容
cdef extern from "src/mt19937/mt19937.h":

    # 定义C结构体s_mt19937_state，包含uint32_t类型的key数组和int类型的pos变量
    struct s_mt19937_state:
        uint32_t key[624]
        int pos

    # 将s_mt19937_state重命名为mt19937_state
    ctypedef s_mt19937_state mt19937_state

    # 定义以下C函数原型，使用nogil标记以避免GIL的影响
    uint64_t mt19937_next64(mt19937_state *state)  nogil
    uint32_t mt19937_next32(mt19937_state *state)  nogil
    double mt19937_next_double(mt19937_state *state)  nogil
    void mt19937_init_by_array(mt19937_state *state, uint32_t *init_key, int key_length)
    void mt19937_seed(mt19937_state *state, uint32_t seed)
    void mt19937_jump(mt19937_state *state)

    # 定义常量RK_STATE_LEN
    enum:
        RK_STATE_LEN

# 定义C函数mt19937_uint64，返回mt19937_next64的结果，避免C++异常处理
cdef uint64_t mt19937_uint64(void *st) noexcept nogil:
    return mt19937_next64(<mt19937_state *> st)

# 定义C函数mt19937_uint32，返回mt19937_next32的结果，避免C++异常处理
cdef uint32_t mt19937_uint32(void *st) noexcept nogil:
    return mt19937_next32(<mt19937_state *> st)

# 定义C函数mt19937_double，返回mt19937_next_double的结果，避免C++异常处理
cdef double mt19937_double(void *st) noexcept nogil:
    return mt19937_next_double(<mt19937_state *> st)

# 定义C函数mt19937_raw，返回mt19937_next32的结果作为uint64_t，避免C++异常处理
cdef uint64_t mt19937_raw(void *st) noexcept nogil:
    return <uint64_t>mt19937_next32(<mt19937_state *> st)

# 定义MT19937类，作为BitGenerator的子类
cdef class MT19937(BitGenerator):
    """
    MT19937(seed=None)

    Mersenne Twister伪随机数生成器的容器类。

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        初始化BitGenerator的种子。如果为None，则从操作系统获取新的不可预测熵。
        如果传递一个int或array_like[ints]，则将传递给SeedSequence以派生初始BitGenerator状态。
        也可以传递SeedSequence实例。

    Attributes
    ----------
    lock: threading.Lock
        共享的锁实例，以便在多个生成器中使用相同的位生成器而不会损坏状态。
        从位生成器生成值的代码应持有位生成器的锁。

    Notes
    -----
    MT19937提供一个包含函数指针的胶囊，用于生成双精度数，以及无符号32位和64位整数。
    这些在Python中不能直接消耗，必须由Generator或类似对象消耗，支持低级访问。

    Python标准库模块"random"也包含一个Mersenne Twister伪随机数生成器。

    **状态和种子**

    MT19937状态向量包括一个624元素的32位无符号整数数组，以及一个介于0和624之间的单个整数值，
    用于索引主数组中的当前位置。

    输入种子由SeedSequence处理以填充整个状态。第一个元素被重置，只设置其最高位。

    **并行特性**

    在并行应用程序中使用BitGenerator的首选方法是使用

    """
    """
    cdef mt19937_state rng_state
    """

    # 定义 MT19937 状态结构体变量 rng_state

    def __init__(self, seed=None):
        # 调用父类 BitGenerator 的初始化方法，传入种子值
        BitGenerator.__init__(self, seed)
        # 使用种子序列生成状态值，长度为 RK_STATE_LEN，数据类型为 np.uint32
        val = self._seed_seq.generate_state(RK_STATE_LEN, np.uint32)
        # 将状态数组的第一个元素设为 0x80000000UL，确保非零初始数组
        self.rng_state.key[0] = 0x80000000UL
        # 遍历生成的状态值，将其赋给 rng_state 的 key 数组
        for i in range(1, RK_STATE_LEN):
            self.rng_state.key[i] = val[i]
        # 设置 rng_state 的 pos 属性为 i
        self.rng_state.pos = i

        # 将 _bitgen 的 state 指向 rng_state 的地址
        self._bitgen.state = &self.rng_state
        # 将 _bitgen 的各种生成函数指向相应的 MT19937 版本的函数
        self._bitgen.next_uint64 = &mt19937_uint64
        self._bitgen.next_uint32 = &mt19937_uint32
        self._bitgen.next_double = &mt19937_double
        self._bitgen.next_raw = &mt19937_raw
    # 使用_legacy_seeding方法来对随机数生成器进行向后兼容的种子设置
    def _legacy_seeding(self, seed):
        """
        _legacy_seeding(seed)

        Seed the generator in a backward compatible way. For modern
        applications, creating a new instance is preferable. Calling this
        overrides self._seed_seq

        Parameters
        ----------
        seed : {None, int, array_like}
            Random seed initializing the pseudo-random number generator.
            Can be an integer in [0, 2**32-1], array of integers in
            [0, 2**32-1], a `SeedSequence, or ``None``. If `seed`
            is ``None``, then fresh, unpredictable entropy will be pulled from
            the OS.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        # 定义一个NumPy数组对象
        cdef np.ndarray obj
        # 使用self.lock进行线程安全操作
        with self.lock:
            try:
                # 如果seed为None，则使用SeedSequence生成种子
                if seed is None:
                    seed = SeedSequence()
                    # 生成长度为RK_STATE_LEN的状态值
                    val = seed.generate_state(RK_STATE_LEN)
                    # 最高位设置为1，确保初始数组非零
                    self.rng_state.key[0] = 0x80000000UL
                    # 将生成的状态值填充到rng_state.key数组中
                    for i in range(1, RK_STATE_LEN):
                        self.rng_state.key[i] = val[i]
                else:
                    # 如果seed具有squeeze属性，则调用squeeze方法
                    if hasattr(seed, 'squeeze'):
                        seed = seed.squeeze()
                    # 将seed转换为整数索引
                    idx = operator.index(seed)
                    # 如果索引超出范围[0, 2**32 - 1]，则抛出ValueError异常
                    if idx > int(2**32 - 1) or idx < 0:
                        raise ValueError("Seed must be between 0 and 2**32 - 1")
                    # 使用mt19937_seed函数对rng_state进行种子初始化
                    mt19937_seed(&self.rng_state, seed)
            # 捕获TypeError异常
            except TypeError:
                # 将seed转换为NumPy数组对象
                obj = np.asarray(seed)
                # 如果数组大小为0，则抛出ValueError异常
                if obj.size == 0:
                    raise ValueError("Seed must be non-empty")
                # 将数组转换为64位整数类型，安全类型转换
                obj = obj.astype(np.int64, casting='safe')
                # 如果数组不是一维的，则抛出ValueError异常
                if obj.ndim != 1:
                    raise ValueError("Seed array must be 1-d")
                # 如果数组中的值超出范围[0, 2**32 - 1]，则抛出ValueError异常
                if ((obj > int(2**32 - 1)) | (obj < 0)).any():
                    raise ValueError("Seed must be between 0 and 2**32 - 1")
                # 将数组转换为32位无符号整数类型，不安全类型转换，按C顺序存储
                obj = obj.astype(np.uint32, casting='unsafe', order='C')
                # 使用mt19937_init_by_array函数对rng_state进行数组初始化
                mt19937_init_by_array(&self.rng_state, <uint32_t*> obj.data, np.PyArray_DIM(obj, 0))
        # 将_seed_seq设置为None
        self._seed_seq = None

    # 使用jump_inplace方法来就地跳转状态
    cdef jump_inplace(self, iter):
        """
        Jump state in-place

        Not part of public API

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.
        """
        # 定义一个NumPy数组索引对象
        cdef np.npy_intp i
        # 循环iter次，调用mt19937_jump函数跳转rng_state的状态
        for i in range(iter):
            mt19937_jump(&self.rng_state)
    def jumped(self, np.npy_intp jumps=1):
        """
        jumped(jumps=1)

        Returns a new bit generator with the state jumped

        The state of the returned bit generator is jumped as-if
        2**(128 * jumps) random numbers have been generated.

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : MT19937
            New instance of generator jumped iter times

        Notes
        -----
        The jump step is computed using a modified version of Matsumoto's
        implementation of Horner's method. The step polynomial is precomputed
        to perform 2**128 steps. The jumped state has been verified to match
        the state produced using Matsumoto's original code.

        References
        ----------
        .. [1] Matsumoto, M, Generating multiple disjoint streams of
           pseudorandom number sequences.  Accessed on: May 6, 2020.
           http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/JUMP/
        .. [2] Hiroshi Haramoto, Makoto Matsumoto, Takuji Nishimura, François
           Panneton, Pierre L\'Ecuyer, "Efficient Jump Ahead for F2-Linear
           Random Number Generators", INFORMS JOURNAL ON COMPUTING, Vol. 20,
           No. 3, Summer 2008, pp. 385-390.
        """
        cdef MT19937 bit_generator

        # 创建一个新的 MT19937 实例
        bit_generator = self.__class__()
        # 将当前对象的状态复制到新实例
        bit_generator.state = self.state
        # 调用 jump_inplace 方法，跳转状态 jumps 次
        bit_generator.jump_inplace(jumps)

        return bit_generator

    @property
    def state(self):
        """
        Get or set the PRNG state

        Returns
        -------
        state : dict
            Dictionary containing the information required to describe the
            state of the PRNG
        """
        # 创建一个包含当前 PRNG 状态信息的字典
        key = np.zeros(624, dtype=np.uint32)
        for i in range(624):
            key[i] = self.rng_state.key[i]

        return {'bit_generator': self.__class__.__name__,
                'state': {'key': key, 'pos': self.rng_state.pos}}

    @state.setter
    def state(self, value):
        # 如果传入的状态是元组类型，则转换为字典
        if isinstance(value, tuple):
            if value[0] != 'MT19937' or len(value) not in (3, 5):
                raise ValueError('state is not a legacy MT19937 state')
            value ={'bit_generator': 'MT19937',
                    'state': {'key': value[1], 'pos': value[2]}}

        # 如果状态不是字典类型，则抛出类型错误
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        # 获取状态字典中的 bit_generator 键，验证是否匹配当前类名
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        # 将状态字典中的 key 和 pos 值分别复制到当前对象的 rng_state 中
        key = value['state']['key']
        for i in range(624):
            self.rng_state.key[i] = key[i]
        self.rng_state.pos = value['state']['pos']
```
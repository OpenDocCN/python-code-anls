# `D:\src\scipysrc\numpy\numpy\random\_sfc64.pyx`

```py
#cython: binding=True
# 导入所需的库
import numpy as np
# 导入Cython特定的numpy功能
cimport numpy as np

# 从C标准库中导入数据类型定义
from libc.stdint cimport uint32_t, uint64_t
# 导入Cython模块中的函数
from ._common cimport uint64_to_double
# 从numpy中导入BitGenerator类
from numpy.random cimport BitGenerator

# 定义外部C语言头文件中的数据结构和函数接口
__all__ = ['SFC64']

cdef extern from "src/sfc64/sfc64.h":
    # 定义C语言中的结构体s_sfc64_state
    struct s_sfc64_state:
        uint64_t s[4]
        int has_uint32
        uint32_t uinteger

    # 使用Cython类型定义sfc64_state结构体
    ctypedef s_sfc64_state sfc64_state
    # 声明调用C语言函数的原型
    uint64_t sfc64_next64(sfc64_state *state)  nogil
    uint32_t sfc64_next32(sfc64_state *state)  nogil
    void sfc64_set_seed(sfc64_state *state, uint64_t *seed)
    void sfc64_get_state(sfc64_state *state, uint64_t *state_arr, int *has_uint32, uint32_t *uinteger)
    void sfc64_set_state(sfc64_state *state, uint64_t *state_arr, int has_uint32, uint32_t uinteger)

# 定义C语言函数的Cython包装器函数
cdef uint64_t sfc64_uint64(void* st) noexcept nogil:
    return sfc64_next64(<sfc64_state *>st)

cdef uint32_t sfc64_uint32(void *st) noexcept nogil:
    return sfc64_next32(<sfc64_state *> st)

cdef double sfc64_double(void* st) noexcept nogil:
    return uint64_to_double(sfc64_next64(<sfc64_state *>st))


# 定义SFC64类，继承自BitGenerator类
cdef class SFC64(BitGenerator):
    """
    SFC64(seed=None)

    BitGenerator for Chris Doty-Humphrey's Small Fast Chaotic PRNG.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.

    Notes
    -----
    `SFC64` is a 256-bit implementation of Chris Doty-Humphrey's Small Fast
    Chaotic PRNG ([1]_). `SFC64` has a few different cycles that one might be
    on, depending on the seed; the expected period will be about
    :math:`2^{255}` ([2]_). `SFC64` incorporates a 64-bit counter which means
    that the absolute minimum cycle length is :math:`2^{64}` and that distinct
    seeds will not run into each other for at least :math:`2^{64}` iterations.

    `SFC64` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a `Generator`
    or similar object that supports low-level access.

    **State and Seeding**

    The `SFC64` state vector consists of 4 unsigned 64-bit values. The last
    is a 64-bit counter that increments by 1 each iteration.

    The input seed is processed by `SeedSequence` to generate the first
    3 values, then the `SFC64` algorithm is iterated a small number of times
    to mix.

    **Compatibility Guarantee**

    `SFC64` makes a guarantee that a fixed seed will always produce the same
    random integer stream.

    References
    ----------
    .. [1] `"PractRand"
            <https://pracrand.sourceforge.net/RNG_engines.txt>`_

    """
    pass
    # 引用和描述
    .. [2] `"Random Invertible Mapping Statistics"
            <https://www.pcg-random.org/posts/random-invertible-mapping-statistics.html>`_
    """

    # 定义 C 扩展类型变量 rng_state，用于存储 SFC64 状态
    cdef sfc64_state rng_state

    # 初始化函数，用于创建 SFC64 PRNG 对象
    def __init__(self, seed=None):
        # 调用 BitGenerator 的初始化函数，传递种子参数
        BitGenerator.__init__(self, seed)
        # 将 RNG 状态绑定到 _bitgen
        self._bitgen.state = <void *>&self.rng_state
        # 设置 _bitgen 的下一个随机数生成函数为 SFC64 的具体实现函数
        self._bitgen.next_uint64 = &sfc64_uint64
        self._bitgen.next_uint32 = &sfc64_uint32
        self._bitgen.next_double = &sfc64_double
        self._bitgen.next_raw = &sfc64_uint64
        # 通过种子序列生成状态向量
        val = self._seed_seq.generate_state(3, np.uint64)
        # 使用生成的状态向量设置 SFC64 RNG 的初始状态
        sfc64_set_seed(&self.rng_state, <uint64_t*>np.PyArray_DATA(val))
        # 重置对象的状态变量
        self._reset_state_variables()

    # 重置对象状态变量的私有方法
    cdef _reset_state_variables(self):
        # 将状态变量标志为无效状态
        self.rng_state.has_uint32 = 0
        # 将整数缓存设置为 0
        self.rng_state.uinteger = 0

    # state 属性的 getter 方法，用于获取 PRNG 的状态
    @property
    def state(self):
        """
        获取或设置 PRNG 的状态

        Returns
        -------
        state : dict
            包含描述 PRNG 状态所需信息的字典
        """
        cdef np.ndarray state_vec
        cdef int has_uint32
        cdef uint32_t uinteger

        # 创建一个包含四个 uint64 类型元素的数组作为状态向量
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        # 获取当前 RNG 状态并存储在 state_vec 中
        sfc64_get_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        &has_uint32, &uinteger)
        # 返回包含 PRNG 状态信息的字典
        return {'bit_generator': self.__class__.__name__,
                'state': {'state': state_vec},
                'has_uint32': has_uint32,
                'uinteger': uinteger}

    # state 属性的 setter 方法，用于设置 PRNG 的状态
    @state.setter
    def state(self, value):
        cdef np.ndarray state_vec
        cdef int has_uint32
        cdef uint32_t uinteger
        # 如果 value 不是字典类型，则抛出类型错误
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        # 获取字典中的 bit_generator 键值，检查是否与当前类名匹配
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'RNG'.format(self.__class__.__name__))
        # 创建一个包含四个 uint64 类型元素的数组 state_vec，并将状态向量值复制到其中
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        state_vec[:] = value['state']['state']
        # 获取 has_uint32 和 uinteger 值
        has_uint32 = value['has_uint32']
        uinteger = value['uinteger']
        # 使用提供的状态设置 SFC64 RNG 的状态
        sfc64_set_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        has_uint32, uinteger)
```
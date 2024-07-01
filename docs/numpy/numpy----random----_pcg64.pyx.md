# `.\numpy\numpy\random\_pcg64.pyx`

```py
#cython: binding=True
# 导入必要的库和模块
import numpy as np
# 导入 C 扩展中需要的 numpy 模块
cimport numpy as np

# 从 C 标准库中导入指定类型
from libc.stdint cimport uint32_t, uint64_t
# 从 _common 模块中导入特定函数和类型
from ._common cimport uint64_to_double, wrap_int
# 从 numpy.random 中导入 BitGenerator 类
from numpy.random cimport BitGenerator

# 定义公开的类和函数列表
__all__ = ['PCG64']

# 从 C 头文件 pcg64.h 中声明 extern 函数和类型
cdef extern from "src/pcg64/pcg64.h":
    # 使用 int 作为通用类型，具体类型从 pcg64.h 中读取，与平台相关
    ctypedef int pcg64_random_t

    # 定义结构体 s_pcg64_state，包含 PCG 状态、是否有未使用的 uint32 和 uint32 值
    struct s_pcg64_state:
        pcg64_random_t *pcg_state
        int has_uint32
        uint32_t uinteger

    ctypedef s_pcg64_state pcg64_state

    # 定义 PCG64 的各种操作函数原型
    uint64_t pcg64_next64(pcg64_state *state)  nogil
    uint32_t pcg64_next32(pcg64_state *state)  nogil
    void pcg64_jump(pcg64_state *state)
    void pcg64_advance(pcg64_state *state, uint64_t *step)
    void pcg64_set_seed(pcg64_state *state, uint64_t *seed, uint64_t *inc)
    void pcg64_get_state(pcg64_state *state, uint64_t *state_arr, int *has_uint32, uint32_t *uinteger)
    void pcg64_set_state(pcg64_state *state, uint64_t *state_arr, int has_uint32, uint32_t uinteger)

    uint64_t pcg64_cm_next64(pcg64_state *state)  noexcept nogil
    uint32_t pcg64_cm_next32(pcg64_state *state)  noexcept nogil
    void pcg64_cm_advance(pcg64_state *state, uint64_t *step)

# 定义 C 函数用于返回 PCG64 生成的 uint64 随机数，输入为指向 PCG64 状态的指针
cdef uint64_t pcg64_uint64(void* st) noexcept nogil:
    return pcg64_next64(<pcg64_state *>st)

# 定义 C 函数用于返回 PCG64 生成的 uint32 随机数，输入为指向 PCG64 状态的指针
cdef uint32_t pcg64_uint32(void *st) noexcept nogil:
    return pcg64_next32(<pcg64_state *> st)

# 定义 C 函数用于返回 PCG64 生成的 double 类型随机数，输入为指向 PCG64 状态的指针
cdef double pcg64_double(void* st) noexcept nogil:
    return uint64_to_double(pcg64_next64(<pcg64_state *>st))

# 定义 C 函数用于返回 PCG64 生成的 uint64 随机数（另一种变体），输入为指向 PCG64 状态的指针
cdef uint64_t pcg64_cm_uint64(void* st) noexcept nogil:
    return pcg64_cm_next64(<pcg64_state *>st)

# 定义 C 函数用于返回 PCG64 生成的 uint32 随机数（另一种变体），输入为指向 PCG64 状态的指针
cdef uint32_t pcg64_cm_uint32(void *st) noexcept nogil:
    return pcg64_cm_next32(<pcg64_state *> st)

# 定义 C 函数用于返回 PCG64 生成的 double 类型随机数（另一种变体），输入为指向 PCG64 状态的指针
cdef double pcg64_cm_double(void* st) noexcept nogil:
    return uint64_to_double(pcg64_cm_next64(<pcg64_state *>st))

# 定义 PCG64 类，继承自 BitGenerator 类
cdef class PCG64(BitGenerator):
    """
    PCG64(seed=None)

    BitGenerator for the PCG-64 pseudo-random number generator.

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
    PCG-64 is a 128-bit implementation of O'Neill's permutation congruential
    generator ([1]_, [2]_). PCG-64 has a period of :math:`2^{128}` and supports
    advancing an arbitrary number of steps as well as :math:`2^{127}` streams.
    The specific member of the PCG family that we use is PCG XSL RR 128/64
    as described in the paper ([2]_).

    `PCG64` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a `Generator`
    """
    # 定义 C 语言扩展中使用的 PCG64 RNG 的状态结构
    cdef pcg64_state rng_state
    # 定义 PCG64 随机数生成器对象的结构
    cdef pcg64_random_t pcg64_random_state

    # 初始化方法，可选参数 seed 用于设置随机数种子
    def __init__(self, seed=None):
        # 调用 BitGenerator 的初始化方法，并传递种子参数
        BitGenerator.__init__(self, seed)
        # 将 RNG 状态中的 pcg_state 成员设置为 pcg64_random_state 的地址
        self.rng_state.pcg_state = &self.pcg64_random_state

        # 设置 _bitgen 对象的状态指针为 rng_state 的地址
        self._bitgen.state = <void *>&self.rng_state
        # 设置 _bitgen 对象使用的下一个 uint64 随机数生成函数
        self._bitgen.next_uint64 = &pcg64_uint64
        # 设置 _bitgen 对象使用的下一个 uint32 随机数生成函数
        self._bitgen.next_uint32 = &pcg64_uint32
        # 设置 _bitgen 对象使用的下一个 double 随机数生成函数
        self._bitgen.next_double = &pcg64_double
        # 设置 _bitgen 对象使用的下一个 raw 随机数生成函数
        self._bitgen.next_raw = &pcg64_uint64
        # 使用种子生成状态，并设置给 _bitgen
        val = self._seed_seq.generate_state(4, np.uint64)
        # 调用 C 函数 pcg64_set_seed 来设置 RNG 的种子
        pcg64_set_seed(&self.rng_state,
                       <uint64_t *>np.PyArray_DATA(val),
                       (<uint64_t *>np.PyArray_DATA(val) + 2))
        # 重置状态变量
        self._reset_state_variables()

    # 重置 RNG 的状态变量
    cdef _reset_state_variables(self):
        # 将 RNG 的 has_uint32 状态变量设置为 0
        self.rng_state.has_uint32 = 0
        # 将 RNG 的 uinteger 状态变量设置为 0

    # 在原地进行 RNG 状态的跳跃
    # 注意：不是公共 API 的一部分
    cdef jump_inplace(self, jumps):
        """
        在原地跳跃 RNG 的状态

        参数
        ----------
        jumps : 整数，正数
            要跳跃 RNG 状态的次数。

        注意
        -----
        当乘以 2**128 时，步长为黄金分割率 phi-1。
        """
        # 使用常量步长进行 RNG 的 advance 操作
        step = 0x9e3779b97f4a7c15f39cc0605cedc835
        self.advance(step * int(jumps))
    def jumped(self, jumps=1):
        """
        jumped(jumps=1)

        Returns a new bit generator with the state jumped.

        Jumps the state as-if jumps * 210306068529402873165736369884012333109
        random numbers have been generated.

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : PCG64
            New instance of generator jumped iter times

        Notes
        -----
        The step size is phi-1 when multiplied by 2**128 where phi is the
        golden ratio.
        """
        # 创建一个新的 PCG64 位生成器对象
        cdef PCG64 bit_generator
        bit_generator = self.__class__()

        # 将当前对象的状态赋给新生成的 bit_generator
        bit_generator.state = self.state

        # 调用 jump_inplace 方法来跳跃状态 jumps 次
        bit_generator.jump_inplace(jumps)

        # 返回新的 bit_generator 对象
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
        # 定义相关变量和数据类型
        cdef np.ndarray state_vec
        cdef int has_uint32
        cdef uint32_t uinteger

        # 创建一个长度为 4 的空 np.uint64 类型的数组 state_vec
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)

        # 调用底层函数 pcg64_get_state 获取 PRNG 的状态信息
        pcg64_get_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        &has_uint32, &uinteger)

        # 从 state_vec 数组中提取状态值和增量值
        state = int(state_vec[0]) * 2**64 + int(state_vec[1])
        inc = int(state_vec[2]) * 2**64 + int(state_vec[3])

        # 返回包含 PRNG 状态信息的字典
        return {'bit_generator': self.__class__.__name__,
                'state': {'state': state, 'inc': inc},
                'has_uint32': has_uint32,
                'uinteger': uinteger}

    @state.setter
    def state(self, value):
        # 定义相关变量和数据类型
        cdef np.ndarray state_vec
        cdef int has_uint32
        cdef uint32_t uinteger

        # 检查输入的状态是否为字典类型
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')

        # 检查状态字典中的生成器类型是否与当前类匹配
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'RNG'.format(self.__class__.__name__))

        # 创建一个长度为 4 的空 np.uint64 类型的数组 state_vec
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)

        # 将状态字典中的状态和增量值分别存入 state_vec 数组的不同位置
        state_vec[0] = value['state']['state'] // 2 ** 64
        state_vec[1] = value['state']['state'] % 2 ** 64
        state_vec[2] = value['state']['inc'] // 2 ** 64
        state_vec[3] = value['state']['inc'] % 2 ** 64

        # 获取状态字典中的 has_uint32 和 uinteger 值
        has_uint32 = value['has_uint32']
        uinteger = value['uinteger']

        # 调用底层函数 pcg64_set_state 设置 PRNG 的状态信息
        pcg64_set_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        has_uint32, uinteger)
    def advance(self, delta):
        """
        advance(delta)

        Advance the underlying RNG as-if delta draws have occurred.

        Parameters
        ----------
        delta : integer, positive
            Number of draws to advance the RNG. Must be less than the
            size state variable in the underlying RNG.

        Returns
        -------
        self : PCG64
            RNG advanced delta steps

        Notes
        -----
        Advancing a RNG updates the underlying RNG state as-if a given
        number of calls to the underlying RNG have been made. In general
        there is not a one-to-one relationship between the number output
        random values from a particular distribution and the number of
        draws from the core RNG.  This occurs for two reasons:

        * The random values are simulated using a rejection-based method
          and so, on average, more than one value from the underlying
          RNG is required to generate an single draw.
        * The number of bits required to generate a simulated value
          differs from the number of bits generated by the underlying
          RNG.  For example, two 16-bit integer values can be simulated
          from a single draw of a 32-bit RNG.

        Advancing the RNG state resets any pre-computed random numbers.
        This is required to ensure exact reproducibility.
        """
        # Ensure delta is wrapped to fit within 128-bit integer size
        delta = wrap_int(delta, 128)

        # Create a numpy array d to hold two 64-bit unsigned integers
        cdef np.ndarray d = np.empty(2, dtype=np.uint64)
        # Split delta into two parts: high and low 64-bit integers
        d[0] = delta // 2**64
        d[1] = delta % 2**64
        # Call C function pcg64_advance with address of rng_state and data in d
        pcg64_advance(&self.rng_state, <uint64_t *>np.PyArray_DATA(d))
        # Reset internal state variables after advancing RNG state
        self._reset_state_variables()
        # Return self to allow method chaining and indicate RNG advanced
        return self
# 定义一个 Cython 类 `PCG64DXSM`，继承自 `BitGenerator` 类
cdef class PCG64DXSM(BitGenerator):
    """
    PCG64DXSM(seed=None)

    BitGenerator for the PCG-64 DXSM pseudo-random number generator.

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
    PCG-64 DXSM is a 128-bit implementation of O'Neill's permutation congruential
    generator ([1]_, [2]_). PCG-64 DXSM has a period of :math:`2^{128}` and supports
    advancing an arbitrary number of steps as well as :math:`2^{127}` streams.
    The specific member of the PCG family that we use is PCG CM DXSM 128/64. It
    differs from `PCG64` in that it uses the stronger DXSM output function,
    a 64-bit "cheap multiplier" in the LCG, and outputs from the state before
    advancing it rather than advance-then-output.

    `PCG64DXSM` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a `Generator`
    or similar object that supports low-level access.

    Supports the method :meth:`advance` to advance the RNG an arbitrary number of
    steps. The state of the PCG-64 DXSM RNG is represented by 2 128-bit unsigned
    integers.

    **State and Seeding**

    The `PCG64DXSM` state vector consists of 2 unsigned 128-bit values,
    which are represented externally as Python ints. One is the state of the
    PRNG, which is advanced by a linear congruential generator (LCG). The
    second is a fixed odd increment used in the LCG.

    The input seed is processed by `SeedSequence` to generate both values. The
    increment is not independently settable.

    **Parallel Features**

    The preferred way to use a BitGenerator in parallel applications is to use
    the `SeedSequence.spawn` method to obtain entropy values, and to use these
    to generate new BitGenerators:

    >>> from numpy.random import Generator, PCG64DXSM, SeedSequence
    >>> sg = SeedSequence(1234)
    >>> rg = [Generator(PCG64DXSM(s)) for s in sg.spawn(10)]

    **Compatibility Guarantee**

    `PCG64DXSM` makes a guarantee that a fixed seed will always produce
    the same random integer stream.

    References
    ----------
    .. [1] `"PCG, A Family of Better Random Number Generators"
           <http://www.pcg-random.org/>`_
    .. [2] O'Neill, Melissa E. `"PCG: A Family of Simple Fast Space-Efficient
           Statistically Good Algorithms for Random Number Generation"
           <https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf>`_
    """
    # 定义 Cython 结构体变量 rng_state 和 pcg64_random_state
    cdef pcg64_state rng_state
    cdef pcg64_random_t pcg64_random_state
    # 初始化方法，用于创建一个新的随机数生成器对象
    def __init__(self, seed=None):
        # 调用父类 BitGenerator 的初始化方法，传入种子值
        BitGenerator.__init__(self, seed)
        # 设置当前对象的 rng_state.pcg_state 指向 self.pcg64_random_state 的地址
        self.rng_state.pcg_state = &self.pcg64_random_state

        # 将 self._bitgen.state 设置为指向 self.rng_state 的空指针
        self._bitgen.state = <void *>&self.rng_state
        # 设置不同类型的随机数生成函数，如整数、双精度数等
        self._bitgen.next_uint64 = &pcg64_cm_uint64
        self._bitgen.next_uint32 = &pcg64_cm_uint32
        self._bitgen.next_double = &pcg64_cm_double
        self._bitgen.next_raw = &pcg64_cm_uint64
        
        # 使用种子序列生成状态，设置初始种子值
        # 生成一个包含 4 个 np.uint64 类型数据的数组
        val = self._seed_seq.generate_state(4, np.uint64)
        # 调用 pcg64_set_seed 函数，设置随机数生成器状态的种子
        pcg64_set_seed(&self.rng_state,
                       <uint64_t *>np.PyArray_DATA(val),
                       (<uint64_t *>np.PyArray_DATA(val) + 2))
        # 重置对象的状态变量
        self._reset_state_variables()

    # 重置状态变量的私有方法
    cdef _reset_state_variables(self):
        # 将 rng_state.has_uint32 置为 0
        self.rng_state.has_uint32 = 0
        # 将 rng_state.uinteger 置为 0
        self.rng_state.uinteger = 0

    # 跳转状态的私有方法，不是公共 API 的一部分
    cdef jump_inplace(self, jumps):
        """
        Jump state in-place
        Not part of public API

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the rng.

        Notes
        -----
        The step size is phi-1 when multiplied by 2**128 where phi is the
        golden ratio.
        """
        # 定义步长为固定的十六进制数
        step = 0x9e3779b97f4a7c15f39cc0605cedc835
        # 调用 advance 方法，以 step * jumps 的步长前进状态
        self.advance(step * int(jumps))

    # 返回经过跳转的新随机数生成器对象
    def jumped(self, jumps=1):
        """
        jumped(jumps=1)

        Returns a new bit generator with the state jumped.

        Jumps the state as-if jumps * 210306068529402873165736369884012333109
        random numbers have been generated.

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : PCG64DXSM
            New instance of generator jumped iter times

        Notes
        -----
        The step size is phi-1 when multiplied by 2**128 where phi is the
        golden ratio.
        """
        # 声明一个新的 PCG64DXSM 类型的 bit_generator 对象
        cdef PCG64DXSM bit_generator

        # 使用当前类的构造函数创建一个新的 bit_generator 对象
        bit_generator = self.__class__()
        # 将新对象的状态设置为当前对象的状态
        bit_generator.state = self.state
        # 调用 jump_inplace 方法，跳转状态 jumps 次
        bit_generator.jump_inplace(jumps)

        # 返回跳转后的新 bit_generator 对象
        return bit_generator
    def state(self):
        """
        Get or set the PRNG state

        Returns
        -------
        state : dict
            Dictionary containing the information required to describe the
            state of the PRNG
        """
        # 定义 Cython 变量：状态向量和布尔值
        cdef np.ndarray state_vec
        cdef int has_uint32
        cdef uint32_t uinteger

        # state_vec 包含 state.high, state.low, inc.high, inc.low 的状态信息
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        
        # 调用 C 函数 pcg64_get_state 从 RNG 状态中获取信息并存入 state_vec
        pcg64_get_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        &has_uint32, &uinteger)
        
        # 将 state_vec 中的数据解析为整数形式的 state 和 inc
        state = int(state_vec[0]) * 2**64 + int(state_vec[1])
        inc = int(state_vec[2]) * 2**64 + int(state_vec[3])
        
        # 返回描述 PRNG 状态的字典
        return {'bit_generator': self.__class__.__name__,
                'state': {'state': state, 'inc': inc},
                'has_uint32': has_uint32,
                'uinteger': uinteger}

    @state.setter
    def state(self, value):
        # 定义 Cython 变量：状态向量和布尔值
        cdef np.ndarray state_vec
        cdef int has_uint32
        cdef uint32_t uinteger
        
        # 检查输入的状态是否为字典类型
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        
        # 检查状态字典中的 bit_generator 是否匹配当前 RNG 类型
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'RNG'.format(self.__class__.__name__))
        
        # 创建一个空的状态向量数组，用来存储状态信息
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        
        # 将 state 和 inc 分别存入状态向量数组的前两个和后两个位置
        state_vec[0] = value['state']['state'] // 2 ** 64
        state_vec[1] = value['state']['state'] % 2 ** 64
        state_vec[2] = value['state']['inc'] // 2 ** 64
        state_vec[3] = value['state']['inc'] % 2 ** 64
        
        # 将 has_uint32 和 uinteger 从输入字典中获取并赋值
        has_uint32 = value['has_uint32']
        uinteger = value['uinteger']
        
        # 调用 C 函数 pcg64_set_state 来设置 RNG 的状态
        pcg64_set_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        has_uint32, uinteger)
    def advance(self, delta):
        """
        advance(delta)

        Advance the underlying RNG as-if delta draws have occurred.

        Parameters
        ----------
        delta : integer, positive
            Number of draws to advance the RNG. Must be less than the
            size state variable in the underlying RNG.

        Returns
        -------
        self : PCG64
            RNG advanced delta steps

        Notes
        -----
        Advancing a RNG updates the underlying RNG state as-if a given
        number of calls to the underlying RNG have been made. In general
        there is not a one-to-one relationship between the number output
        random values from a particular distribution and the number of
        draws from the core RNG.  This occurs for two reasons:

        * The random values are simulated using a rejection-based method
          and so, on average, more than one value from the underlying
          RNG is required to generate an single draw.
        * The number of bits required to generate a simulated value
          differs from the number of bits generated by the underlying
          RNG.  For example, two 16-bit integer values can be simulated
          from a single draw of a 32-bit RNG.

        Advancing the RNG state resets any pre-computed random numbers.
        This is required to ensure exact reproducibility.
        """
        # 对输入的 delta 进行处理，确保其在 128 位整数范围内
        delta = wrap_int(delta, 128)

        # 创建一个长度为 2 的无符号 64 位整数数组 d
        cdef np.ndarray d = np.empty(2, dtype=np.uint64)
        # 将 delta 拆分成两部分，存入数组 d 中
        d[0] = delta // 2**64
        d[1] = delta % 2**64
        # 调用 C 函数 pcg64_cm_advance，更新 RNG 状态，传入拆分后的数组 d
        pcg64_cm_advance(&self.rng_state, <uint64_t *>np.PyArray_DATA(d))
        # 重置内部状态变量，以确保精确可复现性
        self._reset_state_variables()
        # 返回当前对象 self，表示 RNG 已经前进了 delta 步
        return self
```
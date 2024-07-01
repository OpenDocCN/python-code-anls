# `.\numpy\numpy\random\_philox.pyx`

```py
#cython: binding=True
# 导入CPython的PyCapsule_New函数，用于创建Python对象的C扩展对象

from cpython.pycapsule cimport PyCapsule_New
# 导入NumPy库，并在Cython中声明其使用的接口

import numpy as np
cimport numpy as np

# 从C标准库中导入特定类型的整数
from libc.stdint cimport uint32_t, uint64_t

# 导入自定义Cython模块中的函数和类型声明
from ._common cimport uint64_to_double, int_to_array, wrap_int
from numpy.random cimport BitGenerator

# 将Philox类包含在__all__列表中，表示此类为公共接口的一部分
__all__ = ['Philox']

# 导入NumPy C API
np.import_array()

# 定义一个Cython的整数常量PHILOX_BUFFER_SIZE
cdef int PHILOX_BUFFER_SIZE=4

# 从头文件'src/philox/philox.h'中导入结构体和函数声明
cdef extern from 'src/philox/philox.h':
    struct s_r123array2x64:
        uint64_t v[2]

    struct s_r123array4x64:
        uint64_t v[4]

    ctypedef s_r123array4x64 philox4x64_ctr_t
    ctypedef s_r123array2x64 philox4x64_key_t

    struct s_philox_state:
        philox4x64_ctr_t *ctr
        philox4x64_key_t *key
        int buffer_pos
        uint64_t *buffer
        int has_uint32
        uint32_t uinteger

    ctypedef s_philox_state philox_state

    # 导入用于Philox算法的一些函数声明
    uint64_t philox_next64(philox_state *state)  noexcept nogil
    uint32_t philox_next32(philox_state *state)  noexcept nogil
    void philox_jump(philox_state *state)
    void philox_advance(uint64_t *step, philox_state *state)

# 定义C函数philox_uint64，返回64位无符号整数，接受void*类型的参数
cdef uint64_t philox_uint64(void* st) noexcept nogil:
    return philox_next64(<philox_state *> st)

# 定义C函数philox_uint32，返回32位无符号整数，接受void*类型的参数
cdef uint32_t philox_uint32(void *st) noexcept nogil:
    return philox_next32(<philox_state *> st)

# 定义C函数philox_double，返回双精度浮点数，接受void*类型的参数
cdef double philox_double(void* st) noexcept nogil:
    return uint64_to_double(philox_next64(<philox_state *> st))

# 定义Philox类，继承自BitGenerator类，用于实现Philox (4x64) PRNG
cdef class Philox(BitGenerator):
    """
    Philox(seed=None, counter=None, key=None)

    Container for the Philox (4x64) pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.
    counter : {None, int, array_like}, optional
        Counter to use in the Philox state. Can be either
        a Python int (long in 2.x) in [0, 2**256) or a 4-element uint64 array.
        If not provided, the RNG is initialized at 0.
    key : {None, int, array_like}, optional
        Key to use in the Philox state.  Unlike ``seed``, the value in key is
        directly set. Can be either a Python int in [0, 2**128) or a 2-element
        uint64 array. `key` and ``seed`` cannot both be used.

    Attributes
    ----------
    lock: threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.

    Notes
    -----
    Philox is a 64-bit PRNG that uses a counter-based design based on weaker
    (and faster) versions of cryptographic functions [1]_. Instances using
    """
    # 类文档字符串，描述Philox类及其初始化参数和特性
    pass
    # 导入必要的模块和类
    from numpy.random import BitGenerator
    
    # 定义 Philox 类，继承自 BitGenerator 类
    class Philox(BitGenerator):
        """
        Philox is a cryptographic random number generator.
        
        Philox can generate independent sequences by using different keys. It has a 
        period of 2^256 - 1 and supports arbitrary advancing and jumping by 2^128 increments.
    
        Philox provides function pointers for producing doubles, and unsigned 32-bit and 64-bit integers.
        These are not directly usable in Python and require a `Generator` or similar object for access.
    
        State and Seeding:
        - Philox state includes a 256-bit counter as a 4-element uint64 array and a 128-bit key 
          as a 2-element uint64 array.
        - The counter is incremented by 1 for every 4 64-bit randoms produced.
        - Different keys produce independent sequences.
    
        Usage:
        - Seed is processed by `SeedSequence` to generate the key. Counter is set to 0.
        - Alternatively, you can set the key and counter directly.
    
        Parallel Features:
        - Use `SeedSequence.spawn` for parallel applications to obtain entropy values and generate new BitGenerators.
        - `Philox.jumped()` advances state by 2^128 random numbers.
        - `Philox.advance(step)` advances counter by `step` (0 to 2^256-1).
        - Chaining `Philox` instances ensures segments are from the same sequence.
    
        Compatibility Guarantee:
        - Philox guarantees that the same seed will produce the same random stream.
    
        Examples:
        - Generate a random number using Philox:
          >>> from numpy.random import Generator, Philox
          >>> rg = Generator(Philox(1234))
          >>> rg.standard_normal()
    
        References:
        - [Link to references]
        """
        
        # 初始化方法，接受一个 seed 参数，默认为 None
        def __init__(self, seed=None):
            """
            Initialize Philox with a seed.
    
            Parameters
            ----------
            seed : {None, int, array_like}, optional
                Seed for generating random numbers. Default is None.
            """
            # 调用父类的初始化方法
            super().__init__()
            # 如果 seed 为 None，则初始化 key 和 counter 为空数组
            if seed is None:
                self.key = []
                self.counter = []
            else:
                # 否则使用 SeedSequence 处理 seed 生成 key 和设置 counter 为 0
                self.key = SeedSequence(seed).generate_state(2)
                self.counter = [0, 0]
    
        # 生成随机浮点数的方法
        def random(self):
            """Generate random floats."""
            # 返回随机浮点数
            return self._philox()
    
        # 生成随机 32 位无符号整数的方法
        def random_raw(self):
            """Generate random 32-bit unsigned integers."""
            # 返回随机 32 位无符号整数
            return self._philox()
    
        # 生成随机 64 位无符号整数的方法
        def random_raw_long(self):
            """Generate random 64-bit unsigned integers."""
            # 返回随机 64 位无符号整数
            return self._philox()
    
        # 用于调整状态的方法，接受一个步长参数
        def advance(self, step):
            """
            Advance the counter by a specified step.
    
            Parameters
            ----------
            step : int
                Step to advance the counter by.
            """
            # 将 counter[0] 增加 step
            self.counter[0] += step
    
        # 用于跳跃状态的方法
        def jumped(self):
            """Advance the state as if 2^128 random numbers have been generated."""
            # 调用 advance 方法，步长为 2^128
            self.advance(2 ** 128)
            # 返回当前实例
            return self
    
        # 内部方法，用于生成随机数
        def _philox(self):
            """Internal method for generating random numbers."""
            # 返回一个占位符，实际的实现细节未提供
            return NotImplemented
    """
        .. [1] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw,
               "Parallel Random Numbers: As Easy as 1, 2, 3," Proceedings of
               the International Conference for High Performance Computing,
               Networking, Storage and Analysis (SC11), New York, NY: ACM, 2011.
        """
    
        # 定义 Cython 类型变量
        cdef philox_state rng_state
        cdef philox4x64_key_t philox_key
        cdef philox4x64_ctr_t philox_ctr
    
        # 初始化函数，接受种子、计数器和密钥参数
        def __init__(self, seed=None, counter=None, key=None):
            # 如果种子和密钥都不为空，则引发 ValueError 异常
            if seed is not None and key is not None:
                raise ValueError('seed and key cannot be both used')
    
            # 调用 BitGenerator 的初始化方法，传递种子参数
            BitGenerator.__init__(self, seed)
    
            # 将 rng_state 的 ctr 和 key 成员指向对应的 Philox 状态的成员
            self.rng_state.ctr = &self.philox_ctr
            self.rng_state.key = &self.philox_key
    
            # 如果提供了密钥，则将其转换为数组并设置为 rng_state 的 key 成员
            if key is not None:
                key = int_to_array(key, 'key', 128, 64)
                for i in range(2):
                    self.rng_state.key.v[i] = key[i]
                # 设置种子序列为无效状态
                self._seed_seq = None
            else:
                # 否则，使用 _seed_seq 生成长度为 2 的 64 位无符号整数密钥
                key = self._seed_seq.generate_state(2, np.uint64)
                for i in range(2):
                    self.rng_state.key.v[i] = key[i]
    
            # 如果未提供计数器，默认为 0；将其转换为数组并设置为 rng_state 的 ctr 成员
            counter = 0 if counter is None else counter
            counter = int_to_array(counter, 'counter', 256, 64)
            for i in range(4):
                self.rng_state.ctr.v[i] = counter[i]
    
            # 重置状态变量
            self._reset_state_variables()
    
            # 将 _bitgen 的 state 成员指向 rng_state，并设置其余的方法指针
            self._bitgen.state = <void *>&self.rng_state
            self._bitgen.next_uint64 = &philox_uint64
            self._bitgen.next_uint32 = &philox_uint32
            self._bitgen.next_double = &philox_double
            self._bitgen.next_raw = &philox_uint64
    
        # 重置状态变量的私有方法
        cdef _reset_state_variables(self):
            cdef philox_state *rng_state = &self.rng_state
             
            # 初始化 rng_state 的各个成员变量
            rng_state[0].has_uint32 = 0
            rng_state[0].uinteger = 0
            rng_state[0].buffer_pos = PHILOX_BUFFER_SIZE
            for i in range(PHILOX_BUFFER_SIZE):
                rng_state[0].buffer[i] = 0
    
        # 状态属性的 getter 方法，返回 PRNG 的状态信息
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
            # 从 rng_state 中提取 ctr、key 和 buffer 的数据到 numpy 数组中
            ctr = np.empty(4, dtype=np.uint64)
            key = np.empty(2, dtype=np.uint64)
            buffer = np.empty(PHILOX_BUFFER_SIZE, dtype=np.uint64)
            for i in range(4):
                ctr[i] = self.rng_state.ctr.v[i]
                if i < 2:
                    key[i] = self.rng_state.key.v[i]
            for i in range(PHILOX_BUFFER_SIZE):
                buffer[i] = self.rng_state.buffer[i]
    
            # 构建并返回包含 PRNG 状态信息的字典
            state = {'counter': ctr, 'key': key}
            return {'bit_generator': self.__class__.__name__,
                    'state': state,
                    'buffer': buffer,
                    'buffer_pos': self.rng_state.buffer_pos,
                    'has_uint32': self.rng_state.has_uint32,
                    'uinteger': self.rng_state.uinteger}
    
        # 状态属性的 setter 方法
        @state.setter
    def state(self, value):
        # 检查输入值是否为字典类型，如果不是则抛出类型错误异常
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        
        # 从输入字典中获取 'bit_generator' 键对应的值
        bitgen = value.get('bit_generator', '')
        
        # 检查获取的 'bit_generator' 值是否与当前类名相同，如果不同则抛出数值错误异常
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} PRNG'.format(self.__class__.__name__))
        
        # 将输入字典中的 'state' 字典中的 'counter' 和 'key' 分别复制到 rng_state 对象中
        for i in range(4):
            self.rng_state.ctr.v[i] = <uint64_t> value['state']['counter'][i]
            if i < 2:
                self.rng_state.key.v[i] = <uint64_t> value['state']['key'][i]
        
        # 将输入字典中的 'buffer' 列表复制到 rng_state 对象的 buffer 数组中
        for i in range(PHILOX_BUFFER_SIZE):
            self.rng_state.buffer[i] = <uint64_t> value['buffer'][i]
        
        # 将输入字典中的 'has_uint32', 'uinteger' 和 'buffer_pos' 三个键对应的值分别赋给 rng_state 对象的相应属性
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']
        self.rng_state.buffer_pos = value['buffer_pos']

    cdef jump_inplace(self, iter):
        """
        Jump state in-place

        Not part of public API

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.
        """
        # 以 (2 ** 128) * iter 的步长向前推进状态
        self.advance(iter * int(2 ** 128))

    def jumped(self, jumps=1):
        """
        jumped(jumps=1)

        Returns a new bit generator with the state jumped

        The state of the returned bit generator is jumped as-if
        (2**128) * jumps random numbers have been generated.

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : Philox
            New instance of generator jumped iter times
        """
        # 创建一个新的 Philox 实例 bit_generator
        cdef Philox bit_generator
        bit_generator = self.__class__()
        
        # 将当前实例的状态复制到新实例的状态中
        bit_generator.state = self.state
        
        # 调用 jump_inplace 方法，跳跃指定次数的状态
        bit_generator.jump_inplace(jumps)

        # 返回跳跃后的新的 bit_generator 实例
        return bit_generator
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
        self : Philox
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
        # 将 delta 包装为 256 范围内的整数
        delta = wrap_int(delta, 256)

        # 创建一个 numpy 数组 delta_a，将 delta 转换为 'step' 形式的 uint64 数组
        cdef np.ndarray delta_a
        delta_a = int_to_array(delta, 'step', 256, 64)

        # 调用底层函数 philox_advance，更新 RNG 状态
        philox_advance(<uint64_t *> delta_a.data, &self.rng_state)

        # 重置 Philox 对象的状态变量
        self._reset_state_variables()

        # 返回更新后的 Philox 对象自身
        return self
```
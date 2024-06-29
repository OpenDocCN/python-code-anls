# `D:\src\scipysrc\numpy\numpy\random\bit_generator.pyx`

```
#cython: binding=True
"""
BitGenerator base class and SeedSequence used to seed the BitGenerators.

SeedSequence is derived from Melissa E. O'Neill's C++11 `std::seed_seq`
implementation, as it has a lot of nice properties that we want.

https://gist.github.com/imneme/540829265469e673d045
https://www.pcg-random.org/posts/developing-a-seed_seq-alternative.html

The MIT License (MIT)

Copyright (c) 2015 Melissa E. O'Neill
Copyright (c) 2019 NumPy Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import abc
import sys
from itertools import cycle
import re
from secrets import randbits

from threading import Lock

from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from ._common cimport (random_raw, benchmark, prepare_ctypes, prepare_cffi)

__all__ = ['SeedSequence', 'BitGenerator']

# Initialize NumPy C API
np.import_array()

# Regular expression to match decimal numbers
DECIMAL_RE = re.compile(r'[0-9]+')

# Constants used for the SeedSequence class
cdef uint32_t DEFAULT_POOL_SIZE = 4  # Appears also in docstring for pool_size
cdef uint32_t INIT_A = 0x43b0d7e5
cdef uint32_t MULT_A = 0x931e8875
cdef uint32_t INIT_B = 0x8b51f9dd
cdef uint32_t MULT_B = 0x58f38ded
cdef uint32_t MIX_MULT_L = 0xca01f9dd
cdef uint32_t MIX_MULT_R = 0x4973f715
cdef uint32_t XSHIFT = np.dtype(np.uint32).itemsize * 8 // 2
cdef uint32_t MASK32 = 0xFFFFFFFF

# Function to convert a Python integer to a uint32 NumPy array
def _int_to_uint32_array(n):
    arr = []
    if n < 0:
        raise ValueError("expected non-negative integer")
    if n == 0:
        arr.append(np.uint32(n))

    # Convert the integer `n` into an array of uint32 values
    n = int(n)
    while n > 0:
        arr.append(np.uint32(n & MASK32))
        n //= 2**32
    return np.array(arr, dtype=np.uint32)

# Function to coerce an input into a uint32 NumPy array
def _coerce_to_uint32_array(x):
    """ Coerce an input to a uint32 array.

    If `x` is already a uint32 array, return it directly.
    If `x` is a non-negative integer, break it into uint32 words, starting
    from the lowest bits.
    If `x` is a string starting with "0x", interpret it as a hexadecimal integer.

    """
    如果输入是一个 numpy 数组且其类型是 uint32，则直接返回其拷贝。
    如果输入是一个字符串：
        - 如果以 '0x' 开头，则将其视为十六进制数并转换为整数。
        - 否则，如果匹配十进制数字的正则表达式，则将其转换为整数。
        - 否则，引发数值错误异常，提示字符串无法识别。
    如果输入是一个整数或者 numpy 的整数类型：
        - 调用 _int_to_uint32_array 函数将其转换为 uint32 类型的数组并返回。
    如果输入是浮点数或者 numpy 的非精确类型：
        - 抛出类型错误异常，提示种子必须是整数。
    否则：
        - 如果输入为空序列，则返回一个空的 uint32 类型数组。
        - 否则，对每个元素递归调用 _coerce_to_uint32_array 函数，并将结果拼接成一个大数组返回。
cdef uint32_t hashmix(uint32_t value, uint32_t * hash_const):
    # 对 value 进行哈希混合运算，使用 hash_const 数组的第一个元素进行异或操作
    value ^= hash_const[0]
    # 更新 hash_const 的第一个元素为当前值乘以常数 MULT_A
    hash_const[0] *= MULT_A
    # 将 value 乘以更新后的 hash_const 的第一个元素
    value *=  hash_const[0]
    # 对 value 进行位移异或操作
    value ^= value >> XSHIFT
    return value

cdef uint32_t mix(uint32_t x, uint32_t y):
    # 计算混合结果，使用给定的常数 MIX_MULT_L 和 MIX_MULT_R
    cdef uint32_t result = (MIX_MULT_L * x - MIX_MULT_R * y)
    # 对混合结果进行位移异或操作
    result ^= result >> XSHIFT
    return result


class ISeedSequence(abc.ABC):
    """
    Abstract base class for seed sequences.

    ``BitGenerator`` implementations should treat any object that adheres to
    this interface as a seed sequence.

    See Also
    --------
    SeedSequence, SeedlessSeedSequence
    """

    @abc.abstractmethod
    def generate_state(self, n_words, dtype=np.uint32):
        """
        generate_state(n_words, dtype=np.uint32)

        Return the requested number of words for PRNG seeding.

        A BitGenerator should call this method in its constructor with
        an appropriate `n_words` parameter to properly seed itself.

        Parameters
        ----------
        n_words : int
        dtype : np.uint32 or np.uint64, optional
            The size of each word. This should only be either `uint32` or
            `uint64`. Strings (`'uint32'`, `'uint64'`) are fine. Note that
            requesting `uint64` will draw twice as many bits as `uint32` for
            the same `n_words`. This is a convenience for `BitGenerator`\ s that
            express their states as `uint64` arrays.

        Returns
        -------
        state : uint32 or uint64 array, shape=(n_words,)
        """
        pass


class ISpawnableSeedSequence(ISeedSequence):
    """
    Abstract base class for seed sequences that can spawn.
    """

    @abc.abstractmethod
    def spawn(self, n_children):
        """
        spawn(n_children)

        Spawn a number of child `SeedSequence` s by extending the
        `spawn_key`.

        See :ref:`seedsequence-spawn` for additional notes on spawning
        children.

        Parameters
        ----------
        n_children : int

        Returns
        -------
        seqs : list of `SeedSequence` s
        """
        pass


cdef class SeedlessSeedSequence():
    """
    A seed sequence for BitGenerators with no need for seed state.

    See Also
    --------
    SeedSequence, ISeedSequence
    """

    def generate_state(self, n_words, dtype=np.uint32):
        raise NotImplementedError('seedless SeedSequences cannot generate state')

    def spawn(self, n_children):
        return [self] * n_children


# We cannot directly subclass a `cdef class` type from an `ABC` in Cython, so
# we must register it after the fact.
ISpawnableSeedSequence.register(SeedlessSeedSequence)


cdef class SeedSequence():
    """
    SeedSequence(entropy=None, *, spawn_key=(), pool_size=4)

    SeedSequence mixes sources of entropy in a reproducible way to set the
    initial state for independent and very probably non-overlapping
    BitGenerators.
    """
    # `SeedSequence`类的文档字符串，说明了如何使用和实例化该类的信息
    Once the SeedSequence is instantiated, you can call the `generate_state`
    method to get an appropriately sized seed. Calling `spawn(n) <spawn>` will
    create ``n`` SeedSequences that can be used to seed independent
    BitGenerators, i.e. for different threads.

    # 参数部分的说明文档
    Parameters
    ----------
    entropy : {None, int, sequence[int]}, optional
        The entropy for creating a `SeedSequence`.
        All integer values must be non-negative.
    spawn_key : {(), sequence[int]}, optional
        An additional source of entropy based on the position of this
        `SeedSequence` in the tree of such objects created with the
        `SeedSequence.spawn` method. Typically, only `SeedSequence.spawn` will
        set this, and users will not.
    pool_size : {int}, optional
        Size of the pooled entropy to store. Default is 4 to give a 128-bit
        entropy pool. 8 (for 256 bits) is another reasonable choice if working
        with larger PRNGs, but there is very little to be gained by selecting
        another value.
    n_children_spawned : {int}, optional
        The number of children already spawned. Only pass this if
        reconstructing a `SeedSequence` from a serialized form.

    # 注意事项部分，提供了最佳实践和使用建议
    Notes
    -----
    Best practice for achieving reproducible bit streams is to use
    the default ``None`` for the initial entropy, and then use
    `SeedSequence.entropy` to log/pickle the `entropy` for reproducibility:

    # 示例说明如何使用`SeedSequence`类来生成可重现的随机数种子
    >>> sq1 = np.random.SeedSequence()
    >>> sq1.entropy
    243799254704924441050048792905230269161  # random
    >>> sq2 = np.random.SeedSequence(sq1.entropy)
    >>> np.all(sq1.generate_state(10) == sq2.generate_state(10))
    True
    """

    # `SeedSequence`类的初始化方法，接受多个可选参数，并进行相应的初始化操作
    def __init__(self, entropy=None, *, spawn_key=(),
                 pool_size=DEFAULT_POOL_SIZE, n_children_spawned=0):
        # 检查并设置熵池的大小，若小于默认值则引发错误
        if pool_size < DEFAULT_POOL_SIZE:
            raise ValueError("The size of the entropy pool should be at least "
                             f"{DEFAULT_POOL_SIZE}")
        # 如果未提供熵的值，则生成一个默认大小的随机比特串作为熵的值
        if entropy is None:
            entropy = randbits(pool_size * 32)
        # 如果提供的熵的类型不是整数或整数序列，则引发类型错误
        elif not isinstance(entropy, (int, np.integer, list, tuple, range,
                                      np.ndarray)):
            raise TypeError('SeedSequence expects int or sequence of ints for '
                            'entropy not {}'.format(entropy))
        
        # 设置对象的属性，包括熵值、生成键、熵池大小和已生成子代的数量
        self.entropy = entropy
        self.spawn_key = tuple(spawn_key)
        self.pool_size = pool_size
        self.n_children_spawned = n_children_spawned

        # 初始化一个全为零的熵池，数据类型为32位无符号整数
        self.pool = np.zeros(pool_size, dtype=np.uint32)
        # 调用方法，将初始化时设置的熵值混入到熵池中
        self.mix_entropy(self.pool, self.get_assembled_entropy())
    def __repr__(self):
        # 定义一个字符串列表，用于存放对象表示的各行内容
        lines = [
            f'{type(self).__name__}(',  # 打印对象类型名称
            f'    entropy={self.entropy!r},',  # 打印entropy属性值
        ]
        # 如果spawn_key属性非空，则添加其字符串表示到lines列表中
        if self.spawn_key:
            lines.append(f'    spawn_key={self.spawn_key!r},')
        # 如果pool_size属性不等于默认值DEFAULT_POOL_SIZE，则添加其字符串表示到lines列表中
        if self.pool_size != DEFAULT_POOL_SIZE:
            lines.append(f'    pool_size={self.pool_size!r},')
        # 如果n_children_spawned属性不等于0，则添加其字符串表示到lines列表中
        if self.n_children_spawned != 0:
            lines.append(f'    n_children_spawned={self.n_children_spawned!r},')
        # 添加表示对象构造函数结尾的字符串到lines列表中
        lines.append(')')
        # 将lines列表中的所有字符串拼接成一个完整的表示对象的文本
        text = '\n'.join(lines)
        # 返回构造好的对象表示文本
        return text

    @property
    def state(self):
        # 返回一个字典，包含对象的指定属性及其对应的值
        return {k:getattr(self, k) for k in
                ['entropy', 'spawn_key', 'pool_size',
                 'n_children_spawned']
                if getattr(self, k) is not None}

    cdef mix_entropy(self, np.ndarray[np.npy_uint32, ndim=1] mixer,
                     np.ndarray[np.npy_uint32, ndim=1] entropy_array):
        """ Mix in the given entropy to mixer.

        Parameters
        ----------
        mixer : 1D uint32 array, modified in-place
            要混合的uint32类型的一维数组，将会就地修改
        entropy_array : 1D uint32 array
            uint32类型的一维数组，包含要混合的熵数据
        """
        cdef uint32_t hash_const[1]
        hash_const[0] = INIT_A

        # 将熵数据混合到mixer数组中，直到达到池大小
        for i in range(len(mixer)):
            if i < len(entropy_array):
                mixer[i] = hashmix(entropy_array[i], hash_const)
            else:
                # 如果我们的池大小大于熵数据的长度，继续运行哈希函数
                mixer[i] = hashmix(0, hash_const)

        # 将所有位混合在一起，以便后面的位可以影响前面的位
        for i_src in range(len(mixer)):
            for i_dst in range(len(mixer)):
                if i_src != i_dst:
                    mixer[i_dst] = mix(mixer[i_dst],
                                       hashmix(mixer[i_src], hash_const))

        # 添加剩余的熵数据，将每个新的熵词与每个池词混合
        for i_src in range(len(mixer), len(entropy_array)):
            for i_dst in range(len(mixer)):
                mixer[i_dst] = mix(mixer[i_dst],
                                   hashmix(entropy_array[i_src], hash_const))
    # 定义一个方法，用于获取汇总的熵源并将它们转换成统一的 uint32 数组

    def get_assembled_entropy(self):
        """ Convert and assemble all entropy sources into a uniform uint32
        array.

        Returns
        -------
        entropy_array : 1D uint32 array
        """
        
        # 确保至少有一些运行熵（run-entropy），其他是可选的
        assert self.entropy is not None
        
        # 将运行熵（run-entropy）转换为 uint32 数组
        run_entropy = _coerce_to_uint32_array(self.entropy)
        
        # 将生成密钥（spawn key）转换为 uint32 数组
        spawn_entropy = _coerce_to_uint32_array(self.spawn_key)
        
        # 如果生成密钥（spawn key）的长度大于0，并且运行熵（run-entropy）长度小于池大小（pool_size）
        if len(spawn_entropy) > 0 and len(run_entropy) < self.pool_size:
            # 显式使用0填充熵直到达到池大小，以避免与生成密钥（spawn key）冲突
            # 我们在1.19.0中更改了这一点以修复 gh-16539。为了保持与未生成的 SeedSequence
            # 具有小熵输入的流兼容性，仅在指定 spawn_key 时才执行此操作。
            diff = self.pool_size - len(run_entropy)
            run_entropy = np.concatenate(
                [run_entropy, np.zeros(diff, dtype=np.uint32)])
        
        # 将运行熵（run-entropy）和生成密钥（spawn key）数组拼接起来
        entropy_array = np.concatenate([run_entropy, spawn_entropy])
        
        # 返回汇总后的熵数组
        return entropy_array

    @np.errstate(over='ignore')
    def generate_state(self, n_words, dtype=np.uint32):
        """
        generate_state(n_words, dtype=np.uint32)

        Return the requested number of words for PRNG seeding.

        A BitGenerator should call this method in its constructor with
        an appropriate `n_words` parameter to properly seed itself.

        Parameters
        ----------
        n_words : int
            Number of words (32-bit or 64-bit) to generate for seeding.
        dtype : np.uint32 or np.uint64, optional
            The size of each word. This should only be either `uint32` or
            `uint64`. Strings (`'uint32'`, `'uint64'`) are acceptable.

        Returns
        -------
        state : uint32 or uint64 array, shape=(n_words,)
            Array of generated seed values in uint32 or uint64 format.
        """
        # Initialize the constant hash value for seeding
        cdef uint32_t hash_const = INIT_B
        # Variable to hold each data value during processing
        cdef uint32_t data_val

        # Determine the output data type
        out_dtype = np.dtype(dtype)
        if out_dtype == np.dtype(np.uint32):
            # No action needed for uint32
            pass
        elif out_dtype == np.dtype(np.uint64):
            # Double the number of words for uint64
            n_words *= 2
        else:
            # Raise an error for unsupported types
            raise ValueError("only support uint32 or uint64")
        
        # Initialize an array to store the generated state
        state = np.zeros(n_words, dtype=np.uint32)
        
        # Create a cycling iterator over the entropy pool
        src_cycle = cycle(self.pool)
        
        # Generate the seed values
        for i_dst in range(n_words):
            data_val = next(src_cycle)
            data_val ^= hash_const
            hash_const *= MULT_B
            data_val *= hash_const
            data_val ^= data_val >> XSHIFT
            state[i_dst] = data_val
        
        # If output type is uint64, ensure endianness consistency
        if out_dtype == np.dtype(np.uint64):
            # View as little-endian and then convert to native endianness
            state = state.astype('<u4').view('<u8').astype(np.uint64)
        
        # Return the generated state array
        return state

    def spawn(self, n_children):
        """
        spawn(n_children)

        Spawn a number of child `SeedSequence`s by extending the
        `spawn_key`.

        Parameters
        ----------
        n_children : int
            Number of child SeedSequences to spawn.

        Returns
        -------
        seqs : list of `SeedSequence`s
            List containing the spawned SeedSequence objects.

        See Also
        --------
        random.Generator.spawn, random.BitGenerator.spawn :
            Equivalent method on the generator and bit generator.
        """
        # Variable to hold the index during iteration
        cdef uint32_t i

        # List to store spawned SeedSequence objects
        seqs = []
        
        # Iterate over the range of children to spawn
        for i in range(self.n_children_spawned,
                       self.n_children_spawned + n_children):
            # Append a new SeedSequence object to the list
            seqs.append(type(self)(
                self.entropy,
                spawn_key=self.spawn_key + (i,),
                pool_size=self.pool_size,
            ))
        
        # Update the count of spawned children
        self.n_children_spawned += n_children
        
        # Return the list of spawned SeedSequence objects
        return seqs
# 注册 SeedSequence 到 ISpawnableSeedSequence 接口
ISpawnableSeedSequence.register(SeedSequence)


# 定义 BitGenerator 类
cdef class BitGenerator():
    """
    BitGenerator(seed=None)

    Base Class for generic BitGenerators, which provide a stream
    of random bits based on different algorithms. Must be overridden.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `~numpy.random.SeedSequence` to derive the initial `BitGenerator` state.
        One may also pass in a `SeedSequence` instance.
        All integer values must be non-negative.

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same BitGenerator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.

    See Also
    --------
    SeedSequence
    """

    def __init__(self, seed=None):
        # 初始化一个线程锁
        self.lock = Lock()
        # 初始化 _bitgen 状态为 void 指针的 0 值
        self._bitgen.state = <void *>0
        # 如果实例化的类是 BitGenerator 自身，则抛出未实现错误
        if type(self) is BitGenerator:
            raise NotImplementedError('BitGenerator is a base class and cannot be instantized')

        # 初始化 _ctypes 和 _cffi 为 None
        self._ctypes = None
        self._cffi = None

        # 定义常量字符串 name 为 "BitGenerator"
        cdef const char *name = "BitGenerator"
        # 创建一个 Python Capsule 对象，封装 self._bitgen 的 void 指针
        self.capsule = PyCapsule_New(<void *>&self._bitgen, name, NULL)
        # 如果 seed 不是 ISeedSequence 的实例，则将其转换为 SeedSequence 实例
        if not isinstance(seed, ISeedSequence):
            seed = SeedSequence(seed)
        # 设置 _seed_seq 为传入的 seed
        self._seed_seq = seed

    # Pickling support:
    def __getstate__(self):
        # 返回对象的状态和 _seed_seq
        return self.state, self._seed_seq

    def __setstate__(self, state_seed_seq):
        # 如果状态是字典，则为兼容性路径，之前的版本只保存了底层位生成器的状态
        if isinstance(state_seed_seq, dict):
            self.state = state_seed_seq
        else:
            # 否则设置 _seed_seq 和 state
            self._seed_seq = state_seed_seq[1]
            self.state = state_seed_seq[0]

    def __reduce__(self):
        from ._pickle import __bit_generator_ctor

        # 返回对象的 pickle 反序列化信息
        return (
            __bit_generator_ctor,
            (type(self), ),
            (self.state, self._seed_seq)
        )

    @property
    def state(self):
        """
        Get or set the PRNG state

        The base BitGenerator.state must be overridden by a subclass

        Returns
        -------
        state : dict
            Dictionary containing the information required to describe the
            state of the PRNG
        """
        # 抛出未实现错误，子类必须重写此方法
        raise NotImplementedError('Not implemented in base BitGenerator')

    @state.setter
    def state(self, value):
        # 抛出未实现错误，子类必须重写此方法
        raise NotImplementedError('Not implemented in base BitGenerator')

    @property
    def seed_seq(self):
        """
        Get the seed sequence used to initialize the bit generator.

        .. versionadded:: 1.25.0

        Returns
        -------
        seed_seq : ISeedSequence
            The SeedSequence object used to initialize the BitGenerator.
            This is normally a `np.random.SeedSequence` instance.

        """
        # 返回当前对象的 _seed_seq 属性，用于初始化位生成器的种子序列
        return self._seed_seq

    def spawn(self, int n_children):
        """
        spawn(n_children)

        Create new independent child bit generators.

        See :ref:`seedsequence-spawn` for additional notes on spawning
        children.  Some bit generators also implement ``jumped``
        as a different approach for creating independent streams.

        .. versionadded:: 1.25.0

        Parameters
        ----------
        n_children : int
            Number of child bit generators to create.

        Returns
        -------
        child_bit_generators : list of BitGenerators
            List containing newly created child BitGenerator instances.

        Raises
        ------
        TypeError
            When the underlying SeedSequence does not implement spawning.

        See Also
        --------
        random.Generator.spawn, random.SeedSequence.spawn :
            Equivalent method on the generator and seed sequence.

        """
        # 检查 _seed_seq 属性是否实现了 ISpawnableSeedSequence 接口，如果没有则抛出 TypeError 异常
        if not isinstance(self._seed_seq, ISpawnableSeedSequence):
            raise TypeError(
                "The underlying SeedSequence does not implement spawning.")

        # 使用 _seed_seq 的 spawn 方法创建指定数量的子位生成器，并返回这些生成器组成的列表
        return [type(self)(seed=s) for s in self._seed_seq.spawn(n_children)]

    def random_raw(self, size=None, output=True):
        """
        random_raw(self, size=None)

        Return randoms as generated by the underlying BitGenerator

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        output : bool, optional
            Output values.  Used for performance testing since the generated
            values are not returned.

        Returns
        -------
        out : uint or ndarray
            Drawn samples.

        Notes
        -----
        This method directly exposes the raw underlying pseudo-random
        number generator. All values are returned as unsigned 64-bit
        values irrespective of the number of bits produced by the PRNG.

        See the class docstring for the number of bits returned.
        """
        # 调用 random_raw 函数，使用 _bitgen 和 self.lock 作为参数，生成指定大小的随机数或数组，并返回结果
        return random_raw(&self._bitgen, self.lock, size, output)

    def _benchmark(self, Py_ssize_t cnt, method='uint64'):
        """Used in tests"""
        # 调用 _benchmark 函数，使用 _bitgen、self.lock 和 cnt 作为参数，并返回基准测试结果
        return benchmark(&self._bitgen, self.lock, cnt, method)

    @property
    # 定义一个方法 ctypes，用于获取 ctypes 接口
    def ctypes(self):
        """
        ctypes interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing ctypes wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * bitgen - pointer to the bit generator struct
        """
        # 如果 self._ctypes 属性为 None，则调用 prepare_ctypes 函数初始化它
        if self._ctypes is None:
            self._ctypes = prepare_ctypes(&self._bitgen)

        # 返回 self._ctypes 属性，即包含 ctypes 接口的命名元组
        return self._ctypes

    @property
    # 定义一个属性 cffi，用于获取 CFFI 接口
    def cffi(self):
        """
        CFFI interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing CFFI wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * bitgen - pointer to the bit generator struct
        """
        # 如果 self._cffi 属性为 None，则调用 prepare_cffi 函数初始化它
        if self._cffi is None:
            self._cffi = prepare_cffi(&self._bitgen)
        
        # 返回 self._cffi 属性，即包含 CFFI 接口的命名元组
        return self._cffi
```
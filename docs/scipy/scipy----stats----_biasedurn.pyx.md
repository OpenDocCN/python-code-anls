# `D:\src\scipysrc\scipy\scipy\stats\_biasedurn.pyx`

```
# 导入自定义扩展模块中的 CFishersNCHypergeometric 和 StochasticLib3 类
from ._biasedurn cimport CFishersNCHypergeometric, StochasticLib3
# 导入 NumPy 库并声明其命名空间为 np
cimport numpy as np
import numpy as np
# 导入 C 语言标准库中的 uint32_t 和 uint64_t 类型定义
from libc.stdint cimport uint32_t, uint64_t
# 导入 C++ 语言标准库中的 unique_ptr 类
from libcpp.memory cimport unique_ptr

# 调用 NumPy 库的 import_array 函数，初始化 NumPy 数组接口
np.import_array()

# 导入 Python Capsule API 中的 PyCapsule_GetPointer 和 PyCapsule_IsValid 函数
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

# 声明 C 函数 bitgen_t 和 random_normal 的外部接口
cdef extern from "distributions.h":
    ctypedef struct bitgen_t:
        void *state
        uint64_t (*next_uint64)(void *st) nogil
        uint32_t (*next_uint32)(void *st) nogil
        double (*next_double)(void *st) nogil
        uint64_t (*next_raw)(void *st) nogil

    double random_normal(bitgen_t*, double, double) nogil

# 定义 Python 包装类 _PyFishersNCHypergeometric，封装 CFishersNCHypergeometric 对象
cdef class _PyFishersNCHypergeometric:
    cdef unique_ptr[CFishersNCHypergeometric] c_fnch

    def __cinit__(self, int n, int m, int N, double odds, double accuracy):
        # 初始化 c_fnch 成员变量，使用 CFishersNCHypergeometric 类创建对象
        self.c_fnch = unique_ptr[CFishersNCHypergeometric](new CFishersNCHypergeometric(n, m, N, odds, accuracy))

    def mode(self):
        # 调用 CFishersNCHypergeometric 对象的 mode 方法，返回众数
        return self.c_fnch.get().mode()

    def mean(self):
        # 调用 CFishersNCHypergeometric 对象的 mean 方法，返回均值
        return self.c_fnch.get().mean()

    def variance(self):
        # 调用 CFishersNCHypergeometric 对象的 variance 方法，返回方差
        return self.c_fnch.get().variance()

    def probability(self, int x):
        # 调用 CFishersNCHypergeometric 对象的 probability 方法，返回概率
        return self.c_fnch.get().probability(x)

    def moments(self):
        cdef double mean, var
        # 调用 CFishersNCHypergeometric 对象的 moments 方法，获取均值和方差
        self.c_fnch.get().moments(&mean, &var)
        return mean, var


# 定义 Python 包装类 _PyWalleniusNCHypergeometric，封装 CWalleniusNCHypergeometric 对象
cdef class _PyWalleniusNCHypergeometric:
    cdef unique_ptr[CWalleniusNCHypergeometric] c_wnch

    def __cinit__(self, int n, int m, int N, double odds, double accuracy):
        # 初始化 c_wnch 成员变量，使用 CWalleniusNCHypergeometric 类创建对象
        self.c_wnch = unique_ptr[CWalleniusNCHypergeometric](new CWalleniusNCHypergeometric(n, m, N, odds, accuracy))

    def mode(self):
        # 调用 CWalleniusNCHypergeometric 对象的 mode 方法，返回众数
        return self.c_wnch.get().mode()

    def mean(self):
        # 调用 CWalleniusNCHypergeometric 对象的 mean 方法，返回均值
        return self.c_wnch.get().mean()

    def variance(self):
        # 调用 CWalleniusNCHypergeometric 对象的 variance 方法，返回方差
        return self.c_wnch.get().variance()

    def probability(self, int x):
        # 调用 CWalleniusNCHypergeometric 对象的 probability 方法，返回概率
        return self.c_wnch.get().probability(x)

    def moments(self):
        cdef double mean, var
        # 调用 CWalleniusNCHypergeometric 对象的 moments 方法，获取均值和方差
        self.c_wnch.get().moments(&mean, &var)
        return mean, var


# 声明全局变量 _glob_rng，类型为 bitgen_t 指针
cdef bitgen_t* _glob_rng

# 定义 next_double 函数，返回全局 RNG 的下一个双精度随机数
cdef double next_double() noexcept nogil:
    global _glob_rng
    return _glob_rng.next_double(_glob_rng.state)

# 定义 next_normal 函数，返回全局 RNG 的下一个正态分布随机数
cdef double next_normal(const double m, const double s) noexcept nogil:
    global _glob_rng
    return random_normal(_glob_rng, m, s)

# 定义 make_rng 函数，根据 random_state 返回相应的随机数生成器 Capsule 对象
cdef object make_rng(random_state=None):
    # 根据 random_state 类型不同，获取相应的随机数生成器对象
    if random_state is None or isinstance(random_state, int):
        bg = np.random.RandomState(random_state)._bit_generator
    elif isinstance(random_state, np.random.RandomState):
        bg = random_state._bit_generator
    elif isinstance(random_state, np.random.Generator):
        bg = random_state.bit_generator
    else:
        raise ValueError('random_state is not one of None, int, RandomState, Generator')
    # 获取随机数生成器对象的 Capsule
    capsule = bg.capsule
    return capsule


# 定义 Python 包装类 _PyStochasticLib3，封装 StochasticLib3 对象
cdef class _PyStochasticLib3:
    cdef unique_ptr[StochasticLib3] c_sl3
    cdef object capsule
    cdef bitgen_t* bit_generator
    # 初始化函数，用于设置随机数生成器的初始状态
    def __cinit__(self):
        # 使用StochasticLib3类创建一个unique_ptr对象，并初始化其值为StochasticLib3(0)
        self.c_sl3 = unique_ptr[StochasticLib3](new StochasticLib3(0))
        # 将next_double函数指针设置为StochasticLib3对象的成员函数next_double
        self.c_sl3.get().next_double = &next_double
        # 将next_normal函数指针设置为StochasticLib3对象的成员函数next_normal

    # 获取随机数
    def Random(self):
        return self.c_sl3.get().Random()

    # 设置精度
    def SetAccuracy(self, double accur):
        return self.c_sl3.get().SetAccuracy(accur)

    # 处理随机数生成器的状态
    cdef void HandleRng(self, random_state=None) noexcept:
        # 使用make_rng函数创建一个随机数生成器的capsule对象，并存储在self.capsule中
        self.capsule = make_rng(random_state)

        # 获取bitgen_t指针
        cdef const char *capsule_name = "BitGenerator"
        # 检查self.capsule是否是有效的capsule对象
        if not PyCapsule_IsValid(self.capsule, capsule_name):
            # 如果不是有效的capsule对象，则抛出ValueError异常
            raise ValueError("Invalid pointer to anon_func_state")
        # 从self.capsule中获取bitgen_t指针，并存储在self.bit_generator中
        self.bit_generator = <bitgen_t *> PyCapsule_GetPointer(self.capsule, capsule_name)
        # 将self.bit_generator存储在全局变量_glob_rng中
        global _glob_rng
        _glob_rng = self.bit_generator

    # 生成Fisher's noncentral 超几何分布的随机变量
    def rvs_fisher(self, int n, int m, int N, double odds, int size, random_state=None):
        # 处理随机数生成器的状态
        self.HandleRng(random_state)

        # 为rvs分配内存空间，存储Fisher's noncentral 超几何分布的随机变量
        rvs = np.empty(size, dtype=np.float64)
        # 循环调用StochasticLib3对象的FishersNCHyp方法生成随机变量，并存储在rvs中
        for ii in range(size):
            rvs[ii] = self.c_sl3.get().FishersNCHyp(n, m, N, odds)
        return rvs

    # 生成Wallenius noncentral 超几何分布的随机变量
    def rvs_wallenius(self, int n, int m, int N, double odds, int size, random_state=None):
        # 处理随机数生成器的状态
        self.HandleRng(random_state)

        # 为rvs分配内存空间，存储Wallenius noncentral 超几何分布的随机变量
        rvs = np.empty(size, dtype=np.float64)
        # 循环调用StochasticLib3对象的WalleniusNCHyp方法生成随机变量，并存储在rvs中
        for ii in range(size):
            rvs[ii] = self.c_sl3.get().WalleniusNCHyp(n, m, N, odds)
        return rvs

    # 调用Fisher's noncentral 超几何分布的随机变量生成方法
    def FishersNCHyp(self, int n, int m, int N, double odds):
        # 获取默认的随机数生成器状态
        self.HandleRng(None)
        # 调用StochasticLib3对象的FishersNCHyp方法生成随机变量
        return self.c_sl3.get().FishersNCHyp(n, m, N, odds)

    # 调用Wallenius noncentral 超几何分布的随机变量生成方法
    def WalleniusNCHyp(self, int n, int m, int N, double odds):
        # 获取默认的随机数生成器状态
        self.HandleRng(None)
        # 调用StochasticLib3对象的WalleniusNCHyp方法生成随机变量
        return self.c_sl3.get().WalleniusNCHyp(n, m, N, odds)
```
# `.\numpy\numpy\typing\tests\data\pass\random.py`

```
from __future__ import annotations  # 导入未来的类型注释功能

from typing import Any  # 导入 Any 类型，表示可以是任意类型
import numpy as np  # 导入 NumPy 库，用于科学计算

SEED_NONE = None  # 定义空种子
SEED_INT = 4579435749574957634658964293569  # 定义整数种子
SEED_ARR: np.ndarray[Any, np.dtype[np.int64]] = np.array([1, 2, 3, 4], dtype=np.int64)  # 定义 NumPy 整数数组种子
SEED_ARRLIKE: list[int] = [1, 2, 3, 4]  # 定义整数列表种子
SEED_SEED_SEQ: np.random.SeedSequence = np.random.SeedSequence(0)  # 创建 SeedSequence 类型的种子序列对象
SEED_MT19937: np.random.MT19937 = np.random.MT19937(0)  # 创建 MT19937 类型的随机数生成器对象
SEED_PCG64: np.random.PCG64 = np.random.PCG64(0)  # 创建 PCG64 类型的随机数生成器对象
SEED_PHILOX: np.random.Philox = np.random.Philox(0)  # 创建 Philox 类型的随机数生成器对象
SEED_SFC64: np.random.SFC64 = np.random.SFC64(0)  # 创建 SFC64 类型的随机数生成器对象

# 创建默认的随机数生成器对象，没有指定种子
np.random.default_rng()
# 使用 SEED_NONE 作为种子创建默认的随机数生成器对象
np.random.default_rng(SEED_NONE)
# 使用 SEED_INT 作为种子创建默认的随机数生成器对象
np.random.default_rng(SEED_INT)
# 使用 SEED_ARR 作为种子创建默认的随机数生成器对象
np.random.default_rng(SEED_ARR)
# 使用 SEED_ARRLIKE 作为种子创建默认的随机数生成器对象
np.random.default_rng(SEED_ARRLIKE)
# 使用 SEED_SEED_SEQ 作为种子创建默认的随机数生成器对象
np.random.default_rng(SEED_SEED_SEQ)
# 使用 SEED_MT19937 作为种子创建默认的随机数生成器对象
np.random.default_rng(SEED_MT19937)
# 使用 SEED_PCG64 作为种子创建默认的随机数生成器对象
np.random.default_rng(SEED_PCG64)
# 使用 SEED_PHILOX 作为种子创建默认的随机数生成器对象
np.random.default_rng(SEED_PHILOX)
# 使用 SEED_SFC64 作为种子创建默认的随机数生成器对象
np.random.default_rng(SEED_SFC64)

# 使用 SEED_NONE 创建 SeedSequence 对象
np.random.SeedSequence(SEED_NONE)
# 使用 SEED_INT 创建 SeedSequence 对象
np.random.SeedSequence(SEED_INT)
# 使用 SEED_ARR 创建 SeedSequence 对象
np.random.SeedSequence(SEED_ARR)
# 使用 SEED_ARRLIKE 创建 SeedSequence 对象
np.random.SeedSequence(SEED_ARRLIKE)

# 使用 SEED_NONE 创建 MT19937 随机数生成器对象
np.random.MT19937(SEED_NONE)
# 使用 SEED_INT 创建 MT19937 随机数生成器对象
np.random.MT19937(SEED_INT)
# 使用 SEED_ARR 创建 MT19937 随机数生成器对象
np.random.MT19937(SEED_ARR)
# 使用 SEED_ARRLIKE 创建 MT19937 随机数生成器对象
np.random.MT19937(SEED_ARRLIKE)
# 使用 SEED_SEED_SEQ 创建 MT19937 随机数生成器对象
np.random.MT19937(SEED_SEED_SEQ)

# 使用 SEED_NONE 创建 PCG64 随机数生成器对象
np.random.PCG64(SEED_NONE)
# 使用 SEED_INT 创建 PCG64 随机数生成器对象
np.random.PCG64(SEED_INT)
# 使用 SEED_ARR 创建 PCG64 随机数生成器对象
np.random.PCG64(SEED_ARR)
# 使用 SEED_ARRLIKE 创建 PCG64 随机数生成器对象
np.random.PCG64(SEED_ARRLIKE)
# 使用 SEED_SEED_SEQ 创建 PCG64 随机数生成器对象
np.random.PCG64(SEED_SEED_SEQ)

# 使用 SEED_NONE 创建 Philox 随机数生成器对象
np.random.Philox(SEED_NONE)
# 使用 SEED_INT 创建 Philox 随机数生成器对象
np.random.Philox(SEED_INT)
# 使用 SEED_ARR 创建 Philox 随机数生成器对象
np.random.Philox(SEED_ARR)
# 使用 SEED_ARRLIKE 创建 Philox 随机数生成器对象
np.random.Philox(SEED_ARRLIKE)
# 使用 SEED_SEED_SEQ 创建 Philox 随机数生成器对象
np.random.Philox(SEED_SEED_SEQ)

# 使用 SEED_NONE 创建 SFC64 随机数生成器对象
np.random.SFC64(SEED_NONE)
# 使用 SEED_INT 创建 SFC64 随机数生成器对象
np.random.SFC64(SEED_INT)
# 使用 SEED_ARR 创建 SFC64 随机数生成器对象
np.random.SFC64(SEED_ARR)
# 使用 SEED_ARRLIKE 创建 SFC64 随机数生成器对象
np.random.SFC64(SEED_ARRLIKE)
# 使用 SEED_SEED_SEQ 创建 SFC64 随机数生成器对象
np.random.SFC64(SEED_SEED_SEQ)

# 创建 SeedSequence 类型的种子序列对象
seed_seq: np.random.bit_generator.SeedSequence = np.random.SeedSequence(SEED_NONE)
# 使用 spawn 方法生成新的种子序列对象
seed_seq.spawn(10)
# 使用 generate_state 方法生成状态数组，指定数据类型为 "u4"
seed_seq.generate_state(3, "u4")
# 使用 generate_state 方法生成状态数组，指定数据类型为 "uint32"
seed_seq.generate_state(3, "uint32")
# 使用 generate_state 方法生成状态数组，指定数据类型为 "u8"
seed_seq.generate_state(3, "u8")
# 使用 generate_state 方法生成状态数组，指定数据类型为 "uint64"
seed_seq.generate_state(3, "uint64")
# 使用 generate_state 方法生成状态数组，指定数据类型为 np.uint32
seed_seq.generate_state(3, np.uint32)
# 使用 generate_state 方法生成状态数组，指定数据类型为 np.uint64
seed_seq.generate_state(3, np.uint64)

# 创建默认的随机数生成器对象
def_gen: np.random.Generator = np.random.default_rng()

# 创建包含单个浮点数 0.1 的 NumPy 浮点数数组
D_arr_0p1: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.1])
# 创建包含单个浮点数 0.5 的 NumPy 浮点数数组
D_arr_0p5: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.5])
# 创建包含单个浮点数 0.9 的 NumPy 浮点数数组
D_arr_0p9: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.9])
# 创建包含单个浮点数 1.5 的 NumPy 浮点数数组
D_arr_1p5: np.ndarray[Any, np.dtype[np.float64]] = np.array([1.5])
# 创建包含单个整数 10 的 NumPy 整数数组
I_arr_10: np.ndarray[Any, np.dtype[np.int_]] = np.array([10], dtype=np.int_)
# 创建包含单个整数 20 的 NumPy 整数数组
I_arr_20: np.ndarray[Any, np.dtype[np.int_]] = np.array([20], dtype=np.int_)
# 创建包含单个浮点数 0.1 的浮点数列表
D_arr_like_0p1: list[float] = [0.1]
# 创建包含单个浮点数 0.5 的浮点数列表
D_arr_like_0p5: list[float] = [0.5]
# 创建包含单个浮点数 0.9 的浮点数列表
D_arr_like_0p9: list[float] = [0.9]
# 创建包含单个浮点数 1.5 的浮点数列表
D_arr_like_1p5: list[float] = [1.5]
# 创建包含单个整数 10 的整数列表
I_arr_like_10: list[int] = [10]
# 创建包含单个整数 20 的整数列表
I_arr
# 生成一个形状为标准正态分布的随机数，数据类型为np.float32
def_gen.standard_normal(dtype=np.float32)

# 生成一个形状为标
# 生成符合标准 t 分布的随机变量
def_gen.standard_t(D_arr_like_0p5)
# 生成指定大小符合标准 t 分布的随机变量
def_gen.standard_t(D_arr_like_0p5, size=1)

# 生成符合泊松分布的随机变量
def_gen.poisson(0.5)
# 生成一个符合泊松分布的随机变量，大小为 None
def_gen.poisson(0.5, size=None)
# 生成指定大小符合泊松分布的随机变量
def_gen.poisson(0.5, size=1)
# 根据数组 D_arr_0p5 生成符合泊松分布的随机变量
def_gen.poisson(D_arr_0p5)
# 根据数组 D_arr_0p5 生成指定大小符合泊松分布的随机变量
def_gen.poisson(D_arr_0p5, size=1)
# 根据类似于 D_arr_0p5 的数组生成符合泊松分布的随机变量
def_gen.poisson(D_arr_like_0p5)
# 根据类似于 D_arr_0p5 的数组生成指定大小符合泊松分布的随机变量
def_gen.poisson(D_arr_like_0p5, size=1)

# 生成符合幂分布的随机变量
def_gen.power(0.5)
# 生成一个符合幂分布的随机变量，大小为 None
def_gen.power(0.5, size=None)
# 生成指定大小符合幂分布的随机变量
def_gen.power(0.5, size=1)
# 根据数组 D_arr_0p5 生成符合幂分布的随机变量
def_gen.power(D_arr_0p5)
# 根据数组 D_arr_0p5 生成指定大小符合幂分布的随机变量
def_gen.power(D_arr_0p5, size=1)
# 根据类似于 D_arr_0p5 的数组生成符合幂分布的随机变量
def_gen.power(D_arr_like_0p5)
# 根据类似于 D_arr_0p5 的数组生成指定大小符合幂分布的随机变量
def_gen.power(D_arr_like_0p5, size=1)

# 生成符合帕累托分布的随机变量
def_gen.pareto(0.5)
# 生成一个符合帕累托分布的随机变量，大小为 None
def_gen.pareto(0.5, size=None)
# 生成指定大小符合帕累托分布的随机变量
def_gen.pareto(0.5, size=1)
# 根据数组 D_arr_0p5 生成符合帕累托分布的随机变量
def_gen.pareto(D_arr_0p5)
# 根据数组 D_arr_0p5 生成指定大小符合帕累托分布的随机变量
def_gen.pareto(D_arr_0p5, size=1)
# 根据类似于 D_arr_0p5 的数组生成符合帕累托分布的随机变量
def_gen.pareto(D_arr_like_0p5)
# 根据类似于 D_arr_0p5 的数组生成指定大小符合帕累托分布的随机变量
def_gen.pareto(D_arr_like_0p5, size=1)

# 生成符合卡方分布的随机变量
def_gen.chisquare(0.5)
# 生成一个符合卡方分布的随机变量，大小为 None
def_gen.chisquare(0.5, size=None)
# 生成指定大小符合卡方分布的随机变量
def_gen.chisquare(0.5, size=1)
# 根据数组 D_arr_0p5 生成符合卡方分布的随机变量
def_gen.chisquare(D_arr_0p5)
# 根据数组 D_arr_0p5 生成指定大小符合卡方分布的随机变量
def_gen.chisquare(D_arr_0p5, size=1)
# 根据类似于 D_arr_0p5 的数组生成符合卡方分布的随机变量
def_gen.chisquare(D_arr_like_0p5)
# 根据类似于 D_arr_0p5 的数组生成指定大小符合卡方分布的随机变量
def_gen.chisquare(D_arr_like_0p5, size=1)

# 生成符合指数分布的随机变量
def_gen.exponential(0.5)
# 生成一个符合指数分布的随机变量，大小为 None
def_gen.exponential(0.5, size=None)
# 生成指定大小符合指数分布的随机变量
def_gen.exponential(0.5, size=1)
# 根据数组 D_arr_0p5 生成符合指数分布的随机变量
def_gen.exponential(D_arr_0p5)
# 根据数组 D_arr_0p5 生成指定大小符合指数分布的随机变量
def_gen.exponential(D_arr_0p5, size=1)
# 根据类似于 D_arr_0p5 的数组生成符合指数分布的随机变量
def_gen.exponential(D_arr_like_0p5)
# 根据类似于 D_arr_0p5 的数组生成指定大小符合指数分布的随机变量
def_gen.exponential(D_arr_like_0p5, size=1)

# 生成符合几何分布的随机变量
def_gen.geometric(0.5)
# 生成一个符合几何分布的随机变量，大小为 None
def_gen.geometric(0.5, size=None)
# 生成指定大小符合几何分布的随机变量
def_gen.geometric(0.5, size=1)
# 根据数组 D_arr_0p5 生成符合几何分布的随机变量
def_gen.geometric(D_arr_0p5)
# 根据数组 D_arr_0p5 生成指定大小符合几何分布的随机变量
def_gen.geometric(D_arr_0p5, size=1)
# 根据类似于 D_arr_0p5 的数组生成符合几何分布的随机变量
def_gen.geometric(D_arr_like_0p5)
# 根据类似于 D_arr_0p5 的数组生成指定大小符合几何分布的随机变量
def_gen.geometric(D_arr_like_0p5, size=1)

# 生成符合对数系列分布的随机变量
def_gen.logseries(0.5)
# 生成一个符合对数系列分布的随机变量，大小为 None
def_gen.logseries(0.5, size=None)
# 生成指定大小符合对数系列分布的随机变量
def_gen.logseries(0.5, size=1)
# 根据数组 D_arr_0p5 生成符合对数系列分布的随机变量
def_gen.logseries(D_arr_0p5)
# 根据数组 D_arr_0p5 生成指定大小符合对数系列分布的随机变量
def_gen.logseries(D_arr_0p5, size=1)
# 根据类似于 D_arr_0p5 的数组生成符合对数系列分布的随机变量
def_gen.logseries(D_arr_like_0p5)
# 根据类似于 D_arr_0p5 的数组生成指定大小符合对数系列分布的随机变量
def_gen.logseries(D_arr_like_0p5, size=1)

# 生成符合瑞利分布的随机变量
def_gen.rayleigh(0.5)
# 生成一个符合瑞利分布的随机变量，大小为 None
def_gen.rayleigh(0.5
# 生成服从 von Mises 分布的随机数
def_gen.vonmises(D_arr_like_0p5, D_arr_like_0p5)

# 生成服从 von Mises 分布的随机数，可以指定生成的随机数个数
def_gen.vonmises(D_arr_0p5, D_arr_0p5, size=1)

# 生成服从 von Mises 分布的随机数，可以指定生成的随机数个数，输入参数为类似数组的结构
def_gen.vonmises(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 生成 Wald 分布的随机数
def_gen.wald(0.5, 0.5)

# 生成 Wald 分布的随机数，可以指定生成的随机数个数
def_gen.wald(0.5, 0.5, size=None)

# 生成 Wald 分布的随机数，生成一个随机数
def_gen.wald(0.5, 0.5, size=1)

# 生成 Wald 分布的随机数，其中一个参数为类似数组的结构
def_gen.wald(D_arr_0p5, 0.5)

# 生成 Wald 分布的随机数，其中一个参数为类似数组的结构
def_gen.wald(0.5, D_arr_0p5)

# 生成 Wald 分布的随机数，可以指定生成的随机数个数，其中一个参数为类似数组的结构
def_gen.wald(D_arr_0p5, 0.5, size=1)

# 生成 Wald 分布的随机数，可以指定生成的随机数个数，其中一个参数为类似数组的结构
def_gen.wald(0.5, D_arr_0p5, size=1)

# 生成 Wald 分布的随机数，其中一个参数为类似数组的结构
def_gen.wald(D_arr_like_0p5, 0.5)

# 生成 Wald 分布的随机数，其中一个参数为类似数组的结构
def_gen.wald(0.5, D_arr_like_0p5)

# 生成 Wald 分布的随机数，两个参数均为类似数组的结构
def_gen.wald(D_arr_0p5, D_arr_0p5)

# 生成 Wald 分布的随机数，两个参数均为类似数组的结构
def_gen.wald(D_arr_like_0p5, D_arr_like_0p5)

# 生成 Wald 分布的随机数，可以指定生成的随机数个数，两个参数均为类似数组的结构
def_gen.wald(D_arr_0p5, D_arr_0p5, size=1)

# 生成 Wald 分布的随机数，可以指定生成的随机数个数，两个参数均为类似数组的结构
def_gen.wald(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 生成服从均匀分布的随机数
def_gen.uniform(0.5, 0.5)

# 生成服从均匀分布的随机数，可以指定生成的随机数个数
def_gen.uniform(0.5, 0.5, size=None)

# 生成服从均匀分布的随机数，生成一个随机数
def_gen.uniform(0.5, 0.5, size=1)

# 生成服从均匀分布的随机数，其中一个参数为类似数组的结构
def_gen.uniform(D_arr_0p5, 0.5)

# 生成服从均匀分布的随机数，其中一个参数为类似数组的结构
def_gen.uniform(0.5, D_arr_0p5)

# 生成服从均匀分布的随机数，可以指定生成的随机数个数，其中一个参数为类似数组的结构
def_gen.uniform(D_arr_0p5, 0.5, size=1)

# 生成服从均匀分布的随机数，可以指定生成的随机数个数，其中一个参数为类似数组的结构
def_gen.uniform(0.5, D_arr_0p5, size=1)

# 生成服从均匀分布的随机数，其中一个参数为类似数组的结构
def_gen.uniform(D_arr_like_0p5, 0.5)

# 生成服从均匀分布的随机数，其中一个参数为类似数组的结构
def_gen.uniform(0.5, D_arr_like_0p5)

# 生成服从均匀分布的随机数，两个参数均为类似数组的结构
def_gen.uniform(D_arr_0p5, D_arr_0p5)

# 生成服从均匀分布的随机数，两个参数均为类似数组的结构
def_gen.uniform(D_arr_like_0p5, D_arr_like_0p5)

# 生成服从均匀分布的随机数，可以指定生成的随机数个数，两个参数均为类似数组的结构
def_gen.uniform(D_arr_0p5, D_arr_0p5, size=1)

# 生成服从均匀分布的随机数，可以指定生成的随机数个数，两个参数均为类似数组的结构
def_gen.uniform(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 生成 Beta 分布的随机数
def_gen.beta(0.5, 0.5)

# 生成 Beta 分布的随机数，可以指定生成的随机数个数
def_gen.beta(0.5, 0.5, size=None)

# 生成 Beta 分布的随机数，生成一个随机数
def_gen.beta(0.5, 0.5, size=1)

# 生成 Beta 分布的随机数，其中一个参数为类似数组的结构
def_gen.beta(D_arr_0p5, 0.5)

# 生成 Beta 分布的随机数，其中一个参数为类似数组的结构
def_gen.beta(0.5, D_arr_0p5)

# 生成 Beta 分布的随机数，可以指定生成的随机数个数，其中一个参数为类似数组的结构
def_gen.beta(D_arr_0p5, 0.5, size=1)

# 生成 Beta 分布的随机数，可以指定生成的随机数个数，其中一个参数为类似数组的结构
def_gen.beta(0.5, D_arr_0p5, size=1)

# 生成 Beta 分布的随机数，其中一个参数为类似数组的结构
def_gen.beta(D_arr_like_0p5, 0.5)

# 生成 Beta 分布的随机数，其中一个参数为类似数组的结构
def_gen.beta(0.5, D_arr_like_0p5)

# 生成 Beta 分布的随机数，两个参数均为类似数组的结构
def_gen.beta(D_arr_0p5, D_arr_0p5)

# 生成 Beta 分布的随机数，两个参数均为类似数组的结构
def_gen.beta(D_arr_like_0p5, D_arr_like_0p5)

# 生成 Beta 分布的随机数，可以指定生成的随机数个数，两个参数均为类似数组的结构
def_gen.beta(D_arr_0p5, D_arr_0p5, size=1)

# 生成 Beta 分布的随机数，可以指定生成的随机数个数，两个参数均为类似数组的结构
def_gen.beta(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 生成 F 分布的随机数
def_gen.f(0.5, 0.5)

# 生成 F 分布的随机数，可以指定生成的随机数个数
def_gen.f(0.5, 0.5, size=None)

# 生成 F 分布的随机数，生成一个随
# 使用 laplace 分布生成一个随机数，参数为 loc=0.5, scale=0.5
def_gen.laplace(0.5, 0.5)

# 使用 laplace 分布生成一个随机数数组，参数为 loc=0.5, scale=0.5，size 可选
def_gen.laplace(0.5, 0.5, size=None)

# 使用 laplace 分布生成一个单一随机数，参数为 loc=0.5, scale=0.5
def_gen.laplace(0.5, 0.5, size=1)

# 使用 laplace 分布生成一个随机数数组，第一个参数是数组 D_arr_0p5，loc=0.5, scale=0.5
def_gen.laplace(D_arr_0p5, 0.5)

# 使用 laplace 分布生成一个随机数数组，loc=0.5, scale=0.5，第一个参数是数组 D_arr_0p5，size=1
def_gen.laplace(0.5, D_arr_0p5, size=1)

# 使用 laplace 分布生成一个单一随机数，第一个参数是数组 D_arr_0p5，loc=0.5, scale=0.5
def_gen.laplace(D_arr_0p5, 0.5, size=1)

# 使用 laplace 分布生成一个随机数数组，第一个参数是数组 D_arr_like_0p5，loc=0.5, scale=0.5
def_gen.laplace(0.5, D_arr_like_0p5)

# 使用 laplace 分布生成一个单一随机数，两个参数都是数组 D_arr_0p5
def_gen.laplace(D_arr_0p5, D_arr_0p5)

# 使用 laplace 分布生成一个随机数数组，两个参数都是数组 D_arr_like_0p5，size=1
def_gen.laplace(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 使用 logistic 分布生成一个随机数，参数为 loc=0.5, scale=0.5
def_gen.logistic(0.5, 0.5)

# 使用 logistic 分布生成一个随机数数组，参数为 loc=0.5, scale=0.5，size 可选
def_gen.logistic(0.5, 0.5, size=None)

# 使用 logistic 分布生成一个单一随机数，参数为 loc=0.5, scale=0.5
def_gen.logistic(0.5, 0.5, size=1)

# 使用 logistic 分布生成一个随机数数组，第一个参数是数组 D_arr_0p5，loc=0.5, scale=0.5
def_gen.logistic(D_arr_0p5, 0.5)

# 使用 logistic 分布生成一个随机数数组，loc=0.5, scale=0.5，第一个参数是数组 D_arr_0p5，size=1
def_gen.logistic(0.5, D_arr_0p5, size=1)

# 使用 logistic 分布生成一个单一随机数，第一个参数是数组 D_arr_0p5，loc=0.5, scale=0.5
def_gen.logistic(D_arr_0p5, 0.5, size=1)

# 使用 logistic 分布生成一个随机数数组，第一个参数是数组 D_arr_like_0p5，loc=0.5, scale=0.5
def_gen.logistic(0.5, D_arr_like_0p5)

# 使用 logistic 分布生成一个单一随机数，两个参数都是数组 D_arr_0p5
def_gen.logistic(D_arr_0p5, D_arr_0p5)

# 使用 logistic 分布生成一个随机数数组，两个参数都是数组 D_arr_like_0p5，size=1
def_gen.logistic(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 使用 lognormal 分布生成一个随机数，参数为 mean=0.5, sigma=0.5
def_gen.lognormal(0.5, 0.5)

# 使用 lognormal 分布生成一个随机数数组，参数为 mean=0.5, sigma=0.5，size 可选
def_gen.lognormal(0.5, 0.5, size=None)

# 使用 lognormal 分布生成一个单一随机数，参数为 mean=0.5, sigma=0.5
def_gen.lognormal(0.5, 0.5, size=1)

# 使用 lognormal 分布生成一个随机数数组，第一个参数是数组 D_arr_0p5，mean=0.5, sigma=0.5
def_gen.lognormal(D_arr_0p5, 0.5)

# 使用 lognormal 分布生成一个随机数数组，mean=0.5, sigma=0.5，第一个参数是数组 D_arr_0p5，size=1
def_gen.lognormal(0.5, D_arr_0p5, size=1)

# 使用 lognormal 分布生成一个单一随机数，第一个参数是数组 D_arr_0p5，mean=0.5, sigma=0.5
def_gen.lognormal(D_arr_0p5, 0.5, size=1)

# 使用 lognormal 分布生成一个随机数数组，第一个参数是数组 D_arr_like_0p5，mean=0.5, sigma=0.5
def_gen.lognormal(0.5, D_arr_like_0p5)

# 使用 lognormal 分布生成一个单一随机数，两个参数都是数组 D_arr_0p5
def_gen.lognormal(D_arr_0p5, D_arr_0p5)

# 使用 lognormal 分布生成一个随机数数组，两个参数都是数组 D_arr_like_0p5，size=1
def_gen.lognormal(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 使用 noncentral_chisquare 分布生成一个随机数，参数为 df=0.5, nonc=0.5
def_gen.noncentral_chisquare(0.5, 0.5)

# 使用 noncentral_chisquare 分布生成一个随机数数组，参数为 df=0.5, nonc=0.5，size 可选
def_gen.noncentral_chisquare(0.5, 0.5, size=None)

# 使用 noncentral_chisquare 分布生成一个单一随机数，参数为 df=0.5, nonc=0.5
def_gen.noncentral_chisquare(0.5, 0.5, size=1)

# 使用 noncentral_chisquare 分布生成一个随机数数组，第一个参数是数组 D_arr_0p5，df=0.5, nonc=0.5
def_gen.noncentral_chisquare(D_arr_0p5, 0.5)

# 使用 noncentral_chisquare 分布生成一个随机数数组，df=0.5, nonc=0.5，第一个参数是数组 D_arr_0p5，size=1
def_gen.noncentral_chisquare(0.5, D_arr_0p5, size=1)

# 使用 noncentral_chisquare 分布生成一个单一随机数，第一个参数是数组 D_arr_0p5，df=0.5, nonc=0.5
def_gen.noncentral_chisquare(D_arr_0p5, 0.5, size=1)

# 使用 noncentral_chisquare 分布生成一个随机数数组，第一个参数是数组 D_arr_like_0p5，df=0.5, nonc=0.5
def_gen.noncentral_chisquare(0.5, D_arr_like_0p5)

# 使用 noncentral_chisquare 分布生成一个单一随机数，两个参数都是数组 D_arr_0p5
def_gen.noncentral_chisquare(D_arr_0p5, D_arr_0p5)

# 使用 noncentral_chisquare 分布生成一个随机数数组，两个参数都是数组 D_arr_like_0p5，size=1
def_gen.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 使用 normal 分布生成一个随机数，参数为 loc=0.5, scale=0.5
def_gen.normal(0.5, 0.5)

# 使用 normal 分布生成一个随机数数组，参数为 loc=0.5, scale=0.5，size 可选
def_gen.normal(0.5, 0.5, size=None)

# 使用 normal 分布生成一个单一随机
# 生成一个从 0.1 到 0.9 之间的三角分布的随机数，返回一个包含一个元素的数组
def_gen.triangular(0.1, D_arr_0p5, 0.9, size=1)

# 生成一个从 D_arr_like_0p1 到 0.5 之间的三角分布的随机数，返回一个包含一个元素的数组
def_gen.triangular(D_arr_like_0p1, 0.5, D_arr_0p9)

# 生成一个从 0.5 到 D_arr_like_0p5 之间的三角分布的随机数，返回一个包含一个元素的数组
def_gen.triangular(0.5, D_arr_like_0p5, 0.9)

# 生成一个从 D_arr_0p1 到 D_arr_0p5 之间的三角分布的随机数，返回一个包含一个元素的数组
def_gen.triangular(D_arr_0p1, D_arr_0p5, 0.9)

# 生成一个从 D_arr_like_0p1 到 D_arr_like_0p5 之间的三角分布的随机数，返回一个包含一个元素的数组
def_gen.triangular(D_arr_like_0p1, D_arr_like_0p5, 0.9)

# 生成一个从 D_arr_0p1 到 D_arr_0p5 之间的三角分布的随机数，返回一个包含一个元素的数组
def_gen.triangular(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1)

# 生成一个从 D_arr_like_0p1 到 D_arr_like_0p5 之间的三角分布的随机数，返回一个包含一个元素的数组
def_gen.triangular(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1)

# 生成一个自由度为 0.5 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(0.1, 0.5, 0.9)

# 生成一个自由度为 0.5 的非中心 F 分布的随机数，返回一个数组，其大小为 None
def_gen.noncentral_f(0.1, 0.5, 0.9, size=None)

# 生成一个自由度为 0.5 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(0.1, 0.5, 0.9, size=1)

# 生成一个自由度为 D_arr_0p1 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(D_arr_0p1, 0.5, 0.9)

# 生成一个自由度为 0.5 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(0.1, D_arr_0p5, 0.9)

# 生成一个自由度为 D_arr_0p1 和 D_arr_like_0p9 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(D_arr_0p1, 0.5, D_arr_like_0p9, size=1)

# 生成一个自由度为 0.5 和 D_arr_0p5 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(0.1, D_arr_0p5, 0.9, size=1)

# 生成一个自由度为 D_arr_like_0p1 和 D_arr_0p9 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(D_arr_like_0p1, 0.5, D_arr_0p9)

# 生成一个自由度为 0.5 和 D_arr_like_0p5 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(0.5, D_arr_like_0p5, 0.9)

# 生成一个自由度为 D_arr_0p1 和 D_arr_0p5 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, 0.9)

# 生成一个自由度为 D_arr_like_0p1 和 D_arr_like_0p5 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, 0.9)

# 生成一个自由度为 D_arr_0p1 和 D_arr_0p5 和 D_arr_0p9 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1)

# 生成一个自由度为 D_arr_like_0p1 和 D_arr_like_0p5 和 D_arr_like_0p9 的非中心 F 分布的随机数，返回一个包含一个元素的数组
def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1)

# 生成一个参数为 10 和 0.5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(10, 0.5)

# 生成一个参数为 10 和 0.5 的二项分布的随机数，返回一个数组，其大小为 None
def_gen.binomial(10, 0.5, size=None)

# 生成一个参数为 10 和 0.5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(10, 0.5, size=1)

# 生成一个参数为 I_arr_10 和 0.5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(I_arr_10, 0.5)

# 生成一个参数为 10 和 D_arr_0p5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(10, D_arr_0p5)

# 生成一个参数为 I_arr_10 和 0.5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(I_arr_10, 0.5, size=1)

# 生成一个参数为 10 和 D_arr_0p5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(10, D_arr_0p5, size=1)

# 生成一个参数为 I_arr_like_10 和 0.5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(I_arr_like_10, 0.5)

# 生成一个参数为 10 和 D_arr_like_0p5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(10, D_arr_like_0p5)

# 生成一个参数为 I_arr_10 和 D_arr_0p5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(I_arr_10, D_arr_0p5)

# 生成一个参数为 I_arr_like_10 和 D_arr_like_0p5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(I_arr_like_10, D_arr_like_0p5)

# 生成一个参数为 I_arr_10 和 D_arr_0p5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(I_arr_10, D_arr_0p5, size=1)

# 生成一个参数为 I_arr_like_10 和 D_arr_like_0p5 的二项分布的随机数，返回一个包含一个元素的数组
def_gen.binomial(I_arr_like_10, D_arr_like_0p5, size
# 定义一个随机数生成器，生成整数范围在 [0, 100) 内的随机数
def_gen.integers(0, [100])

# 定义一个布尔类型的 numpy 数组，包含一个元素 0
I_bool_low: np.ndarray[Any, np.dtype[np.bool]] = np.array([0], dtype=np.bool)
# 定义一个与上述 numpy 数组相似的 Python 列表，包含一个整数 0
I_bool_low_like: list[int] = [0]
# 定义一个布尔类型的 numpy 数组，包含一个元素 1
I_bool_high_open: np.ndarray[Any, np.dtype[np.bool]] = np.array([1], dtype=np.bool)
# 定义一个布尔类型的 numpy 数组，包含一个元素 1
I_bool_high_closed: np.ndarray[Any, np.dtype[np.bool]] = np.array([1], dtype=np.bool)

# 生成一个布尔类型的随机数
def_gen.integers(2, dtype=bool)
# 生成一个布尔类型的随机数，取值范围 [0, 2)
def_gen.integers(0, 2, dtype=bool)
# 生成一个布尔类型的随机数，取值范围 [0, 1]
def_gen.integers(1, dtype=bool, endpoint=True)
# 生成一个布尔类型的随机数，取值范围 [0, 1]
def_gen.integers(0, 1, dtype=bool, endpoint=True)
# 生成一个布尔类型的随机数，取值范围 [0, 1]
def_gen.integers(I_bool_low_like, 1, dtype=bool, endpoint=True)
# 生成一个布尔类型的随机数，取值范围 [0, 1)
def_gen.integers(I_bool_high_open, dtype=bool)
# 生成一个布尔类型的随机数，取值范围 [0, 1)
def_gen.integers(I_bool_low, I_bool_high_open, dtype=bool)
# 生成一个布尔类型的随机数，取值范围 [0, 1)
def_gen.integers(0, I_bool_high_open, dtype=bool)
# 生成一个布尔类型的随机数，取值范围 [0, 2]
def_gen.integers(I_bool_high_closed, dtype=bool, endpoint=True)
# 生成一个布尔类型的随机数，取值范围 [0, 2]
def_gen.integers(I_bool_low, I_bool_high_closed, dtype=bool, endpoint=True)
# 生成一个布尔类型的随机数，取值范围 [0, 2]
def_gen.integers(0, I_bool_high_closed, dtype=bool, endpoint=True)

# 定义一个无符号 8 位整数的 numpy 数组，包含一个元素 0
I_u1_low: np.ndarray[Any, np.dtype[np.uint8]] = np.array([0], dtype=np.uint8)
# 定义一个与上述 numpy 数组相似的 Python 列表，包含一个整数 0
I_u1_low_like: list[int] = [0]
# 定义一个无符号 8 位整数的 numpy 数组，包含一个元素 255
I_u1_high_open: np.ndarray[Any, np.dtype[np.uint8]] = np.array([255], dtype=np.uint8)
# 定义一个无符号 8 位整数的 numpy 数组，包含一个元素 255
I_u1_high_closed: np.ndarray[Any, np.dtype[np.uint8]] = np.array([255], dtype=np.uint8)

# 生成一个无符号 8 位整数的随机数，取值范围 [0, 256)
def_gen.integers(256, dtype="u1")
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 256)
def_gen.integers(0, 256, dtype="u1")
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255]
def_gen.integers(255, dtype="u1", endpoint=True)
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255]
def_gen.integers(0, 255, dtype="u1", endpoint=True)
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255]
def_gen.integers(I_u1_low_like, 255, dtype="u1", endpoint=True)
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255)
def_gen.integers(I_u1_high_open, dtype="u1")
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255)
def_gen.integers(I_u1_low, I_u1_high_open, dtype="u1")
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255)
def_gen.integers(0, I_u1_high_open, dtype="u1")
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 256)
def_gen.integers(I_u1_high_closed, dtype="u1", endpoint=True)
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 256)
def_gen.integers(I_u1_low, I_u1_high_closed, dtype="u1", endpoint=True)
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 256)
def_gen.integers(0, I_u1_high_closed, dtype="u1", endpoint=True)

# 生成一个无符号 8 位整数的随机数，取值范围 [0, 256)
def_gen.integers(256, dtype="uint8")
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 256)
def_gen.integers(0, 256, dtype="uint8")
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255]
def_gen.integers(255, dtype="uint8", endpoint=True)
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255]
def_gen.integers(0, 255, dtype="uint8", endpoint=True)
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255]
def_gen.integers(I_u1_low_like, 255, dtype="uint8", endpoint=True)
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255)
def_gen.integers(I_u1_high_open, dtype="uint8")
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255)
def_gen.integers(I_u1_low, I_u1_high_open, dtype="uint8")
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 255)
def_gen.integers(0, I_u1_high_open, dtype="uint8")
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 256)
def_gen.integers(I_u1_high_closed, dtype="uint8", endpoint=True)
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 256)
def_gen.integers(I_u1_low, I_u1_high_closed, dtype="uint8", endpoint=True)
# 生成一个无符号 8 位整数的随机数，取值范围 [0, 256)
def_gen.integers(0, I_u1_high_closed, dtype="uint8", endpoint=True)
# 生成一个指定范围内的 uint8 类型的随机整数，包括上限值
def_gen.integers(0, I_u1_high_closed, dtype="uint8", endpoint=True)

# 生成一个范围在 0 到 255 之间的 uint8 类型的随机整数
def_gen.integers(256, dtype=np.uint8)

# 生成一个范围在 0 到 255 之间的 uint8 类型的随机整数，包括上限值 255
def_gen.integers(0, 256, dtype=np.uint8)

# 生成一个范围在 0 到 255 之间的 uint8 类型的随机整数，包括上限值 255
def_gen.integers(255, dtype=np.uint8, endpoint=True)

# 生成一个范围在 0 到 255 之间的 uint8 类型的随机整数，包括上限值 255
def_gen.integers(0, 255, dtype=np.uint8, endpoint=True)

# 生成一个范围在 I_u1_low_like 到 255 之间的 uint8 类型的随机整数，包括上限值 255
def_gen.integers(I_u1_low_like, 255, dtype=np.uint8, endpoint=True)

# 生成一个范围在 I_u1_high_open 到 255 之间的 uint8 类型的随机整数
def_gen.integers(I_u1_high_open, dtype=np.uint8)

# 生成一个范围在 I_u1_low 到 I_u1_high_open 之间的 uint8 类型的随机整数
def_gen.integers(I_u1_low, I_u1_high_open, dtype=np.uint8)

# 生成一个范围在 0 到 I_u1_high_open 之间的 uint8 类型的随机整数
def_gen.integers(0, I_u1_high_open, dtype=np.uint8)

# 生成一个范围在 I_u1_high_closed 到 255 之间的 uint8 类型的随机整数，包括上限值 I_u1_high_closed
def_gen.integers(I_u1_high_closed, dtype=np.uint8, endpoint=True)

# 生成一个范围在 I_u1_low 到 I_u1_high_closed 之间的 uint8 类型的随机整数，包括上限值 I_u1_high_closed
def_gen.integers(I_u1_low, I_u1_high_closed, dtype=np.uint8, endpoint=True)

# 生成一个范围在 0 到 I_u1_high_closed 之间的 uint8 类型的随机整数，包括上限值 I_u1_high_closed
def_gen.integers(0, I_u1_high_closed, dtype=np.uint8, endpoint=True)

# 创建一个 uint16 类型的数组，包含一个元素 0
I_u2_low: np.ndarray[Any, np.dtype[np.uint16]] = np.array([0], dtype=np.uint16)

# 创建一个包含一个元素 0 的整数列表
I_u2_low_like: list[int] = [0]

# 创建一个 uint16 类型的数组，包含一个元素 65535
I_u2_high_open: np.ndarray[Any, np.dtype[np.uint16]] = np.array([65535], dtype=np.uint16)

# 创建一个 uint16 类型的数组，包含一个元素 65535
I_u2_high_closed: np.ndarray[Any, np.dtype[np.uint16]] = np.array([65535], dtype=np.uint16)

# 生成一个范围在 0 到 65536 之间的 u2 类型的随机整数
def_gen.integers(65536, dtype="u2")

# 生成一个范围在 0 到 65536 之间的 u2 类型的随机整数
def_gen.integers(0, 65536, dtype="u2")

# 生成一个范围在 0 到 65535 之间的 u2 类型的随机整数，包括上限值 65535
def_gen.integers(65535, dtype="u2", endpoint=True)

# 生成一个范围在 0 到 65535 之间的 u2 类型的随机整数，包括上限值 65535
def_gen.integers(0, 65535, dtype="u2", endpoint=True)

# 生成一个范围在 I_u2_low_like 到 65535 之间的 u2 类型的随机整数，包括上限值 65535
def_gen.integers(I_u2_low_like, 65535, dtype="u2", endpoint=True)

# 生成一个范围在 I_u2_high_open 到 65535 之间的 u2 类型的随机整数
def_gen.integers(I_u2_high_open, dtype="u2")

# 生成一个范围在 I_u2_low 到 I_u2_high_open 之间的 u2 类型的随机整数
def_gen.integers(I_u2_low, I_u2_high_open, dtype="u2")

# 生成一个范围在 0 到 I_u2_high_open 之间的 u2 类型的随机整数
def_gen.integers(0, I_u2_high_open, dtype="u2")

# 生成一个范围在 I_u2_high_closed 到 65535 之间的 u2 类型的随机整数，包括上限值 I_u2_high_closed
def_gen.integers(I_u2_high_closed, dtype="u2", endpoint=True)

# 生成一个范围在 I_u2_low 到 I_u2_high_closed 之间的 u2 类型的随机整数，包括上限值 I_u2_high_closed
def_gen.integers(I_u2_low, I_u2_high_closed, dtype="u2", endpoint=True)

# 生成一个范围在 0 到 I_u2_high_closed 之间的 u2 类型的随机整数，包括上限值 I_u2_high_closed
def_gen.integers(0, I_u2_high_closed, dtype="u2", endpoint=True)

# 生成一个范围在 0 到 65536 之间的 uint16 类型的随机整数
def_gen.integers(65536, dtype="uint16")

# 生成一个范围在 0 到 65536 之间的 uint16 类型的随机整数
def_gen.integers(0, 65536, dtype="uint16")

# 生成一个范围在 0 到 65535 之间的 uint16 类型的随机整数，包括上限值 65535
def_gen.integers(65535, dtype="uint16", endpoint=True)

# 生成一个范围在 0 到 65535 之间的 uint16 类型的随机整数，包括上限值 65535
def_gen.integers(0, 65535, dtype="uint16", endpoint=True)

# 生成一个范围在 I_u2_low_like 到 65535 之间的 uint16 类型的随机整数，包括上限值 65535
def_gen.integers(I_u2_low_like, 65535, dtype="uint16", endpoint=True)

# 生成一个范围在 I_u2_high_open 到 65535 之间的 uint16 类型的随机整数
def_gen.integers(I_u2_high_open, dtype="uint16")

# 生成一个范围在 I_u2_low 到 I_u2_high_open 之间的 uint16 类型的随机整数
def_gen.integers(I_u2_low, I_u2_high_open, dtype="uint16")

# 生成一个范围在 0 到 I_u2_high_open 之间的 uint16 类型的随机整数
def_gen.integers(0, I_u2_high_open, dtype="uint16")

# 生成一个范围在 I_u2_high_closed 到 65535 之间的 uint16 类型的随机整数，包括上限值 I_u2_high_closed
def_gen.integers(I_u2_high_closed, dtype="uint16", endpoint=True)

# 生成一个范围在 I_u2_low 到 I_u2_high_closed 之间的 uint16 类型的随机整数，包括上限值 I_u2_high_closed
def_gen.integers(I_u2_low, I_u2_high_closed, dtype="uint16", endpoint=True)

# 生成一个范围在 0
# 定义一个包含单个元素 4294967295 的 uint32 类型的 numpy 数组
I_u4_high_open: np.ndarray[Any, np.dtype[np.uint32]] = np.array([4294967295], dtype=np.uint32)
# 定义一个包含单个元素 4294967295 的 uint32 类型的 numpy 数组
I_u4_high_closed: np.ndarray[Any, np.dtype[np.uint32]] = np.array([4294967295], dtype=np.uint32)

# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967296)，不包含 4294967296
def_gen.integers(4294967296, dtype="u4")
# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967296)，不包含 4294967296
def_gen.integers(0, 4294967296, dtype="u4")
# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967295]，包含 4294967295
def_gen.integers(4294967295, dtype="u4", endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967295]，包含 4294967295
def_gen.integers(0, 4294967295, dtype="u4", endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [I_u4_low_like, 4294967295]，包含 4294967295
def_gen.integers(I_u4_low_like, 4294967295, dtype="u4", endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [0, I_u4_high_open)，不包含 I_u4_high_open
def_gen.integers(I_u4_high_open, dtype="u4")
# 生成一个 uint32 类型的随机整数，范围为 [I_u4_low, I_u4_high_open)，不包含 I_u4_high_open
def_gen.integers(I_u4_low, I_u4_high_open, dtype="u4")
# 生成一个 uint32 类型的随机整数，范围为 [0, I_u4_high_open)，不包含 I_u4_high_open
def_gen.integers(0, I_u4_high_open, dtype="u4")
# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967295]，包含 4294967295
def_gen.integers(I_u4_high_closed, dtype="u4", endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [I_u4_low, 4294967295]，包含 4294967295
def_gen.integers(I_u4_low, I_u4_high_closed, dtype="u4", endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967295]，包含 4294967295
def_gen.integers(0, I_u4_high_closed, dtype="u4", endpoint=True)

# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967296)，不包含 4294967296
def_gen.integers(4294967296, dtype="uint32")
# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967296)，不包含 4294967296
def_gen.integers(0, 4294967296, dtype="uint32")
# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967295]，包含 4294967295
def_gen.integers(4294967295, dtype="uint32", endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967295]，包含 4294967295
def_gen.integers(0, 4294967295, dtype="uint32", endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [I_u4_low_like, 4294967295]，包含 4294967295
def_gen.integers(I_u4_low_like, 4294967295, dtype="uint32", endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [0, I_u4_high_open)，不包含 I_u4_high_open
def_gen.integers(I_u4_high_open, dtype="uint32")
# 生成一个 uint32 类型的随机整数，范围为 [I_u4_low, I_u4_high_open)，不包含 I_u4_high_open
def_gen.integers(I_u4_low, I_u4_high_open, dtype="uint32")
# 生成一个 uint32 类型的随机整数，范围为 [0, I_u4_high_open)，不包含 I_u4_high_open
def_gen.integers(0, I_u4_high_open, dtype="uint32")
# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967295]，包含 4294967295
def_gen.integers(I_u4_high_closed, dtype="uint32", endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [I_u4_low, 4294967295]，包含 4294967295
def_gen.integers(I_u4_low, I_u4_high_closed, dtype="uint32", endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [0, 4294967295]，包含 4294967295
def_gen.integers(0, I_u4_high_closed, dtype="uint32", endpoint=True)

# 生成一个 uint32 类型的随机整数，范围为 [0, 18446744073709551616)，不包含 18446744073709551616
def_gen.integers(18446744073709551616, dtype=np.uint64)
# 生成一个 uint32 类型的随机整数，范围为 [0, 18446744073709551616)，不包含 18446744073709551616
def_gen.integers(0, 18446744073709551616, dtype=np.uint64)
# 生成一个 uint32 类型的随机整数，范围为 [0, 18446744073709551615]，包含 18446744073709551615
def_gen.integers(18446744073709551615, dtype=np.uint64, endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [0, 18446744073709551615]，包含 18446744073709551615
def_gen.integers(0, 18446744073709551615, dtype=np.uint64, endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [I_u8_low_like, 18446744073709551615]，包含 18446744073709551615
def_gen.integers(I_u8_low_like, 18446744073709551615, dtype=np.uint64, endpoint=True)
# 生成一个 uint32 类型的随机整数，范围为 [0, I_u8_high_open)，不包含 I_u8_high_open
def_gen.integers(I_u8_high_open, dtype=np.uint64)
# 生成一个 uint32 类型的随机整数，范围为 [I_u8_low, I_u8_high_open)，不包含 I_u8_high_open
def_gen.integers(I_u8_low, I_u8_high_open, dtype=np.uint64)
# 生成一个 uint32 类型的随机整数，范围为 [0, I_u8_high_open)，不包含 I_u8_high_open
def_gen.integers(0, I_u8_high_open, dtype=np.uint64)
# 生成一个随机整数，范围是从 I_u8_high_closed 到最大无符号 8 位整数，包含上界
def_gen.integers(I_u8_high_closed, dtype="u8", endpoint=True)

# 生成一个随机整数，范围是从 I_u8_low 到 I_u8_high_closed，包含上界，数据类型为无符号 8 位整数
def_gen.integers(I_u8_low, I_u8_high_closed, dtype="u8", endpoint=True)

# 生成一个随机整数，范围是从 0 到 I_u8_high_closed，包含上界，数据类型为无符号 8 位整数
def_gen.integers(0, I_u8_high_closed, dtype="u8", endpoint=True)

# 生成一个随机整数，大于等于 18446744073709551616，数据类型为无符号 64 位整数
def_gen.integers(18446744073709551616, dtype="uint64")

# 生成一个随机整数，范围是从 0 到 18446744073709551616，数据类型为无符号 64 位整数
def_gen.integers(0, 18446744073709551616, dtype="uint64")

# 生成一个随机整数，范围是从 0 到 18446744073709551615，包含上界，数据类型为无符号 64 位整数
def_gen.integers(18446744073709551615, dtype="uint64", endpoint=True)

# 生成一个随机整数，范围是从 0 到 18446744073709551615，包含上界，数据类型为无符号 64 位整数
def_gen.integers(0, 18446744073709551615, dtype="uint64", endpoint=True)

# 生成一个随机整数，范围是从 I_u8_low_like 到 18446744073709551615，包含上界，数据类型为无符号 64 位整数
def_gen.integers(I_u8_low_like, 18446744073709551615, dtype="uint64", endpoint=True)

# 生成一个随机整数，大于等于 I_u8_high_open，数据类型为无符号 64 位整数
def_gen.integers(I_u8_high_open, dtype="uint64")

# 生成一个随机整数，范围是从 I_u8_low 到 I_u8_high_open，数据类型为无符号 64 位整数
def_gen.integers(I_u8_low, I_u8_high_open, dtype="uint64")

# 生成一个随机整数，范围是从 0 到 I_u8_high_open，数据类型为无符号 64 位整数
def_gen.integers(0, I_u8_high_open, dtype="uint64")

# 生成一个随机整数，小于等于 I_u8_high_closed，数据类型为无符号 64 位整数
def_gen.integers(I_u8_high_closed, dtype="uint64", endpoint=True)

# 生成一个随机整数，范围是从 I_u8_low 到 I_u8_high_closed，包含上界，数据类型为无符号 64 位整数
def_gen.integers(I_u8_low, I_u8_high_closed, dtype="uint64", endpoint=True)

# 生成一个随机整数，范围是从 0 到 I_u8_high_closed，包含上界，数据类型为无符号 64 位整数
def_gen.integers(0, I_u8_high_closed, dtype="uint64", endpoint=True)

# 生成一个随机整数，大于等于 18446744073709551616，数据类型为 numpy 的无符号 64 位整数
def_gen.integers(18446744073709551616, dtype=np.uint64)

# 生成一个随机整数，范围是从 0 到 18446744073709551616，数据类型为 numpy 的无符号 64 位整数
def_gen.integers(0, 18446744073709551616, dtype=np.uint64)

# 生成一个随机整数，范围是从 0 到 18446744073709551615，包含上界，数据类型为 numpy 的无符号 64 位整数
def_gen.integers(18446744073709551615, dtype=np.uint64, endpoint=True)

# 生成一个随机整数，范围是从 0 到 18446744073709551615，包含上界，数据类型为 numpy 的无符号 64 位整数
def_gen.integers(0, 18446744073709551615, dtype=np.uint64, endpoint=True)

# 生成一个随机整数，范围是从 I_i1_low_like 到 127，包含上界，数据类型为 8 位整数
def_gen.integers(I_i1_low_like, 127, dtype="i1", endpoint=True)

# 生成一个随机整数，大于等于 I_i1_high_open，数据类型为 8 位整数
def_gen.integers(I_i1_high_open, dtype="i1")

# 生成一个随机整数，范围是从 I_i1_low 到 I_i1_high_open，数据类型为 8 位整数
def_gen.integers(I_i1_low, I_i1_high_open, dtype="i1")

# 生成一个随机整数，小于等于 I_i1_high_open，数据类型为 8 位整数
def_gen.integers(-128, I_i1_high_open, dtype="i1")

# 生成一个随机整数，大于等于 I_i1_high_closed，数据类型为 8 位整数
def_gen.integers(I_i1_high_closed, dtype="i1", endpoint=True)

# 生成一个随机整数，范围是从 I_i1_low 到 I_i1_high_closed，包含上界，数据类型为 8 位整数
def_gen.integers(I_i1_low, I_i1_high_closed, dtype="i1", endpoint=True)

# 生成一个随机整数，范围是从 0 到 I_i1_high_closed，包含上界，数据类型为 8 位整数
def_gen.integers(-128, I_i1_high_closed, dtype="i1", endpoint=True)

# 生成一个随机整数，范围是从 0 到 127，包含上界，数据类型为 8 位整数
def_gen.integers(127, dtype="i1", endpoint=True)

# 生成一个随机整数，范围是从 -128 到 127，包含上界，数据类型为 8 位整数
def_gen.integers(-128, 127, dtype="i1", endpoint=True)

# 生成一个随机整数，范围是从 I_i1_low_like 到 127，包含上界，数据类型为 8 位整数
def_gen.integers(I_i1_low_like, 127, dtype="int8", endpoint=True)

# 生成一个随机整数，大于等于 I_i1_high_open，数据类型为 8 位整数
def_gen.integers(I_i1_high_open, dtype="int8")

# 生成一个随机整数，范围是从 I_i1_low 到 I_i1_high_open，数据类型为 8 位整数
def_gen.integers(I_i1_low, I_i1_high_open, dtype="int8")

# 生成一个随机整数，小于等于 I_i1_high_open，数据类型为 8 位整数
def_gen.integers(-128, I_i1_high_open, dtype="int8")

# 生成一个随机整数，大于等于 I_i1_high_closed，数据类型为 8 位整数
def_gen.integers(I_i1_high_closed, dtype="int8", endpoint=True)
    
# 生成一个随机整数，范围是从 I_i1_low 到 I_i1_high_closed，包含上界，数据类型为 8 位整数
def_gen.integers(I_i1_low, I_i1_high_closed, dtype="int8", endpoint=True)

# 生成一个随机整数，范围是从 -128 到 I_i1_high_closed，包含上界，数据类型为 8 位整数
def_gen.integers(-128, I_i1_high_closed, dtype="int8", endpoint=True)

# 创建一个 numpy 的 8 位整数类型的数组，包
# 使用 numpy.random.default_rng() 创建一个新的生成器对象
def_gen.integers(I_i1_low, I_i1_high_closed, dtype="int8", endpoint=True)

# 使用 np.int8 类型生成一个随机整数，范围从 -128 到 I_i1_high_closed
def_gen.integers(-128, I_i1_high_closed, dtype="int8", endpoint=True)

# 使用 np.int8 类型生成一个随机整数，范围从 128 到 +127
def_gen.integers(128, dtype=np.int8)

# 使用 np.int8 类型生成一个随机整数，范围从 -128 到 +127
def_gen.integers(-128, 128, dtype=np.int8)

# 使用 np.int8 类型生成一个随机整数，范围从 0 到 +127
def_gen.integers(127, dtype=np.int8, endpoint=True)

# 使用 np.int8 类型生成一个随机整数，范围从 -128 到 +127
def_gen.integers(-128, 127, dtype=np.int8, endpoint=True)

# 使用 np.int8 类型生成一个随机整数，范围从 I_i1_low_like 到 +127
def_gen.integers(I_i1_low_like, 127, dtype=np.int8, endpoint=True)

# 使用 np.int8 类型生成一个随机整数，范围从 I_i1_high_open 到 +127
def_gen.integers(I_i1_high_open, dtype=np.int8)

# 使用 np.int8 类型生成一个随机整数，范围从 I_i1_low 到 I_i1_high_open
def_gen.integers(I_i1_low, I_i1_high_open, dtype=np.int8)

# 使用 np.int8 类型生成一个随机整数，范围从 -128 到 I_i1_high_open
def_gen.integers(-128, I_i1_high_open, dtype=np.int8)

# 使用 np.int8 类型生成一个随机整数，范围从 I_i1_high_closed 到 +127
def_gen.integers(I_i1_high_closed, dtype=np.int8, endpoint=True)

# 使用 np.int8 类型生成一个随机整数，范围从 I_i1_low 到 I_i1_high_closed
def_gen.integers(I_i1_low, I_i1_high_closed, dtype=np.int8, endpoint=True)

# 使用 np.int8 类型生成一个随机整数，范围从 -128 到 I_i1_high_closed
def_gen.integers(-128, I_i1_high_closed, dtype=np.int8, endpoint=True)

# 创建一个包含单个元素 -32768 的 np.int16 类型的数组
I_i2_low: np.ndarray[Any, np.dtype[np.int16]] = np.array([-32768], dtype=np.int16)

# 创建一个包含单个元素 -32768 的 list[int] 类型列表
I_i2_low_like: list[int] = [-32768]

# 创建一个包含单个元素 32767 的 np.int16 类型的数组
I_i2_high_open: np.ndarray[Any, np.dtype[np.int16]] = np.array([32767], dtype=np.int16)

# 创建一个包含单个元素 32767 的 np.int16 类型的数组
I_i2_high_closed: np.ndarray[Any, np.dtype[np.int16]] = np.array([32767], dtype=np.int16)

# 使用 dtype="i2" 类型生成一个随机整数，范围从 32768 到 +32767
def_gen.integers(32768, dtype="i2")

# 使用 dtype="i2" 类型生成一个随机整数，范围从 -32768 到 32768
def_gen.integers(-32768, 32768, dtype="i2")

# 使用 dtype="i2" 类型生成一个随机整数，范围从 32767 到 +32767
def_gen.integers(32767, dtype="i2", endpoint=True)

# 使用 dtype="i2" 类型生成一个随机整数，范围从 -32768 到 32767
def_gen.integers(-32768, 32767, dtype="i2", endpoint=True)

# 使用 dtype="i2" 类型生成一个随机整数，范围从 I_i2_low_like 到 32767
def_gen.integers(I_i2_low_like, 32767, dtype="i2", endpoint=True)

# 使用 dtype="i2" 类型生成一个随机整数，范围从 I_i2_high_open 到 +32767
def_gen.integers(I_i2_high_open, dtype="i2")

# 使用 dtype="i2" 类型生成一个随机整数，范围从 I_i2_low 到 I_i2_high_open
def_gen.integers(I_i2_low, I_i2_high_open, dtype="i2")

# 使用 dtype="i2" 类型生成一个随机整数，范围从 -32768 到 I_i2_high_open
def_gen.integers(-32768, I_i2_high_open, dtype="i2")

# 使用 dtype="i2" 类型生成一个随机整数，范围从 I_i2_high_closed 到 +32767
def_gen.integers(I_i2_high_closed, dtype="i2", endpoint=True)

# 使用 dtype="i2" 类型生成一个随机整数，范围从 I_i2_low 到 I_i2_high_closed
def_gen.integers(I_i2_low, I_i2_high_closed, dtype="i2", endpoint=True)

# 使用 dtype="i2" 类型生成一个随机整数，范围从 -32768 到 I_i2_high_closed
def_gen.integers(-32768, I_i2_high_closed, dtype="i2", endpoint=True)

# 使用 dtype="int16" 类型生成一个随机整数，范围从 32768 到 +32767
def_gen.integers(32768, dtype="int16")

# 使用 dtype="int16" 类型生成一个随机整数，范围从 -32768 到 32768
def_gen.integers(-32768, 32768, dtype="int16")

# 使用 dtype="int16" 类型生成一个随机整数，范围从 32767 到 +32767
def_gen.integers(32767, dtype="int16", endpoint=True)

# 使用 dtype="int16" 类型生成一个随机整数，范围从 -32768 到 32767
def_gen.integers(-32768, 32767, dtype="int16", endpoint=True)

# 使用 dtype="int16" 类型生成一个随机整数，范围从 I_i2_low_like 到 32767
def_gen.integers(I_i2_low_like, 32767, dtype="int16", endpoint=True)

# 使用 dtype="int16" 类型生成一个随机整数，范围从 I_i2_high_open 到 +32767
def_gen.integers(I_i2_high_open, dtype="int16")

# 使用 dtype="int16" 类型生成一个随机整数，范围从 I_i2_low 到 I_i2_high_open
def_gen.integers(I_i2_low, I_i2_high_open, dtype="int16")

# 使用 dtype="int16" 类型生成一个随机整数，范围从 -32768 到 I_i2_high_open
def_gen.integers(-32768, I_i2_high_open, dtype="int16")

# 使用 dtype="int16" 类型生成一个随机整数，范围从 I_i2_high_closed 到 +32767
def_gen.integers(I_i2_high_closed, dtype="int16", endpoint=True)

# 使用 dtype="int16" 类型生成一个随机整数，范围从 I_i2_low 到 I_i2_high_closed
def_gen.integers(I_i2_low, I_i2_high_closed, dtype="int16", endpoint=True)

# 使用 dtype="int16" 类型生成一个随机整数，范围从 -32768 到 I_i2_high_closed
def_gen.integers(-32768, I_i2_high_closed, dtype="int16", endpoint=True)

# 使用 dtype=np.int16 类型生成一个随机整数，范围从 32768 到 +32767
def_gen.integers(32768, dtype=np.int16)

# 使用 dtype=np.int16 类型生成一个随机整数，范围从 -32768 到 32768
def_gen.integers(-32768, 32768, dtype=np.int16)

# 使用 dtype=np.int16 类型生成一个随机整数，范围从 32767 到 +32767
def_gen.integers(32767, dtype=np.int16,
# 创建一个包含单个值 -2147483648 的 int32 类型的 NumPy 数组
I_i4_low: np.ndarray[Any, np.dtype[np.int32]] = np.array([-2147483648], dtype=np.int32)
# 创建一个包含单个值 -2147483648 的 int32 类型的列表
I_i4_low_like: list[int] = [-2147483648]
# 创建一个包含单个值 2147483647 的 int32 类型的 NumPy 数组
I_i4_high_open: np.ndarray[Any, np.dtype[np.int32]] = np.array([2147483647], dtype=np.int32)
# 创建一个包含单个值 2147483647 的 int32 类型的 NumPy 数组
I_i4_high_closed: np.ndarray[Any, np.dtype[np.int32]] = np.array([2147483647], dtype=np.int32)

# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，默认情况下是开区间
def_gen.integers(2147483648, dtype="i4")
# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，指定了范围
def_gen.integers(-2147483648, 2147483648, dtype="i4")
# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，端点包含在内
def_gen.integers(2147483647, dtype="i4", endpoint=True)
# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，指定了范围且端点包含在内
def_gen.integers(-2147483648, 2147483647, dtype="i4", endpoint=True)
# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，起始值为列表中的值
def_gen.integers(I_i4_low_like, 2147483647, dtype="i4", endpoint=True)
# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，默认情况下是开区间，起始值为数组中的值
def_gen.integers(I_i4_high_open, dtype="i4")
# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，起始值和结束值分别为数组中的值
def_gen.integers(I_i4_low, I_i4_high_open, dtype="i4")
# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，起始值为 -2147483648，结束值为数组中的值
def_gen.integers(-2147483648, I_i4_high_open, dtype="i4")
# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，端点包含在内，结束值为数组中的值
def_gen.integers(I_i4_high_closed, dtype="i4", endpoint=True)
# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，指定了范围且端点包含在内，起始值和结束值分别为数组中的值
def_gen.integers(I_i4_low, I_i4_high_closed, dtype="i4", endpoint=True)
# 使用 def_gen.integers 函数生成一个 int32 类型的随机整数，起始值为 -2147483648，结束值为数组中的值，端点包含在内
def_gen.integers(-2147483648, I_i4_high_closed, dtype="i4", endpoint=True)

# 创建一个包含单个值 -9223372036854775808 的 int64 类型的 NumPy 数组
I_i8_low: np.ndarray[Any, np.dtype[np.int64]] = np.array([-9223372036854775808], dtype=np.int64)
# 创建一个包含单个值 -9223372036854775808 的 int64 类型的列表
I_i8_low_like: list[int] = [-9223372036854775808]
# 创建一个包含单个值 9223372036854775807 的 int64 类型的 NumPy 数组
I_i8_high_open: np.ndarray[Any, np.dtype[np.int64]] = np.array([9223372036854775807], dtype=np.int64)
# 创建一个包含单个值 9223372036854775807 的 int64 类型的 NumPy 数组
I_i8_high_closed: np.ndarray[Any, np.dtype[np.int64]] = np.array([9223372036854775807], dtype=np.int64)

# 使用 def_gen.integers 函数生成一个 int64 类型的随机整数，默认情况下是开区间
def_gen.integers(9223372036854775808, dtype="i8")
# 使用 def_gen.integers 函数生成一个 int64 类型的随机整数，指定了范围
def_gen.integers(-9223372036854775808, 9223372036854775808, dtype="i8")
# 使用 def_gen.integers 函数生成一个 int64 类型的随机整数，端点包含在内
def_gen.integers(9223372036854775807, dtype="i8", endpoint=True)
# 使用 def_gen.integers 函数生成一个 int64 类型的随机整数，指定了范围且端点包含在内
def_gen.integers(-9223372036854775808, 9223372036854775807, dtype="i8", endpoint=True)
# 生成一个从 I_i8_low_like 到 9223372036854775807 的整数，数据类型为 'i8'，包括端点
def_gen.integers(I_i8_low_like, 9223372036854775807, dtype="i8", endpoint=True)

# 生成一个从 I_i8_high_open 开始的整数，数据类型为 'i8'
def_gen.integers(I_i8_high_open, dtype="i8")

# 生成一个从 I_i8_low 到 I_i8_high_open 的整数，数据类型为 'i8'
def_gen.integers(I_i8_low, I_i8_high_open, dtype="i8")

# 生成一个从 -9223372036854775808 到 I_i8_high_open 的整数，数据类型为 'i8'
def_gen.integers(-9223372036854775808, I_i8_high_open, dtype="i8")

# 生成一个从 I_i8_high_closed 开始的整数，数据类型为 'i8'，不包括端点
def_gen.integers(I_i8_high_closed, dtype="i8", endpoint=True)

# 生成一个从 I_i8_low 到 I_i8_high_closed 的整数，数据类型为 'i8'，包括端点
def_gen.integers(I_i8_low, I_i8_high_closed, dtype="i8", endpoint=True)

# 生成一个从 -9223372036854775808 到 I_i8_high_closed 的整数，数据类型为 'i8'，包括端点
def_gen.integers(-9223372036854775808, I_i8_high_closed, dtype="i8", endpoint=True)

# 生成一个大于 9223372036854775808 的整数，数据类型为 'int64'
def_gen.integers(9223372036854775808, dtype="int64")

# 生成一个从 -9223372036854775808 到 9223372036854775808 的整数，数据类型为 'int64'
def_gen.integers(-9223372036854775808, 9223372036854775808, dtype="int64")

# 生成一个从 9223372036854775807 开始的整数，数据类型为 'int64'，包括端点
def_gen.integers(9223372036854775807, dtype="int64", endpoint=True)

# 生成一个从 -9223372036854775808 到 9223372036854775807 的整数，数据类型为 'int64'，包括端点
def_gen.integers(-9223372036854775808, 9223372036854775807, dtype="int64", endpoint=True)

# 生成一个从 I_i8_low_like 到 9223372036854775807 的整数，数据类型为 'int64'，包括端点
def_gen.integers(I_i8_low_like, 9223372036854775807, dtype="int64", endpoint=True)

# 生成一个从 I_i8_high_open 开始的整数，数据类型为 'int64'
def_gen.integers(I_i8_high_open, dtype="int64")

# 生成一个从 I_i8_low 到 I_i8_high_open 的整数，数据类型为 'int64'
def_gen.integers(I_i8_low, I_i8_high_open, dtype="int64")

# 生成一个从 -9223372036854775808 到 I_i8_high_open 的整数，数据类型为 'int64'
def_gen.integers(-9223372036854775808, I_i8_high_open, dtype="int64")

# 生成一个从 I_i8_high_closed 开始的整数，数据类型为 'int64'，不包括端点
def_gen.integers(I_i8_high_closed, dtype="int64", endpoint=True)

# 生成一个从 I_i8_low 到 I_i8_high_closed 的整数，数据类型为 'int64'，包括端点
def_gen.integers(I_i8_low, I_i8_high_closed, dtype="int64", endpoint=True)

# 生成一个从 -9223372036854775808 到 I_i8_high_closed 的整数，数据类型为 'int64'，包括端点
def_gen.integers(-9223372036854775808, I_i8_high_closed, dtype="int64", endpoint=True)

# 获取默认的随机数生成器的位生成器
def_gen.bit_generator

# 生成一个长度为 2 的随机字节数组
def_gen.bytes(2)

# 从 0 到 4 中随机选择一个整数
def_gen.choice(5)

# 从 0 到 4 中随机选择 3 个整数，允许重复选择
def_gen.choice(5, 3)

# 从 0 到 4 中随机选择 3 个整数，允许重复选择
def_gen.choice(5, 3, replace=True)

# 从 0 到 4 中随机选择 3 个整数，每个选择的概率相等
def_gen.choice(5, 3, p=[1 / 5] * 5)

# 从 0 到 4 中随机选择 3 个整数，每个选择的概率相等，且不允许重复选择
def_gen.choice(5, 3, p=[1 / 5] * 5, replace=False)

# 从给定的列表中随机选择一个元素
def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"])

# 从给定的列表中随机选择 3 个元素
def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3)

# 从给定的列表中随机选择 3 个元素，每个选择的概率相等
def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, p=[1 / 4] * 4)

# 从给定的列表中随机选择 3 个元素，允许重复选择
def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=True)

# 从给定的列表中随机选择 3 个元素，不允许重复选择，且每个选择的概率由给定的 numpy 数组决定
def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=False, p=np.array([1 / 8, 1 / 8, 1 / 2, 1 / 4]))

# 生成一个二维 Dirichlet 分布样本，参数为 [0.5, 0.5]
def_gen.dirichlet([0.5, 0.5])

# 生成一个二维 Dirichlet 分布样本，参数为 np.array([0.5, 0.5])
def_gen.dirichlet(np.array([0.5, 0.5]))

# 生成三个二维 Dirichlet 分布样本，参数为 np.array([0.5, 0.5])
def_gen.dirichlet(np.array([0.5, 0.5]), size=3)

# 生成一个多项式分布样本，参数为总共 20 次试验，每个结果的概率为 [1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0]
def_gen.multinomial(20, [1 / 6.0] * 6)

# 生成一个多项式分布样本，参数为总共 20 次试验，每个结果的概率为 np.array([0.5, 0.5])
def_gen.multinomial(20, np.array([0.5, 0.5]))

# 生成两个多项式分布样本，每个样本为总共 20 次试验，每个结果的概率为 [1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0]
def_gen.multinomial(20, [1 / 6.0] * 6, size=2)

# 生成一个二维多项式分布样本，参数为分别有两个试验组，每个组中有 [10] 和 [20] 次试验，每个结果的概率为 [1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0]
def_gen.multinomial([[10], [20]], [1 / 6.0] * 6, size=(2,
# 从 np.random 模块中导入 RandomState 类别名为 random_st
random_st: np.random.RandomState = np.random.RandomState()

# 生成一个指定形状的多项式分布样本
def_gen.multinomial(np.array([[10], [20]]), np.array([0.5, 0.5]), size=(2, 2))

# 使用多元超几何分布生成样本，传入列表参数
def_gen.multivariate_hypergeometric([3, 5, 7], 2)
def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2)
def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, size=4)
def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, size=(4, 7))
def_gen.multivariate_hypergeometric([3, 5, 7], 2, method="count")
def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, method="marginals")

# 使用多元正态分布生成样本，传入均值和协方差矩阵
def_gen.multivariate_normal([0.0], [[1.0]])
def_gen.multivariate_normal([0.0], np.array([[1.0]]))
def_gen.multivariate_normal(np.array([0.0]), [[1.0]])
def_gen.multivariate_normal([0.0], np.array([[1.0]]))

# 生成一个序列的随机排列
def_gen.permutation(10)
def_gen.permutation([1, 2, 3, 4])
def_gen.permutation(np.array([1, 2, 3, 4]))
def_gen.permutation(D_2D, axis=1)
def_gen.permuted(D_2D)
def_gen.permuted(D_2D_like)
def_gen.permuted(D_2D, axis=1)
def_gen.permuted(D_2D, out=D_2D)
def_gen.permuted(D_2D_like, out=D_2D)
def_gen.permuted(D_2D_like, out=D_2D)
def_gen.permuted(D_2D, axis=1, out=D_2D)

# 对数组或列表进行就地随机排列
def_gen.shuffle(np.arange(10))
def_gen.shuffle([1, 2, 3, 4, 5])
def_gen.shuffle(D_2D, axis=1)

# 返回 RandomState 对象的字符串表示形式
def_gen.__str__()
def_gen.__repr__()
def_gen.__setstate__(dict(def_gen.bit_generator.state))

# 以下是 RandomState 对象的各种随机分布方法的使用示例

# 标准正态分布
random_st.standard_normal()
random_st.standard_normal(size=None)
random_st.standard_normal(size=1)

# 均匀分布
random_st.random()
random_st.random(size=None)
random_st.random(size=1)

# 标准柯西分布
random_st.standard_cauchy()
random_st.standard_cauchy(size=None)
random_st.standard_cauchy(size=1)

# 标准指数分布
random_st.standard_exponential()
random_st.standard_exponential(size=None)
random_st.standard_exponential(size=1)

# Zipf 分布
random_st.zipf(1.5)
random_st.zipf(1.5, size=None)
random_st.zipf(1.5, size=1)
random_st.zipf(D_arr_1p5)
random_st.zipf(D_arr_1p5, size=1)
random_st.zipf(D_arr_like_1p5)
random_st.zipf(D_arr_like_1p5, size=1)

# Weibull 分布
random_st.weibull(0.5)
random_st.weibull(0.5, size=None)
random_st.weibull(0.5, size=1)
random_st.weibull(D_arr_0p5)
random_st.weibull(D_arr_0p5, size=1)
random_st.weibull(D_arr_like_0p5)
random_st.weibull(D_arr_like_0p5, size=1)

# t 分布
random_st.standard_t(0.5)
random_st.standard_t(0.5, size=None)
random_st.standard_t(0.5, size=1)
random_st.standard_t(D_arr_0p5)
random_st.standard_t(D_arr_0p5, size=1)
random_st.standard_t(D_arr_like_0p5)
random_st.standard_t(D_arr_like_0p5, size=1)

# 泊松分布
random_st.poisson(0.5)
random_st.poisson(0.5, size=None)
random_st.poisson(0.5, size=1)
random_st.poisson(D_arr_0p5)
random_st.poisson(D_arr_0p5, size=1)
random_st.poisson(D_arr_like_0p5)
random_st.poisson(D_arr_like_0p5, size=1)

# 功率分布
random_st.power(0.5)
random_st.power(0.5, size=None)
random_st.power(0.5, size=1)
random_st.power(D_arr_0p5)
random_st.power(D_arr_0p5, size=1)
random_st.power(D_arr_like_0p5)
random_st.power(D_arr_like_0p5, size=1)

# 帕累托分布
random_st.pareto(0.5)
random_st.pareto(0.5, size=None)
random_st.pareto(0.5, size=1)
random_st.pareto(D_arr_0p5)
random_st.pareto(D_arr_0p5, size=1)
# 生成服从帕累托分布的随机数
random_st.pareto(D_arr_like_0p5)

# 生成一个服从帕累托分布的随机数，返回一个大小为1的数组
random_st.pareto(D_arr_like_0p5, size=1)

# 生成服从卡方分布的随机数，自由度为0.5
random_st.chisquare(0.5)

# 生成一个服从卡方分布的随机数，自由度为0.5，默认返回单个数值
random_st.chisquare(0.5, size=None)

# 生成一个服从卡方分布的随机数，自由度为0.5，返回一个大小为1的数组
random_st.chisquare(0.5, size=1)

# 生成服从卡方分布的随机数，自由度为D_arr_0p5
random_st.chisquare(D_arr_0p5)

# 生成一个服从卡方分布的随机数，自由度为D_arr_0p5，返回一个大小为1的数组
random_st.chisquare(D_arr_0p5, size=1)

# 生成服从卡方分布的随机数，自由度为D_arr_like_0p5
random_st.chisquare(D_arr_like_0p5)

# 生成一个服从卡方分布的随机数，自由度为D_arr_like_0p5，返回一个大小为1的数组
random_st.chisquare(D_arr_like_0p5, size=1)

# 生成服从指数分布的随机数，比例参数为0.5
random_st.exponential(0.5)

# 生成一个服从指数分布的随机数，比例参数为0.5，默认返回单个数值
random_st.exponential(0.5, size=None)

# 生成一个服从指数分布的随机数，比例参数为0.5，返回一个大小为1的数组
random_st.exponential(0.5, size=1)

# 生成服从指数分布的随机数，比例参数为D_arr_0p5
random_st.exponential(D_arr_0p5)

# 生成一个服从指数分布的随机数，比例参数为D_arr_0p5，返回一个大小为1的数组
random_st.exponential(D_arr_0p5, size=1)

# 生成服从指数分布的随机数，比例参数为D_arr_like_0p5
random_st.exponential(D_arr_like_0p5)

# 生成一个服从指数分布的随机数，比例参数为D_arr_like_0p5，返回一个大小为1的数组
random_st.exponential(D_arr_like_0p5, size=1)

# 生成服从几何分布的随机数，成功概率为0.5
random_st.geometric(0.5)

# 生成一个服从几何分布的随机数，成功概率为0.5，默认返回单个数值
random_st.geometric(0.5, size=None)

# 生成一个服从几何分布的随机数，成功概率为0.5，返回一个大小为1的数组
random_st.geometric(0.5, size=1)

# 生成服从几何分布的随机数，成功概率为D_arr_0p5
random_st.geometric(D_arr_0p5)

# 生成一个服从几何分布的随机数，成功概率为D_arr_0p5，返回一个大小为1的数组
random_st.geometric(D_arr_0p5, size=1)

# 生成服从几何分布的随机数，成功概率为D_arr_like_0p5
random_st.geometric(D_arr_like_0p5)

# 生成一个服从几何分布的随机数，成功概率为D_arr_like_0p5，返回一个大小为1的数组
random_st.geometric(D_arr_like_0p5, size=1)

# 生成服从对数系列分布的随机数，成功概率为0.5
random_st.logseries(0.5)

# 生成一个服从对数系列分布的随机数，成功概率为0.5，默认返回单个数值
random_st.logseries(0.5, size=None)

# 生成一个服从对数系列分布的随机数，成功概率为0.5，返回一个大小为1的数组
random_st.logseries(0.5, size=1)

# 生成服从对数系列分布的随机数，成功概率为D_arr_0p5
random_st.logseries(D_arr_0p5)

# 生成一个服从对数系列分布的随机数，成功概率为D_arr_0p5，返回一个大小为1的数组
random_st.logseries(D_arr_0p5, size=1)

# 生成服从对数系列分布的随机数，成功概率为D_arr_like_0p5
random_st.logseries(D_arr_like_0p5)

# 生成一个服从对数系列分布的随机数，成功概率为D_arr_like_0p5，返回一个大小为1的数组
random_st.logseries(D_arr_like_0p5, size=1)

# 生成服从瑞利分布的随机数，比例参数为0.5
random_st.rayleigh(0.5)

# 生成一个服从瑞利分布的随机数，比例参数为0.5，默认返回单个数值
random_st.rayleigh(0.5, size=None)

# 生成一个服从瑞利分布的随机数，比例参数为0.5，返回一个大小为1的数组
random_st.rayleigh(0.5, size=1)

# 生成服从瑞利分布的随机数，比例参数为D_arr_0p5
random_st.rayleigh(D_arr_0p5)

# 生成一个服从瑞利分布的随机数，比例参数为D_arr_0p5，返回一个大小为1的数组
random_st.rayleigh(D_arr_0p5, size=1)

# 生成服从瑞利分布的随机数，比例参数为D_arr_like_0p5
random_st.rayleigh(D_arr_like_0p5)

# 生成一个服从瑞利分布的随机数，比例参数为D_arr_like_0p5，返回一个大小为1的数组
random_st.rayleigh(D_arr_like_0p5, size=1)

# 生成服从标准伽马分布的随机数，形状参数为0.5
random_st.standard_gamma(0.5)

# 生成一个服从标准伽马分布的随机数，形状参数为0.5，默认返回单个数值
random_st.standard_gamma(0.5, size=None)

# 生成一个服从标准伽马分布的随机数，形状参数为0.5，返回一个大小为1的数组
random_st.standard_gamma(0.5, size=1)

# 生成服从标准伽马分布的随机数，形状参数为D_arr_0p5
random_st.standard_gamma(D_arr_0p5)

# 生成一个服从标准伽马分布的随机数，形状参数为D_arr_0p5，返回一个大小为1的数组
random_st.standard_gamma(D_arr_0p5, size=1)

# 生成服从标准伽马分布的随机数，形状参数为D_arr_like_0p5
random_st.standard_gamma(D_arr_like_0p5)

# 生成一个服从标准伽马分布的随机数，形状参数为D_arr_like_0p5，返回一个大小为1的数组
random_st.standard_gamma(D_arr_like_0p5, size=1)

# 生成服从冯·米塞斯分布的随机数，形状参数为0.5，位置参数为0.5
random_st.vonmises(0.5, 0.5)

# 生成一个服从冯·米塞斯分布的随机数
# 生成一个在区间 [0.5, D_arr_like_0p5] 内均匀分布的随机数
random_st.uniform(0.5, D_arr_like_0p5)

# 生成一个在区间 [D_arr_0p5, D_arr_0p5] 内均匀分布的随机数
random_st.uniform(D_arr_0p5, D_arr_0p5)

# 生成一个在区间 [D_arr_like_0p5, D_arr_like_0p5] 内均匀分布的随机数
random_st.uniform(D_arr_like_0p5, D_arr_like_0p5)

# 生成一个在区间 [D_arr_0p5, D_arr_0p5] 内均匀分布的大小为1的随机数数组
random_st.uniform(D_arr_0p5, D_arr_0p5, size=1)

# 生成一个在区间 [D_arr_like_0p5, D_arr_like_0p5] 内均匀分布的大小为1的随机数数组
random_st.uniform(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 生成一个 Beta 分布的随机数，参数为 (0.5, 0.5)
random_st.beta(0.5, 0.5)

# 生成一个 Beta 分布的随机数，参数为 (0.5, 0.5)，大小为 None
random_st.beta(0.5, 0.5, size=None)

# 生成一个 Beta 分布的大小为1的随机数数组，参数为 (0.5, 0.5)
random_st.beta(0.5, 0.5, size=1)

# 生成一个 Beta 分布的随机数，其中一个参数为 D_arr_0p5，另一个为 0.5
random_st.beta(D_arr_0p5, 0.5)

# 生成一个 Beta 分布的随机数，其中一个参数为 0.5，另一个为 D_arr_0p5
random_st.beta(0.5, D_arr_0p5)

# 生成一个 Beta 分布的大小为1的随机数数组，其中一个参数为 D_arr_0p5，另一个为 0.5
random_st.beta(D_arr_0p5, 0.5, size=1)

# 生成一个 Beta 分布的大小为1的随机数数组，其中一个参数为 0.5，另一个为 D_arr_0p5
random_st.beta(0.5, D_arr_0p5, size=1)

# 生成一个 Beta 分布的随机数，其中一个参数为 D_arr_like_0p5，另一个为 0.5
random_st.beta(D_arr_like_0p5, 0.5)

# 生成一个 Beta 分布的随机数，其中一个参数为 0.5，另一个为 D_arr_like_0p5
random_st.beta(0.5, D_arr_like_0p5)

# 生成一个 Beta 分布的随机数，两个参数均为 D_arr_0p5
random_st.beta(D_arr_0p5, D_arr_0p5)

# 生成一个 Beta 分布的随机数，两个参数均为 D_arr_like_0p5
random_st.beta(D_arr_like_0p5, D_arr_like_0p5)

# 生成一个 Beta 分布的大小为1的随机数数组，两个参数均为 D_arr_0p5
random_st.beta(D_arr_0p5, D_arr_0p5, size=1)

# 生成一个 Beta 分布的大小为1的随机数数组，两个参数均为 D_arr_like_0p5
random_st.beta(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 生成一个 F 分布的随机数，参数为 (0.5, 0.5)
random_st.f(0.5, 0.5)

# 生成一个 F 分布的随机数，参数为 (0.5, 0.5)，大小为 None
random_st.f(0.5, 0.5, size=None)

# 生成一个 F 分布的大小为1的随机数数组，参数为 (0.5, 0.5)
random_st.f(0.5, 0.5, size=1)

# 生成一个 F 分布的随机数，其中一个参数为 D_arr_0p5，另一个为 0.5
random_st.f(D_arr_0p5, 0.5)

# 生成一个 F 分布的随机数，其中一个参数为 0.5，另一个为 D_arr_0p5
random_st.f(0.5, D_arr_0p5)

# 生成一个 F 分布的大小为1的随机数数组，其中一个参数为 D_arr_0p5，另一个为 0.5
random_st.f(D_arr_0p5, 0.5, size=1)

# 生成一个 F 分布的大小为1的随机数数组，其中一个参数为 0.5，另一个为 D_arr_0p5
random_st.f(0.5, D_arr_0p5, size=1)

# 生成一个 F 分布的随机数，其中一个参数为 D_arr_like_0p5，另一个为 0.5
random_st.f(D_arr_like_0p5, 0.5)

# 生成一个 F 分布的随机数，其中一个参数为 0.5，另一个为 D_arr_like_0p5
random_st.f(0.5, D_arr_like_0p5)

# 生成一个 F 分布的随机数，两个参数均为 D_arr_0p5
random_st.f(D_arr_0p5, D_arr_0p5)

# 生成一个 F 分布的随机数，两个参数均为 D_arr_like_0p5
random_st.f(D_arr_like_0p5, D_arr_like_0p5)

# 生成一个 F 分布的大小为1的随机数数组，两个参数均为 D_arr_0p5
random_st.f(D_arr_0p5, D_arr_0p5, size=1)

# 生成一个 F 分布的大小为1的随机数数组，两个参数均为 D_arr_like_0p5
random_st.f(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 生成一个 Gamma 分布的随机数，参数为 (0.5, 0.5)
random_st.gamma(0.5, 0.5)

# 生成一个 Gamma 分布的随机数，参数为 (0.5, 0.5)，大小为 None
random_st.gamma(0.5, 0.5, size=None)

# 生成一个 Gamma 分布的大小为1的随机数数组，参数为 (0.5, 0.5)
random_st.gamma(0.5, 0.5, size=1)

# 生成一个 Gamma 分布的随机数，其中一个参数为 D_arr_0p5，另一个为 0.5
random_st.gamma(D_arr_0p5, 0.5)

# 生成一个 Gamma 分布的随机数，其中一个参数为 0.5，另一个为 D_arr_0p5
random_st.gamma(0.5, D_arr_0p5)

# 生成一个 Gamma 分布的大小为1的随机数数组，其中一个参数为 D_arr_0p5，另一个为 0.5
random_st.gamma(D_arr_0p5, 0.5, size=1)

# 生成一个 Gamma 分布的大小为1的随机数数组，其中一个参数为 0.5，另一个为 D_arr_0p5
random_st.gamma(0.5, D_arr_0p5, size=1)

# 生成一个 Gamma 分布的随机数，其中一个参数为 D_arr_like_0p5，另一个为 0.5
random_st.gamma(D_arr_like_0p5, 0.5)

# 生成一个 Gamma 分布的随机数，其中一个参数为 0.5，另一个为 D_arr_like_0p5
random_st.gamma(0.5, D_arr_like_0p5)

# 生成一个 Gamma 分布的随机数，两个参数均为 D_arr_0p5
random_st.gamma(D_arr_0p5, D_arr_0p5)

#
# 从 scipy.stats 模块中调用 logistic 分布函数，生成指定形状参数的随机数
random_st.logistic(D_arr_like_0p5, 0.5)

# 从 scipy.stats 模块中调用 logistic 分布函数，生成指定位置参数的随机数
random_st.logistic(0.5, D_arr_like_0p5)

# 从 scipy.stats 模块中调用 logistic 分布函数，生成两个参数相同的随机数
random_st.logistic(D_arr_0p5, D_arr_0p5)

# 从 scipy.stats 模块中调用 logistic 分布函数，生成两个参数相同的随机数
random_st.logistic(D_arr_like_0p5, D_arr_like_0p5)

# 从 scipy.stats 模块中调用 logistic 分布函数，生成指定形状参数的单个随机数
random_st.logistic(D_arr_0p5, D_arr_0p5, size=1)

# 从 scipy.stats 模块中调用 logistic 分布函数，生成指定形状参数的单个随机数
random_st.logistic(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成指定位置和形状参数的随机数
random_st.lognormal(0.5, 0.5)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成指定位置和形状参数的随机数
random_st.lognormal(0.5, 0.5, size=None)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成指定位置和形状参数的单个随机数
random_st.lognormal(0.5, 0.5, size=1)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成一个参数为数组的随机数
random_st.lognormal(D_arr_0p5, 0.5)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成一个参数为数组的随机数
random_st.lognormal(0.5, D_arr_0p5)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成一个参数为数组的单个随机数
random_st.lognormal(D_arr_0p5, 0.5, size=1)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成一个参数为数组的单个随机数
random_st.lognormal(0.5, D_arr_0p5, size=1)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成一个参数为类似数组的随机数
random_st.lognormal(D_arr_like_0p5, 0.5)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成一个参数为类似数组的随机数
random_st.lognormal(0.5, D_arr_like_0p5)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成两个参数相同的随机数
random_st.lognormal(D_arr_0p5, D_arr_0p5)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成两个参数相同的随机数
random_st.lognormal(D_arr_like_0p5, D_arr_like_0p5)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成两个参数相同的单个随机数
random_st.lognormal(D_arr_0p5, D_arr_0p5, size=1)

# 从 scipy.stats 模块中调用 lognormal 分布函数，生成两个参数相同的单个随机数
random_st.lognormal(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成指定自由度和非中心参数的随机数
random_st.noncentral_chisquare(0.5, 0.5)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成指定自由度和非中心参数的随机数
random_st.noncentral_chisquare(0.5, 0.5, size=None)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成指定自由度和非中心参数的单个随机数
random_st.noncentral_chisquare(0.5, 0.5, size=1)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成一个参数为数组的随机数
random_st.noncentral_chisquare(D_arr_0p5, 0.5)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成一个参数为数组的随机数
random_st.noncentral_chisquare(0.5, D_arr_0p5)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成一个参数为数组的单个随机数
random_st.noncentral_chisquare(D_arr_0p5, 0.5, size=1)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成一个参数为数组的单个随机数
random_st.noncentral_chisquare(0.5, D_arr_0p5, size=1)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成一个参数为类似数组的随机数
random_st.noncentral_chisquare(D_arr_like_0p5, 0.5)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成一个参数为类似数组的随机数
random_st.noncentral_chisquare(0.5, D_arr_like_0p5)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成两个参数相同的随机数
random_st.noncentral_chisquare(D_arr_0p5, D_arr_0p5)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成两个参数相同的随机数
random_st.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成两个参数相同的单个随机数
random_st.noncentral_chisquare(D_arr_0p5, D_arr_0p5, size=1)

# 从 scipy.stats 模块中调用 noncentral_chisquare 分布函数，生成两个参数相同的单个随机数
random_st.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5, size=1)

# 从 scipy.stats 模块中调用 normal 分布函数，生成指定位置和形状参数的随机数
random_st.normal(0.5, 0.5)

# 从 scipy.stats 模块中调用 normal 分布函数，生成指定位置和形状参数的随机数
random_st.normal(0.5, 0.5, size=None)

# 从 scipy.stats 模块中调用 normal 分布函数，生成指定位置和形状参数的单个随机数
random_st.normal(0.5, 0.5, size=1)

# 从 scipy.stats 模块中调用 normal 分布函数，生成一个参数为数组的随机数
random_st.normal(D_arr_0p5, 0.5)

# 从 scipy.stats 模块中调用 normal 分布函数，生成一个参数为数组的随机数
random_st.normal(0.5, D_arr_0p5)

# 从 scipy.stats 模块中调用 normal 分布函数，生成一个参数为数组的单个随机数
random_st.normal(D_arr_0p5, 0.5, size=1)

# 从 scipy.stats 模块中调用 normal 分布函数，生成一个参数为数组的单个随机数
random_st.normal(0.5, D_arr_0p5, size=1)

# 从 scipy.stats 模块中调用 normal 分布函数，生成一个参数为类似数组的随机数
random_st.normal(D_arr_like_0
# 使用 Scipy 中的 random_st 对象生成非中心 F 分布的随机变量，参数分别为 D_arr_0p1, 0.5, D_arr_like_0p9，生成一个随机数
random_st.noncentral_f(D_arr_0p1, 0.5, D_arr_like_0p9, size=1)

# 使用 Scipy 中的 random_st 对象生成非中心 F 分布的随机变量，参数分别为 0.1, D_arr_0p5, 0.9，生成一个随机数
random_st.noncentral_f(0.1, D_arr_0p5, 0.9, size=1)

# 使用 Scipy 中的 random_st 对象生成非中心 F 分布的随机变量，参数分别为 D_arr_like_0p1, 0.5, D_arr_0p9
random_st.noncentral_f(D_arr_like_0p1, 0.5, D_arr_0p9)

# 使用 Scipy 中的 random_st 对象生成非中心 F 分布的随机变量，参数分别为 0.5, D_arr_like_0p5, 0.9
random_st.noncentral_f(0.5, D_arr_like_0p5, 0.9)

# 使用 Scipy 中的 random_st 对象生成非中心 F 分布的随机变量，参数分别为 D_arr_0p1, D_arr_0p5, 0.9
random_st.noncentral_f(D_arr_0p1, D_arr_0p5, 0.9)

# 使用 Scipy 中的 random_st 对象生成非中心 F 分布的随机变量，参数分别为 D_arr_like_0p1, D_arr_like_0p5, 0.9
random_st.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, 0.9)

# 使用 Scipy 中的 random_st 对象生成非中心 F 分布的随机变量，参数分别为 D_arr_0p1, D_arr_0p5, D_arr_0p9，生成一个随机数
random_st.noncentral_f(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1)

# 使用 Scipy 中的 random_st 对象生成非中心 F 分布的随机变量，参数分别为 D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9，生成一个随机数
random_st.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 10, 0.5，返回一个随机数
random_st.binomial(10, 0.5)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 10, 0.5，返回一个指定大小的数组
random_st.binomial(10, 0.5, size=None)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 10, 0.5，返回一个大小为 1 的数组
random_st.binomial(10, 0.5, size=1)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 I_arr_10, 0.5
random_st.binomial(I_arr_10, 0.5)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 10, D_arr_0p5
random_st.binomial(10, D_arr_0p5)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 I_arr_10, 0.5，返回一个大小为 1 的数组
random_st.binomial(I_arr_10, 0.5, size=1)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 10, D_arr_0p5，返回一个大小为 1 的数组
random_st.binomial(10, D_arr_0p5, size=1)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 I_arr_like_10, 0.5
random_st.binomial(I_arr_like_10, 0.5)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 10, D_arr_like_0p5
random_st.binomial(10, D_arr_like_0p5)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 I_arr_10, D_arr_0p5
random_st.binomial(I_arr_10, D_arr_0p5)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 I_arr_like_10, D_arr_like_0p5
random_st.binomial(I_arr_like_10, D_arr_like_0p5)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 I_arr_10, D_arr_0p5，返回一个大小为 1 的数组
random_st.binomial(I_arr_10, D_arr_0p5, size=1)

# 使用 Scipy 中的 random_st 对象生成二项分布的随机变量，参数分别为 I_arr_like_10, D_arr_like_0p5，返回一个大小为 1 的数组
random_st.binomial(I_arr_like_10, D_arr_like_0p5, size=1)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 10, 0.5
random_st.negative_binomial(10, 0.5)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 10, 0.5，返回一个指定大小的数组
random_st.negative_binomial(10, 0.5, size=None)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 10, 0.5，返回一个大小为 1 的数组
random_st.negative_binomial(10, 0.5, size=1)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 I_arr_10, 0.5
random_st.negative_binomial(I_arr_10, 0.5)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 10, D_arr_0p5
random_st.negative_binomial(10, D_arr_0p5)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 I_arr_10, 0.5，返回一个大小为 1 的数组
random_st.negative_binomial(I_arr_10, 0.5, size=1)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 10, D_arr_0p5，返回一个大小为 1 的数组
random_st.negative_binomial(10, D_arr_0p5, size=1)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 I_arr_like_10, 0.5
random_st.negative_binomial(I_arr_like_10, 0.5)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 10, D_arr_like_0p5
random_st.negative_binomial(10, D_arr_like_0p5)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 I_arr_10, D_arr_0p5
random_st.negative_binomial(I_arr_10, D_arr_0p5)

# 使用 Scipy 中的 random_st 对象生成负二项分布的随机变量，参数分别为 I_arr_like_10, D_arr_like_0p5
random_st.negative_binomial(I_arr_like_10, D_arr_like_0p5)

# 使用 Scipy 中的 random_st 对象生成负二项分布的
# 生成一个随机整数，范围是[I_u1_high_open, inf)，数据类型是'u1'
random_st.randint(I_u1_high_open, dtype="u1")

# 生成一个随机整数，范围是[I_u1_low, I_u1_high_open)，数据类型是'u1'
random_st.randint(I_u1_low, I_u1_high_open, dtype="u1")

# 生成一个随机整数，范围是[0, I_u1_high_open)，数据类型是'u1'
random_st.randint(0, I_u1_high_open, dtype="u1")

# 生成一个随机整数，范围是[0, 256)，数据类型是'uint8'
random_st.randint(256, dtype="uint8")

# 生成一个随机整数，范围是[0, 256)，数据类型是'uint8'
random_st.randint(0, 256, dtype="uint8")

# 生成一个随机整数，范围是[I_u1_high_open, inf)，数据类型是'uint8'
random_st.randint(I_u1_high_open, dtype="uint8")

# 生成一个随机整数，范围是[I_u1_low, I_u1_high_open)，数据类型是'uint8'
random_st.randint(I_u1_low, I_u1_high_open, dtype="uint8")

# 生成一个随机整数，范围是[0, I_u1_high_open)，数据类型是'uint8'
random_st.randint(0, I_u1_high_open, dtype="uint8")

# 生成一个随机整数，范围是[0, 256)，数据类型是'uint8'
random_st.randint(256, dtype=np.uint8)

# 生成一个随机整数，范围是[0, 256)，数据类型是'uint8'
random_st.randint(0, 256, dtype=np.uint8)

# 生成一个随机整数，范围是[I_u1_high_open, inf)，数据类型是'uint8'
random_st.randint(I_u1_high_open, dtype=np.uint8)

# 生成一个随机整数，范围是[I_u1_low, I_u1_high_open)，数据类型是'uint8'
random_st.randint(I_u1_low, I_u1_high_open, dtype=np.uint8)

# 生成一个随机整数，范围是[0, I_u1_high_open)，数据类型是'uint8'
random_st.randint(0, I_u1_high_open, dtype=np.uint8)

# 生成一个随机整数，范围是[0, 65536)，数据类型是'u2'
random_st.randint(65536, dtype="u2")

# 生成一个随机整数，范围是[0, 65536)，数据类型是'u2'
random_st.randint(0, 65536, dtype="u2")

# 生成一个随机整数，范围是[I_u2_high_open, inf)，数据类型是'u2'
random_st.randint(I_u2_high_open, dtype="u2")

# 生成一个随机整数，范围是[I_u2_low, I_u2_high_open)，数据类型是'u2'
random_st.randint(I_u2_low, I_u2_high_open, dtype="u2")

# 生成一个随机整数，范围是[0, I_u2_high_open)，数据类型是'u2'
random_st.randint(0, I_u2_high_open, dtype="u2")

# 生成一个随机整数，范围是[0, 65536)，数据类型是'uint16'
random_st.randint(65536, dtype="uint16")

# 生成一个随机整数，范围是[0, 65536)，数据类型是'uint16'
random_st.randint(0, 65536, dtype="uint16")

# 生成一个随机整数，范围是[I_u2_high_open, inf)，数据类型是'uint16'
random_st.randint(I_u2_high_open, dtype="uint16")

# 生成一个随机整数，范围是[I_u2_low, I_u2_high_open)，数据类型是'uint16'
random_st.randint(I_u2_low, I_u2_high_open, dtype="uint16")

# 生成一个随机整数，范围是[0, I_u2_high_open)，数据类型是'uint16'
random_st.randint(0, I_u2_high_open, dtype="uint16")

# 生成一个随机整数，范围是[0, 65536)，数据类型是'uint16'
random_st.randint(65536, dtype=np.uint16)

# 生成一个随机整数，范围是[0, 65536)，数据类型是'uint16'
random_st.randint(0, 65536, dtype=np.uint16)

# 生成一个随机整数，范围是[I_u2_high_open, inf)，数据类型是'uint16'
random_st.randint(I_u2_high_open, dtype=np.uint16)

# 生成一个随机整数，范围是[I_u2_low, I_u2_high_open)，数据类型是'uint16'
random_st.randint(I_u2_low, I_u2_high_open, dtype=np.uint16)

# 生成一个随机整数，范围是[0, I_u2_high_open)，数据类型是'uint16'
random_st.randint(0, I_u2_high_open, dtype=np.uint16)

# 生成一个随机整数，范围是[0, 4294967296)，数据类型是'u4'
random_st.randint(4294967296, dtype="u4")

# 生成一个随机整数，范围是[0, 4294967296)，数据类型是'u4'
random_st.randint(0, 4294967296, dtype="u4")

# 生成一个随机整数，范围是[I_u4_high_open, inf)，数据类型是'u4'
random_st.randint(I_u4_high_open, dtype="u4")

# 生成一个随机整数，范围是[I_u4_low, I_u4_high_open)，数据类型是'u4'
random_st.randint(I_u4_low, I_u4_high_open, dtype="u4")

# 生成一个随机整数，范围是[0, I_u4_high_open)，数据类型是'u4'
random_st.randint(0, I_u4_high_open, dtype="u4")

# 生成一个随机整数，范围是[0, 4294967296)，数据类型是'uint32'
random_st.randint(4294967296, dtype="uint32")

# 生成一个随机整数，范围是[0, 4294967296)，数据类型是'uint32'
random_st.randint(0, 4294967296, dtype="uint32")

# 生成一个随机整数，范围是[I_u4_high_open, inf)，数据类型是'uint32'
random_st.randint(I_u4_high_open, dtype="uint32")

# 生成一个随机整数，范围是[I_u4_low, I_u4_high_open)，数据类型是'uint32'
random_st.randint(I_u4_low, I_u4_high_open, dtype="uint32")

# 生成一个随机整数，范围是[0, I_u4_high_open)，数据类型是'uint32'
random_st.randint(0, I_u4_high_open, dtype="uint32")

# 生成一个随机整数，范围是[0, 4294967296)，数据类型是'uint32'
random_st.randint(4294967296, dtype=np.uint32)

# 生成一个随机整数，范围是[0, 4294967296)，数据类型是'uint32'
random_st.randint(0, 4294967296, dtype=np.uint32)

# 生成一个随机整数，范围是[I_u4_high_open, inf)，数据类型是'uint32'
random_st.randint(I_u4_high_open, dtype=np.uint32)

# 生成一个随机整数，范围是[I_u4_low, I_u4_high_open)，数据类型是'uint32'
random_st.randint(I_u4_low, I_u4_high_open, dtype=np.uint32)

# 生成一个随机整数，范围是[0, I_u4_high_open)，数据类型是'uint32'
random_st.randint(0, I_u4_high_open, dtype=np.uint32)

# 生成一个随机整数，范围是[0, 18446744073709551616)，数据类型是'u8'
random_st.randint(18446744073709551616, dtype="u8")

# 生成一个随机整数，范围是
# 生成一个随机的8位有符号整数（int8）在范围[-128, 127]之间
random_st.randint(-128, 128, dtype="i1")

# 生成一个随机的8位有符号整数（int8），范围从0到I_i1_high_open-1
random_st.randint(I_i1_high_open, dtype="i1")

# 生成一个随机的8位有符号整数（int8），范围从I_i1_low到I_i1_high_open-1
random_st.randint(I_i1_low, I_i1_high_open, dtype="i1")

# 生成一个随机的8位有符号整数（int8），范围从-128到I_i1_high_open-1
random_st.randint(-128, I_i1_high_open, dtype="i1")

# 生成一个随机的8位有符号整数（int8），范围从0到127
random_st.randint(128, dtype="int8")

# 生成一个随机的8位有符号整数（int8），范围从-128到127
random_st.randint(-128, 128, dtype="int8")

# 生成一个随机的8位有符号整数（int8），范围从0到I_i1_high_open-1
random_st.randint(I_i1_high_open, dtype="int8")

# 生成一个随机的8位有符号整数（int8），范围从I_i1_low到I_i1_high_open-1
random_st.randint(I_i1_low, I_i1_high_open, dtype="int8")

# 生成一个随机的8位有符号整数（int8），范围从-128到I_i1_high_open-1
random_st.randint(-128, I_i1_high_open, dtype="int8")

# 生成一个随机的16位有符号整数（int16），范围从-32768到32767
random_st.randint(32768, dtype="i2")

# 生成一个随机的16位有符号整数（int16），范围从-32768到32767
random_st.randint(-32768, 32768, dtype="i2")

# 生成一个随机的16位有符号整数（int16），范围从0到I_i2_high_open-1
random_st.randint(I_i2_high_open, dtype="i2")

# 生成一个随机的16位有符号整数（int16），范围从I_i2_low到I_i2_high_open-1
random_st.randint(I_i2_low, I_i2_high_open, dtype="i2")

# 生成一个随机的16位有符号整数（int16），范围从-32768到I_i2_high_open-1
random_st.randint(-32768, I_i2_high_open, dtype="i2")

# 生成一个随机的16位有符号整数（int16），范围从0到32767
random_st.randint(32768, dtype="int16")

# 生成一个随机的16位有符号整数（int16），范围从-32768到32767
random_st.randint(-32768, 32768, dtype="int16")

# 生成一个随机的16位有符号整数（int16），范围从0到I_i2_high_open-1
random_st.randint(I_i2_high_open, dtype="int16")

# 生成一个随机的16位有符号整数（int16），范围从I_i2_low到I_i2_high_open-1
random_st.randint(I_i2_low, I_i2_high_open, dtype="int16")

# 生成一个随机的16位有符号整数（int16），范围从-32768到I_i2_high_open-1
random_st.randint(-32768, I_i2_high_open, dtype="int16")

# 生成一个随机的16位有符号整数（int16），范围从0到32767
random_st.randint(32768, dtype=np.int16)

# 生成一个随机的16位有符号整数（int16），范围从-32768到32767
random_st.randint(-32768, 32768, dtype=np.int16)

# 生成一个随机的16位有符号整数（int16），范围从0到I_i2_high_open-1
random_st.randint(I_i2_high_open, dtype=np.int16)

# 生成一个随机的16位有符号整数（int16），范围从I_i2_low到I_i2_high_open-1
random_st.randint(I_i2_low, I_i2_high_open, dtype=np.int16)

# 生成一个随机的16位有符号整数（int16），范围从-32768到I_i2_high_open-1
random_st.randint(-32768, I_i2_high_open, dtype=np.int16)

# 生成一个随机的32位有符号整数（int32），范围从-2147483648到2147483647
random_st.randint(2147483648, dtype="i4")

# 生成一个随机的32位有符号整数（int32），范围从-2147483648到2147483647
random_st.randint(-2147483648, 2147483648, dtype="i4")

# 生成一个随机的32位有符号整数（int32），范围从0到I_i4_high_open-1
random_st.randint(I_i4_high_open, dtype="i4")

# 生成一个随机的32位有符号整数（int32），范围从I_i4_low到I_i4_high_open-1
random_st.randint(I_i4_low, I_i4_high_open, dtype="i4")

# 生成一个随机的32位有符号整数（int32），范围从-2147483648到I_i4_high_open-1
random_st.randint(-2147483648, I_i4_high_open, dtype="i4")

# 生成一个随机的32位有符号整数（int32），范围从0到2147483647
random_st.randint(2147483648, dtype="int32")

# 生成一个随机的32位有符号整数（int32），范围从-2147483648到2147483647
random_st.randint(-2147483648, 2147483648, dtype="int32")

# 生成一个随机的32位有符号整数（int32），范围从0到I_i4_high_open-1
random_st.randint(I_i4_high_open, dtype="int32")

# 生成一个随机的32位有符号整数（int32），范围从I_i4_low到I_i4_high_open-1
random_st.randint(I_i4_low, I_i4_high_open, dtype="int32")

# 生成一个随机的32位有符号整数（int32），范围从-2147483648到I_i4_high_open-1
random_st.randint(-2147483648, I_i4_high_open, dtype="int32")

# 生成一个随机的32位有符号整数（int32），范围从0到2147483647
random_st.randint(2147483648, dtype=np.int32)

# 生成一个随机的32位有符号整数（int32），范围从-2147483648到2147483647
random_st.randint(-2147483648, 2147483648, dtype=np.int32)

# 生成一个随机的32位有符号整数（int32），范围从0到I_i4_high_open-1
random_st.randint(I_i4_high_open, dtype=np.int32)

# 生成一个随机的32位有符号整数（int32），范围从I_i4_low到I_i4_high_open-1
random_st.randint(I_i4_low, I_i4_high_open, dtype=np.int32)

# 生成一个随机的32位有符号整数（int32），范围从-2147483648到I_i4_high_open-1
random_st.randint(-2147483648, I_i4_high_open, dtype=np.int32)

# 生成一个随机的64位有符号整数（int64），范围从-9223372036854775808到9223372036854775807
random_st.randint(9223372036854775808, dtype="i8")

# 生成一个随机的64位有符号整数
# 生成一个 64 位整数，范围在 I_i8_high_open 之间
random_st.randint(I_i8_high_open, dtype=np.int64)

# 生成一个 64 位整数，范围在 I_i8_low 到 I_i8_high_open 之间
random_st.randint(I_i8_low, I_i8_high_open, dtype=np.int64)

# 生成一个 64 位整数，范围在 -9223372036854775808 到 I_i8_high_open 之间
random_st.randint(-9223372036854775808, I_i8_high_open, dtype=np.int64)

# 设置背景为 np.random.BitGenerator 类型的变量，值为 random_st._bit_generator
bg: np.random.BitGenerator = random_st._bit_generator

# 生成一个包含 2 个字节的随机字节串
random_st.bytes(2)

# 从 0 到 4 中随机选择一个整数
random_st.choice(5)

# 从 0 到 4 中随机选择 3 个整数，可以重复选择
random_st.choice(5, 3)

# 从 0 到 4 中随机选择 3 个整数，允许重复选择，按给定的概率分布 p 选择
random_st.choice(5, 3, replace=True, p=[1 / 5] * 5)

# 从 0 到 4 中随机选择 3 个整数，不允许重复选择，按给定的概率分布 p 选择
random_st.choice(5, 3, replace=False, p=[1 / 5] * 5)

# 从列表 ["pooh", "rabbit", "piglet", "Christopher"] 中随机选择一个元素
random_st.choice(["pooh", "rabbit", "piglet", "Christopher"])

# 从列表 ["pooh", "rabbit", "piglet", "Christopher"] 中随机选择 3 个元素
random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3)

# 从列表 ["pooh", "rabbit", "piglet", "Christopher"] 中随机选择 3 个元素，按给定的概率分布 p 选择
random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, p=[1 / 4] * 4)

# 从列表 ["pooh", "rabbit", "piglet", "Christopher"] 中随机选择 3 个元素，允许重复选择
random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=True)

# 从列表 ["pooh", "rabbit", "piglet", "Christopher"] 中随机选择 3 个元素，不允许重复选择，按给定的概率分布 p 选择
random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=False, p=np.array([1 / 8, 1 / 8, 1 / 2, 1 / 4]))

# 生成一个二维 Dirichlet 分布样本，参数为 [0.5, 0.5]
random_st.dirichlet([0.5, 0.5])

# 生成一个二维 Dirichlet 分布样本，参数为 np.array([0.5, 0.5])
random_st.dirichlet(np.array([0.5, 0.5]))

# 生成三个二维 Dirichlet 分布样本，参数为 np.array([0.5, 0.5])
random_st.dirichlet(np.array([0.5, 0.5]), size=3)

# 生成一个多项式分布样本，总样本数为 20，每个类别的概率为 [1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0]
random_st.multinomial(20, [1 / 6.0] * 6)

# 生成一个多项式分布样本，总样本数为 20，每个类别的概率为 np.array([0.5, 0.5])
random_st.multinomial(20, np.array([0.5, 0.5]))

# 生成两个多项式分布样本，每个样本总数为 20，每个类别的概率为 [1/6.0] * 6
random_st.multinomial(20, [1 / 6.0] * 6, size=2)

# 生成一个多变量正态分布样本，均值为 [0.0]，协方差矩阵为 [[1.0]]
random_st.multivariate_normal([0.0], [[1.0]])

# 生成一个多变量正态分布样本，均值为 [0.0]，协方差矩阵为 np.array([[1.0]])
random_st.multivariate_normal([0.0], np.array([[1.0]]))

# 生成一个多变量正态分布样本，均值为 np.array([0.0])，协方差矩阵为 [[1.0]]
random_st.multivariate_normal(np.array([0.0]), [[1.0]])

# 生成一个多变量正态分布样本，均值为 [0.0]，协方差矩阵为 np.array([[1.0]])
random_st.multivariate_normal([0.0], np.array([[1.0]]))

# 对整数序列进行随机排列
random_st.permutation(10)

# 对列表 [1, 2, 3, 4] 进行随机排列
random_st.permutation([1, 2, 3, 4])

# 对数组 np.array([1, 2, 3, 4]) 进行随机排列
random_st.permutation(np.array([1, 2, 3, 4]))

# 对二维数组 D_2D 进行随机排列
random_st.permutation(D_2D)

# 随机打乱数组 np.arange(10)
random_st.shuffle(np.arange(10))

# 随机打乱列表 [1, 2, 3, 4, 5]
random_st.shuffle([1, 2, 3, 4, 5])

# 随机打乱二维数组 D_2D
random_st.shuffle(D_2D)

# 创建一个新的 RandomState 实例，使用 SEED_PCG64 作为种子
np.random.RandomState(SEED_PCG64)

# 创建一个新的 RandomState 实例，使用 0 作为种子
np.random.RandomState(0)

# 创建一个新的 RandomState 实例，使用 [0, 1, 2] 作为种子
np.random.RandomState([0, 1, 2])

# 返回 random_st 的字符串表示形式
random_st.__str__()

# 返回 random_st 的详细表示形式
random_st.__repr__()

# 获取 random_st 的状态信息，并存储在 random_st_state 中
random_st_state = random_st.__getstate__()

# 恢复 random_st 的状态信息，使用之前存储的 random_st_state
random_st.__setstate__(random_st_state)

# 使用默认种子初始化随机数生成器
random_st.seed()

# 使用指定种子（1）初始化随机数生成器
random_st.seed(1)

# 使用指定种子（[0, 1]）初始化随机数生成器
random_st.seed([0, 1])

# 获取 random_st 的当前状态信息，并存储在 random_st_get_state 中
random_st_get_state = random_st.get_state()

# 获取 random_st 的当前状态信息（兼容模式），并存储在 random_st_get_state_legacy 中
random_st_get_state_legacy = random_st.get_state(legacy=True)

# 设置 random_st 的状态信息，使用之前存储的状态 random_st_get_state
random_st.set_state(random_st_get_state)

# 生成一个 [0, 1) 范围内的随机浮点数
random_st.rand()

# 生成一个包含一个 [0, 1) 范围内随机浮点数的数组
random_st.rand(1)

# 生成一个包含一个 [0, 1) 范围内随机浮点数的 1x2 数组
random_st.rand(1, 2)

# 生成一个标准正态分布的随机数
random_st.randn()

# 生成一个包含一个标准正态分布
```
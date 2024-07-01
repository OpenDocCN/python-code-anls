# `.\numpy\numpy\typing\tests\data\reveal\random.pyi`

```py
import sys
import threading
from typing import Any
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.random._generator import Generator
from numpy.random._mt19937 import MT19937
from numpy.random._pcg64 import PCG64
from numpy.random._sfc64 import SFC64
from numpy.random._philox import Philox
from numpy.random.bit_generator import SeedSequence, SeedlessSeedSequence

# 如果 Python 版本大于等于 3.11，则使用标准库中的 assert_type
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    # 否则，从 typing_extensions 导入 assert_type
    from typing_extensions import assert_type

# 创建默认的随机数生成器
def_rng = np.random.default_rng()
# 创建种子序列
seed_seq = np.random.SeedSequence()
# 创建 MT19937 随机数生成器
mt19937 = np.random.MT19937()
# 创建 PCG64 随机数生成器
pcg64 = np.random.PCG64()
# 创建 SFC64 随机数生成器
sfc64 = np.random.SFC64()
# 创建 Philox 随机数生成器
philox = np.random.Philox()
# 创建无种子的种子序列
seedless_seq = SeedlessSeedSequence()

# 使用 assert_type 函数确保对象类型正确
assert_type(def_rng, Generator)
assert_type(mt19937, MT19937)
assert_type(pcg64, PCG64)
assert_type(sfc64, SFC64)
assert_type(philox, Philox)
assert_type(seed_seq, SeedSequence)
assert_type(seedless_seq, SeedlessSeedSequence)

# 对 MT19937 随机数生成器进行不同操作，并检查返回类型
mt19937_jumped = mt19937.jumped()
mt19937_jumped3 = mt19937.jumped(3)
mt19937_raw = mt19937.random_raw()
mt19937_raw_arr = mt19937.random_raw(5)

assert_type(mt19937_jumped, MT19937)
assert_type(mt19937_jumped3, MT19937)
assert_type(mt19937_raw, int)
assert_type(mt19937_raw_arr, npt.NDArray[np.uint64])
assert_type(mt19937.lock, threading.Lock)

# 对 PCG64 随机数生成器进行不同操作，并检查返回类型
pcg64_jumped = pcg64.jumped()
pcg64_jumped3 = pcg64.jumped(3)
pcg64_adv = pcg64.advance(3)
pcg64_raw = pcg64.random_raw()
pcg64_raw_arr = pcg64.random_raw(5)

assert_type(pcg64_jumped, PCG64)
assert_type(pcg64_jumped3, PCG64)
assert_type(pcg64_adv, PCG64)
assert_type(pcg64_raw, int)
assert_type(pcg64_raw_arr, npt.NDArray[np.uint64])
assert_type(pcg64.lock, threading.Lock)

# 对 Philox 随机数生成器进行不同操作，并检查返回类型
philox_jumped = philox.jumped()
philox_jumped3 = philox.jumped(3)
philox_adv = philox.advance(3)
philox_raw = philox.random_raw()
philox_raw_arr = philox.random_raw(5)

assert_type(philox_jumped, Philox)
assert_type(philox_jumped3, Philox)
assert_type(philox_adv, Philox)
assert_type(philox_raw, int)
assert_type(philox_raw_arr, npt.NDArray[np.uint64])
assert_type(philox.lock, threading.Lock)

# 对 SFC64 随机数生成器进行不同操作，并检查返回类型
sfc64_raw = sfc64.random_raw()
sfc64_raw_arr = sfc64.random_raw(5)

assert_type(sfc64_raw, int)
assert_type(sfc64_raw_arr, npt.NDArray[np.uint64])
assert_type(sfc64.lock, threading.Lock)

# 对种子序列进行不同操作，并检查返回类型
assert_type(seed_seq.pool, npt.NDArray[np.uint32])
assert_type(seed_seq.entropy, None | int | Sequence[int])
assert_type(seed_seq.spawn(1), list[np.random.SeedSequence])
assert_type(seed_seq.generate_state(8, "uint32"), npt.NDArray[np.uint32 | np.uint64])
assert_type(seed_seq.generate_state(8, "uint64"), npt.NDArray[np.uint32 | np.uint64])

# 创建默认的随机数生成器，并指定其类型
def_gen: np.random.Generator = np.random.default_rng()

# 创建不同的 numpy 数组并指定其类型
D_arr_0p1: npt.NDArray[np.float64] = np.array([0.1])
D_arr_0p5: npt.NDArray[np.float64] = np.array([0.5])
D_arr_0p9: npt.NDArray[np.float64] = np.array([0.9])
D_arr_1p5: npt.NDArray[np.float64] = np.array([1.5])
I_arr_10: npt.NDArray[np.int_] = np.array([10], dtype=np.int_)
I_arr_20: npt.NDArray[np.int_] = np.array([20], dtype=np.int_)
D_arr_like_0p1: list[float] = [0.1]
D_arr_like_0p5: list[float] = [0.5]
D_arr_like_0p9: list[float] = [0.9]
D_arr_like_1p5: list[float] = [1.5]
I_arr_like_10: list[int] = [10]
I_arr_like_20: list[int] = [20]
D_2D_like: list[list[float]] = [[1, 2], [2, 3], [3, 4], [4, 5.1]]
D_2D: npt.NDArray[np.float64] = np.array(D_2D_like)
S_out: npt.NDArray[np.float32] = np.empty(1, dtype=np.float32)
D_out: npt.NDArray[np.float64] = np.empty(1)

assert_type(def_gen.standard_normal(), float)
# 检查 def_gen.standard_normal() 返回类型是否为 float

assert_type(def_gen.standard_normal(dtype=np.float32), float)
# 检查 def_gen.standard_normal(dtype=np.float32) 返回类型是否为 float

assert_type(def_gen.standard_normal(dtype="float32"), float)
# 检查 def_gen.standard_normal(dtype="float32") 返回类型是否为 float

assert_type(def_gen.standard_normal(dtype="double"), float)
# 检查 def_gen.standard_normal(dtype="double") 返回类型是否为 float

assert_type(def_gen.standard_normal(dtype=np.float64), float)
# 检查 def_gen.standard_normal(dtype=np.float64) 返回类型是否为 float

assert_type(def_gen.standard_normal(size=None), float)
# 检查 def_gen.standard_normal(size=None) 返回类型是否为 float

assert_type(def_gen.standard_normal(size=1), npt.NDArray[np.float64])
# 检查 def_gen.standard_normal(size=1) 返回类型是否为 numpy float64 数组

assert_type(def_gen.standard_normal(size=1, dtype=np.float32), npt.NDArray[np.float32])
# 检查 def_gen.standard_normal(size=1, dtype=np.float32) 返回类型是否为 numpy float32 数组

assert_type(def_gen.standard_normal(size=1, dtype="f4"), npt.NDArray[np.float32])
# 检查 def_gen.standard_normal(size=1, dtype="f4") 返回类型是否为 numpy float32 数组

assert_type(def_gen.standard_normal(size=1, dtype="float32", out=S_out), npt.NDArray[np.float32])
# 检查 def_gen.standard_normal(size=1, dtype="float32", out=S_out) 返回类型是否为 numpy float32 数组

assert_type(def_gen.standard_normal(dtype=np.float32, out=S_out), npt.NDArray[np.float32])
# 检查 def_gen.standard_normal(dtype=np.float32, out=S_out) 返回类型是否为 numpy float32 数组

assert_type(def_gen.standard_normal(size=1, dtype=np.float64), npt.NDArray[np.float64])
# 检查 def_gen.standard_normal(size=1, dtype=np.float64) 返回类型是否为 numpy float64 数组

assert_type(def_gen.standard_normal(size=1, dtype="float64"), npt.NDArray[np.float64])
# 检查 def_gen.standard_normal(size=1, dtype="float64") 返回类型是否为 numpy float64 数组

assert_type(def_gen.standard_normal(size=1, dtype="f8"), npt.NDArray[np.float64])
# 检查 def_gen.standard_normal(size=1, dtype="f8") 返回类型是否为 numpy float64 数组

assert_type(def_gen.standard_normal(out=D_out), npt.NDArray[np.float64])
# 检查 def_gen.standard_normal(out=D_out) 返回类型是否为 numpy float64 数组

assert_type(def_gen.standard_normal(size=1, dtype="float64"), npt.NDArray[np.float64])
# 检查 def_gen.standard_normal(size=1, dtype="float64") 返回类型是否为 numpy float64 数组

assert_type(def_gen.standard_normal(size=1, dtype="float64", out=D_out), npt.NDArray[np.float64])
# 检查 def_gen.standard_normal(size=1, dtype="float64", out=D_out) 返回类型是否为 numpy float64 数组

assert_type(def_gen.random(), float)
# 检查 def_gen.random() 返回类型是否为 float

assert_type(def_gen.random(dtype=np.float32), float)
# 检查 def_gen.random(dtype=np.float32) 返回类型是否为 float

assert_type(def_gen.random(dtype="float32"), float)
# 检查 def_gen.random(dtype="float32") 返回类型是否为 float

assert_type(def_gen.random(dtype="double"), float)
# 检查 def_gen.random(dtype="double") 返回类型是否为 float

assert_type(def_gen.random(dtype=np.float64), float)
# 检查 def_gen.random(dtype=np.float64) 返回类型是否为 float

assert_type(def_gen.random(size=None), float)
# 检查 def_gen.random(size=None) 返回类型是否为 float

assert_type(def_gen.random(size=1), npt.NDArray[np.float64])
# 检查 def_gen.random(size=1) 返回类型是否为 numpy float64 数组

assert_type(def_gen.random(size=1, dtype=np.float32), npt.NDArray[np.float32])
# 检查 def_gen.random(size=1, dtype=np.float32) 返回类型是否为 numpy float32 数组

assert_type(def_gen.random(size=1, dtype="f4"), npt.NDArray[np.float32])
# 检查 def_gen.random(size=1, dtype="f4") 返回类型是否为 numpy float32 数组

assert_type(def_gen.random(size=1, dtype="float32", out=S_out), npt.NDArray[np.float32])
# 检查 def_gen.random(size=1, dtype="float32", out=S_out) 返回类型是否为 numpy float32 数组

assert_type(def_gen.random(dtype=np.float32, out=S_out), npt.NDArray[np.float32])
# 检查 def_gen.random(dtype=np.float32, out=S_out) 返回类型是否为 numpy float32 数组

assert_type(def_gen.random(size=1, dtype=np.float64), npt.NDArray[np.float64])
# 检查 def_gen.random(size=1, dtype=np.float64) 返回类型是否为 numpy float64 数组

assert_type(def_gen.random(size=1, dtype="float64"), npt.NDArray[np.float64])
# 检查 def_gen.random(size=1, dtype="float64") 返回类型是否为 numpy float64 数组

assert_type(def_gen.random(size=1, dtype="f8"), npt.NDArray[np.float64])
# 检查 def_gen.random(size=1, dtype="f8") 返回类型是否为 numpy float64 数组

assert_type(def_gen.random(out=D_out), npt.NDArray[np.float64])
# 检查 def_gen.random(out=D_out) 返回类型是否为 numpy float64 数组

assert_type(def_gen.random(size=1, dtype="float64"), npt.NDArray[np.float64])
# 检查 def_gen.random(size=1, dtype="float64") 返回类型是否为 numpy float64 数组

assert_type(def_gen.random(size=1, dtype="float64", out=D_out), npt.NDArray[np.float64])
# 检查 def_gen.random(size=1, dtype="float64", out=D_out) 返回类型是否为 numpy float64 数组

assert_type(def_gen.standard_cauchy(), float)
# 检查 def_gen.standard_cauchy() 返回类型是否为 float
# 验证 def_gen.standard_cauchy() 函数返回值类型是否为 float
assert_type(def_gen.standard_cauchy(size=None), float)
# 验证 def_gen.standard_cauchy(size=1) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_cauchy(size=1), npt.NDArray[np.float64])

# 验证 def_gen.standard_exponential() 函数返回值类型是否为 float
assert_type(def_gen.standard_exponential(), float)
# 验证 def_gen.standard_exponential(method="inv") 函数返回值类型是否为 float
assert_type(def_gen.standard_exponential(method="inv"), float)
# 验证 def_gen.standard_exponential(dtype=np.float32) 函数返回值类型是否为 float
assert_type(def_gen.standard_exponential(dtype=np.float32), float)
# 验证 def_gen.standard_exponential(dtype="float32") 函数返回值类型是否为 float
assert_type(def_gen.standard_exponential(dtype="float32"), float)
# 验证 def_gen.standard_exponential(dtype="double") 函数返回值类型是否为 float
assert_type(def_gen.standard_exponential(dtype="double"), float)
# 验证 def_gen.standard_exponential(dtype=np.float64) 函数返回值类型是否为 float
assert_type(def_gen.standard_exponential(dtype=np.float64), float)
# 验证 def_gen.standard_exponential(size=None) 函数返回值类型是否为 float
assert_type(def_gen.standard_exponential(size=None), float)
# 验证 def_gen.standard_exponential(size=None, method="inv") 函数返回值类型是否为 float
assert_type(def_gen.standard_exponential(size=None, method="inv"), float)
# 验证 def_gen.standard_exponential(size=1, method="inv") 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_exponential(size=1, method="inv"), npt.NDArray[np.float64])
# 验证 def_gen.standard_exponential(size=1, dtype=np.float32) 函数返回值类型是否为 numpy 数组中的 float32 类型
assert_type(def_gen.standard_exponential(size=1, dtype=np.float32), npt.NDArray[np.float32])
# 验证 def_gen.standard_exponential(size=1, dtype="f4", method="inv") 函数返回值类型是否为 numpy 数组中的 float32 类型
assert_type(def_gen.standard_exponential(size=1, dtype="f4", method="inv"), npt.NDArray[np.float32])
# 验证 def_gen.standard_exponential(size=1, dtype="float32", out=S_out) 函数返回值类型是否为 numpy 数组中的 float32 类型
assert_type(def_gen.standard_exponential(size=1, dtype="float32", out=S_out), npt.NDArray[np.float32])
# 验证 def_gen.standard_exponential(dtype=np.float32, out=S_out) 函数返回值类型是否为 numpy 数组中的 float32 类型
assert_type(def_gen.standard_exponential(dtype=np.float32, out=S_out), npt.NDArray[np.float32])
# 验证 def_gen.standard_exponential(size=1, dtype=np.float64, method="inv") 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_exponential(size=1, dtype=np.float64, method="inv"), npt.NDArray[np.float64])
# 验证 def_gen.standard_exponential(size=1, dtype="float64") 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_exponential(size=1, dtype="float64"), npt.NDArray[np.float64])
# 验证 def_gen.standard_exponential(size=1, dtype="f8") 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_exponential(size=1, dtype="f8"), npt.NDArray[np.float64])
# 验证 def_gen.standard_exponential(out=D_out) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_exponential(out=D_out), npt.NDArray[np.float64])
# 验证 def_gen.standard_exponential(size=1, dtype="float64") 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_exponential(size=1, dtype="float64"), npt.NDArray[np.float64])
# 验证 def_gen.standard_exponential(size=1, dtype="float64", out=D_out) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_exponential(size=1, dtype="float64", out=D_out), npt.NDArray[np.float64])

# 验证 def_gen.zipf(1.5) 函数返回值类型是否为 int
assert_type(def_gen.zipf(1.5), int)
# 验证 def_gen.zipf(1.5, size=None) 函数返回值类型是否为 int
assert_type(def_gen.zipf(1.5, size=None), int)
# 验证 def_gen.zipf(1.5, size=1) 函数返回值类型是否为 numpy 数组中的 int64 类型
assert_type(def_gen.zipf(1.5, size=1), npt.NDArray[np.int64])
# 验证 def_gen.zipf(D_arr_1p5) 函数返回值类型是否为 numpy 数组中的 int64 类型
assert_type(def_gen.zipf(D_arr_1p5), npt.NDArray[np.int64])
# 验证 def_gen.zipf(D_arr_1p5, size=1) 函数返回值类型是否为 numpy 数组中的 int64 类型
assert_type(def_gen.zipf(D_arr_1p5, size=1), npt.NDArray[np.int64])
# 验证 def_gen.zipf(D_arr_like_1p5) 函数返回值类型是否为 numpy 数组中的 int64 类型
assert_type(def_gen.zipf(D_arr_like_1p5), npt.NDArray[np.int64])
# 验证 def_gen.zipf(D_arr_like_1p5, size=1) 函数返回值类型是否为 numpy 数组中的 int64 类型
assert_type(def_gen.zipf(D_arr_like_1p5, size=1), npt.NDArray[np.int64])

# 验证 def_gen.weibull(0.5) 函数返回值类型是否为 float
assert_type(def_gen.weibull(0.5), float)
# 验证 def_gen.weibull(0.5, size=None) 函数返回值类型是否为 float
assert_type(def_gen.weibull(0.5, size=None), float)
# 验证 def_gen.weibull(0.5, size=1) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.weibull(0.5, size=1), npt.NDArray[np.float64])
# 验证 def_gen.weibull(D_arr_0p5) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.weibull(D_arr_0p5), npt.NDArray[np.float64])
# 验证 def_gen.weibull(D_arr_0p5, size=1) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.weibull(D_arr_0p5, size=1), npt.NDArray[np.float64])
# 验证 def_gen.weibull(D_arr_like_0p5) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.weibull(D_arr_like_0p5), npt.NDArray[np.float64])
# 验证 def_gen.weibull(D_arr_like_0p5, size=1) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.weibull(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 验证 def_gen.standard_t(0.5) 函数返回值类型是否为 float
assert_type(def_gen.standard_t(0.5), float)
# 验证 def_gen.standard_t(0.5, size=None) 函数返回值类型是否为 float
assert_type(def_gen.standard_t(0.5, size=None), float)
# 验证 def_gen.standard_t(0.5, size=1) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_t(0.5, size=1), npt.NDArray[np.float64])
# 验证 def_gen.standard_t(D_arr_0p5) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_t(D_arr_0p5), npt.NDArray[np.float64])
# 验证 def_gen.standard_t(D_arr_0p5, size=1) 函数返回值类型是否为 numpy 数组中的 float64 类型
assert_type(def_gen.standard_t(D_arr_0p5, size=1),
# 确保 def_gen.standard_t 返回的结果是 npt.NDArray[np.float64] 类型
assert_type(def_gen.standard_t(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 确保 def_gen.poisson 函数返回的结果是 int 类型
assert_type(def_gen.poisson(0.5), int)
# 确保 def_gen.poisson 函数返回的结果是 int 类型，并且 size 参数为 None
assert_type(def_gen.poisson(0.5, size=None), int)
# 确保 def_gen.poisson 函数返回的结果是 npt.NDArray[np.int64] 类型，并且 size 参数为 1
assert_type(def_gen.poisson(0.5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.poisson 函数返回的结果是 npt.NDArray[np.int64] 类型，参数为 D_arr_0p5
assert_type(def_gen.poisson(D_arr_0p5), npt.NDArray[np.int64])
# 确保 def_gen.poisson 函数返回的结果是 npt.NDArray[np.int64] 类型，参数为 D_arr_0p5，并且 size 参数为 1
assert_type(def_gen.poisson(D_arr_0p5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.poisson 函数返回的结果是 npt.NDArray[np.int64] 类型，参数为 D_arr_like_0p5
assert_type(def_gen.poisson(D_arr_like_0p5), npt.NDArray[np.int64])
# 确保 def_gen.poisson 函数返回的结果是 npt.NDArray[np.int64] 类型，参数为 D_arr_like_0p5，并且 size 参数为 1
assert_type(def_gen.poisson(D_arr_like_0p5, size=1), npt.NDArray[np.int64])

# 确保 def_gen.power 函数返回的结果是 float 类型
assert_type(def_gen.power(0.5), float)
# 确保 def_gen.power 函数返回的结果是 float 类型，并且 size 参数为 None
assert_type(def_gen.power(0.5, size=None), float)
# 确保 def_gen.power 函数返回的结果是 npt.NDArray[np.float64] 类型，并且 size 参数为 1
assert_type(def_gen.power(0.5, size=1), npt.NDArray[np.float64])
# 确保 def_gen.power 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_0p5
assert_type(def_gen.power(D_arr_0p5), npt.NDArray[np.float64])
# 确保 def_gen.power 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_0p5，并且 size 参数为 1
assert_type(def_gen.power(D_arr_0p5, size=1), npt.NDArray[np.float64])
# 确保 def_gen.power 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_like_0p5
assert_type(def_gen.power(D_arr_like_0p5), npt.NDArray[np.float64])
# 确保 def_gen.power 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_like_0p5，并且 size 参数为 1
assert_type(def_gen.power(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 确保 def_gen.pareto 函数返回的结果是 float 类型
assert_type(def_gen.pareto(0.5), float)
# 确保 def_gen.pareto 函数返回的结果是 float 类型，并且 size 参数为 None
assert_type(def_gen.pareto(0.5, size=None), float)
# 确保 def_gen.pareto 函数返回的结果是 npt.NDArray[np.float64] 类型，并且 size 参数为 1
assert_type(def_gen.pareto(0.5, size=1), npt.NDArray[np.float64])
# 确保 def_gen.pareto 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_0p5
assert_type(def_gen.pareto(D_arr_0p5), npt.NDArray[np.float64])
# 确保 def_gen.pareto 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_0p5，并且 size 参数为 1
assert_type(def_gen.pareto(D_arr_0p5, size=1), npt.NDArray[np.float64])
# 确保 def_gen.pareto 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_like_0p5
assert_type(def_gen.pareto(D_arr_like_0p5), npt.NDArray[np.float64])
# 确保 def_gen.pareto 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_like_0p5，并且 size 参数为 1
assert_type(def_gen.pareto(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 确保 def_gen.chisquare 函数返回的结果是 float 类型
assert_type(def_gen.chisquare(0.5), float)
# 确保 def_gen.chisquare 函数返回的结果是 float 类型，并且 size 参数为 None
assert_type(def_gen.chisquare(0.5, size=None), float)
# 确保 def_gen.chisquare 函数返回的结果是 npt.NDArray[np.float64] 类型，并且 size 参数为 1
assert_type(def_gen.chisquare(0.5, size=1), npt.NDArray[np.float64])
# 确保 def_gen.chisquare 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_0p5
assert_type(def_gen.chisquare(D_arr_0p5), npt.NDArray[np.float64])
# 确保 def_gen.chisquare 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_0p5，并且 size 参数为 1
assert_type(def_gen.chisquare(D_arr_0p5, size=1), npt.NDArray[np.float64])
# 确保 def_gen.chisquare 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_like_0p5
assert_type(def_gen.chisquare(D_arr_like_0p5), npt.NDArray[np.float64])
# 确保 def_gen.chisquare 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_like_0p5，并且 size 参数为 1
assert_type(def_gen.chisquare(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 确保 def_gen.exponential 函数返回的结果是 float 类型
assert_type(def_gen.exponential(0.5), float)
# 确保 def_gen.exponential 函数返回的结果是 float 类型，并且 size 参数为 None
assert_type(def_gen.exponential(0.5, size=None), float)
# 确保 def_gen.exponential 函数返回的结果是 npt.NDArray[np.float64] 类型，并且 size 参数为 1
assert_type(def_gen.exponential(0.5, size=1), npt.NDArray[np.float64])
# 确保 def_gen.exponential 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_0p5
assert_type(def_gen.exponential(D_arr_0p5), npt.NDArray[np.float64])
# 确保 def_gen.exponential 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_0p5，并且 size 参数为 1
assert_type(def_gen.exponential(D_arr_0p5, size=1), npt.NDArray[np.float64])
# 确保 def_gen.exponential 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_like_0p5
assert_type(def_gen.exponential(D_arr_like_0p5), npt.NDArray[np.float64])
# 确保 def_gen.exponential 函数返回的结果是 npt.NDArray[np.float64] 类型，参数为 D_arr_like_0p5，并且 size 参数为 1
assert_type(def_gen.exponential(D_arr_like_0p5, size=1), npt.NDArray[np.float64
# 调用自定义生成器模块中的logseries函数，返回结果类型为np.int64的NumPy数组
assert_type(def_gen.logseries(D_arr_0p5, size=1), npt.NDArray[np.int64])

# 调用自定义生成器模块中的logseries函数，返回结果类型为np.int64的NumPy数组
assert_type(def_gen.logseries(D_arr_like_0p5), npt.NDArray[np.int64])

# 调用自定义生成器模块中的logseries函数，返回结果类型为np.int64的NumPy数组
assert_type(def_gen.logseries(D_arr_like_0p5, size=1), npt.NDArray[np.int64])

# 调用自定义生成器模块中的rayleigh函数，返回结果类型为float
assert_type(def_gen.rayleigh(0.5), float)

# 调用自定义生成器模块中的rayleigh函数，返回结果类型为float
assert_type(def_gen.rayleigh(0.5, size=None), float)

# 调用自定义生成器模块中的rayleigh函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.rayleigh(0.5, size=1), npt.NDArray[np.float64])

# 调用自定义生成器模块中的rayleigh函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.rayleigh(D_arr_0p5), npt.NDArray[np.float64])

# 调用自定义生成器模块中的rayleigh函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.rayleigh(D_arr_0p5, size=1), npt.NDArray[np.float64])

# 调用自定义生成器模块中的rayleigh函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.rayleigh(D_arr_like_0p5), npt.NDArray[np.float64])

# 调用自定义生成器模块中的rayleigh函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.rayleigh(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为float
assert_type(def_gen.standard_gamma(0.5), float)

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为float
assert_type(def_gen.standard_gamma(0.5, size=None), float)

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为float32
assert_type(def_gen.standard_gamma(0.5, dtype="float32"), float)

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为float32
assert_type(def_gen.standard_gamma(0.5, size=None, dtype="float32"), float)

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.standard_gamma(0.5, size=1), npt.NDArray[np.float64])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.standard_gamma(D_arr_0p5), npt.NDArray[np.float64])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float32的NumPy数组
assert_type(def_gen.standard_gamma(D_arr_0p5, dtype="f4"), npt.NDArray[np.float32])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float32的NumPy数组
assert_type(def_gen.standard_gamma(0.5, size=1, dtype="float32", out=S_out), npt.NDArray[np.float32])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float32的NumPy数组
assert_type(def_gen.standard_gamma(D_arr_0p5, dtype=np.float32, out=S_out), npt.NDArray[np.float32])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.standard_gamma(D_arr_0p5, size=1), npt.NDArray[np.float64])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.standard_gamma(D_arr_like_0p5), npt.NDArray[np.float64])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.standard_gamma(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.standard_gamma(0.5, out=D_out), npt.NDArray[np.float64])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.standard_gamma(D_arr_like_0p5, out=D_out), npt.NDArray[np.float64])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.standard_gamma(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 调用自定义生成器模块中的standard_gamma函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.standard_gamma(D_arr_like_0p5, size=1, out=D_out, dtype=np.float64), npt.NDArray[np.float64])

# 调用自定义生成器模块中的vonmises函数，返回结果类型为float
assert_type(def_gen.vonmises(0.5, 0.5), float)

# 调用自定义生成器模块中的vonmises函数，返回结果类型为float
assert_type(def_gen.vonmises(0.5, 0.5, size=None), float)

# 调用自定义生成器模块中的vonmises函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.vonmises(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 调用自定义生成器模块中的vonmises函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.vonmises(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 调用自定义生成器模块中的vonmises函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.vonmises(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 调用自定义生成器模块中的vonmises函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.vonmises(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 调用自定义生成器模块中的vonmises函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.vonmises(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 调用自定义生成器模块中的vonmises函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.vonmises(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 调用自定义生成器模块中的vonmises函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.vonmises(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 调用自定义生成器模块中的vonmises函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.vonmises(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 调用自定义生成器模块中的vonmises函数，返回结果类型为np.float64的NumPy数组
assert_type(def_gen.vonmises(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 调用自定义生成器模块中的vonmises函数，
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 float
assert_type(def_gen.wald(0.5, 0.5, size=None), float)
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组
assert_type(def_gen.wald(0.5, 0.5, size=1), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5
assert_type(def_gen.wald(D_arr_0p5, 0.5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5
assert_type(def_gen.wald(0.5, D_arr_0p5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5，且样本大小为 1
assert_type(def_gen.wald(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5，且样本大小为 1
assert_type(def_gen.wald(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组，其中一个参数为形状类似于 D_arr_0p5 的数组
assert_type(def_gen.wald(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组，其中一个参数为形状类似于 D_arr_0p5 的数组
assert_type(def_gen.wald(0.5, D_arr_like_0p5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组，两个参数均为数组 D_arr_0p5
assert_type(def_gen.wald(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组，两个参数均为形状类似于 D_arr_0p5 的数组
assert_type(def_gen.wald(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组，两个参数均为数组 D_arr_0p5，且样本大小为 1
assert_type(def_gen.wald(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Wald 分布样本的类型是否为 numpy float64 数组，两个参数均为形状类似于 D_arr_0p5 的数组，且样本大小为 1
assert_type(def_gen.wald(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 检查使用自定义生成器生成的均匀分布样本的类型是否为 float
assert_type(def_gen.uniform(0.5, 0.5), float)
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 float
assert_type(def_gen.uniform(0.5, 0.5, size=None), float)
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5
assert_type(def_gen.uniform(0.5, 0.5, size=1), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5
assert_type(def_gen.uniform(D_arr_0p5, 0.5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5
assert_type(def_gen.uniform(0.5, D_arr_0p5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5，且样本大小为 1
assert_type(def_gen.uniform(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5，且样本大小为 1
assert_type(def_gen.uniform(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，其中一个参数为形状类似于 D_arr_0p5 的数组
assert_type(def_gen.uniform(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，其中一个参数为形状类似于 D_arr_0p5 的数组
assert_type(def_gen.uniform(0.5, D_arr_like_0p5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，两个参数均为数组 D_arr_0p5
assert_type(def_gen.uniform(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，两个参数均为形状类似于 D_arr_0p5 的数组
assert_type(def_gen.uniform(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，两个参数均为数组 D_arr_0p5，且样本大小为 1
assert_type(def_gen.uniform(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的均匀分布样本的类型是否为 numpy float64 数组，两个参数均为形状类似于 D_arr_0p5 的数组，且样本大小为 1
assert_type(def_gen.uniform(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 检查使用自定义生成器生成的 Beta 分布样本的类型是否为 float
assert_type(def_gen.beta(0.5, 0.5), float)
# 检查使用自定义生成器生成的 Beta 分布样本的类型是否为 float
assert_type(def_gen.beta(0.5, 0.5, size=None), float)
# 检查使用自定义生成器生成的 Beta 分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5
assert_type(def_gen.beta(0.5, 0.5, size=1), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Beta 分布样本的类型是否为 numpy float64 数组，其中一个参数为数组 D_arr_0p5
assert_type(def_gen.beta(D_arr_0p5, 0.5), npt.NDArray[np.float64])
# 检查使用自定义生成器生成的 Beta 分布样本的
# 使用 assert_type 函数检查 def_gen.f 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.f(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.f 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.f(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.f 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.f(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.f 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.f(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.f 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.f(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.f 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.f(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.f 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.f(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.f 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.f(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.f 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.f(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 float
assert_type(def_gen.gamma(0.5, 0.5), float)

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 float
assert_type(def_gen.gamma(0.5, 0.5, size=None), float)

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gamma 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gamma(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 float
assert_type(def_gen.gumbel(0.5, 0.5), float)

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 float
assert_type(def_gen.gumbel(0.5, 0.5, size=None), float)

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.gumbel 函数返回值的类型是否为 npt.NDArray[np.float64]
assert_type(def_gen.gumbel(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数检查 def_gen.laplace 函数返回值的类型是否为 float
assert_type(def_gen.laplace(0.5, 0.5), float)

# 使用 assert_type 函数检查 def_gen.laplace 函数返回值的类型是否为 float
assert_type
# 检查 laplace 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.laplace(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 检查 laplace 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.laplace(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 检查 laplace 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.laplace(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 检查 laplace 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.laplace(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 检查 laplace 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.laplace(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 检查 laplace 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.laplace(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 检查 laplace 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.laplace(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 float 的单元测试
assert_type(def_gen.logistic(0.5, 0.5), float)

# 检查 logistic 函数调用，验证返回结果类型为 float 的单元测试
assert_type(def_gen.logistic(0.5, 0.5, size=None), float)

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 检查 logistic 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.logistic(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 float 的单元测试
assert_type(def_gen.lognormal(0.5, 0.5), float)

# 检查 lognormal 函数调用，验证返回结果类型为 float 的单元测试
assert_type(def_gen.lognormal(0.5, 0.5, size=None), float)

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 检查 lognormal 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.lognormal(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 检查 noncentral_chisquare 函数调用，验证返回结果类型为 float 的单元测试
assert_type(def_gen.noncentral_chisquare(0.5, 0.5), float)

# 检查 noncentral_chisquare 函数调用，验证返回结果类型为 float 的单元测试
assert_type(def_gen.noncentral_chisquare(0.5, 0.5, size=None), float)

# 检查 noncentral_chisquare 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.noncentral_chisquare(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 检查 noncentral_chisquare 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.noncentral_chisquare(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 检查 noncentral_chisquare 函数调用，验证返回结果类型为 np.float64 的单元测试
assert_type(def_gen.noncentral_chisquare(0.5, D_arr_0p5), npt.NDArray[np.float64])
# 使用自定义生成器(def_gen)生成非中心卡方分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.noncentral_chisquare(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成非中心卡方分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.noncentral_chisquare(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成非中心卡方分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.noncentral_chisquare(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成非中心卡方分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.noncentral_chisquare(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成非中心卡方分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.noncentral_chisquare(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成非中心卡方分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成非中心卡方分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.noncentral_chisquare(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成非中心卡方分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 float
assert_type(def_gen.normal(0.5, 0.5), float)

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 float
assert_type(def_gen.normal(0.5, 0.5, size=None), float)

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成正态分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.normal(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 float
assert_type(def_gen.triangular(0.1, 0.5, 0.9), float)

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 float
assert_type(def_gen.triangular(0.1, 0.5, 0.9, size=None), float)

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.triangular(0.1, 0.5, 0.9, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.triangular(D_arr_0p1, 0.5, 0.9), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.triangular(0.1, D_arr_0p5, 0.9), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.triangular(D_arr_0p1, 0.5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.triangular(0.1, D_arr_0p5, 0.9, size=1), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.triangular(D_arr_like_0p1, 0.5, D_arr_0p9), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.triangular(0.5, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.triangular(D_arr_0p1, D_arr_0p5, 0.9), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.triangular(D_arr_like_0p1, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])

# 使用自定义生成器(def_gen)生成三角分布的随机变量，返回类型为 numpy.float64 数组
assert_type(def_gen.triangular(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1), npt.NDArray[np.float64])

# 使用
#`
# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, 0.5, 0.9, size=1), npt.NDArray[np.float64])

# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, 0.5, 0.9), npt.NDArray[np.float64])

# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9), npt.NDArray[np.float64])

# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, 0.5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])

# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9, size=1), npt.NDArray[np.float64])

# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, 0.5, D_arr_0p9), npt.NDArray[np.float64])

# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.5, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])

# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, 0.9), npt.NDArray[np.float64])

# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])

# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1), npt.NDArray[np.float64])

# 确保 def_gen.noncentral_f 函数返回的结果类型是 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])


# 确保 def_gen.binomial 函数返回的结果类型是 int
assert_type(def_gen.binomial(10, 0.5), int)

# 确保 def_gen.binomial 函数返回的结果类型是 int
assert_type(def_gen.binomial(10, 0.5, size=None), int)

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, 0.5, size=1), npt.NDArray[np.int64])

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, 0.5), npt.NDArray[np.int64])

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_0p5), npt.NDArray[np.int64])

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, 0.5, size=1), npt.NDArray[np.int64])

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_0p5, size=1), npt.NDArray[np.int64])

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_like_10, 0.5), npt.NDArray[np.int64])

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_like_0p5), npt.NDArray[np.int64])

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, D_arr_0p5), npt.NDArray[np.int64])

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_like_10, D_arr_like_0p5), npt.NDArray[np.int64])

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, D_arr_0p5, size=1), npt.NDArray[np.int64])

# 确保 def_gen.binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_like_10, D_arr_like_0p5, size=1), npt.NDArray[np.int64])


# 确保 def_gen.negative_binomial 函数返回的结果类型是 int
assert_type(def_gen.negative_binomial(10, 0.5), int)

# 确保 def_gen.negative_binomial 函数返回的结果类型是 int
assert_type(def_gen.negative_binomial(10, 0.5, size=None), int)

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(10, 0.5, size=1), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_10, 0.5), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(10, D_arr_0p5), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_10, 0.5, size=1), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(10, D_arr_0p5, size=1), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_like_10, 0.5), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(10, D_arr_like_0p5), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_10, D_arr_0p5), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_like_10, D_arr_like_0p5), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_10, D_arr_0p5, size=1), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 函数返回的结果类型是 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_like_10, D_arr_like_0p5, size=1), npt.NDArray```python
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, 0.5, 0.9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, 0.5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, 0.5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, 0.5, D_arr_0p9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.5, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])

# 确保 def_gen.binomial 返回的结果类型为 int
assert_type(def_gen.binomial(10, 0.5), int)
# 确保 def_gen.binomial 返回的结果类型为 int
assert_type(def_gen.binomial(10, 0.5, size=None), int)
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, 0.5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, 0.5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, 0.5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_0p5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_like_10, 0.5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_like_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, D_arr_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_like_10, D_arr_like_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, D_arr_0p5, size=1), n```py
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, 0.5, 0.9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, 0.5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, 0.5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, 0.5, D_arr_0p9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.5, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])

# 确保 def_gen.binomial 返回的结果类型为 int
assert_type(def_gen.binomial(10, 0.5), int)
# 确保 def_gen.binomial 返回的结果类型为 int
assert_type(def_gen.binomial(10, 0.5, size=None), int)
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, 0.5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, 0.5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, 0.5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_0p5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_like_10, 0.5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_like_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, D_arr_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_like_10, D_arr_like_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, D_arr_0p5, size=1), npt```python
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, 0.5, 0.9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, 0.5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, 0.5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, 0.5, D_arr_0p9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(0.5, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1), npt.NDArray[np.float64])
# 确保 def_gen.noncentral_f 返回的结果类型为 npt.NDArray[np.float64]
assert_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])

# 确保 def_gen.binomial 返回的结果类型为 int
assert_type(def_gen.binomial(10, 0.5), int)
# 确保 def_gen.binomial 返回的结果类型为 int
assert_type(def_gen.binomial(10, 0.5, size=None), int)
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, 0.5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, 0.5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, 0.5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_0p5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_like_10, 0.5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(10, D_arr_like_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, D_arr_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_like_10, D_arr_like_0p5), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_10, D_arr_0p5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.binomial(I_arr_like_10, D_arr_like_0p5, size=1), npt.NDArray[np.int64])

# 确保 def_gen.negative_binomial 返回的结果类型为 int
assert_type(def_gen.negative_binomial(10, 0.5), int)
# 确保 def_gen.negative_binomial 返回的结果类型为 int
assert_type(def_gen.negative_binomial(10, 0.5, size=None), int)
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(10, 0.5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_10, 0.5), npt.NDArray[np.int64])
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(10, D_arr_0p5), npt.NDArray[np.int64])
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_10, 0.5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(10, D_arr_0p5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_like_10, 0.5), npt.NDArray[np.int64])
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(10, D_arr_like_0p5), npt.NDArray[np.int64])
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_10, D_arr_0p5), npt.NDArray[np.int64])
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_like_10, D_arr_like_0p5), npt.NDArray[np.int64])
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_10, D_arr_0p5, size=1), npt.NDArray[np.int64])
# 确保 def_gen.negative_binomial 返回的结果类型为 npt.NDArray[np.int64]
assert_type(def_gen.negative_binomial(I_arr_like_10, D_arr_like_0p5, size=1), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 int
assert_type(def_gen.hypergeometric(20, 20, 10), int)
# 确保函数 def_gen.hypergeometric 的返回类型为 int
assert_type(def_gen.hypergeometric(20, 20, 10, size=None), int)
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(20, 20, 10, size=1), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(I_arr_20, 20, 10), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(20, I_arr_20, 10), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(I_arr_20, 20, I_arr_like_10, size=1), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(20, I_arr_20, 10, size=1), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(I_arr_like_20, 20, I_arr_10), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(20, I_arr_like_20, 10), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(I_arr_20, I_arr_20, 10), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(I_arr_like_20, I_arr_like_20, 10), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(I_arr_20, I_arr_20, I_arr_10, size=1), npt.NDArray[np.int64])
# 确保函数 def_gen.hypergeometric 的返回类型为 numpy int64 数组
assert_type(def_gen.hypergeometric(I_arr_like_20, I_arr_like_20, I_arr_like_10, size=1), npt.NDArray[np.int64])

# 定义一个包含单个整数 100 的 numpy int64 数组
I_int64_100: npt.NDArray[np.int64] = np.array([100], dtype=np.int64)

# 确保函数 def_gen.integers 的返回类型为 int
assert_type(def_gen.integers(0, 100), int)
# 确保函数 def_gen.integers 的返回类型为 int
assert_type(def_gen.integers(100), int)
# 确保函数 def_gen.integers 的返回类型为 numpy int64 数组
assert_type(def_gen.integers([100]), npt.NDArray[np.int64])
# 确保函数 def_gen.integers 的返回类型为 numpy int64 数组
assert_type(def_gen.integers(0, [100]), npt.NDArray[np.int64])

# 定义一个包含单个布尔值 False 的 numpy bool 数组
I_bool_low: npt.NDArray[np.bool] = np.array([0], dtype=np.bool)
# 定义一个包含单个整数 0 的列表
I_bool_low_like: list[int] = [0]
# 定义一个包含单个布尔值 True 的 numpy bool 数组
I_bool_high_open: npt.NDArray[np.bool] = np.array([1], dtype=np.bool)
# 定义一个包含单个布尔值 True 的 numpy bool 数组
I_bool_high_closed: npt.NDArray[np.bool] = np.array([1], dtype=np.bool)

# 确保函数 def_gen.integers 的返回类型为 bool
assert_type(def_gen.integers(2, dtype=bool), bool)
# 确保函数 def_gen.integers 的返回类型为 bool
assert_type(def_gen.integers(0, 2, dtype=bool), bool)
# 确保函数 def_gen.integers 的返回类型为 bool
assert_type(def_gen.integers(1, dtype=bool, endpoint=True), bool)
# 确保函数 def_gen.integers 的返回类型为 bool
assert_type(def_gen.integers(0, 1, dtype=bool, endpoint=True), bool)
# 确保函数 def_gen.integers 的返回类型为 numpy bool 数组
assert_type(def_gen.integers(I_bool_low_like, 1, dtype=bool, endpoint=True), npt.NDArray[np.bool])
# 确保函数 def_gen.integers 的返回类型为 numpy bool 数组
assert_type(def_gen.integers(I_bool_high_open, dtype=bool), npt.NDArray[np.bool])
# 确保函数 def_gen.integers 的返回类型为 numpy bool 数组
assert_type(def_gen.integers(I_bool_low, I_bool_high_open, dtype=bool), npt.NDArray[np.bool])
# 确保函数 def_gen.integers 的返回类型为 numpy bool 数组
assert_type(def_gen.integers(0, I_bool_high_open, dtype=bool), npt.NDArray[np.bool])
# 确保函数 def_gen.integers 的返回类型为 numpy bool 数组
assert_type(def_gen.integers(I_bool_high_closed, dtype=bool, endpoint=True), npt.NDArray[np.bool])
# 确保函数 def_gen.integers 的返回类型为 numpy bool 数组
assert_type(def_gen.integers(I_bool_low, I_bool_high_closed, dtype=bool, endpoint=True), npt.NDArray[np.bool])
# 确保函数 def_gen.integers 的返回类型为 numpy bool 数组
assert_type(def_gen.integers(0, I_bool_high_closed, dtype=bool, endpoint=True), npt.NDArray[np.bool])

# 确保函数 def_gen.integers 的返回类型为 numpy bool
assert_type(def_gen.integers(2, dtype=np.bool), np.bool)
# 确保函数 def_gen.integers 的返回类型为 numpy bool
assert_type(def_gen.integers(0, 2, dtype=np.bool), np.bool)
# 确保函数 def_gen.integers 的返回类型为 numpy bool
assert_type(def_gen.integers(1, dtype=np.bool, endpoint=True), np.bool)
# 确保函数 def_gen.integers 的返回类型为 numpy bool
assert_type(def_gen.integers(0, 1, dtype=np.bool, endpoint=True), np.bool)
# 确保函数 def_gen.integers 的返回类型为 numpy bool 数组
assert_type(def_gen.integers(I_bool_low_like, 1, dtype=np.bool, endpoint=True), npt.NDArray[np.bool])
# 确保函数 def_gen.integers 的返回类型为 numpy bool 数组
assert_type(def_gen.integers(I_bool_high_open, dtype=np.bool), npt.NDArray[np.bool])
# 断言：生成一个布尔类型的随机整数，范围是[I_bool_low, I_bool_high_open)，返回结果的类型应为np.bool
assert_type(def_gen.integers(I_bool_low, I_bool_high_open, dtype=np.bool), npt.NDArray[np.bool])

# 断言：生成一个布尔类型的随机整数，范围是[0, I_bool_high_open)，返回结果的类型应为np.bool
assert_type(def_gen.integers(0, I_bool_high_open, dtype=np.bool), npt.NDArray[np.bool])

# 断言：生成一个布尔类型的随机整数，范围是[I_bool_high_closed, ∞)，返回结果的类型应为np.bool
assert_type(def_gen.integers(I_bool_high_closed, dtype=np.bool, endpoint=True), npt.NDArray[np.bool])

# 断言：生成一个布尔类型的随机整数，范围是[I_bool_low, I_bool_high_closed]，返回结果的类型应为np.bool
assert_type(def_gen.integers(I_bool_low, I_bool_high_closed, dtype=np.bool, endpoint=True), npt.NDArray[np.bool])

# 断言：生成一个布尔类型的随机整数，范围是[0, I_bool_high_closed]，返回结果的类型应为np.bool
assert_type(def_gen.integers(0, I_bool_high_closed, dtype=np.bool, endpoint=True), npt.NDArray[np.bool])

# 定义一个uint8类型的数组，其值为[0]
I_u1_low: npt.NDArray[np.uint8] = np.array([0], dtype=np.uint8)

# 定义一个int类型的列表，其值为[0]
I_u1_low_like: list[int] = [0]

# 定义一个uint8类型的数组，其值为[255]
I_u1_high_open: npt.NDArray[np.uint8] = np.array([255], dtype=np.uint8)

# 定义一个uint8类型的数组，其值为[255]
I_u1_high_closed: npt.NDArray[np.uint8] = np.array([255], dtype=np.uint8)

# 断言：生成一个uint8类型的随机整数，范围是[256, ∞)，返回结果的类型应为np.uint8
assert_type(def_gen.integers(256, dtype="u1"), np.uint8)

# 断言：生成一个uint8类型的随机整数，范围是[0, 256)，返回结果的类型应为np.uint8
assert_type(def_gen.integers(0, 256, dtype="u1"), np.uint8)

# 断言：生成一个uint8类型的随机整数，范围是[0, 255]，返回结果的类型应为np.uint8
assert_type(def_gen.integers(255, dtype="u1", endpoint=True), np.uint8)

# 断言：生成一个uint8类型的随机整数，范围是[0, 255]，返回结果的类型应为np.uint8
assert_type(def_gen.integers(0, 255, dtype="u1", endpoint=True), np.uint8)

# 断言：生成一个uint8类型的随机整数，范围是[I_u1_low_like, 255]，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(I_u1_low_like, 255, dtype="u1", endpoint=True), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[I_u1_high_open, ∞)，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(I_u1_high_open, dtype="u1"), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[I_u1_low, I_u1_high_open)，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(I_u1_low, I_u1_high_open, dtype="u1"), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[0, I_u1_high_open)，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(0, I_u1_high_open, dtype="u1"), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[I_u1_high_closed, ∞)，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(I_u1_high_closed, dtype="u1", endpoint=True), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[I_u1_low, I_u1_high_closed]，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(I_u1_low, I_u1_high_closed, dtype="u1", endpoint=True), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[0, I_u1_high_closed]，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(0, I_u1_high_closed, dtype="u1", endpoint=True), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[256, ∞)，返回结果的类型应为np.uint8
assert_type(def_gen.integers(256, dtype="uint8"), np.uint8)

# 断言：生成一个uint8类型的随机整数，范围是[0, 256)，返回结果的类型应为np.uint8
assert_type(def_gen.integers(0, 256, dtype="uint8"), np.uint8)

# 断言：生成一个uint8类型的随机整数，范围是[0, 255]，返回结果的类型应为np.uint8
assert_type(def_gen.integers(255, dtype="uint8", endpoint=True), np.uint8)

# 断言：生成一个uint8类型的随机整数，范围是[0, 255]，返回结果的类型应为np.uint8
assert_type(def_gen.integers(0, 255, dtype="uint8", endpoint=True), np.uint8)

# 断言：生成一个uint8类型的随机整数，范围是[I_u1_low_like, 255]，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(I_u1_low_like, 255, dtype="uint8", endpoint=True), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[I_u1_high_open, ∞)，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(I_u1_high_open, dtype="uint8"), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[I_u1_low, I_u1_high_open)，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(I_u1_low, I_u1_high_open, dtype="uint8"), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[0, I_u1_high_open)，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(0, I_u1_high_open, dtype="uint8"), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[I_u1_high_closed, ∞)，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(I_u1_high_closed, dtype="uint8", endpoint=True), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[I_u1_low, I_u1_high_closed]，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(I_u1_low, I_u1_high_closed, dtype="uint8", endpoint=True), npt.NDArray[np.uint8])

# 断言：生成一个uint8类型的随机整数，范围是[0, I_u1_high_closed]，返回结果的类型应为npt.NDArray[np.uint8]
assert_type(def_gen.integers(0, I_u1_high_closed, dtype="uint8", endpoint=True
# 使用 def_gen.integers 函数生成一个 uint8 类型的随机整数数组，范围是从 I_u1_low_like 到 255，包含 255，生成的结果应为 np.uint8 类型的数组
assert_type(def_gen.integers(I_u1_low_like, 255, dtype=np.uint8, endpoint=True), npt.NDArray[np.uint8])

# 使用 def_gen.integers 函数生成一个 uint8 类型的随机整数数组，范围是从 I_u1_high_open 到 255，不包含 255，生成的结果应为 np.uint8 类型的数组
assert_type(def_gen.integers(I_u1_high_open, dtype=np.uint8), npt.NDArray[np.uint8])

# 使用 def_gen.integers 函数生成一个 uint8 类型的随机整数数组，范围是从 I_u1_low 到 I_u1_high_open，不包含 I_u1_high_open，生成的结果应为 np.uint8 类型的数组
assert_type(def_gen.integers(I_u1_low, I_u1_high_open, dtype=np.uint8), npt.NDArray[np.uint8])

# 使用 def_gen.integers 函数生成一个 uint8 类型的随机整数数组，范围是从 0 到 I_u1_high_open，不包含 I_u1_high_open，生成的结果应为 np.uint8 类型的数组
assert_type(def_gen.integers(0, I_u1_high_open, dtype=np.uint8), npt.NDArray[np.uint8])

# 使用 def_gen.integers 函数生成一个 uint8 类型的随机整数数组，范围是从 I_u1_high_closed 到 255，包含 255，生成的结果应为 np.uint8 类型的数组
assert_type(def_gen.integers(I_u1_high_closed, dtype=np.uint8, endpoint=True), npt.NDArray[np.uint8])

# 使用 def_gen.integers 函数生成一个 uint8 类型的随机整数数组，范围是从 I_u1_low 到 I_u1_high_closed，包含 I_u1_high_closed，生成的结果应为 np.uint8 类型的数组
assert_type(def_gen.integers(I_u1_low, I_u1_high_closed, dtype=np.uint8, endpoint=True), npt.NDArray[np.uint8])

# 使用 def_gen.integers 函数生成一个 uint8 类型的随机整数数组，范围是从 0 到 I_u1_high_closed，包含 I_u1_high_closed，生成的结果应为 np.uint8 类型的数组
assert_type(def_gen.integers(0, I_u1_high_closed, dtype=np.uint8, endpoint=True), npt.NDArray[np.uint8])

# 定义一个名为 I_u2_low 的变量，类型为 np.uint16，其值为 [0] 的数组
I_u2_low: npt.NDArray[np.uint16] = np.array([0], dtype=np.uint16)

# 定义一个名为 I_u2_low_like 的变量，类型为 list[int]，其值为 [0] 的列表
I_u2_low_like: list[int] = [0]

# 定义一个名为 I_u2_high_open 的变量，类型为 np.uint16，其值为 [65535] 的数组
I_u2_high_open: npt.NDArray[np.uint16] = np.array([65535], dtype=np.uint16)

# 定义一个名为 I_u2_high_closed 的变量，类型为 np.uint16，其值为 [65535] 的数组
I_u2_high_closed: npt.NDArray[np.uint16] = np.array([65535], dtype=np.uint16)

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数，范围是从 65536 到 65536，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(65536, dtype="u2"), np.uint16)

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数，范围是从 0 到 65536，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(0, 65536, dtype="u2"), np.uint16)

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数，范围是从 65535 到 65535，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(65535, dtype="u2", endpoint=True), np.uint16)

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数，范围是从 0 到 65535，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(0, 65535, dtype="u2", endpoint=True), np.uint16)

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数数组，范围是从 I_u2_low_like 到 65535，包含 65535，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(I_u2_low_like, 65535, dtype="u2", endpoint=True), npt.NDArray[np.uint16])

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数数组，范围是从 I_u2_high_open 到 65535，不包含 65535，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(I_u2_high_open, dtype="u2"), npt.NDArray[np.uint16])

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数数组，范围是从 I_u2_low 到 I_u2_high_open，不包含 I_u2_high_open，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(I_u2_low, I_u2_high_open, dtype="u2"), npt.NDArray[np.uint16])

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数数组，范围是从 0 到 I_u2_high_open，不包含 I_u2_high_open，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(0, I_u2_high_open, dtype="u2"), npt.NDArray[np.uint16])

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数数组，范围是从 I_u2_high_closed 到 65535，包含 65535，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(I_u2_high_closed, dtype="u2", endpoint=True), npt.NDArray[np.uint16])

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数数组，范围是从 I_u2_low 到 I_u2_high_closed，包含 I_u2_high_closed，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(I_u2_low, I_u2_high_closed, dtype="u2", endpoint=True), npt.NDArray[np.uint16])

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数数组，范围是从 0 到 I_u2_high_closed，包含 I_u2_high_closed，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(0, I_u2_high_closed, dtype="u2", endpoint=True), npt.NDArray[np.uint16])

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数，范围是从 65536 到 65536，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(65536, dtype="uint16"), np.uint16)

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数，范围是从 0 到 65536，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(0, 65536, dtype="uint16"), np.uint16)

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数，范围是从 65535 到 65535，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(65535, dtype="uint16", endpoint=True), np.uint16)

# 使用 def_gen.integers 函数生成一个 uint16 类型的随机整数，范围是从 0 到 65535，生成的结果应为 np.uint16 类型的数组
assert_type(def_gen.integers(0, 65535, dtype="uint16", endpoint=True), np.uint16)

# 使用 def_gen.integers 函数
# 确保生成的整数符合指定的数据类型 np.uint16
assert_type(def_gen.integers(0, 65536, dtype=np.uint16), np.uint16)

# 确保生成的整数符合指定的数据类型 np.uint16，包括终点值 65535
assert_type(def_gen.integers(65535, dtype=np.uint16, endpoint=True), np.uint16)

# 确保生成的整数符合指定的数据类型 np.uint16，包括起始和终点值 0 和 65535
assert_type(def_gen.integers(0, 65535, dtype=np.uint16, endpoint=True), np.uint16)

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.uint16]，起始值为 I_u2_low_like，终点值为 65535
assert_type(def_gen.integers(I_u2_low_like, 65535, dtype=np.uint16, endpoint=True), npt.NDArray[np.uint16])

# 确保生成的整数符合指定的数据类型 np.uint16，起始值为 I_u2_high_open
assert_type(def_gen.integers(I_u2_high_open, dtype=np.uint16), npt.NDArray[np.uint16])

# 确保生成的整数符合指定的数据类型 np.uint16，起始值为 I_u2_low，终点值为 I_u2_high_open
assert_type(def_gen.integers(I_u2_low, I_u2_high_open, dtype=np.uint16), npt.NDArray[np.uint16])

# 确保生成的整数符合指定的数据类型 np.uint16，起始值为 0，终点值为 I_u2_high_open
assert_type(def_gen.integers(0, I_u2_high_open, dtype=np.uint16), npt.NDArray[np.uint16])

# 确保生成的整数符合指定的数据类型 np.uint16，起始值为 I_u2_high_closed，包括终点值
assert_type(def_gen.integers(I_u2_high_closed, dtype=np.uint16, endpoint=True), npt.NDArray[np.uint16])

# 确保生成的整数符合指定的数据类型 np.uint16，起始值为 I_u2_low，终点值为 I_u2_high_closed，包括终点值
assert_type(def_gen.integers(I_u2_low, I_u2_high_closed, dtype=np.uint16, endpoint=True), npt.NDArray[np.uint16])

# 确保生成的整数符合指定的数据类型 np.uint16，起始值为 0，终点值为 I_u2_high_closed，包括终点值
assert_type(def_gen.integers(0, I_u2_high_closed, dtype=np.uint16, endpoint=True), npt.NDArray[np.uint16])

# 定义起始值为 0 的 np.uint32 类型的数组
I_u4_low: npt.NDArray[np.uint32] = np.array([0], dtype=np.uint32)

# 定义起始值为 0 的 int 类型的列表
I_u4_low_like: list[int] = [0]

# 定义起始值为 4294967295 的 np.uint32 类型的数组
I_u4_high_open: npt.NDArray[np.uint32] = np.array([4294967295], dtype=np.uint32)

# 定义起始值为 4294967295 的 np.uint32 类型的数组
I_u4_high_closed: npt.NDArray[np.uint32] = np.array([4294967295], dtype=np.uint32)

# 确保生成的整数符合指定的数据类型 np.int_，起始值为 4294967296
assert_type(def_gen.integers(4294967296, dtype=np.int_), np.int_)

# 确保生成的整数符合指定的数据类型 np.int_，起始值为 0，终点值为 4294967296
assert_type(def_gen.integers(0, 4294967296, dtype=np.int_), np.int_)

# 确保生成的整数符合指定的数据类型 np.int_，起始值为 4294967295，包括终点值
assert_type(def_gen.integers(4294967295, dtype=np.int_, endpoint=True), np.int_)

# 确保生成的整数符合指定的数据类型 np.int_，起始值为 0，终点值为 4294967295，包括终点值
assert_type(def_gen.integers(0, 4294967295, dtype=np.int_, endpoint=True), np.int_)

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.int_]，起始值为 I_u4_low_like，终点值为 4294967295，包括终点值
assert_type(def_gen.integers(I_u4_low_like, 4294967295, dtype=np.int_, endpoint=True), npt.NDArray[np.int_])

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.int_]，起始值为 I_u4_high_open
assert_type(def_gen.integers(I_u4_high_open, dtype=np.int_), npt.NDArray[np.int_])

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.int_]，起始值为 I_u4_low，终点值为 I_u4_high_open
assert_type(def_gen.integers(I_u4_low, I_u4_high_open, dtype=np.int_), npt.NDArray[np.int_])

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.int_]，起始值为 0，终点值为 I_u4_high_open
assert_type(def_gen.integers(0, I_u4_high_open, dtype=np.int_), npt.NDArray[np.int_])

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.int_]，起始值为 I_u4_high_closed，包括终点值
assert_type(def_gen.integers(I_u4_high_closed, dtype=np.int_, endpoint=True), npt.NDArray[np.int_])

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.int_]，起始值为 I_u4_low，终点值为 I_u4_high_closed，包括终点值
assert_type(def_gen.integers(I_u4_low, I_u4_high_closed, dtype=np.int_, endpoint=True), npt.NDArray[np.int_])

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.int_]，起始值为 0，终点值为 I_u4_high_closed，包括终点值
assert_type(def_gen.integers(0, I_u4_high_closed, dtype=np.int_, endpoint=True), npt.NDArray[np.int_])

# 确保生成的整数符合指定的数据类型 np.uint32，起始值为 4294967296
assert_type(def_gen.integers(4294967296, dtype="u4"), np.uint32)

# 确保生成的整数符合指定的数据类型 np.uint32，起始值为 0，终点值为 4294967296
assert_type(def_gen.integers(0, 4294967296, dtype="u4"), np.uint32)

# 确保生成的整数符合指定的数据类型 np.uint32，起始值为 4294967295，包括终点值
assert_type(def_gen.integers(4294967295, dtype="u4", endpoint=True), np.uint32)

# 确保生成的整数符合指定的数据类型 np.uint32，起始值为 0，终点值为 4294967295，包括终点值
assert_type(def_gen.integers(0, 4294967295, dtype="u4", endpoint=True), np.uint32)

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.uint32]，起始值为 I_u4_low_like，终点值为 4294967295，包括终点值
assert_type(def_gen.integers(I_u4_low_like, 4294967295, dtype="u4", endpoint=True), npt.NDArray[np.uint32])

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.uint32]，起始值为 I_u4_high_open
assert_type(def_gen.integers(I_u4_high_open, dtype="u4"), npt.NDArray[np.uint32])

# 确保生成的整数符合指定的数据类型 npt.NDArray[np.uint32]，起始值为 I_u4_low，
# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组
assert_type(def_gen.integers(I_u4_low, I_u4_high_closed, dtype="u4", endpoint=True), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组
assert_type(def_gen.integers(0, I_u4_high_closed, dtype="u4", endpoint=True), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32
assert_type(def_gen.integers(4294967296, dtype="uint32"), np.uint32)

# 确保生成的随机整数数组的元素类型为 np.uint32
assert_type(def_gen.integers(0, 4294967296, dtype="uint32"), np.uint32)

# 确保生成的随机整数数组的元素类型为 np.uint32，并且包含终点值
assert_type(def_gen.integers(4294967295, dtype="uint32", endpoint=True), np.uint32)

# 确保生成的随机整数数组的元素类型为 np.uint32，并且包含起点和终点值
assert_type(def_gen.integers(0, 4294967295, dtype="uint32", endpoint=True), np.uint32)

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组，并且包含起点和终点值
assert_type(def_gen.integers(I_u4_low_like, 4294967295, dtype="uint32", endpoint=True), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组
assert_type(def_gen.integers(I_u4_high_open, dtype="uint32"), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组
assert_type(def_gen.integers(I_u4_low, I_u4_high_open, dtype="uint32"), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组
assert_type(def_gen.integers(0, I_u4_high_open, dtype="uint32"), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32，并且包含终点值
assert_type(def_gen.integers(I_u4_high_closed, dtype="uint32", endpoint=True), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组，并且包含起点和终点值
assert_type(def_gen.integers(I_u4_low, I_u4_high_closed, dtype="uint32", endpoint=True), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组，并且包含起点和终点值
assert_type(def_gen.integers(0, I_u4_high_closed, dtype="uint32", endpoint=True), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32
assert_type(def_gen.integers(4294967296, dtype=np.uint32), np.uint32)

# 确保生成的随机整数数组的元素类型为 np.uint32
assert_type(def_gen.integers(0, 4294967296, dtype=np.uint32), np.uint32)

# 确保生成的随机整数数组的元素类型为 np.uint32，并且包含终点值
assert_type(def_gen.integers(4294967295, dtype=np.uint32, endpoint=True), np.uint32)

# 确保生成的随机整数数组的元素类型为 np.uint32，并且包含起点和终点值
assert_type(def_gen.integers(0, 4294967295, dtype=np.uint32, endpoint=True), np.uint32)

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组，并且包含起点和终点值
assert_type(def_gen.integers(I_u4_low_like, 4294967295, dtype=np.uint32, endpoint=True), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组
assert_type(def_gen.integers(I_u4_high_open, dtype=np.uint32), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组
assert_type(def_gen.integers(I_u4_low, I_u4_high_open, dtype=np.uint32), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组
assert_type(def_gen.integers(0, I_u4_high_open, dtype=np.uint32), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32，并且包含终点值
assert_type(def_gen.integers(I_u4_high_closed, dtype=np.uint32, endpoint=True), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组，并且包含起点和终点值
assert_type(def_gen.integers(I_u4_low, I_u4_high_closed, dtype=np.uint32, endpoint=True), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint32 的 numpy 数组，并且包含起点和终点值
assert_type(def_gen.integers(0, I_u4_high_closed, dtype=np.uint32, endpoint=True), npt.NDArray[np.uint32])

# 确保生成的随机整数数组的元素类型为 np.uint 的 numpy 数组
assert_type(def_gen.integers(4294967296, dtype=np.uint), np.uint)

# 确保生成的随机整数数组的元素类型为 np.uint 的 numpy 数组
assert_type(def_gen.integers(0, 4294967296, dtype=np.uint), np.uint)

# 确保生成的随机整数数组的元素类型为 np.uint，并且包含终点值
assert_type(def_gen.integers(4294967295, dtype=np.uint, endpoint=True), np.uint)

# 确保生成的随机整数数组的元素类型为 np.uint，并且包含起点和终点值
assert_type(def_gen.integers(0, 4294967295, dtype=np.uint, endpoint=True), np.uint)

# 确保生成的随机整数数组的元素类型为 np.uint 的 numpy 数组，并且包含起点和终点值
assert_type(def_gen.integers(I_u4_low_like, 4294967295, dtype=np.uint, endpoint=True), npt.NDArray[np.uint])

# 确保生成的随机整数数组的元素类型为 np.uint 的 numpy 数组
assert_type(def_gen.integers(I_u4_high_open, dtype=np.uint), npt.NDArray[np.uint])

# 确保生成的随机整数数组的元素类型为 np.uint 的 numpy 数组
assert_type(def_gen.integers(I_u4_low, I_u4_high_open, dtype=np.uint), npt.NDArray[np.uint])

# 确保生成的随机整数数组的元素类型为 np.uint 的 numpy 数组
assert_type(def_gen.integers(0, I_u4_high_open, dtype=np.uint), npt.NDArray[np.uint])
# 确保生成的随机整数数组的类型为 np.uint，并且上限为 I_u4_high_closed，包含该上限值
assert_type(def_gen.integers(I_u4_high_closed, dtype=np.uint, endpoint=True), npt.NDArray[np.uint])

# 确保生成的随机整数数组的类型为 np.uint，并且上限在 I_u4_low 和 I_u4_high_closed 之间，包含上限值
assert_type(def_gen.integers(I_u4_low, I_u4_high_closed, dtype=np.uint, endpoint=True), npt.NDArray[np.uint])

# 确保生成的随机整数数组的类型为 np.uint，并且上限为 I_u4_high_closed，包含该上限值
assert_type(def_gen.integers(0, I_u4_high_closed, dtype=np.uint, endpoint=True), npt.NDArray[np.uint])

# 创建一个包含单个元素 0 的 np.uint64 数组
I_u8_low: npt.NDArray[np.uint64] = np.array([0], dtype=np.uint64)

# 创建一个包含单个元素 0 的 int 列表
I_u8_low_like: list[int] = [0]

# 创建一个包含单个元素 18446744073709551615 的 np.uint64 数组
I_u8_high_open: npt.NDArray[np.uint64] = np.array([18446744073709551615], dtype=np.uint64)

# 创建一个包含单个元素 18446744073709551615 的 np.uint64 数组
I_u8_high_closed: npt.NDArray[np.uint64] = np.array([18446744073709551615], dtype=np.uint64)

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 18446744073709551616
assert_type(def_gen.integers(18446744073709551616, dtype="u8"), np.uint64)

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 0 和 18446744073709551616 之间，包含上限值
assert_type(def_gen.integers(0, 18446744073709551616, dtype="u8"), np.uint64)

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 18446744073709551615，包含该上限值
assert_type(def_gen.integers(18446744073709551615, dtype="u8", endpoint=True), np.uint64)

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 0 和 18446744073709551615 之间，包含上限值
assert_type(def_gen.integers(0, 18446744073709551615, dtype="u8", endpoint=True), np.uint64)

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 18446744073709551615，包含该上限值
assert_type(def_gen.integers(I_u8_low_like, 18446744073709551615, dtype="u8", endpoint=True), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 I_u8_high_open 的元素值
assert_type(def_gen.integers(I_u8_high_open, dtype="u8"), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 I_u8_low 和 I_u8_high_open 之间，不包含上限值
assert_type(def_gen.integers(I_u8_low, I_u8_high_open, dtype="u8"), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 0 和 I_u8_high_open 之间，不包含上限值
assert_type(def_gen.integers(0, I_u8_high_open, dtype="u8"), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 I_u8_high_closed 的元素值，包含该上限值
assert_type(def_gen.integers(I_u8_high_closed, dtype="u8", endpoint=True), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 I_u8_low 和 I_u8_high_closed 之间，包含上限值
assert_type(def_gen.integers(I_u8_low, I_u8_high_closed, dtype="u8", endpoint=True), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 0 和 I_u8_high_closed 之间，包含上限值
assert_type(def_gen.integers(0, I_u8_high_closed, dtype="u8", endpoint=True), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 18446744073709551616
assert_type(def_gen.integers(18446744073709551616, dtype="uint64"), np.uint64)

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 0 和 18446744073709551616 之间，包含上限值
assert_type(def_gen.integers(0, 18446744073709551616, dtype="uint64"), np.uint64)

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 18446744073709551615，包含该上限值
assert_type(def_gen.integers(18446744073709551615, dtype="uint64", endpoint=True), np.uint64)

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 0 和 18446744073709551615 之间，包含上限值
assert_type(def_gen.integers(0, 18446744073709551615, dtype="uint64", endpoint=True), np.uint64)

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 I_u8_low_like 和 18446744073709551615 之间，包含上限值
assert_type(def_gen.integers(I_u8_low_like, 18446744073709551615, dtype="uint64", endpoint=True), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 I_u8_high_open 的元素值
assert_type(def_gen.integers(I_u8_high_open, dtype="uint64"), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 I_u8_low 和 I_u8_high_open 之间，不包含上限值
assert_type(def_gen.integers(I_u8_low, I_u8_high_open, dtype="uint64"), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 0 和 I_u8_high_open 之间，不包含上限值
assert_type(def_gen.integers(0, I_u8_high_open, dtype="uint64"), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 I_u8_high_closed 的元素值，包含该上限值
assert_type(def_gen.integers(I_u8_high_closed, dtype="uint64", endpoint=True), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 I_u8_low 和 I_u8_high_closed 之间，包含上限值
assert_type(def_gen.integers(I_u8_low, I_u8_high_closed, dtype="uint64", endpoint=True), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限在 0 和 I_u8_high_closed 之间，包含上限值
assert_type(def_gen.integers(0, I_u8_high_closed, dtype="uint64", endpoint=True), npt.NDArray[np.uint64])

# 确保生成的随机整数数组的类型为 np.uint64，并且上限为 18446744073709551616
assert_type(def_gen.integers(18446744073709551616, dtype=np.uint64), np.uint64)

# 确保生成的
assert_type(def_gen.integers(0, 18446744073709551615, dtype=np.uint64, endpoint=True), np.uint64)
# 检查从0到18446744073709551615范围内的无符号64位整数生成器类型是否为np.uint64

assert_type(def_gen.integers(I_u8_low_like, 18446744073709551615, dtype=np.uint64, endpoint=True), npt.NDArray[np.uint64])
# 检查从I_u8_low_like到18446744073709551615范围内的无符号64位整数生成器类型是否为npt.NDArray[np.uint64]

assert_type(def_gen.integers(I_u8_high_open, dtype=np.uint64), npt.NDArray[np.uint64])
# 检查从0到I_u8_high_open范围内的无符号64位整数生成器类型是否为npt.NDArray[np.uint64]

assert_type(def_gen.integers(I_u8_low, I_u8_high_open, dtype=np.uint64), npt.NDArray[np.uint64])
# 检查从I_u8_low到I_u8_high_open范围内的无符号64位整数生成器类型是否为npt.NDArray[np.uint64]

assert_type(def_gen.integers(0, I_u8_high_open, dtype=np.uint64), npt.NDArray[np.uint64])
# 检查从0到I_u8_high_open范围内的无符号64位整数生成器类型是否为npt.NDArray[np.uint64]

assert_type(def_gen.integers(I_u8_high_closed, dtype=np.uint64, endpoint=True), npt.NDArray[np.uint64])
# 检查从0到I_u8_high_closed范围内的无符号64位整数生成器类型是否为npt.NDArray[np.uint64]

assert_type(def_gen.integers(I_u8_low, I_u8_high_closed, dtype=np.uint64, endpoint=True), npt.NDArray[np.uint64])
# 检查从I_u8_low到I_u8_high_closed范围内的无符号64位整数生成器类型是否为npt.NDArray[np.uint64]

assert_type(def_gen.integers(0, I_u8_high_closed, dtype=np.uint64, endpoint=True), npt.NDArray[np.uint64])
# 检查从0到I_u8_high_closed范围内的无符号64位整数生成器类型是否为npt.NDArray[np.uint64]


I_i1_low: npt.NDArray[np.int8] = np.array([-128], dtype=np.int8)
# 创建一个包含值-128的np.int8类型的数组赋值给I_i1_low

I_i1_low_like: list[int] = [-128]
# 创建一个包含-128值的int类型列表赋值给I_i1_low_like

I_i1_high_open: npt.NDArray[np.int8] = np.array([127], dtype=np.int8)
# 创建一个包含值127的np.int8类型的数组赋值给I_i1_high_open

I_i1_high_closed: npt.NDArray[np.int8] = np.array([127], dtype=np.int8)
# 创建一个包含值127的np.int8类型的数组赋值给I_i1_high_closed

assert_type(def_gen.integers(128, dtype="i1"), np.int8)
# 检查从-128到127范围内的有符号8位整数生成器类型是否为np.int8

assert_type(def_gen.integers(-128, 128, dtype="i1"), np.int8)
# 检查从-128到128范围内的有符号8位整数生成器类型是否为np.int8

assert_type(def_gen.integers(127, dtype="i1", endpoint=True), np.int8)
# 检查从-128到127范围内的有符号8位整数生成器类型是否为np.int8

assert_type(def_gen.integers(-128, 127, dtype="i1", endpoint=True), np.int8)
# 检查从-128到127范围内的有符号8位整数生成器类型是否为np.int8

assert_type(def_gen.integers(I_i1_low_like, 127, dtype="i1", endpoint=True), npt.NDArray[np.int8])
# 检查从I_i1_low_like到127范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def_gen.integers(I_i1_high_open, dtype="i1"), npt.NDArray[np.int8])
# 检查从-128到I_i1_high_open范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def_gen.integers(I_i1_low, I_i1_high_open, dtype="i1"), npt.NDArray[np.int8])
# 检查从I_i1_low到I_i1_high_open范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def_gen.integers(-128, I_i1_high_open, dtype="i1"), npt.NDArray[np.int8])
# 检查从-128到I_i1_high_open范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def_gen.integers(I_i1_high_closed, dtype="i1", endpoint=True), npt.NDArray[np.int8])
# 检查从-128到I_i1_high_closed范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def_gen.integers(I_i1_low, I_i1_high_closed, dtype="i1", endpoint=True), npt.NDArray[np.int8])
# 检查从I_i1_low到I_i1_high_closed范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def_gen.integers(-128, I_i1_high_closed, dtype="i1", endpoint=True), npt.NDArray[np.int8])
# 检查从-128到I_i1_high_closed范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]


assert_type(def_gen.integers(128, dtype="int8"), np.int8)
# 检查从-128到127范围内的有符号8位整数生成器类型是否为np.int8

assert_type(def_gen.integers(-128, 128, dtype="int8"), np.int8)
# 检查从-128到128范围内的有符号8位整数生成器类型是否为np.int8

assert_type(def_gen.integers(127, dtype="int8", endpoint=True), np.int8)
# 检查从-128到127范围内的有符号8位整数生成器类型是否为np.int8

assert_type(def_gen.integers(-128, 127, dtype="int8", endpoint=True), np.int8)
# 检查从-128到127范围内的有符号8位整数生成器类型是否为np.int8

assert_type(def_gen.integers(I_i1_low_like, 127, dtype="int8", endpoint=True), npt.NDArray[np.int8])
# 检查从I_i1_low_like到127范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def_gen.integers(I_i1_high_open, dtype="int8"), npt.NDArray[np.int8])
# 检查从-128到I_i1_high_open范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def_gen.integers(I_i1_low, I_i1_high_open, dtype="int8"), npt.NDArray[np.int8])
# 检查从I_i1_low到I_i1_high_open范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def_gen.integers(-128, I_i1_high_open, dtype="int8"), npt.NDArray[np.int8])
# 检查从-128到I_i1_high_open范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def_gen.integers(I_i1_high_closed, dtype="int8", endpoint=True), npt.NDArray[np.int8])
# 检查从-128到I_i1_high_closed范围内的有符号8位整数生成器类型是否为npt.NDArray[np.int8]

assert_type(def
# 断言生成的随机整数类型为 np.int8
assert_type(def_gen.integers(128, dtype=np.int8), np.int8)
# 断言生成的随机整数类型为 np.int8，范围在 [-128, 128)
assert_type(def_gen.integers(-128, 128, dtype=np.int8), np.int8)
# 断言生成的随机整数类型为 np.int8，范围在 [0, 127]
assert_type(def_gen.integers(127, dtype=np.int8, endpoint=True), np.int8)
# 断言生成的随机整数类型为 np.int8，范围在 [-128, 127]
assert_type(def_gen.integers(-128, 127, dtype=np.int8, endpoint=True), np.int8)
# 断言生成的随机整数类型为 npt.NDArray[np.int8]，范围在 [I_i1_low_like, 127]
assert_type(def_gen.integers(I_i1_low_like, 127, dtype=np.int8, endpoint=True), npt.NDArray[np.int8])
# 断言生成的随机整数类型为 npt.NDArray[np.int8]，范围在 [I_i1_high_open, inf)
assert_type(def_gen.integers(I_i1_high_open, dtype=np.int8), npt.NDArray[np.int8])
# 断言生成的随机整数类型为 npt.NDArray[np.int8]，范围在 [I_i1_low, I_i1_high_open)
assert_type(def_gen.integers(I_i1_low, I_i1_high_open, dtype=np.int8), npt.NDArray[np.int8])
# 断言生成的随机整数类型为 npt.NDArray[np.int8]，范围在 [-128, I_i1_high_open)
assert_type(def_gen.integers(-128, I_i1_high_open, dtype=np.int8), npt.NDArray[np.int8])
# 断言生成的随机整数类型为 npt.NDArray[np.int8]，范围在 [I_i1_high_closed, 127]
assert_type(def_gen.integers(I_i1_high_closed, dtype=np.int8, endpoint=True), npt.NDArray[np.int8])
# 断言生成的随机整数类型为 npt.NDArray[np.int8]，范围在 [I_i1_low, I_i1_high_closed]
assert_type(def_gen.integers(I_i1_low, I_i1_high_closed, dtype=np.int8, endpoint=True), npt.NDArray[np.int8])
# 断言生成的随机整数类型为 npt.NDArray[np.int8]，范围在 [-128, I_i1_high_closed]
assert_type(def_gen.integers(-128, I_i1_high_closed, dtype=np.int8, endpoint=True), npt.NDArray[np.int8])

# 定义 np.int16 类型的变量和常量
I_i2_low: npt.NDArray[np.int16] = np.array([-32768], dtype=np.int16)
I_i2_low_like: list[int] = [-32768]
I_i2_high_open: npt.NDArray[np.int16] = np.array([32767], dtype=np.int16)
I_i2_high_closed: npt.NDArray[np.int16] = np.array([32767], dtype=np.int16)

# 断言生成的随机整数类型为 np.int16，范围在 [32768, inf)
assert_type(def_gen.integers(32768, dtype="i2"), np.int16)
# 断言生成的随机整数类型为 np.int16，范围在 [-32768, 32768)
assert_type(def_gen.integers(-32768, 32768, dtype="i2"), np.int16)
# 断言生成的随机整数类型为 np.int16，范围在 [0, 32767]
assert_type(def_gen.integers(32767, dtype="i2", endpoint=True), np.int16)
# 断言生成的随机整数类型为 np.int16，范围在 [-32768, 32767]
assert_type(def_gen.integers(-32768, 32767, dtype="i2", endpoint=True), np.int16)
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [I_i2_low_like, 32767]
assert_type(def_gen.integers(I_i2_low_like, 32767, dtype="i2", endpoint=True), npt.NDArray[np.int16])
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [I_i2_high_open, inf)
assert_type(def_gen.integers(I_i2_high_open, dtype="i2"), npt.NDArray[np.int16])
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [I_i2_low, I_i2_high_open)
assert_type(def_gen.integers(I_i2_low, I_i2_high_open, dtype="i2"), npt.NDArray[np.int16])
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [-32768, I_i2_high_open)
assert_type(def_gen.integers(-32768, I_i2_high_open, dtype="i2"), npt.NDArray[np.int16])
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [I_i2_high_closed, 32767]
assert_type(def_gen.integers(I_i2_high_closed, dtype="i2", endpoint=True), npt.NDArray[np.int16])
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [I_i2_low, I_i2_high_closed]
assert_type(def_gen.integers(I_i2_low, I_i2_high_closed, dtype="i2", endpoint=True), npt.NDArray[np.int16])
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [-32768, I_i2_high_closed]
assert_type(def_gen.integers(-32768, I_i2_high_closed, dtype="i2", endpoint=True), npt.NDArray[np.int16])

# 断言生成的随机整数类型为 np.int16，范围在 [32768, inf)
assert_type(def_gen.integers(32768, dtype="int16"), np.int16)
# 断言生成的随机整数类型为 np.int16，范围在 [-32768, 32768)
assert_type(def_gen.integers(-32768, 32768, dtype="int16"), np.int16)
# 断言生成的随机整数类型为 np.int16，范围在 [0, 32767]
assert_type(def_gen.integers(32767, dtype="int16", endpoint=True), np.int16)
# 断言生成的随机整数类型为 np.int16，范围在 [-32768, 32767]
assert_type(def_gen.integers(-32768, 32767, dtype="int16", endpoint=True), np.int16)
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [I_i2_low_like, 32767]
assert_type(def_gen.integers(I_i2_low_like, 32767, dtype="int16", endpoint=True), npt.NDArray[np.int16])
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [I_i2_high_open, inf)
assert_type(def_gen.integers(I_i2_high_open, dtype="int16"), npt.NDArray[np.int16])
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [I_i2_low, I_i2_high_open)
assert_type(def_gen.integers(I_i2_low, I_i2_high_open, dtype="int16"), npt.NDArray[np.int16])
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [-32768, I_i2_high_open)
assert_type(def_gen.integers(-32768, I_i2_high_open, dtype="int16"), npt.NDArray[np.int16])
# 断言生成的随机整数类型为 npt.NDArray[np.int16]，范围在 [I_i2_high_closed, 32767]
assert_type(def_gen.integers(I_i2_high_closed, dtype="int16", endpoint=True), npt.NDArray[np.int16])
# 确保生成的整数数组的数据类型为 int16，并使用 endpoint=True 包含最后一个元素
assert_type(def_gen.integers(I_i2_low, I_i2_high_closed, dtype="int16", endpoint=True), npt.NDArray[np.int16])

# 确保生成的整数数组的数据类型为 int16，范围从 -32768 到 I_i2_high_closed，并包含最后一个元素
assert_type(def_gen.integers(-32768, I_i2_high_closed, dtype="int16", endpoint=True), npt.NDArray[np.int16])

# 确保生成的整数数组的数据类型为 int16，范围从 32768 到最大值，不包含最后一个元素
assert_type(def_gen.integers(32768, dtype=np.int16), np.int16)

# 确保生成的整数数组的数据类型为 int16，范围从 -32768 到 32768，并包含最后一个元素
assert_type(def_gen.integers(-32768, 32768, dtype=np.int16), np.int16)

# 确保生成的整数数组的数据类型为 int16，范围从 32767 到最大值，并包含最后一个元素
assert_type(def_gen.integers(32767, dtype=np.int16, endpoint=True), np.int16)

# 确保生成的整数数组的数据类型为 int16，范围从 -32768 到 32767，并包含最后一个元素
assert_type(def_gen.integers(-32768, 32767, dtype=np.int16, endpoint=True), np.int16)

# 确保生成的整数数组的数据类型为 int16，范围从 I_i2_low_like 到 32767，并包含最后一个元素
assert_type(def_gen.integers(I_i2_low_like, 32767, dtype=np.int16, endpoint=True), npt.NDArray[np.int16])

# 确保生成的整数数组的数据类型为 int16，范围从 I_i2_high_open 到最大值，不包含最后一个元素
assert_type(def_gen.integers(I_i2_high_open, dtype=np.int16), npt.NDArray[np.int16])

# 确保生成的整数数组的数据类型为 int16，范围从 I_i2_low 到 I_i2_high_open，不包含最后一个元素
assert_type(def_gen.integers(I_i2_low, I_i2_high_open, dtype=np.int16), npt.NDArray[np.int16])

# 确保生成的整数数组的数据类型为 int16，范围从 -32768 到 I_i2_high_open，并包含最后一个元素
assert_type(def_gen.integers(-32768, I_i2_high_open, dtype=np.int16), npt.NDArray[np.int16])

# 确保生成的整数数组的数据类型为 int16，范围从 I_i2_high_closed 到最大值，并包含最后一个元素
assert_type(def_gen.integers(I_i2_high_closed, dtype=np.int16, endpoint=True), npt.NDArray[np.int16])

# 确保生成的整数数组的数据类型为 int16，范围从 I_i2_low 到 I_i2_high_closed，并包含最后一个元素
assert_type(def_gen.integers(I_i2_low, I_i2_high_closed, dtype=np.int16, endpoint=True), npt.NDArray[np.int16])

# 确保生成的整数数组的数据类型为 int16，范围从 -32768 到 I_i2_high_closed，并包含最后一个元素
assert_type(def_gen.integers(-32768, I_i2_high_closed, dtype=np.int16, endpoint=True), npt.NDArray[np.int16])

# 将 -2147483648 赋值给数组 I_i4_low，并指定数据类型为 int32
I_i4_low: npt.NDArray[np.int32] = np.array([-2147483648], dtype=np.int32)

# 将 -2147483648 赋值给列表 I_i4_low_like，元素类型为 int
I_i4_low_like: list[int] = [-2147483648]

# 将 2147483647 赋值给数组 I_i4_high_open，并指定数据类型为 int32
I_i4_high_open: npt.NDArray[np.int32] = np.array([2147483647], dtype=np.int32)

# 将 2147483647 赋值给数组 I_i4_high_closed，并指定数据类型为 int32
I_i4_high_closed: npt.NDArray[np.int32] = np.array([2147483647], dtype=np.int32)

# 确保生成的整数数组的数据类型为 int32，范围从 2147483648 到最大值，不包含最后一个元素
assert_type(def_gen.integers(2147483648, dtype="i4"), np.int32)

# 确保生成的整数数组的数据类型为 int32，范围从 -2147483648 到 2147483648，并包含最后一个元素
assert_type(def_gen.integers(-2147483648, 2147483648, dtype="i4"), np.int32)

# 确保生成的整数数组的数据类型为 int32，范围从 2147483647 到最大值，并包含最后一个元素
assert_type(def_gen.integers(2147483647, dtype="i4", endpoint=True), np.int32)

# 确保生成的整数数组的数据类型为 int32，范围从 -2147483648 到 2147483647，并包含最后一个元素
assert_type(def_gen.integers(-2147483648, 2147483647, dtype="i4", endpoint=True), np.int32)

# 确保生成的整数数组的数据类型为 int32，范围从 I_i4_low_like 到 2147483647，并包含最后一个元素
assert_type(def_gen.integers(I_i4_low_like, 2147483647, dtype="i4", endpoint=True), npt.NDArray[np.int32])

# 确保生成的整数数组的数据类型为 int32，范围从 I_i4_high_open 到最大值，不包含最后一个元素
assert_type(def_gen.integers(I_i4_high_open, dtype="i4"), npt.NDArray[np.int32])

# 确保生成的整数数组的数据类型为 int32，范围从 I_i4_low 到 I_i4_high_open，不包含最后一个元素
assert_type(def_gen.integers(I_i4_low, I_i4_high_open, dtype="i4"), npt.NDArray[np.int32])

# 确保生成的整数数组的数据类型为 int32，范围从 -2147483648 到 I_i4_high_open，并包含最后一个元素
assert_type(def_gen.integers(-2147483648, I_i4_high_open, dtype="i4"), npt.NDArray[np.int32])

# 确保生成的整数数组的数据类型为 int32，范围从 I_i4_high_closed 到最大值，并包含最后一个元素
assert_type(def_gen.integers(I_i4_high_closed, dtype="i4", endpoint=True), npt.NDArray[np.int32])

# 确保生成的整数数组的数据类型为 int32，范围从 I_i4_low 到 I_i4_high_closed，并包含最后一个元素
assert_type(def_gen.integers(I_i4_low, I_i4_high_closed, dtype="i4", endpoint=True), npt.NDArray[np.int32])

# 确保生成的整数数组的数据类型为 int32，范围从 -2147483648 到 I_i4_high_closed，并包含最后一个元素
assert_type(def_gen.integers(-2147483648, I_i4_high_closed, dtype="i4", endpoint=True), npt.NDArray[np.int32])

# 确保生成的整数数组的数据类型为 int32，范围从 2147483648 到最大值，不包含最后一个元素
assert_type(def_gen.integers(2147483648, dtype="int32"), np.int32)

# 确保生成的整数数组的数据类型为 int32，范围从 -2147483648 到 2147483648，并包含最后一个元素
assert_type(def_gen.integers(-2147483648, 2147483648, dtype="int32"), np.int32)

# 确保生成的整数数组的数据类型为 int32，范
# 确保返回值是由指定的integers生成器生成的 numpy 数组，数据类型为 int32 的数组
assert_type(def_gen.integers(I_i4_high_open, dtype="int32"), npt.NDArray[np.int32])
# 确保返回值是由指定范围的integers生成器生成的 numpy 数组，数据类型为 int32 的数组
assert_type(def_gen.integers(I_i4_low, I_i4_high_open, dtype="int32"), npt.NDArray[np.int32])
# 确保返回值是由指定范围的integers生成器生成的 numpy 数组，数据类型为 int32 的数组
assert_type(def_gen.integers(-2147483648, I_i4_high_open, dtype="int32"), npt.NDArray[np.int32])
# 确保返回值是由指定的integers生成器生成的 numpy 数组，数据类型为 int32 的数组
assert_type(def_gen.integers(I_i4_high_closed, dtype="int32", endpoint=True), npt.NDArray[np.int32])
# 确保返回值是由指定范围的integers生成器生成的 numpy 数组，数据类型为 int32 的数组
assert_type(def_gen.integers(I_i4_low, I_i4_high_closed, dtype="int32", endpoint=True), npt.NDArray[np.int32])
# 确保返回值是由指定范围的integers生成器生成的 numpy 数组，数据类型为 int32 的数组
assert_type(def_gen.integers(-2147483648, I_i4_high_closed, dtype="int32", endpoint=True), npt.NDArray[np.int32])

# 确保返回值是由指定的integers生成器生成的 numpy int32 类型数据
assert_type(def_gen.integers(2147483648, dtype=np.int32), np.int32)
# 确保返回值是由指定范围的integers生成器生成的 numpy int32 类型数据
assert_type(def_gen.integers(-2147483648, 2147483648, dtype=np.int32), np.int32)
# 确保返回值是由指定的integers生成器生成的 numpy int32 类型数据
assert_type(def_gen.integers(2147483647, dtype=np.int32, endpoint=True), np.int32)
# 确保返回值是由指定范围的integers生成器生成的 numpy int32 类型数据
assert_type(def_gen.integers(-2147483648, 2147483647, dtype=np.int32, endpoint=True), np.int32)
# 确保返回值是由指定范围的integers生成器生成的 numpy 数组，数据类型为 int32 的数组
assert_type(def_gen.integers(I_i4_low_like, 2147483647, dtype=np.int32, endpoint=True), npt.NDArray[np.int32])
# 确保返回值是由指定的integers生成器生成的 numpy int32 类型数据
assert_type(def_gen.integers(I_i4_high_open, dtype=np.int32), npt.NDArray[np.int32])
# 确保返回值是由指定范围的integers生成器生成的 numpy int32 类型数据
assert_type(def_gen.integers(I_i4_low, I_i4_high_open, dtype=np.int32), npt.NDArray[np.int32])
# 确保返回值是由指定范围的integers生成器生成的 numpy int32 类型数据
assert_type(def_gen.integers(-2147483648, I_i4_high_open, dtype=np.int32), npt.NDArray[np.int32])
# 确保返回值是由指定的integers生成器生成的 numpy int32 类型数据
assert_type(def_gen.integers(I_i4_high_closed, dtype=np.int32, endpoint=True), npt.NDArray[np.int32])
# 确保返回值是由指定范围的integers生成器生成的 numpy int32 类型数据
assert_type(def_gen.integers(I_i4_low, I_i4_high_closed, dtype=np.int32, endpoint=True), npt.NDArray[np.int32])
# 确保返回值是由指定范围的integers生成器生成的 numpy int32 类型数据
assert_type(def_gen.integers(-2147483648, I_i4_high_closed, dtype=np.int32, endpoint=True), npt.NDArray[np.int32])

# 创建一个 numpy int64 类型的数组，包含值 -9223372036854775808
I_i8_low: npt.NDArray[np.int64] = np.array([-9223372036854775808], dtype=np.int64)
# 创建一个列表，包含值 -9223372036854775808
I_i8_low_like: list[int] = [-9223372036854775808]
# 创建一个 numpy int64 类型的数组，包含值 9223372036854775807
I_i8_high_open: npt.NDArray[np.int64] = np.array([9223372036854775807], dtype=np.int64)
# 创建一个 numpy int64 类型的数组，包含值 9223372036854775807
I_i8_high_closed: npt.NDArray[np.int64] = np.array([9223372036854775807], dtype=np.int64)

# 确保返回值是由指定的integers生成器生成的 numpy int64 类型数据
assert_type(def_gen.integers(9223372036854775808, dtype="i8"), np.int64)
# 确保返回值是由指定范围的integers生成器生成的 numpy int64 类型数据
assert_type(def_gen.integers(-9223372036854775808, 9223372036854775808, dtype="i8"), np.int64)
# 确保返回值是由指定的integers生成器生成的 numpy int64 类型数据
assert_type(def_gen.integers(9223372036854775807, dtype="i8", endpoint=True), np.int64)
# 确保返回值是由指定范围的integers生成器生成的 numpy int64 类型数据
assert_type(def_gen.integers(-9223372036854775808, 9223372036854775807, dtype="i8", endpoint=True), np.int64)
# 确保返回值是由指定范围的integers生成器生成的 numpy 数组，数据类型为 int64 的数组
assert_type(def_gen.integers(I_i8_low_like, 9223372036854775807, dtype="i8", endpoint=True), npt.NDArray[np.int64])
# 确保返回值是由指定的integers生成器生成的 numpy int64 类型数据
assert_type(def_gen.integers(I_i8_high_open, dtype="i8"), npt.NDArray[np.int64])
# 确保返回值是由指定范围的integers生成器生成的 numpy int64 类型数据
assert_type(def_gen.integers(I_i8_low, I_i8_high_open, dtype="i8"), npt.NDArray[np.int64])
# 确保返回值是由指定范围的integers生成器生成的 numpy int64 类型数据
assert_type(def_gen.integers(-9223372036854775808, I_i8_high_open, dtype="i8"), npt.NDArray[np.int64])
# 确保返回值是由指定的integers生成器生成的 numpy int64 类型数据
assert_type(def_gen.integers(I_i8_high_closed, dtype="i8", endpoint=True), npt.NDArray[np.int64])
# 确保返回值是由指定范围的integers生成器生成的 numpy int64 类型数据
assert_type(def_gen.integers(I_i8_low, I_i8_high_closed, dtype="i8", endpoint=True), npt.NDArray[np.int64])
# 使用 assert_type 函数检查生成器生成的整数是否为 np.int64 类型，生成范围为从最小整数到 I_i8_high_closed，包含最大值
assert_type(def_gen.integers(-9223372036854775808, I_i8_high_closed, dtype="i8", endpoint=True), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的整数是否为 np.int64 类型，生成范围为从 9223372036854775808 开始的 int64 整数
assert_type(def_gen.integers(9223372036854775808, dtype="int64"), np.int64)

# 使用 assert_type 函数检查生成器生成的整数是否为 np.int64 类型，生成范围为从 -9223372036854775808 到 9223372036854775808 的 int64 整数
assert_type(def_gen.integers(-9223372036854775808, 9223372036854775808, dtype="int64"), np.int64)

# 使用 assert_type 函数检查生成器生成的整数是否为 np.int64 类型，生成范围为从 9223372036854775807 开始的 int64 整数，包含最大值
assert_type(def_gen.integers(9223372036854775807, dtype="int64", endpoint=True), np.int64)

# 使用 assert_type 函数检查生成器生成的整数是否为 np.int64 类型，生成范围为从 -9223372036854775808 到 9223372036854775807 的 int64 整数，包含最大值
assert_type(def_gen.integers(-9223372036854775808, 9223372036854775807, dtype="int64", endpoint=True), np.int64)

# 使用 assert_type 函数检查生成器生成的整数是否为 npt.NDArray[np.int64] 类型，生成范围为从 I_i8_low_like 开始到 9223372036854775807 的 int64 整数，包含最大值
assert_type(def_gen.integers(I_i8_low_like, 9223372036854775807, dtype="int64", endpoint=True), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的整数是否为 npt.NDArray[np.int64] 类型，生成范围为从 I_i8_high_open 开始的 int64 整数
assert_type(def_gen.integers(I_i8_high_open, dtype="int64"), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的整数是否为 npt.NDArray[np.int64] 类型，生成范围为从 I_i8_low 到 I_i8_high_open 的 int64 整数
assert_type(def_gen.integers(I_i8_low, I_i8_high_open, dtype="int64"), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的整数是否为 npt.NDArray[np.int64] 类型，生成范围为从 -9223372036854775808 到 I_i8_high_open 的 int64 整数
assert_type(def_gen.integers(-9223372036854775808, I_i8_high_open, dtype="int64"), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的整数是否为 npt.NDArray[np.int64] 类型，生成范围为从 I_i8_high_closed 开始的 int64 整数，包含最大值
assert_type(def_gen.integers(I_i8_high_closed, dtype="int64", endpoint=True), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的整数是否为 npt.NDArray[np.int64] 类型，生成范围为从 I_i8_low 到 I_i8_high_closed 的 int64 整数，包含最大值
assert_type(def_gen.integers(I_i8_low, I_i8_high_closed, dtype="int64", endpoint=True), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的整数是否为 npt.NDArray[np.int64] 类型，生成范围为从 -9223372036854775808 到 I_i8_high_closed 的 int64 整数，包含最大值
assert_type(def_gen.integers(-9223372036854775808, I_i8_high_closed, dtype="int64", endpoint=True), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的随机位生成器是否为 np.random.BitGenerator 类型
assert_type(def_gen.bit_generator, np.random.BitGenerator)

# 使用 assert_type 函数检查生成器生成的字节串是否为 bytes 类型，生成长度为 2 的字节串
assert_type(def_gen.bytes(2), bytes)

# 使用 assert_type 函数检查生成器生成的随机选择是否为 int 类型，从 0 到 5 中选择一个整数
assert_type(def_gen.choice(5), int)

# 使用 assert_type 函数检查生成器生成的随机选择是否为 npt.NDArray[np.int64] 类型，从 0 到 5 中选择 3 个整数
assert_type(def_gen.choice(5, 3), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的随机选择是否为 npt.NDArray[np.int64] 类型，从 0 到 5 中选择 3 个整数，允许重复选择
assert_type(def_gen.choice(5, 3, replace=True), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的随机选择是否为 npt.NDArray[np.int64] 类型，从 0 到 5 中选择 3 个整数，每个选择的概率为 1/5
assert_type(def_gen.choice(5, 3, p=[1 / 5] * 5), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的随机选择是否为 npt.NDArray[np.int64] 类型，从 0 到 5 中选择 3 个整数，每个选择的概率为 1/5，不允许重复选择
assert_type(def_gen.choice(5, 3, p=[1 / 5] * 5, replace=False), npt.NDArray[np.int64])

# 使用 assert_type 函数检查生成器生成的随机选择是否为 Any 类型，从给定列表中选择一个元素
assert_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"]), Any)

# 使用 assert_type 函数检查生成器生成的随机选择是否为 npt.NDArray[Any] 类型，从给定列表中选择 3 个元素
assert_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3), npt.NDArray[Any])
# 确保从给定列表中随机选择三个元素，返回类型应为 NumPy 数组
assert_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, p=[1 / 4] * 4), npt.NDArray[Any])
# 确保从给定列表中随机选择三个元素（可重复选择），返回类型应为 NumPy 数组
assert_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=True), npt.NDArray[Any])
# 确保从给定列表中随机选择三个元素（不可重复选择，概率不均等），返回类型应为 NumPy 数组
assert_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=False, p=np.array([1 / 8, 1 / 8, 1 / 2, 1 / 4])), npt.NDArray[Any])

# 确保生成 Dirichlet 分布的样本，参数为列表，返回类型应为 NumPy 数组（浮点数）
assert_type(def_gen.dirichlet([0.5, 0.5]), npt.NDArray[np.float64])
# 确保生成 Dirichlet 分布的样本，参数为 NumPy 数组，返回类型应为 NumPy 数组（浮点数）
assert_type(def_gen.dirichlet(np.array([0.5, 0.5])), npt.NDArray[np.float64])
# 确保生成 Dirichlet 分布的多个样本，参数为 NumPy 数组，返回类型应为 NumPy 数组（浮点数）
assert_type(def_gen.dirichlet(np.array([0.5, 0.5]), size=3), npt.NDArray[np.float64])

# 确保生成多项式分布的样本，参数为整数和概率列表，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multinomial(20, [1 / 6.0] * 6), npt.NDArray[np.int64])
# 确保生成多项式分布的样本，参数为整数和 NumPy 数组，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multinomial(20, np.array([0.5, 0.5])), npt.NDArray[np.int64])
# 确保生成多项式分布的多个样本，参数为整数、概率列表和 size 参数，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multinomial(20, [1 / 6.0] * 6, size=2), npt.NDArray[np.int64])
# 确保生成多项式分布的多维样本，参数为二维数组、概率列表和 size 参数，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multinomial([[10], [20]], [1 / 6.0] * 6, size=(2, 2)), npt.NDArray[np.int64])
# 确保生成多项式分布的多维样本，参数为 NumPy 数组、NumPy 数组和 size 参数，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multinomial(np.array([[10], [20]]), np.array([0.5, 0.5]), size=(2, 2)), npt.NDArray[np.int64])

# 确保生成多元超几何分布的样本，参数为列表和整数，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multivariate_hypergeometric([3, 5, 7], 2), npt.NDArray[np.int64])
# 确保生成多元超几何分布的样本，参数为 NumPy 数组和整数，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2), npt.NDArray[np.int64])
# 确保生成多元超几何分布的多个样本，参数为 NumPy 数组、整数和 size 参数，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, size=4), npt.NDArray[np.int64])
# 确保生成多元超几何分布的多维样本，参数为 NumPy 数组、整数、size 参数和 method 参数，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, size=(4, 7)), npt.NDArray[np.int64])
# 确保生成多元超几何分布的样本，参数为列表、整数和 method 参数，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multivariate_hypergeometric([3, 5, 7], 2, method="count"), npt.NDArray[np.int64])
# 确保生成多元超几何分布的样本，参数为 NumPy 数组、整数和 method 参数，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, method="marginals"), npt.NDArray[np.int64])

# 确保生成多元正态分布的样本，参数为均值列表和协方差矩阵，返回类型应为 NumPy 数组（浮点数）
assert_type(def_gen.multivariate_normal([0.0], [[1.0]]), npt.NDArray[np.float64])
# 确保生成多元正态分布的样本，参数为均值列表和 NumPy 数组（协方差矩阵），返回类型应为 NumPy 数组（浮点数）
assert_type(def_gen.multivariate_normal([0.0], np.array([[1.0]])), npt.NDArray[np.float64])
# 确保生成多元正态分布的样本，参数为 NumPy 数组和协方差矩阵，返回类型应为 NumPy 数组（浮点数）
assert_type(def_gen.multivariate_normal(np.array([0.0]), [[1.0]]), npt.NDArray[np.float64])
# 确保生成多元正态分布的样本，参数为均值列表和 NumPy 数组（协方差矩阵），返回类型应为 NumPy 数组（浮点数）
assert_type(def_gen.multivariate_normal([0.0], np.array([[1.0]])), npt.NDArray[np.float64])

# 确保生成给定整数范围内的随机排列，返回类型应为 NumPy 数组（整数）
assert_type(def_gen.permutation(10), npt.NDArray[np.int64])
# 确保生成给定列表的随机排列，返回类型应为 NumPy 数组（任意类型）
assert_type(def_gen.permutation([1, 2, 3, 4]), npt.NDArray[Any])
# 确保生成给定 NumPy 数组的随机排列，返回类型应为 NumPy 数组（任意类型）
assert_type(def_gen.permutation(np.array([1, 2, 3, 4])), npt.NDArray[Any])
# 确保生成给定二维数组按指定轴的随机排列，返回类型应为 NumPy 数组（任意类型）
assert_type(def_gen.permutation(D_2D, axis=1), npt.NDArray[Any])
# 确保生成给定二维数组的随机排列，返回类型应为 NumPy 数组（任意类型）
assert_type(def_gen.permuted(D_2D), npt.NDArray[Any])
# 确保生成给定形状的随机排列，返回类型应为 NumPy 数组（任意类型）
assert_type(def_gen.permuted(D_2D_like), npt.NDArray[Any])
# 确保生成给定二维数组按指定轴的随机排列，返回类型应为 NumPy 数组（任意类型）
assert_type(def_gen.permuted(D_2D, axis=1), npt.NDArray[Any])
# 确保生成给定二维数组按指定输出的随机排列，返回类型应为 NumPy 数组（任意类型）
assert_type(def_gen.permuted(D_2D, out=D_2D), npt.NDArray[Any])
# 确保生成给
assert_type(def_gen.__repr__(), str)
assert_type(def_gen.__setstate__(dict(def_gen.bit_generator.state)), None)

# 创建一个 RandomState 对象，使用 NumPy 的随机数生成器初始化
random_st: np.random.RandomState = np.random.RandomState()

# 验证随机数生成器生成标准正态分布的值是否为浮点数
assert_type(random_st.standard_normal(), float)
# 验证随机数生成器生成指定大小标准正态分布的值是否为浮点数
assert_type(random_st.standard_normal(size=None), float)
# 验证随机数生成器生成指定大小标准正态分布的值是否为浮点数数组
assert_type(random_st.standard_normal(size=1), npt.NDArray[np.float64])

# 验证随机数生成器生成 [0,1) 之间均匀分布的随机数是否为浮点数
assert_type(random_st.random(), float)
# 验证随机数生成器生成指定大小 [0,1) 之间均匀分布的随机数是否为浮点数
assert_type(random_st.random(size=None), float)
# 验证随机数生成器生成指定大小 [0,1) 之间均匀分布的随机数是否为浮点数数组
assert_type(random_st.random(size=1), npt.NDArray[np.float64])

# 验证随机数生成器生成标准 Cauchy 分布的随机数是否为浮点数
assert_type(random_st.standard_cauchy(), float)
# 验证随机数生成器生成指定大小标准 Cauchy 分布的随机数是否为浮点数
assert_type(random_st.standard_cauchy(size=None), float)
# 验证随机数生成器生成指定大小标准 Cauchy 分布的随机数是否为浮点数数组
assert_type(random_st.standard_cauchy(size=1), npt.NDArray[np.float64])

# 验证随机数生成器生成指数分布的随机数是否为浮点数
assert_type(random_st.standard_exponential(), float)
# 验证随机数生成器生成指定大小指数分布的随机数是否为浮点数
assert_type(random_st.standard_exponential(size=None), float)
# 验证随机数生成器生成指定大小指数分布的随机数是否为浮点数数组
assert_type(random_st.standard_exponential(size=1), npt.NDArray[np.float64])

# 验证随机数生成器生成参数为 1.5 的 Zipf 分布的随机整数是否为整数
assert_type(random_st.zipf(1.5), int)
# 验证随机数生成器生成参数为 1.5 的 Zipf 分布的随机整数是否为整数
assert_type(random_st.zipf(1.5, size=None), int)
# 验证随机数生成器生成参数为 1.5 的 Zipf 分布的随机整数是否为整数数组
assert_type(random_st.zipf(1.5, size=1), npt.NDArray[np.long])
# 验证随机数生成器生成参数为 D_arr_1p5 的 Zipf 分布的随机整数是否为整数数组
assert_type(random_st.zipf(D_arr_1p5), npt.NDArray[np.long])
# 验证随机数生成器生成参数为 D_arr_1p5 的 Zipf 分布的随机整数是否为整数数组
assert_type(random_st.zipf(D_arr_1p5, size=1), npt.NDArray[np.long])
# 验证随机数生成器生成参数为 D_arr_like_1p5 的 Zipf 分布的随机整数是否为整数数组
assert_type(random_st.zipf(D_arr_like_1p5), npt.NDArray[np.long])
# 验证随机数生成器生成参数为 D_arr_like_1p5 的 Zipf 分布的随机整数是否为整数数组
assert_type(random_st.zipf(D_arr_like_1p5, size=1), npt.NDArray[np.long])

# 验证随机数生成器生成参数为 0.5 的 Weibull 分布的随机数是否为浮点数
assert_type(random_st.weibull(0.5), float)
# 验证随机数生成器生成参数为 0.5 的 Weibull 分布的随机数是否为浮点数
assert_type(random_st.weibull(0.5, size=None), float)
# 验证随机数生成器生成参数为 0.5 的 Weibull 分布的随机数是否为浮点数数组
assert_type(random_st.weibull(0.5, size=1), npt.NDArray[np.float64])
# 验证随机数生成器生成参数为 D_arr_0p5 的 Weibull 分布的随机数是否为浮点数数组
assert_type(random_st.weibull(D_arr_0p5), npt.NDArray[np.float64])
# 验证随机数生成器生成参数为 D_arr_0p5 的 Weibull 分布的随机数是否为浮点数数组
assert_type(random_st.weibull(D_arr_0p5, size=1), npt.NDArray[np.float64])
# 验证随机数生成器生成参数为 D_arr_like_0p5 的 Weibull 分布的随机数是否为浮点数数组
assert_type(random_st.weibull(D_arr_like_0p5), npt.NDArray[np.float64])
# 验证随机数生成器生成参数为 D_arr_like_0p5 的 Weibull 分布的随机数是否为浮点数数组
assert_type(random_st.weibull(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 验证随机数生成器生成参数为 0.5 的标准 t 分布的随机数是否为浮点数
assert_type(random_st.standard_t(0.5), float)
# 验证随机数生成器生成参数为 0.5 的标准 t 分布的随机数是否为浮点数
assert_type(random_st.standard_t(0.5, size=None), float)
# 验证随机数生成器生成参数为 0.5 的标准 t 分布的随机数是否为浮点数数组
assert_type(random_st.standard_t(0.5, size=1), npt.NDArray[np.float64])
# 验证随机数生成器生成参数为 D_arr_0p5 的标准 t 分布的随机数是否为浮点数数组
assert_type(random_st.standard_t(D_arr_0p5), npt.NDArray[np.float64])
# 验证随机数生成器生成参数为 D_arr_0p5 的标准 t 分布的随机数是否为浮点数数组
assert_type(random_st.standard_t(D_arr_0p5, size=1), npt.NDArray[np.float64])
# 验证随机数生成器生成参数为 D_arr_like_0p5 的标准 t 分布的随机数是否为浮点数数组
assert_type(random_st.standard_t(D_arr_like_0p5), npt.NDArray[np.float64])
# 验证随机数生成器生成参数为 D_arr_like_0p5 的标准 t 分布的随机数是否为浮点数数组
assert_type(random_st.standard_t(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 验证随机数生成器生成参数为 0.5 的 Poisson 分布的随机整数是否为整数
assert_type(random_st.poisson(0.5), int)
# 验证随机数生成器生成参数为 0.5 的 Poisson 分布的随机整数是否为整数
assert_type(random_st.poisson(0.5, size=None), int)
# 验证随机数生成器生成参数为 0.5 的 Poisson 分布的随机整数是否为整数数组
assert_type(random_st.poisson(0.5, size=1), npt.NDArray[np.long])
# 验证随机数生成器生成参数为 D_arr_0p5 的 Poisson 分布的随机整数是否为整数数组
assert_type(random_st.poisson(D_arr_
# 检查 random_st.power(D_arr_like_0p5) 的返回类型是否为 np.float64
assert_type(random_st.power(D_arr_like_0p5), npt.NDArray[np.float64])

# 检查 random_st.power(D_arr_like_0p5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.power(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 检查 random_st.pareto(0.5) 的返回类型是否为 float
assert_type(random_st.pareto(0.5), float)

# 检查 random_st.pareto(0.5, size=None) 的返回类型是否为 float
assert_type(random_st.pareto(0.5, size=None), float)

# 检查 random_st.pareto(0.5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.pareto(0.5, size=1), npt.NDArray[np.float64])

# 检查 random_st.pareto(D_arr_0p5) 的返回类型是否为 np.float64
assert_type(random_st.pareto(D_arr_0p5), npt.NDArray[np.float64])

# 检查 random_st.pareto(D_arr_0p5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.pareto(D_arr_0p5, size=1), npt.NDArray[np.float64])

# 检查 random_st.pareto(D_arr_like_0p5) 的返回类型是否为 np.float64
assert_type(random_st.pareto(D_arr_like_0p5), npt.NDArray[np.float64])

# 检查 random_st.pareto(D_arr_like_0p5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.pareto(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 检查 random_st.chisquare(0.5) 的返回类型是否为 float
assert_type(random_st.chisquare(0.5), float)

# 检查 random_st.chisquare(0.5, size=None) 的返回类型是否为 float
assert_type(random_st.chisquare(0.5, size=None), float)

# 检查 random_st.chisquare(0.5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.chisquare(0.5, size=1), npt.NDArray[np.float64])

# 检查 random_st.chisquare(D_arr_0p5) 的返回类型是否为 np.float64
assert_type(random_st.chisquare(D_arr_0p5), npt.NDArray[np.float64])

# 检查 random_st.chisquare(D_arr_0p5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.chisquare(D_arr_0p5, size=1), npt.NDArray[np.float64])

# 检查 random_st.chisquare(D_arr_like_0p5) 的返回类型是否为 np.float64
assert_type(random_st.chisquare(D_arr_like_0p5), npt.NDArray[np.float64])

# 检查 random_st.chisquare(D_arr_like_0p5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.chisquare(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 检查 random_st.exponential(0.5) 的返回类型是否为 float
assert_type(random_st.exponential(0.5), float)

# 检查 random_st.exponential(0.5, size=None) 的返回类型是否为 float
assert_type(random_st.exponential(0.5, size=None), float)

# 检查 random_st.exponential(0.5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.exponential(0.5, size=1), npt.NDArray[np.float64])

# 检查 random_st.exponential(D_arr_0p5) 的返回类型是否为 np.float64
assert_type(random_st.exponential(D_arr_0p5), npt.NDArray[np.float64])

# 检查 random_st.exponential(D_arr_0p5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.exponential(D_arr_0p5, size=1), npt.NDArray[np.float64])

# 检查 random_st.exponential(D_arr_like_0p5) 的返回类型是否为 np.float64
assert_type(random_st.exponential(D_arr_like_0p5), npt.NDArray[np.float64])

# 检查 random_st.exponential(D_arr_like_0p5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.exponential(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 检查 random_st.geometric(0.5) 的返回类型是否为 int
assert_type(random_st.geometric(0.5), int)

# 检查 random_st.geometric(0.5, size=None) 的返回类型是否为 int
assert_type(random_st.geometric(0.5, size=None), int)

# 检查 random_st.geometric(0.5, size=1) 的返回类型是否为 np.long
assert_type(random_st.geometric(0.5, size=1), npt.NDArray[np.long])

# 检查 random_st.geometric(D_arr_0p5) 的返回类型是否为 np.long
assert_type(random_st.geometric(D_arr_0p5), npt.NDArray[np.long])

# 检查 random_st.geometric(D_arr_0p5, size=1) 的返回类型是否为 np.long
assert_type(random_st.geometric(D_arr_0p5, size=1), npt.NDArray[np.long])

# 检查 random_st.geometric(D_arr_like_0p5) 的返回类型是否为 np.long
assert_type(random_st.geometric(D_arr_like_0p5), npt.NDArray[np.long])

# 检查 random_st.geometric(D_arr_like_0p5, size=1) 的返回类型是否为 np.long
assert_type(random_st.geometric(D_arr_like_0p5, size=1), npt.NDArray[np.long])

# 检查 random_st.logseries(0.5) 的返回类型是否为 int
assert_type(random_st.logseries(0.5), int)

# 检查 random_st.logseries(0.5, size=None) 的返回类型是否为 int
assert_type(random_st.logseries(0.5, size=None), int)

# 检查 random_st.logseries(0.5, size=1) 的返回类型是否为 np.long
assert_type(random_st.logseries(0.5, size=1), npt.NDArray[np.long])

# 检查 random_st.logseries(D_arr_0p5) 的返回类型是否为 np.long
assert_type(random_st.logseries(D_arr_0p5), npt.NDArray[np.long])

# 检查 random_st.logseries(D_arr_0p5, size=1) 的返回类型是否为 np.long
assert_type(random_st.logseries(D_arr_0p5, size=1), npt.NDArray[np.long])

# 检查 random_st.logseries(D_arr_like_0p5) 的返回类型是否为 np.long
assert_type(random_st.logseries(D_arr_like_0p5), npt.NDArray[np.long])

# 检查 random_st.logseries(D_arr_like_0p5, size=1) 的返回类型是否为 np.long
assert_type(random_st.logseries(D_arr_like_0p5, size=1), npt.NDArray[np.long])

# 检查 random_st.rayleigh(0.5) 的返回类型是否为 float
assert_type(random_st.rayleigh(0.5), float)

# 检查 random_st.rayleigh(0.5, size=None) 的返回类型是否为 float
assert_type(random_st.rayleigh(0.5, size=None), float)

# 检查 random_st.rayleigh(0.5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.rayleigh(0.5, size=1), npt.NDArray[np.float64])

# 检查 random_st.rayleigh(D_arr_0p5) 的返回类型是否为 np.float64
assert_type(random_st.rayleigh(D_arr_0p5), npt.NDArray[np.float64])

# 检查 random_st.rayleigh(D_arr_0p5, size=1) 的返回类型是否为 np.float64
assert_type(random_st.rayleigh(D_arr_0p5, size=1), npt.NDArray[np.float64])

# 检查 random_st.rayleigh(D_arr_like_0p5) 的返回类型是否为 np.float64
assert_type(random_st.ray
# 使用 assert_type 函数验证 random_st.standard_gamma 函数返回值的类型是否为 float
assert_type(random_st.standard_gamma(0.5, size=None), float)

# 使用 assert_type 函数验证 random_st.standard_gamma 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64
assert_type(random_st.standard_gamma(0.5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.standard_gamma 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_0p5
assert_type(random_st.standard_gamma(D_arr_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.standard_gamma 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_0p5，并指定生成数组的大小为 1
assert_type(random_st.standard_gamma(D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.standard_gamma 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_like_0p5
assert_type(random_st.standard_gamma(D_arr_like_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.standard_gamma 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_like_0p5，并指定生成数组的大小为 1
assert_type(random_st.standard_gamma(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.standard_gamma 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_like_0p5，并指定生成数组的大小为 1
assert_type(random_st.standard_gamma(D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 float
assert_type(random_st.vonmises(0.5, 0.5), float)

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 float
assert_type(random_st.vonmises(0.5, 0.5, size=None), float)

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64
assert_type(random_st.vonmises(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_0p5 和 0.5
assert_type(random_st.vonmises(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 0.5 和 D_arr_0p5
assert_type(random_st.vonmises(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_0p5 和 0.5，并指定生成数组的大小为 1
assert_type(random_st.vonmises(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 0.5 和 D_arr_like_0p5，并指定生成数组的大小为 1
assert_type(random_st.vonmises(0.5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_like_0p5 和 0.5
assert_type(random_st.vonmises(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 0.5 和 D_arr_like_0p5
assert_type(random_st.vonmises(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_0p5 和 D_arr_0p5
assert_type(random_st.vonmises(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_like_0p5 和 D_arr_like_0p5
assert_type(random_st.vonmises(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_0p5 和 D_arr_0p5，并指定生成数组的大小为 1
assert_type(random_st.vonmises(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.vonmises 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_like_0p5 和 D_arr_like_0p5，并指定生成数组的大小为 1
assert_type(random_st.vonmises(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.wald 函数返回值的类型是否为 float
assert_type(random_st.wald(0.5, 0.5), float)

# 使用 assert_type 函数验证 random_st.wald 函数返回值的类型是否为 float
assert_type(random_st.wald(0.5, 0.5, size=None), float)

# 使用 assert_type 函数验证 random_st.wald 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64
assert_type(random_st.wald(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.wald 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_0p5 和 0.5
assert_type(random_st.wald(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.wald 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 0.5 和 D_arr_0p5
assert_type(random_st.wald(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.wald 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_0p5 和 0.5，并指定生成数组的大小为 1
assert_type(random_st.wald(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.wald 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 0.5 和 D_arr_0p5，并指定生成数组的大小为 1
assert_type(random_st.wald(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.wald 函数返回值的类型是否为 numpy 数组，元素类型为 np.float64，
# 输入参数为 D_arr_like_0p5 和 0.5
assert_type(random_st.wald(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 使用 assert_type 函数验证 random_st.wald 函数返回值的类型是否为 numpy 数组，元素
# 确保 random_st.uniform 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.uniform(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])
# 确保 random_st.uniform 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.uniform(0.5, D_arr_like_0p5), npt.NDArray[np.float64])
# 确保 random_st.uniform 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.uniform(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])
# 确保 random_st.uniform 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.uniform(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])
# 确保 random_st.uniform 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.uniform(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 确保 random_st.uniform 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.uniform(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 确保 random_st.beta 返回的值是 float 类型
assert_type(random_st.beta(0.5, 0.5), float)
# 确保 random_st.beta 返回的值是 float 类型
assert_type(random_st.beta(0.5, 0.5, size=None), float)
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(0.5, 0.5, size=1), npt.NDArray[np.float64])
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(D_arr_0p5, 0.5), npt.NDArray[np.float64])
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(0.5, D_arr_0p5), npt.NDArray[np.float64])
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(0.5, D_arr_like_0p5), npt.NDArray[np.float64])
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 确保 random_st.beta 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.beta(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 确保 random_st.f 返回的值是 float 类型
assert_type(random_st.f(0.5, 0.5), float)
# 确保 random_st.f 返回的值是 float 类型
assert_type(random_st.f(0.5, 0.5, size=None), float)
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(0.5, 0.5, size=1), npt.NDArray[np.float64])
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(D_arr_0p5, 0.5), npt.NDArray[np.float64])
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(0.5, D_arr_0p5), npt.NDArray[np.float64])
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(0.5, D_arr_like_0p5), npt.NDArray[np.float64])
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 确保 random_st.f 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.f(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 确保 random_st.gamma 返回的值是 float 类型
assert_type(random_st.gamma(0.5, 0.5), float)
# 确保 random_st.gamma 返回的值是 float 类型
assert_type(random_st.gamma(0.5, 0.5, size=None), float)
# 确保 random_st.gamma 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.gamma(0.5, 0.5, size=1), npt.NDArray[np.float64])
# 确保 random_st.gamma 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.gamma(D_arr_0p5, 0.5), npt.NDArray[np.float64])
# 确保 random_st.gamma 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.gamma(0.5, D_arr_0p5), npt.NDArray[np.float64])
# 确保 random_st.gamma 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.gamma(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])
# 确保 random_st.gamma 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.gamma(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 确保 random_st.gamma 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.gamma(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])
# 确保 random_st.gamma 返回的值是 npt.NDArray[np.float64] 类型
assert_type(random_st.gamma(0.5, D_arr_like_0p
# 验证 random_st.gamma 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gamma(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])
# 验证 random_st.gamma 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gamma(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])
# 验证 random_st.gamma 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gamma(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 验证 random_st.gamma 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gamma(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.gumbel 函数返回的结果类型为 float
assert_type(random_st.gumbel(0.5, 0.5), float)
# 验证 random_st.gumbel 函数返回的结果类型为 float
assert_type(random_st.gumbel(0.5, 0.5, size=None), float)
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(0.5, 0.5, size=1), npt.NDArray[np.float64])
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(D_arr_0p5, 0.5), npt.NDArray[np.float64])
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(0.5, D_arr_0p5), npt.NDArray[np.float64])
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(0.5, D_arr_like_0p5), npt.NDArray[np.float64])
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 验证 random_st.gumbel 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.gumbel(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.laplace 函数返回的结果类型为 float
assert_type(random_st.laplace(0.5, 0.5), float)
# 验证 random_st.laplace 函数返回的结果类型为 float
assert_type(random_st.laplace(0.5, 0.5, size=None), float)
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(0.5, 0.5, size=1), npt.NDArray[np.float64])
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(D_arr_0p5, 0.5), npt.NDArray[np.float64])
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(0.5, D_arr_0p5), npt.NDArray[np.float64])
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(0.5, D_arr_like_0p5), npt.NDArray[np.float64])
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 验证 random_st.laplace 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.laplace(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.logistic 函数返回的结果类型为 float
assert_type(random_st.logistic(0.5, 0.5), float)
# 验证 random_st.logistic 函数返回的结果类型为 float
assert_type(random_st.logistic(0.5, 0.5, size=None), float)
# 验证 random_st.logistic 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.logistic(0.5, 0.5, size=1), npt.NDArray[np.float64])
# 验证 random_st.logistic 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.logistic(D_arr_0p5, 0.5), npt.NDArray[np.float64])
# 验证 random_st.logistic 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.logistic(0.5, D_arr_0p5), npt.NDArray[np.float64])
# 验证 random_st.logistic 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.logistic(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])
# 验证 random_st.logistic 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.logistic(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])
# 验证 random_st.logistic 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.logistic(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])
# 验证 random_st.logistic 函数返回的结果类型为 np.float64 的 NumPy 数组
assert_type(random_st.logistic(0.5, D_arr_like_0p5), npt.NDArray[np.float64])
# 验证 random_st.logistic 函数返回的结果类型为 np.float64
assert_type(random_st.logistic(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 验证 random_st.logistic 函数返回的结果类型为 np.float64
assert_type(random_st.logistic(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 验证 random_st.logistic 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.logistic(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.logistic 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.logistic(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 float
assert_type(random_st.lognormal(0.5, 0.5), float)

# 验证 random_st.lognormal 函数返回的结果类型为 float，且不指定返回数据的尺寸
assert_type(random_st.lognormal(0.5, 0.5, size=None), float)

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.lognormal(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64
assert_type(random_st.lognormal(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64
assert_type(random_st.lognormal(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.lognormal(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.lognormal(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64
assert_type(random_st.lognormal(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64
assert_type(random_st.lognormal(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64
assert_type(random_st.lognormal(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.lognormal(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.lognormal(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.lognormal 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.lognormal(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 float
assert_type(random_st.noncentral_chisquare(0.5, 0.5), float)

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 float，且不指定返回数据的尺寸
assert_type(random_st.noncentral_chisquare(0.5, 0.5, size=None), float)

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.noncentral_chisquare(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64
assert_type(random_st.noncentral_chisquare(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64
assert_type(random_st.noncentral_chisquare(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.noncentral_chisquare(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.noncentral_chisquare(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64
assert_type(random_st.noncentral_chisquare(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64
assert_type(random_st.noncentral_chisquare(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64
assert_type(random_st.noncentral_chisquare(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.noncentral_chisquare(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.noncentral_chisquare 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 验证 random_st.normal 函数返回的结果类型为 float
assert_type(random_st.normal(0.5, 0.5), float)

# 验证 random_st.normal 函数返回的结果类型为 float，且不指定返回数据的尺寸
assert_type(random_st.normal(0.5, 0.5, size=None), float)

# 验证 random_st.normal 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.normal(0.5, 0.5, size=1), npt.NDArray[np.float64])

# 验证 random_st.normal 函数返回的结果类型为 np.float64
assert_type(random_st.normal(D_arr_0p5, 0.5), npt.NDArray[np.float64])

# 验证 random_st.normal 函数返回的结果类型为 np.float64
assert_type(random_st.normal(0.5, D_arr_0p5), npt.NDArray[np.float64])

# 验证 random_st.normal 函数返回的结果类型为 np.float64，且指定返回数据的尺寸为 1
assert_type(random_st.normal(D_arr_0p5, 0.5, size=1), npt.NDArray[np.float64])
# 验证从正态分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.normal(0.5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 验证从正态分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.normal(D_arr_like_0p5, 0.5), npt.NDArray[np.float64])

# 验证从正态分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.normal(0.5, D_arr_like_0p5), npt.NDArray[np.float64])

# 验证从正态分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.normal(D_arr_0p5, D_arr_0p5), npt.NDArray[np.float64])

# 验证从正态分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.normal(D_arr_like_0p5, D_arr_like_0p5), npt.NDArray[np.float64])

# 验证从正态分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.normal(D_arr_0p5, D_arr_0p5, size=1), npt.NDArray[np.float64])

# 验证从正态分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.normal(D_arr_like_0p5, D_arr_like_0p5, size=1), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 float 类型
assert_type(random_st.triangular(0.1, 0.5, 0.9), float)

# 验证从三角分布中随机采样的值是否为 float 类型
assert_type(random_st.triangular(0.1, 0.5, 0.9, size=None), float)

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(0.1, 0.5, 0.9, size=1), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(D_arr_0p1, 0.5, 0.9), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(0.1, D_arr_0p5, 0.9), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(D_arr_0p1, 0.5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(0.1, D_arr_0p5, 0.9, size=1), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(D_arr_like_0p1, 0.5, D_arr_0p9), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(0.5, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(D_arr_0p1, D_arr_0p5, 0.9), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(D_arr_like_0p1, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1), npt.NDArray[np.float64])

# 验证从三角分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.triangular(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 float 类型
assert_type(random_st.noncentral_f(0.1, 0.5, 0.9), float)

# 验证从非中心 F 分布中随机采样的值是否为 float 类型
assert_type(random_st.noncentral_f(0.1, 0.5, 0.9, size=None), float)

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(0.1, 0.5, 0.9, size=1), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(D_arr_0p1, 0.5, 0.9), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(0.1, D_arr_0p5, 0.9), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(D_arr_0p1, 0.5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(0.1, D_arr_0p5, 0.9, size=1), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(D_arr_like_0p1, 0.5, D_arr_0p9), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(0.5, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(D_arr_0p1, D_arr_0p5, 0.9), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, 0.9), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1), npt.NDArray[np.float64])

# 验证从非中心 F 分布中随机采样的值是否为 numpy.float64 类型
assert_type(random_st.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1), npt.NDArray[np.float64])

# 验证从二项分布中随机采样的值是否为 int 类型
assert_type(random_st.binomial(10, 0.5), int)

# 验证从二项分
# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型
assert_type(random_st.binomial(10, 0.5, size=1), npt.NDArray[np.long])

# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_10 和 0.5
assert_type(random_st.binomial(I_arr_10, 0.5), npt.NDArray[np.long])

# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型，传入参数 10 和 D_arr_0p5
assert_type(random_st.binomial(10, D_arr_0p5), npt.NDArray[np.long])

# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_10、0.5 和 size=1
assert_type(random_st.binomial(I_arr_10, 0.5, size=1), npt.NDArray[np.long])

# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型，传入参数 10、D_arr_0p5 和 size=1
assert_type(random_st.binomial(10, D_arr_0p5, size=1), npt.NDArray[np.long])

# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_like_10 和 0.5
assert_type(random_st.binomial(I_arr_like_10, 0.5), npt.NDArray[np.long])

# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型，传入参数 10 和 D_arr_like_0p5
assert_type(random_st.binomial(10, D_arr_like_0p5), npt.NDArray[np.long])

# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_10 和 D_arr_0p5
assert_type(random_st.binomial(I_arr_10, D_arr_0p5), npt.NDArray[np.long])

# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_like_10、D_arr_like_0p5
assert_type(random_st.binomial(I_arr_like_10, D_arr_like_0p5), npt.NDArray[np.long])

# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_10、D_arr_0p5 和 size=1
assert_type(random_st.binomial(I_arr_10, D_arr_0p5, size=1), npt.NDArray[np.long])

# 验证 random_st.binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_like_10、D_arr_like_0p5 和 size=1
assert_type(random_st.binomial(I_arr_like_10, D_arr_like_0p5, size=1), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 int 类型，传入参数 10 和 0.5
assert_type(random_st.negative_binomial(10, 0.5), int)

# 验证 random_st.negative_binomial 函数的返回类型是否为 int 类型，传入参数 10、0.5 和 size=None
assert_type(random_st.negative_binomial(10, 0.5, size=None), int)

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 10、0.5 和 size=1
assert_type(random_st.negative_binomial(10, 0.5, size=1), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_10 和 0.5
assert_type(random_st.negative_binomial(I_arr_10, 0.5), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 10 和 D_arr_0p5
assert_type(random_st.negative_binomial(10, D_arr_0p5), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_10、0.5 和 size=1
assert_type(random_st.negative_binomial(I_arr_10, 0.5, size=1), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 10、D_arr_0p5 和 size=1
assert_type(random_st.negative_binomial(10, D_arr_0p5, size=1), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_like_10 和 0.5
assert_type(random_st.negative_binomial(I_arr_like_10, 0.5), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 10 和 D_arr_like_0p5
assert_type(random_st.negative_binomial(10, D_arr_like_0p5), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_10 和 D_arr_0p5
assert_type(random_st.negative_binomial(I_arr_10, D_arr_0p5), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_like_10 和 D_arr_like_0p5
assert_type(random_st.negative_binomial(I_arr_like_10, D_arr_like_0p5), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_10、D_arr_0p5 和 size=1
assert_type(random_st.negative_binomial(I_arr_10, D_arr_0p5, size=1), npt.NDArray[np.long])

# 验证 random_st.negative_binomial 函数的返回类型是否为 np.long 类型，传入参数 I_arr_like_10、D_arr_like_0p5 和 size=1
assert_type(random_st.negative_binomial(I_arr_like_10, D_arr_like_0p5, size=1), npt.NDArray[np.long])

# 验证 random_st.hypergeometric 函数的返回类型是否为 int 类型，传入参数 20、20 和 10
assert_type(random_st.hypergeometric(20, 20, 10), int)

# 验证 random_st.hypergeometric 函数的返回类型是否为 int 类型，传入参数 20、20、10 和 size=None
assert_type(random_st.hypergeometric(20, 20, 10, size=None), int)

# 验证 random_st.hypergeometric 函数的返回类型是否为 np.long 类型，传入参数 20、20、10 和 size=1
assert_type(random_st.hypergeometric(20, 20, 10, size=1), npt.NDArray[np.long])

# 验证 random_st.hypergeometric 函数的返回类型是否为 np.long 类型，传入参数 I_arr_20、20 和 10
assert_type(random_st.hypergeometric(I_arr_20, 20, 10), npt.NDArray[np.long])

# 验证 random_st.hypergeometric 函数的返回类型是否为 np.long 类型，传入参数 20、I_arr_20 和 10
assert_type(random_st.hypergeometric(20, I_arr_20, 10), npt.NDArray[np.long])

# 验证 random_st.hypergeometric 函数的返回类型是否为 np.long 类型，传入参数 I_arr_20、20、I_arr_like_10 和 size=1
assert_type(random_st.hypergeometric(I_arr_20, 20, I_arr_like_10, size=1), npt.NDArray[np.long])

# 验证 random_st.hypergeometric 函数的返回类型是否为 np.long 类型，传入参数 20、I_arr_20、10 和 size=1
assert_type(random_st.hypergeometric(20, I_arr_20, 10, size=1), npt.NDArray[np.long])

# 验证 random_st.hypergeometric 函数的返回类型是否为 np.long 类型，传入参数 I_arr_like_20、20 和 I_arr_10
assert_type(random_st.hypergeometric(I_arr_like_20, 20, I_arr_10), npt.NDArray[np.long])

# 验证 random_st.hypergeometric 函数的返回类型是否为 np.long 类型
# 确保 hypergeometric 函数返回的类型为 np.long 的 NumPy 数组
assert_type(random_st.hypergeometric(I_arr_like_20, I_arr_like_20, I_arr_like_10, size=1), npt.NDArray[np.long])

# 确保 randint 函数返回的类型为 int
assert_type(random_st.randint(0, 100), int)
assert_type(random_st.randint(100), int)
# 确保 randint 函数返回的类型为包含 np.long 类型元素的 NumPy 数组
assert_type(random_st.randint([100]), npt.NDArray[np.long])
# 确保 randint 函数返回的类型为包含 np.long 类型元素的 NumPy 数组
assert_type(random_st.randint(0, [100]), npt.NDArray[np.long])

# 确保 randint 函数返回的类型为 bool
assert_type(random_st.randint(2, dtype=bool), bool)
assert_type(random_st.randint(0, 2, dtype=bool), bool)
# 确保 randint 函数返回的类型为包含 np.bool 类型元素的 NumPy 数组
assert_type(random_st.randint(I_bool_high_open, dtype=bool), npt.NDArray[np.bool])
assert_type(random_st.randint(I_bool_low, I_bool_high_open, dtype=bool), npt.NDArray[np.bool])
assert_type(random_st.randint(0, I_bool_high_open, dtype=bool), npt.NDArray[np.bool])

# 确保 randint 函数返回的类型为 np.bool
assert_type(random_st.randint(2, dtype=np.bool), np.bool)
assert_type(random_st.randint(0, 2, dtype=np.bool), np.bool)
# 确保 randint 函数返回的类型为包含 np.bool 类型元素的 NumPy 数组
assert_type(random_st.randint(I_bool_high_open, dtype=np.bool), npt.NDArray[np.bool])
assert_type(random_st.randint(I_bool_low, I_bool_high_open, dtype=np.bool), npt.NDArray[np.bool])
assert_type(random_st.randint(0, I_bool_high_open, dtype=np.bool), npt.NDArray[np.bool])

# 确保 randint 函数返回的类型为 np.uint8
assert_type(random_st.randint(256, dtype="u1"), np.uint8)
assert_type(random_st.randint(0, 256, dtype="u1"), np.uint8)
# 确保 randint 函数返回的类型为包含 np.uint8 类型元素的 NumPy 数组
assert_type(random_st.randint(I_u1_high_open, dtype="u1"), npt.NDArray[np.uint8])
assert_type(random_st.randint(I_u1_low, I_u1_high_open, dtype="u1"), npt.NDArray[np.uint8])
assert_type(random_st.randint(0, I_u1_high_open, dtype="u1"), npt.NDArray[np.uint8])

# 确保 randint 函数返回的类型为 np.uint8
assert_type(random_st.randint(256, dtype="uint8"), np.uint8)
assert_type(random_st.randint(0, 256, dtype="uint8"), np.uint8)
# 确保 randint 函数返回的类型为包含 np.uint8 类型元素的 NumPy 数组
assert_type(random_st.randint(I_u1_high_open, dtype="uint8"), npt.NDArray[np.uint8])
assert_type(random_st.randint(I_u1_low, I_u1_high_open, dtype="uint8"), npt.NDArray[np.uint8])
assert_type(random_st.randint(0, I_u1_high_open, dtype="uint8"), npt.NDArray[np.uint8])

# 确保 randint 函数返回的类型为 np.uint8
assert_type(random_st.randint(256, dtype=np.uint8), np.uint8)
assert_type(random_st.randint(0, 256, dtype=np.uint8), np.uint8)
# 确保 randint 函数返回的类型为包含 np.uint8 类型元素的 NumPy 数组
assert_type(random_st.randint(I_u1_high_open, dtype=np.uint8), npt.NDArray[np.uint8])
assert_type(random_st.randint(I_u1_low, I_u1_high_open, dtype=np.uint8), npt.NDArray[np.uint8])
assert_type(random_st.randint(0, I_u1_high_open, dtype=np.uint8), npt.NDArray[np.uint8])

# 确保 randint 函数返回的类型为 np.uint16
assert_type(random_st.randint(65536, dtype="u2"), np.uint16)
assert_type(random_st.randint(0, 65536, dtype="u2"), np.uint16)
# 确保 randint 函数返回的类型为包含 np.uint16 类型元素的 NumPy 数组
assert_type(random_st.randint(I_u2_high_open, dtype="u2"), npt.NDArray[np.uint16])
assert_type(random_st.randint(I_u2_low, I_u2_high_open, dtype="u2"), npt.NDArray[np.uint16])
assert_type(random_st.randint(0, I_u2_high_open, dtype="u2"), npt.NDArray[np.uint16])

# 确保 randint 函数返回的类型为 np.uint16
assert_type(random_st.randint(65536, dtype="uint16"), np.uint16)
assert_type(random_st.randint(0, 65536, dtype="uint16"), np.uint16)
# 确保 randint 函数返回的类型为包含 np.uint16 类型元素的 NumPy 数组
assert_type(random_st.randint(I_u2_high_open, dtype="uint16"), npt.NDArray[np.uint16])
assert_type(random_st.randint(I_u2_low, I_u2_high_open, dtype="uint16"), npt.NDArray[np.uint16])
# 检查随机生成的无符号 16 位整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(0, I_u2_high_open, dtype="uint16"), npt.NDArray[np.uint16])

# 检查随机生成的无符号 16 位整数类型的随机数是否与 numpy 的 np.uint16 类型相匹配
assert_type(random_st.randint(65536, dtype=np.uint16), np.uint16)

# 检查随机生成的无符号 16 位整数类型的随机数是否在指定范围内，并与 numpy 的 np.uint16 类型相匹配
assert_type(random_st.randint(0, 65536, dtype=np.uint16), np.uint16)

# 检查随机生成的无符号 16 位整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(I_u2_high_open, dtype=np.uint16), npt.NDArray[np.uint16])

# 检查随机生成的无符号 16 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(I_u2_low, I_u2_high_open, dtype=np.uint16), npt.NDArray[np.uint16])

# 检查随机生成的无符号 16 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, I_u2_high_open, dtype=np.uint16), npt.NDArray[np.uint16])

# 检查随机生成的无符号 32 位整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(4294967296, dtype="u4"), np.uint32)

# 检查随机生成的无符号 32 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, 4294967296, dtype="u4"), np.uint32)

# 检查随机生成的无符号 32 位整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(I_u4_high_open, dtype="u4"), npt.NDArray[np.uint32])

# 检查随机生成的无符号 32 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(I_u4_low, I_u4_high_open, dtype="u4"), npt.NDArray[np.uint32])

# 检查随机生成的无符号 32 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, I_u4_high_open, dtype="u4"), npt.NDArray[np.uint32])

# 检查随机生成的无符号 32 位整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(4294967296, dtype="uint32"), np.uint32)

# 检查随机生成的无符号 32 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, 4294967296, dtype="uint32"), np.uint32)

# 检查随机生成的无符号 32 位整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(I_u4_high_open, dtype="uint32"), npt.NDArray[np.uint32])

# 检查随机生成的无符号 32 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(I_u4_low, I_u4_high_open, dtype="uint32"), npt.NDArray[np.uint32])

# 检查随机生成的无符号 32 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, I_u4_high_open, dtype="uint32"), npt.NDArray[np.uint32])

# 检查随机生成的无符号 32 位整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(4294967296, dtype=np.uint32), np.uint32)

# 检查随机生成的无符号 32 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, 4294967296, dtype=np.uint32), np.uint32)

# 检查随机生成的无符号 32 位整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(I_u4_high_open, dtype=np.uint32), npt.NDArray[np.uint32])

# 检查随机生成的无符号 32 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(I_u4_low, I_u4_high_open, dtype=np.uint32), npt.NDArray[np.uint32])

# 检查随机生成的无符号 32 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, I_u4_high_open, dtype=np.uint32), npt.NDArray[np.uint32])

# 检查随机生成的无符号整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(4294967296, dtype=np.uint), np.uint)

# 检查随机生成的无符号整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, 4294967296, dtype=np.uint), np.uint)

# 检查随机生成的无符号整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(I_u4_high_open, dtype=np.uint), npt.NDArray[np.uint])

# 检查随机生成的无符号整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(I_u4_low, I_u4_high_open, dtype=np.uint), npt.NDArray[np.uint])

# 检查随机生成的无符号整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, I_u4_high_open, dtype=np.uint), npt.NDArray[np.uint])

# 检查随机生成的无符号 64 位整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(18446744073709551616, dtype="u8"), np.uint64)

# 检查随机生成的无符号 64 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, 18446744073709551616, dtype="u8"), np.uint64)

# 检查随机生成的无符号 64 位整数类型的随机数是否与给定的类型相匹配
assert_type(random_st.randint(I_u8_high_open, dtype="u8"), npt.NDArray[np.uint64])

# 检查随机生成的无符号 64 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(I_u8_low, I_u8_high_open, dtype="u8"), npt.NDArray[np.uint64])

# 检查随机生成的无符号 64 位整数类型的随机数是否在指定范围内，并与给定的类型相匹配
assert_type(random_st.randint(0, I_u8_high_open, dtype="u8"), npt.NDArray[np.uint64])

# 检
# 验证从随机数生成器中生成的无符号64位整数是否为numpy数组中的uint64类型
assert_type(random_st.randint(0, I_u8_high_open, dtype="uint64"), npt.NDArray[np.uint64])

# 验证从随机数生成器中生成的np.uint64类型整数是否为np.uint64类型
assert_type(random_st.randint(18446744073709551616, dtype=np.uint64), np.uint64)

# 验证从随机数生成器中生成的范围在[0, 18446744073709551616)之间的np.uint64类型整数是否为np.uint64类型
assert_type(random_st.randint(0, 18446744073709551616, dtype=np.uint64), np.uint64)

# 验证从随机数生成器中生成的大于I_u8_high_open的np.uint64类型整数是否为numpy数组中的uint64类型
assert_type(random_st.randint(I_u8_high_open, dtype=np.uint64), npt.NDArray[np.uint64])

# 验证从随机数生成器中生成的范围在[I_u8_low, I_u8_high_open)之间的np.uint64类型整数是否为numpy数组中的uint64类型
assert_type(random_st.randint(I_u8_low, I_u8_high_open, dtype=np.uint64), npt.NDArray[np.uint64])

# 验证从随机数生成器中生成的范围在[0, I_u8_high_open)之间的np.uint64类型整数是否为numpy数组中的uint64类型
assert_type(random_st.randint(0, I_u8_high_open, dtype=np.uint64), npt.NDArray[np.uint64])

# 验证从随机数生成器中生成的int8类型整数是否为np.int8类型
assert_type(random_st.randint(128, dtype="i1"), np.int8)

# 验证从随机数生成器中生成的范围在[-128, 128)之间的np.int8类型整数是否为np.int8类型
assert_type(random_st.randint(-128, 128, dtype="i1"), np.int8)

# 验证从随机数生成器中生成的大于I_i1_high_open的np.int8类型整数是否为numpy数组中的int8类型
assert_type(random_st.randint(I_i1_high_open, dtype="i1"), npt.NDArray[np.int8])

# 验证从随机数生成器中生成的范围在[I_i1_low, I_i1_high_open)之间的np.int8类型整数是否为numpy数组中的int8类型
assert_type(random_st.randint(I_i1_low, I_i1_high_open, dtype="i1"), npt.NDArray[np.int8])

# 验证从随机数生成器中生成的范围在[-128, I_i1_high_open)之间的np.int8类型整数是否为numpy数组中的int8类型
assert_type(random_st.randint(-128, I_i1_high_open, dtype="i1"), npt.NDArray[np.int8])

# 验证从随机数生成器中生成的int8类型整数是否为np.int8类型
assert_type(random_st.randint(128, dtype="int8"), np.int8)

# 验证从随机数生成器中生成的范围在[-128, 128)之间的np.int8类型整数是否为np.int8类型
assert_type(random_st.randint(-128, 128, dtype="int8"), np.int8)

# 验证从随机数生成器中生成的大于I_i1_high_open的np.int8类型整数是否为numpy数组中的int8类型
assert_type(random_st.randint(I_i1_high_open, dtype="int8"), npt.NDArray[np.int8])

# 验证从随机数生成器中生成的范围在[I_i1_low, I_i1_high_open)之间的np.int8类型整数是否为numpy数组中的int8类型
assert_type(random_st.randint(I_i1_low, I_i1_high_open, dtype="int8"), npt.NDArray[np.int8])

# 验证从随机数生成器中生成的范围在[-128, I_i1_high_open)之间的np.int8类型整数是否为numpy数组中的int8类型
assert_type(random_st.randint(-128, I_i1_high_open, dtype="int8"), npt.NDArray[np.int8])

# 验证从随机数生成器中生成的int8类型整数是否为np.int8类型
assert_type(random_st.randint(128, dtype=np.int8), np.int8)

# 验证从随机数生成器中生成的范围在[-128, 128)之间的np.int8类型整数是否为np.int8类型
assert_type(random_st.randint(-128, 128, dtype=np.int8), np.int8)

# 验证从随机数生成器中生成的大于I_i1_high_open的np.int8类型整数是否为numpy数组中的int8类型
assert_type(random_st.randint(I_i1_high_open, dtype=np.int8), npt.NDArray[np.int8])

# 验证从随机数生成器中生成的范围在[I_i1_low, I_i1_high_open)之间的np.int8类型整数是否为numpy数组中的int8类型
assert_type(random_st.randint(I_i1_low, I_i1_high_open, dtype=np.int8), npt.NDArray[np.int8])

# 验证从随机数生成器中生成的范围在[-128, I_i1_high_open)之间的np.int8类型整数是否为numpy数组中的int8类型
assert_type(random_st.randint(-128, I_i1_high_open, dtype=np.int8), npt.NDArray[np.int8])

# 验证从随机数生成器中生成的int16类型整数是否为np.int16类型
assert_type(random_st.randint(32768, dtype="i2"), np.int16)

# 验证从随机数生成器中生成的范围在[-32768, 32768)之间的np.int16类型整数是否为np.int16类型
assert_type(random_st.randint(-32768, 32768, dtype="i2"), np.int16)

# 验证从随机数生成器中生成的大于I_i2_high_open的np.int16类型整数是否为numpy数组中的int16类型
assert_type(random_st.randint(I_i2_high_open, dtype="i2"), npt.NDArray[np.int16])

# 验证从随机数生成器中生成的范围在[I_i2_low, I_i2_high_open)之间的np.int16类型整数是否为numpy数组中的int16类型
assert_type(random_st.randint(I_i2_low, I_i2_high_open, dtype="i2"), npt.NDArray[np.int16])

# 验证从随机数生成器中生成的范围在[-32768, I_i2_high_open)之间的np.int16类型整数是否为numpy数组中的int16类型
assert_type(random_st.randint(-32768, I_i2_high_open, dtype="i2"), npt.NDArray[np.int16])

# 验证从随机数生成器中生成的int16类型整数是否为np.int16类型
assert_type(random_st.randint(32768, dtype="int16"), np.int16)

# 验证从随机数生成器中生成的范围在[-32768, 32768)之间的np.int16类型整数是否为np.int16类型
assert_type(random_st.randint(-32768, 32768, dtype="int16"), np.int16)

# 验证从随机数生成器中生成的大于I_i2_high_open的np.int16类型整数是否为numpy数组中的int16类型
assert_type(random_st.randint(I_i2_high_open, dtype="int16"), npt.NDArray[np.int16])

# 验证从随机数生成器中生成的范围在[I_i2_low, I_i2_high_open)之间的np.int16类型整数是否为numpy数组中的int16类型
assert_type(random_st.randint(I_i2_low, I_i2_high_open, dtype="int16"), npt.NDArray[np.int16])

# 验证从随机数生成器中生成的范围在[-32768, I_i2_high_open)之间的np.int16类型整数是否为numpy数组中的int16类型
assert_type(random_st.randint(-32768, I_i2_high_open, dtype="int16"), npt.NDArray[np.int16])

# 验证从随机数生成器中生成的int16类型整数是否为np.int16类型
# 确保生成的随机数类型为 np.int32
assert_type(random_st.randint(-2147483648, 2147483648, dtype="i4"), np.int32)
# 确保生成的随机数类型为 npt.NDArray[np.int32]
assert_type(random_st.randint(I_i4_high_open, dtype="i4"), npt.NDArray[np.int32])
# 确保生成的随机数类型为 npt.NDArray[np.int32]
assert_type(random_st.randint(I_i4_low, I_i4_high_open, dtype="i4"), npt.NDArray[np.int32])
# 确保生成的随机数类型为 npt.NDArray[np.int32]
assert_type(random_st.randint(-2147483648, I_i4_high_open, dtype="i4"), npt.NDArray[np.int32])

# 确保生成的随机数类型为 np.int32
assert_type(random_st.randint(2147483648, dtype="int32"), np.int32)
# 确保生成的随机数类型为 np.int32
assert_type(random_st.randint(-2147483648, 2147483648, dtype="int32"), np.int32)
# 确保生成的随机数类型为 npt.NDArray[np.int32]
assert_type(random_st.randint(I_i4_high_open, dtype="int32"), npt.NDArray[np.int32])
# 确保生成的随机数类型为 npt.NDArray[np.int32]
assert_type(random_st.randint(I_i4_low, I_i4_high_open, dtype="int32"), npt.NDArray[np.int32])
# 确保生成的随机数类型为 npt.NDArray[np.int32]
assert_type(random_st.randint(-2147483648, I_i4_high_open, dtype="int32"), npt.NDArray[np.int32])

# 确保生成的随机数类型为 np.int32
assert_type(random_st.randint(2147483648, dtype=np.int32), np.int32)
# 确保生成的随机数类型为 np.int32
assert_type(random_st.randint(-2147483648, 2147483648, dtype=np.int32), np.int32)
# 确保生成的随机数类型为 npt.NDArray[np.int32]
assert_type(random_st.randint(I_i4_high_open, dtype=np.int32), npt.NDArray[np.int32])
# 确保生成的随机数类型为 npt.NDArray[np.int32]
assert_type(random_st.randint(I_i4_low, I_i4_high_open, dtype=np.int32), npt.NDArray[np.int32])
# 确保生成的随机数类型为 npt.NDArray[np.int32]
assert_type(random_st.randint(-2147483648, I_i4_high_open, dtype=np.int32), npt.NDArray[np.int32])

# 确保生成的随机数类型为 np.int_
assert_type(random_st.randint(9223372036854775808, dtype=np.int_), np.int_)
# 确保生成的随机数类型为 np.int_
assert_type(random_st.randint(-9223372036854775808, 9223372036854775808, dtype=np.int_), np.int_)
# 确保生成的随机数类型为 npt.NDArray[np.int_]
assert_type(random_st.randint(I_i4_high_open, dtype=np.int_), npt.NDArray[np.int_])
# 确保生成的随机数类型为 npt.NDArray[np.int_]
assert_type(random_st.randint(I_i4_low, I_i4_high_open, dtype=np.int_), npt.NDArray[np.int_])
# 确保生成的随机数类型为 npt.NDArray[np.int_]
assert_type(random_st.randint(-2147483648, I_i4_high_open, dtype=np.int_), npt.NDArray[np.int_])

# 确保生成的随机数类型为 np.int64
assert_type(random_st.randint(9223372036854775808, dtype="i8"), np.int64)
# 确保生成的随机数类型为 np.int64
assert_type(random_st.randint(-9223372036854775808, 9223372036854775808, dtype="i8"), np.int64)
# 确保生成的随机数类型为 npt.NDArray[np.int64]
assert_type(random_st.randint(I_i8_high_open, dtype="i8"), npt.NDArray[np.int64])
# 确保生成的随机数类型为 npt.NDArray[np.int64]
assert_type(random_st.randint(I_i8_low, I_i8_high_open, dtype="i8"), npt.NDArray[np.int64])
# 确保生成的随机数类型为 npt.NDArray[np.int64]
assert_type(random_st.randint(-9223372036854775808, I_i8_high_open, dtype="i8"), npt.NDArray[np.int64])

# 确保生成的随机数类型为 np.int64
assert_type(random_st.randint(9223372036854775808, dtype="int64"), np.int64)
# 确保生成的随机数类型为 np.int64
assert_type(random_st.randint(-9223372036854775808, 9223372036854775808, dtype="int64"), np.int64)
# 确保生成的随机数类型为 npt.NDArray[np.int64]
assert_type(random_st.randint(I_i8_high_open, dtype="int64"), npt.NDArray[np.int64])
# 确保生成的随机数类型为 npt.NDArray[np.int64]
assert_type(random_st.randint(I_i8_low, I_i8_high_open, dtype="int64"), npt.NDArray[np.int64])
# 确保生成的随机数类型为 npt.NDArray[np.int64]
assert_type(random_st.randint(-9223372036854775808, I_i8_high_open, dtype="int64"), npt.NDArray[np.int64])

# 确保生成的随机数类型为 np.int64
assert_type(random_st.randint(9223372036854775808, dtype=np.int64), np.int64)
# 确保生成的随机数类型为 np.int64
assert_type(random_st.randint(-9223372036854775808, 9223372036854775808, dtype=np.int64), np.int64)
# 确保生成的随机数类型为 npt.NDArray[np.int64]
assert_type(random_st.randint(I_i8_high_open, dtype=np.int64), npt.NDArray[np.int64])
# 确保生成的随机数类型为 npt.NDArray[np.int64]
assert_type(random_st.randint(I_i8_low, I_i8_high_open, dtype=np.int64), npt.NDArray[np.int64])
# 检查生成的随机整数类型，范围从 -9223372036854775808 到 I_i8_high_open
assert_type(random_st.randint(-9223372036854775808, I_i8_high_open, dtype=np.int64), npt.NDArray[np.int64])

# 检查随机数生成器对象的类型
assert_type(random_st._bit_generator, np.random.BitGenerator)

# 检查生成的随机字节类型
assert_type(random_st.bytes(2), bytes)

# 检查随机选择单个整数的类型
assert_type(random_st.choice(5), int)
# 检查随机选择多个整数的类型
assert_type(random_st.choice(5, 3), npt.NDArray[np.long])
# 检查带有替换的多项式抽样的类型
assert_type(random_st.choice(5, 3, replace=True), npt.NDArray[np.long])
# 检查带有自定义概率分布的多项式抽样的类型
assert_type(random_st.choice(5, 3, p=[1 / 5] * 5), npt.NDArray[np.long])
# 检查不带替换和自定义概率分布的多项式抽样的类型
assert_type(random_st.choice(5, 3, p=[1 / 5] * 5, replace=False), npt.NDArray[np.long])

# 检查随机选择字符串列表中元素的类型
assert_type(random_st.choice(["pooh", "rabbit", "piglet", "Christopher"]), Any)
# 检查随机选择多个字符串列表中元素的类型
assert_type(random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3), npt.NDArray[Any])
# 检查带有自定义概率分布的随机选择多个字符串列表中元素的类型
assert_type(random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, p=[1 / 4] * 4), npt.NDArray[Any])
# 检查带有替换的随机选择多个字符串列表中元素的类型
assert_type(random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=True), npt.NDArray[Any])
# 检查带有自定义概率分布和不带替换的随机选择多个字符串列表中元素的类型
assert_type(random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=False, p=np.array([1 / 8, 1 / 8, 1 / 2, 1 / 4])), npt.NDArray[Any])

# 检查 Dirichlet 分布生成的类型
assert_type(random_st.dirichlet([0.5, 0.5]), npt.NDArray[np.float64])
assert_type(random_st.dirichlet(np.array([0.5, 0.5])), npt.NDArray[np.float64])
assert_type(random_st.dirichlet(np.array([0.5, 0.5]), size=3), npt.NDArray[np.float64])

# 检查多项式分布生成的类型
assert_type(random_st.multinomial(20, [1 / 6.0] * 6), npt.NDArray[np.long])
assert_type(random_st.multinomial(20, np.array([0.5, 0.5])), npt.NDArray[np.long])
assert_type(random_st.multinomial(20, [1 / 6.0] * 6, size=2), npt.NDArray[np.long])

# 检查多变量正态分布生成的类型
assert_type(random_st.multivariate_normal([0.0], [[1.0]]), npt.NDArray[np.float64])
assert_type(random_st.multivariate_normal([0.0], np.array([[1.0]])), npt.NDArray[np.float64])
assert_type(random_st.multivariate_normal(np.array([0.0]), [[1.0]]), npt.NDArray[np.float64])
assert_type(random_st.multivariate_normal([0.0], np.array([[1.0]])), npt.NDArray[np.float64])

# 检查数组元素随机排列生成的类型
assert_type(random_st.permutation(10), npt.NDArray[np.long])
assert_type(random_st.permutation([1, 2, 3, 4]), npt.NDArray[Any])
assert_type(random_st.permutation(np.array([1, 2, 3, 4])), npt.NDArray[Any])
assert_type(random_st.permutation(D_2D), npt.NDArray[Any])

# 检查原地随机打乱数组的类型
assert_type(random_st.shuffle(np.arange(10)), None)
assert_type(random_st.shuffle([1, 2, 3, 4, 5]), None)
assert_type(random_st.shuffle(D_2D), None)

# 检查 RandomState 对象生成的类型
assert_type(np.random.RandomState(pcg64), np.random.RandomState)
assert_type(np.random.RandomState(0), np.random.RandomState)
assert_type(np.random.RandomState([0, 1, 2]), np.random.RandomState)
# 检查对象转换为字符串的类型
assert_type(random_st.__str__(), str)
# 检查对象表示形式的类型
assert_type(random_st.__repr__(), str)
# 检查获取状态字典的类型
random_st_state = random_st.__getstate__()
assert_type(random_st_state, dict[str, Any])
# 检查设置状态的方法返回的类型
assert_type(random_st.__setstate__(random_st_state), None)
# 检查设置种子的方法返回的类型
assert_type(random_st.seed(), None)
assert_type(random_st.seed(1), None)
assert_type(random_st.seed([0, 1]), None)
# 检查获取状态的方法返回的类型
random_st_get_state = random_st.get_state()
assert_type(random_st_state, dict[str, Any])
# 调用 random_st 对象的 get_state 方法，并设置 legacy 参数为 True，返回随机状态的字典或元组
random_st_get_state_legacy = random_st.get_state(legacy=True)

# 断言 random_st_get_state_legacy 的类型为 dict[str, Any] 或 tuple[str, npt.NDArray[np.uint32], int, int, float]
assert_type(random_st_get_state_legacy, dict[str, Any] | tuple[str, npt.NDArray[np.uint32], int, int, float])

# 调用 random_st 对象的 set_state 方法，并传入 random_st_get_state 的返回值作为参数，断言返回值为 None
assert_type(random_st.set_state(random_st_get_state), None)

# 断言 random_st 对象的 rand 方法返回类型为 float
assert_type(random_st.rand(), float)

# 断言 random_st 对象的 rand 方法传入参数 1 后返回类型为 npt.NDArray[np.float64]
assert_type(random_st.rand(1), npt.NDArray[np.float64])

# 断言 random_st 对象的 rand 方法传入参数 (1, 2) 后返回类型为 npt.NDArray[np.float64]
assert_type(random_st.rand(1, 2), npt.NDArray[np.float64])

# 断言 random_st 对象的 randn 方法返回类型为 float
assert_type(random_st.randn(), float)

# 断言 random_st 对象的 randn 方法传入参数 1 后返回类型为 npt.NDArray[np.float64]
assert_type(random_st.randn(1), npt.NDArray[np.float64])

# 断言 random_st 对象的 randn 方法传入参数 (1, 2) 后返回类型为 npt.NDArray[np.float64]
assert_type(random_st.randn(1, 2), npt.NDArray[np.float64])

# 断言 random_st 对象的 random_sample 方法返回类型为 float
assert_type(random_st.random_sample(), float)

# 断言 random_st 对象的 random_sample 方法传入参数 1 后返回类型为 npt.NDArray[np.float64]
assert_type(random_st.random_sample(1), npt.NDArray[np.float64])

# 断言 random_st 对象的 random_sample 方法传入参数 size=(1, 2) 后返回类型为 npt.NDArray[np.float64]
assert_type(random_st.random_sample(size=(1, 2)), npt.NDArray[np.float64])

# 断言 random_st 对象的 tomaxint 方法返回类型为 int
assert_type(random_st.tomaxint(), int)

# 断言 random_st 对象的 tomaxint 方法传入参数 1 后返回类型为 npt.NDArray[np.int64]
assert_type(random_st.tomaxint(1), npt.NDArray[np.int64])

# 断言 random_st 对象的 tomaxint 方法传入参数 (1,) 后返回类型为 npt.NDArray[np.int64]
assert_type(random_st.tomaxint((1,)), npt.NDArray[np.int64])

# 调用 np.random 的 set_bit_generator 方法，传入参数 pcg64，并断言返回值为 None
assert_type(np.random.set_bit_generator(pcg64), None)

# 断言调用 np.random 的 get_bit_generator 方法返回类型为 np.random.BitGenerator
assert_type(np.random.get_bit_generator(), np.random.BitGenerator)
```
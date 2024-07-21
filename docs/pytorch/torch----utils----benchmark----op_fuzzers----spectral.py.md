# `.\pytorch\torch\utils\benchmark\op_fuzzers\spectral.py`

```py
# 引入类型注释允许未定义的函数
# import math 模块用于数学运算
import math

# 引入 PyTorch 库
import torch
# 引入 torch.utils 中的 benchmark 模块
from torch.utils import benchmark
# 从 torch.utils.benchmark 中引入 FuzzedParameter, FuzzedTensor, ParameterAlias 类
from torch.utils.benchmark import FuzzedParameter, FuzzedTensor, ParameterAlias

# __all__ 是一个公共接口的列表，用于控制 from module import * 的导入行为
__all__ = ['SpectralOpFuzzer']

# 定义最小和最大维度尺寸的常量
MIN_DIM_SIZE = 16
MAX_DIM_SIZE = 16 * 1024

# 生成从 1 到给定上限的以给定基数为底的幂的生成器函数
def power_range(upper_bound, base):
    return (base ** i for i in range(int(math.log(upper_bound, base)) + 1))

# 生成正则数列表，这些数仅由质因数 2、3 和 5 组成
# 在 FFT 实现中通常是最快的
REGULAR_SIZES = []
for i in power_range(MAX_DIM_SIZE, 2):
    for j in power_range(MAX_DIM_SIZE // i, 3):
        ij = i * j
        for k in power_range(MAX_DIM_SIZE // ij, 5):
            ijk = ij * k
            if ijk > MIN_DIM_SIZE:
                REGULAR_SIZES.append(ijk)
# 对正则数列表进行排序
REGULAR_SIZES.sort()

# SpectralOpFuzzer 类，继承自 benchmark.Fuzzer 类
class SpectralOpFuzzer(benchmark.Fuzzer):
    def __init__(self, *, seed: int, dtype=torch.float64,
                 cuda: bool = False, probability_regular: float = 1.0):
        super().__init__(
            parameters=[
                # 定义一个模糊参数对象 "ndim"，表示数据的维度（例如1D、2D或3D）
                FuzzedParameter("ndim", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),

                # 生成用于 `x` 的形状参数
                #   测试所有形状是重要的，但是对于FFT特别重要的是常规大小，
                #   因此需要特别关注。这通过生成两个值来完成，
                #   一个从允许的最小到最大整数范围内的分布，另一个从常规数字中
                #   （两个分布都是对数均匀分布）随机选择其中一个。
                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=MIN_DIM_SIZE,
                        maxval=MAX_DIM_SIZE,
                        distribution="loguniform",
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_regular_{i}",
                        distribution={size: 1. / len(REGULAR_SIZES) for size in REGULAR_SIZES}
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        # 分布定义，使用概率正则化参数
                        distribution={
                            ParameterAlias(f"k_regular_{i}"): probability_regular,
                            ParameterAlias(f"k_any_{i}"): 1 - probability_regular,
                        },
                        strict=True,
                    ) for i in range(3)
                ],

                # `x` 的步长参数（基准测试步进内存访问）
                [
                    FuzzedParameter(
                        name=f"step_{i}",
                        distribution={1: 0.8, 2: 0.06, 4: 0.06, 8: 0.04, 16: 0.04},
                    ) for i in range(3)
                ],
            ],
            tensors=[
                # 定义一个模糊的张量对象 "x"
                FuzzedTensor(
                    name="x",
                    size=("k0", "k1", "k2"),  # 使用先前定义的形状参数
                    steps=("step_0", "step_1", "step_2"),  # 使用先前定义的步长参数
                    probability_contiguous=0.75,  # 连续分配内存的概率
                    min_elements=4 * 1024,  # 元素数量下限
                    max_elements=32 * 1024 ** 2,  # 元素数量上限
                    max_allocation_bytes=2 * 1024**3,  # 最大内存分配（2 GB）
                    dim_parameter="ndim",  # 使用先前定义的维度参数
                    dtype=dtype,  # 数据类型
                    cuda=cuda,  # 是否在CUDA上运行
                ),
            ],
            seed=seed,  # 随机数种子
        )
```
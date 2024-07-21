# `.\pytorch\torch\utils\benchmark\op_fuzzers\sparse_unary.py`

```py
# 添加类型检查时允许未注释的函数定义
# mypy: allow-untyped-defs

# 导入 NumPy 库，使用 np 作为别名
import numpy as np

# 导入 PyTorch 库
import torch

# 从 torch.utils.benchmark 模块中导入所需的类和函数
from torch.utils.benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedSparseTensor

# 定义最小维度大小为 16
_MIN_DIM_SIZE = 16
# 定义最大维度大小为 16 * 1024 ** 2
_MAX_DIM_SIZE = 16 * 1024 ** 2

# 创建包含 2 的幂次方维度大小的元组
_POW_TWO_SIZES = tuple(2 ** i for i in range(
    int(np.log2(_MIN_DIM_SIZE)),  # 计算最小维度大小的对数
    int(np.log2(_MAX_DIM_SIZE)) + 1,  # 计算最大维度大小的对数，并加一
))

# 定义一个新类 UnaryOpSparseFuzzer，继承自 Fuzzer 类
class UnaryOpSparseFuzzer(Fuzzer):
    # 初始化函数，用于创建一个对象实例
    def __init__(self, seed, dtype=torch.float32, cuda=False):
        # 调用父类的初始化方法
        super().__init__(
            parameters=[
                # x 的稀疏维度参数。可以是 1D、2D 或 3D。
                FuzzedParameter("dim_parameter", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),
                FuzzedParameter(
                    name="sparse_dim",
                    distribution={1: 0.4, 2: 0.4, 3: 0.2},
                    strict=True
                ),
                # x 的形状参数。
                #   测试所有形状很重要，尤其是二的幂次方，因此特别关注。
                #   通过从允许的最小和最大值之间的所有整数中生成一个值，
                #   以及从二的幂次方中生成另一个值（两个分布都是对数均匀分布），
                #   然后随机选择其中一个来实现这一目的。
                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=_MIN_DIM_SIZE,
                        maxval=_MAX_DIM_SIZE,
                        distribution="loguniform",
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_pow2_{i}",
                        distribution={size: 1. / len(_POW_TWO_SIZES) for size in _POW_TWO_SIZES}
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_any_{i}"): 0.8,
                            ParameterAlias(f"k_pow2_{i}"): 0.2,
                        },
                        strict=True,
                    ) for i in range(3)
                ],
                # 稀疏张量的密度参数
                FuzzedParameter(
                    name="density",
                    distribution={0.1: 0.4, 0.05: 0.3, 0.01: 0.3},
                ),
                # 稀疏张量的合并参数
                FuzzedParameter(
                    name="coalesced",
                    distribution={True: 0.5, False: 0.5},
                ),
                # 随机值参数
                FuzzedParameter(name="random_value", minval=0, maxval=2 ** 32 - 1, distribution="uniform"),
            ],
            tensors=[
                # 创建一个稀疏张量对象 x
                FuzzedSparseTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    dim_parameter="dim_parameter",
                    sparse_dim="sparse_dim",
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    density="density",
                    coalesced="coalesced",
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )
```
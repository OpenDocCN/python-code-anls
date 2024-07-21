# `.\pytorch\benchmarks\operator_benchmark\pt\cat_test.py`

```
# 引入必要的库和模块
import random  # 导入random模块，用于生成随机数
from typing import List  # 导入typing模块中的List类型

import operator_benchmark as op_bench  # 导入operator_benchmark模块，并重命名为op_bench

import torch  # 导入torch模块，用于深度学习框架

"""Microbenchmarks for Cat operator"""

# 配置不同设备上的运行环境
cross_product_configs = {
    "device": ["cpu", "cuda"],  # 包括cpu和cuda两种设备
}

# 针对PT Cat运算符的短期配置
cat_configs_short = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],  # 属性名称为sizes, N, dim
    attrs=[
        [(1, 1, 1), 2, 0],  # noqa: E241  # 第一种配置，大小为(1, 1, 1)，N为2，维度dim为0
        [(512, 512, 2), 2, 1],  # noqa: E241  # 第二种配置，大小为(512, 512, 2)，N为2，维度dim为1
        [(128, 1024, 2), 2, 1],  # noqa: E241  # 第三种配置，大小为(128, 1024, 2)，N为2，维度dim为1
    ],
    cross_product_configs=cross_product_configs,  # 使用之前定义的设备配置
    tags=["short"],  # 添加标签为short
)

# 针对静态运行时特性的配置 - 用于精简模型的快速路径运行时
cat_configs_static_runtime = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],  # 属性名称为sizes, N, dim
    attrs=[
        [[(1, 160), (1, 14)], -1, 1],  # 配置1
        [[(1, 20, 40), (1, 4, 40), (1, 5, 40)], -1, 1],  # 配置2
        [[(1, 580), (1, 174)], -1, 1],  # 配置3
        [[(20, 160), (20, 14)], -1, 1],  # 配置4
        [[(20, 20, 40), (20, 4, 40), (20, 5, 40)], -1, 1],  # 配置5
        [[(20, 580), (20, 174)], -1, 1],  # 配置6
    ],
    cross_product_configs=cross_product_configs,  # 使用之前定义的设备配置
    tags=["static_runtime"],  # 添加标签为static_runtime
)

# 长期配置
cat_configs_long = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],  # 属性名称为sizes, N, dim
    attrs=[
        [(2**10, 2**10, 2), 2, 0],  # noqa: E241  # 第一种配置，大小为(1024, 1024, 2)，N为2，维度dim为0
        [(2**10 + 1, 2**10 - 1, 2), 2, 1],  # noqa: E226,E241  # 第二种配置，大小为(1025, 1023, 2)，N为2，维度dim为1
        [(2**10, 2**10, 2), 2, 2],  # noqa: E241  # 第三种配置，大小为(1024, 1024, 2)，N为2，维度dim为2
        [
            [
                lambda: random.randint(2**6, 2**7),  # 使用lambda函数生成随机数，范围为64到128之间
                2**7 - 17,  # N为111
                2**6 + 1,  # 维度dim为65
            ],  # noqa: E201,E226,E241  # 第四种配置
            5,  # N为5
            0,  # 维度dim为0
        ],
        [
            [
                2**6 + 2**5,  # 大小为96
                lambda: random.randint(2**6, 2**7),  # 使用lambda函数生成随机数，范围为64到128之间
                2**6,  # 维度dim为64
            ],  # noqa: E201,E226,E241,E272  # 第五种配置
            5,  # N为5
            1,  # 维度dim为1
        ],
        [
            [
                2**7,  # 大小为128
                2**6,  # 大小为64
                lambda: random.randint(2**6, 2**7),  # 使用lambda函数生成随机数，范围为64到128之间
            ],  # noqa: E201,E241,E272  # 第六种配置
            5,  # N为5
            2,  # 维度dim为2
        ],
        [[lambda: random.randint(2**5, 2**6), 2**5, 2**6], 50, 0],  # noqa: E241  # 第七种配置，大小为(32到64之间的随机数, 32, 64)，N为50，维度dim为0
        [
            [2**5, lambda: random.randint(2**5, 2**6), 2**6],  # noqa: E241,E272  # 第八种配置，大小为(32, 32到64之间的随机数, 64)，N为50，维度dim为1
            50,  # N为50
            1,  # 维度dim为1
        ],
        [
            [
                2**5 + 1,  # 大小为33
                2**6 + 1,  # 大小为65
                lambda: random.randint(2**5, 2**6),  # 使用lambda函数生成随机数，范围为32到64之间
            ],  # noqa: E226,E241,E272  # 第九种配置，N为50，维度dim为2
            50,  # N为50
            2,  # 维度dim为2
        ],
    ],
    cross_product_configs=cross_product_configs,  # 使用之前定义的设备配置
    tags=["long"],  # 添加标签为long
)

# 在CUDA上，超过4维度的情况有不同的代码路径
cat_configs_multidim = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],  # 属性名称为sizes, N, dim
    attrs=[
        [(2**6, 2**5, 2**2, 2**4, 2**5), 2, 2],  # noqa: E241  # 第一种配置，大小为(64, 32, 4, 16, 32)，N为2，维度dim为2
        [(2**4, 2**5, 2**2, 2**4, 2**5), 8, 2],  # noqa: E241  # 第二种配置，大小为(16, 32, 4, 16, 32)，N为8，维度dim为2
        [
            (2**3 + 1, 2**5 - 1, 2**2 + 1, 2**4 - 1, 2**5 + 1),  # 大小为(9, 31, 5, 15, 33)
            17,  # N为17
            4,  # 维度dim为4
        ],  # noqa: E226,E241  # 第三种配置
    ],
    cross_product_configs=cross_product_configs,  # 使用之前定义的设备配置
    tags=["multidim"],  # 添加标签为multidim
)

# 多输入的配置
cat_configs_manyinputs = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],

    # 定义属性名称列表，包括 "sizes", "N", "dim"
    attrs=[
        # 定义属性列表，每个元素是一个包含三个元素的列表，对应不同的属性配置
        [[lambda: random.randint(1, 10000)], 100, 0],
        [[lambda: random.randint(1, 1000)], 1000, 0],
        [[lambda: random.randint(1, 500)], 2000, 0],
        [[lambda: random.randint(1, 300)], 3000, 0],
    ],

    # 传递预定义的交叉产品配置到参数中
    cross_product_configs=cross_product_configs,

    # 标记此数据集具有 "manyinputs" 标签
    tags=["manyinputs"],
# 导入所需模块
import operator_benchmark as op_bench
import torch
from typing import List

# 创建一个 CatBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class CatBenchmark(op_bench.TorchBenchmarkBase):

    # 初始化方法，设置基准测试的输入数据
    def init(self, sizes, N, dim, device):
        # 固定随机种子以便复现
        random.seed(42)
        inputs = []
        gen_sizes = []

        # 根据传入的 sizes 和 N 来生成输入数据的尺寸
        if type(sizes) == list and N == -1:
            gen_sizes = sizes
        else:
            for i in range(N):
                # 根据 sizes 中的每个元素（可能是函数）生成新的尺寸数据
                gen_sizes.append(
                    [
                        old_size() if callable(old_size) else old_size
                        for old_size in sizes
                    ]
                )

        # 根据生成的尺寸数据创建随机张量作为输入
        for s in gen_sizes:
            inputs.append(torch.rand(s, device=device))

        # 初始化一个空的结果张量，使用给定的设备
        result = torch.empty(0, device=device)
        self.inputs = {"result": result, "inputs": inputs, "dim": dim}
        # 设置当前模块名称为 "cat"
        self.set_module_name("cat")

    # 前向传播方法，执行张量的拼接操作
    def forward(self, result: torch.Tensor, inputs: List[torch.Tensor], dim: int):
        return torch.cat(inputs, dim=dim, out=result)

# 生成 PyTorch 的性能测试，使用不同的配置参数和 CatBenchmark 类
op_bench.generate_pt_test(
    cat_configs_short
    + cat_configs_long
    + cat_configs_multidim
    + cat_configs_manyinputs
    + cat_configs_static_runtime,
    CatBenchmark,
)

# 如果当前脚本作为主程序运行，则执行基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
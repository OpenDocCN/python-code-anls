# `.\pytorch\benchmarks\operator_benchmark\pt\softmax_test.py`

```
# 引入 operator_benchmark 模块，命名为 op_bench
import operator_benchmark as op_bench
# 引入 PyTorch 模块
import torch
import torch.nn as nn


"""
Microbenchmarks for the softmax operators.
"""


# Configs for softmax ops

# 创建一个短配置列表，包含不同的 N, C, H, W 组合和设备类型（CPU 和 CUDA）
softmax_configs_short = op_bench.config_list(
    attr_names=["N", "C", "H", "W"],
    attrs=[
        [1, 3, 256, 256],
        [4, 3, 256, 256],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

# 创建一个长配置列表，包含不同的 N, C, H, W 组合、设备类型和标签（长）
softmax_configs_long = op_bench.cross_product_configs(
    N=[8, 16],
    C=[3],
    H=[256, 512],
    W=[256, 512],
    device=["cpu", "cuda"],
    tags=["long"],
)

# 创建操作列表，包含操作名称和对应的 PyTorch 操作函数
softmax_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Softmax", nn.Softmax],
        ["Softmax2d", nn.Softmax2d],
        ["LogSoftmax", nn.LogSoftmax],
    ],
)

# 创建二维操作列表，包含操作名称和对应的 PyTorch 操作函数（仅限二维）
softmax_two_dims_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Softmax", nn.Softmax],
        ["LogSoftmax", nn.LogSoftmax],
    ],
)

# 创建二维操作的配置列表，包含不同的 M, N, dim 组合、设备类型和标签（长）
softmax_two_dims_configs = op_bench.config_list(
    attr_names=["M", "N", "dim"],
    attrs=[
        [700, 23258, 0],
        [700, 23258, 1],
        [1024, 23258, 1],
        [128, 128, 1],
        [48, 128, 1],
        [16, 1024, 1],
        [32, 1024, 1],
        [48, 1024, 1],
        [16, 512, 1],
        [32, 512, 1],
        [48, 512, 1],
        [16, 256, 1],
        [32, 256, 1],
        [48, 256, 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["long"],
)

# 定义 SoftmaxBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class SoftmaxBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, op_func):
        self.inputs = {"input": torch.rand(N, C, H, W, device=device)}  # 初始化输入数据
        self.op_func = op_func()  # 初始化操作函数实例

    def forward(self, input):
        return self.op_func(input)  # 执行操作函数并返回结果


# 定义 Softmax2DimsBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class Softmax2DimsBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dim, device, op_func):
        self.inputs = {"input": torch.rand(M, N, device=device)}  # 初始化输入数据
        self.op_func = op_func(dim=dim)  # 初始化操作函数实例，设置维度参数

    def forward(self, input):
        return self.op_func(input)  # 执行操作函数并返回结果


# 从操作列表和配置列表生成基于 PyTorch 的性能测试
op_bench.generate_pt_tests_from_op_list(
    softmax_ops_list, softmax_configs_short + softmax_configs_long, SoftmaxBenchmark
)

# 从二维操作列表和二维配置列表生成基于 PyTorch 的性能测试
op_bench.generate_pt_tests_from_op_list(
    softmax_two_dims_ops_list, softmax_two_dims_configs, Softmax2DimsBenchmark
)

# 如果当前文件被直接执行，则运行 operator_benchmark 模块的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
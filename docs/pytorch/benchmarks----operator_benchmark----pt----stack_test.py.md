# `.\pytorch\benchmarks\operator_benchmark\pt\stack_test.py`

```
# 导入必要的库
import random  # 导入随机数生成模块
from typing import List  # 引入类型提示中的列表类型

import operator_benchmark as op_bench  # 导入性能基准测试模块

import torch  # 导入PyTorch深度学习框架

"""Microbenchmarks for Stack operator"""
# 微基准测试堆栈操作符

# 针对静态运行时的PT堆栈操作的配置
stack_configs_static_runtime = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(20, 40), 5],
        [(1, 40), 5],
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(3))},
    tags=["static_runtime"],
)

# 短尺寸的堆栈操作配置
stack_configs_short = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(1, 1, 1), 2],  # noqa: E241
        [(512, 512, 2), 2],  # noqa: E241
        [(128, 1024, 2), 2],  # noqa: E241
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(4))},
    tags=["short"],
)

# 长尺寸的堆栈操作配置
stack_configs_long = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(2**10, 2**10, 2), 2],  # noqa: E241
        [(2**10 + 1, 2**10 - 1, 2), 2],  # noqa: E226,E241
        [(2**10, 2**10, 2), 2],  # noqa: E241
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(4))},
    tags=["long"],
)

# 多维堆栈操作配置，在CUDA上>4维度时有不同的代码路径
stack_configs_multidim = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(2**6, 2**5, 2**2, 2**4, 2**5), 2],  # noqa: E241
        [(2**4, 2**5, 2**2, 2**4, 2**5), 8],  # noqa: E241
        [
            (2**3 + 1, 2**5 - 1, 2**2 + 1, 2**4 - 1, 2**5 + 1),
            17,
        ],  # noqa: E226,E241
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(6))},
    tags=["multidim"],
)


class StackBenchmark(op_bench.TorchBenchmarkBase):
    # 堆栈操作性能基准类继承自op_bench.TorchBenchmarkBase
    def init(self, sizes, N, dim, device):
        random.seed(42)  # 设置随机数种子为42
        inputs = []
        gen_sizes = []
        if type(sizes) == list and N == -1:
            gen_sizes = sizes
        else:
            for i in range(N):
                gen_sizes.append(
                    [
                        old_size() if callable(old_size) else old_size
                        for old_size in sizes
                    ]
                )

        for s in gen_sizes:
            inputs.append(torch.rand(s, device=device))  # 在给定设备上生成随机张量并添加到inputs列表中
        result = torch.rand(gen_sizes[0], device=device)  # 生成一个随机张量作为结果
        self.inputs = {"result": result, "inputs": inputs, "dim": dim}  # 设置输入参数字典
        self.set_module_name("stack")  # 设置模块名称为"stack"

    def forward(self, result: torch.Tensor, inputs: List[torch.Tensor], dim: int):
        # 执行堆栈操作
        return torch.stack(inputs, dim=dim, out=result)


op_bench.generate_pt_test(
    stack_configs_static_runtime
    + stack_configs_short
    + stack_configs_long
    + stack_configs_multidim,
    StackBenchmark,  # 生成PyTorch性能测试
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()  # 运行性能基准测试主程序
```
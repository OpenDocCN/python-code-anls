# `.\pytorch\benchmarks\operator_benchmark\pt\index_select_test.py`

```py
import numpy

import operator_benchmark as op_bench
import torch


"""Microbenchmarks for index_select operator."""

# 定义一个短配置列表，用于测试 index_select 操作符的性能
index_select_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "dim"],
    attrs=[
        [8, 8, 1, 1],
        [256, 512, 1, 1],
        [512, 512, 1, 1],
        [8, 8, 2, 1],
        [256, 512, 2, 1],
        [512, 512, 2, 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],  # 设置标签为 "short"
)


# 定义一个长配置列表，跨越多种参数组合，用于更详细的性能测试
index_select_configs_long = op_bench.cross_product_configs(
    M=[128, 1024],
    N=[128, 1024],
    K=[1, 2],
    dim=[1],
    device=["cpu", "cuda"],
    tags=["long"],  # 设置标签为 "long"
)


class IndexSelectBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, dim, device):
        max_val = N
        numpy.random.seed((1 << 32) - 1)
        index_dim = numpy.random.randint(0, N)
        # 准备测试所需的输入数据
        self.inputs = {
            "input_one": torch.rand(M, N, K, device=device),
            "dim": dim,
            "index": torch.tensor(
                numpy.random.randint(0, max_val, index_dim), device=device
            ),
        }
        self.set_module_name("index_select")  # 设置模块名称为 "index_select"

    def forward(self, input_one, dim, index):
        # 执行 index_select 操作
        return torch.index_select(input_one, dim, index)


# 生成基于配置列表的性能测试
op_bench.generate_pt_test(
    index_select_configs_short + index_select_configs_long, IndexSelectBenchmark
)


if __name__ == "__main__":
    # 运行基准测试
    op_bench.benchmark_runner.main()
```
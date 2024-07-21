# `.\pytorch\benchmarks\operator_benchmark\pt\split_test.py`

```py
# 导入 operator_benchmark 库和 torch 库
import operator_benchmark as op_bench
import torch


"""Microbenchmarks for Split operator"""


# 针对 PT Split 操作符的配置
# 创建一个短配置列表，包括不同的 M、N 和 parts 组合，以及设备为 CPU 或 CUDA
split_configs_short = op_bench.config_list(
    attr_names=["M", "N", "parts"],
    attrs=[
        [8, 8, 2],
        [256, 512, 2],
        [512, 512, 2],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

# 创建一个长配置列表，包括更大的 M、N 和 parts 组合，设备同样为 CPU 或 CUDA
split_configs_long = op_bench.cross_product_configs(
    M=[128, 1024], N=[128, 1024], parts=[2, 4], device=["cpu", "cuda"], tags=["long"]
)


# 定义 SplitBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class SplitBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置输入参数和模块名称
    def init(self, M, N, parts, device):
        self.inputs = {
            "input": torch.rand(M, N, device=device),  # 生成一个随机张量作为输入
            "split_size": int(M * N / parts),  # 计算分割大小
        }
        self.set_module_name("split")  # 设置模块名称为 'split'

    # 前向方法，执行 torch.split 操作
    def forward(self, input, split_size: int):
        return torch.split(input, split_size)


# 生成 PT Split 操作的性能测试用例，包括短配置和长配置
op_bench.generate_pt_test(split_configs_short + split_configs_long, SplitBenchmark)


# 如果作为主程序运行，则执行 operator_benchmark 库中的性能评测主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
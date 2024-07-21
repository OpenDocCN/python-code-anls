# `.\pytorch\benchmarks\operator_benchmark\pt\remainder_test.py`

```
# 导入名为 operator_benchmark 的模块并重命名为 op_bench
import operator_benchmark as op_bench
# 导入 PyTorch 库
import torch


"""Microbenchmarks for remainder operators."""
# 用于求余运算的微基准


# 定义操作列表，包含名称和对应的函数
remainder_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["fmod", torch.fmod],         # 使用 torch.fmod 进行浮点数求余运算
        ["remainder", torch.remainder],   # 使用 torch.remainder 进行元素级求余运算
    ],
)

# 定义短基准测试的配置列表
remainder_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],    # 较小规模的测试配置
        [64, 64, 64],    # 中等规模的测试配置
        [64, 64, 128],   # 中等规模的测试配置
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],    # 在 CPU 和 CUDA 设备上测试
        "dtype": [torch.int32, torch.float, torch.double],   # 使用不同的数据类型进行测试
    },
    tags=["short"],    # 标记为短基准测试
)

# 定义长基准测试的交叉配置
remainder_long_configs = op_bench.cross_product_configs(
    M=[8, 128],    # 不同规模的 M 值
    N=[32, 64],    # 不同规模的 N 值
    K=[256, 512],  # 不同规模的 K 值
    device=["cpu", "cuda"],    # 在 CPU 和 CUDA 设备上测试
    dtype=[torch.int32, torch.float, torch.double],   # 使用不同的数据类型进行测试
    tags=["long"],    # 标记为长基准测试
)


# 定义 RemainderOpBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class RemainderOpBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置测试所需的参数和数据
    def init(self, M, N, K, device, dtype, op_func):
        self.dividend = torch.rand(M, N, K, device=device)   # 随机生成符合设备要求的 dividend 张量
        self.dividend = (self.dividend * 1000 - 500).to(dtype=dtype)   # 调整 dividend 张量的数据类型和数值范围

        self.divisor = torch.rand(M, N, K, device=device)    # 随机生成符合设备要求的 divisor 张量
        # +1 以避免除以零
        self.divisor = (self.divisor * 40 + 1).to(dtype=dtype)   # 调整 divisor 张量的数据类型和数值范围

        self.inputs = {"dividend": self.dividend, "divisor": self.divisor}   # 设置输入字典，包含 dividend 和 divisor 张量

        self.op_func = op_func    # 设置要测试的操作函数

    # 前向方法，执行操作函数并返回结果
    def forward(self, dividend, divisor):
        return self.op_func(dividend, divisor)


# 从操作列表 remainder_ops_list 中生成基准测试，使用短和长基准测试的配置
op_bench.generate_pt_tests_from_op_list(
    remainder_ops_list,
    remainder_short_configs + remainder_long_configs,
    RemainderOpBenchmark,
)


# 如果运行此文件，则执行基准测试运行器的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
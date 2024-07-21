# `.\pytorch\benchmarks\operator_benchmark\pt\binary_test.py`

```py
# 导入 operator_benchmark 模块，并重命名为 op_bench
import operator_benchmark as op_bench
# 导入 torch 模块
import torch

"""Microbenchmarks for binary operators."""
# 用于二元操作的微基准测试

# 定义具有广播功能的二元操作列表
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],  # 添加名为 'add' 的操作，使用 torch.add 函数
    ],
)

# 配置具有广播的测试
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],  # 输入数据的形状配置
    ],
    cross_product_configs={
        "device": ["cpu"],  # 设备选择为 CPU
        "dtype": [torch.float],  # 数据类型选择为 torch.float
    },
    tags=["short"],  # 标签为 'short'
)


class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):
    # 二元操作广播基准测试类继承自 op_bench.TorchBenchmarkBase

    def init(self, in_one, in_two, dtype, device, op_func):
        # 初始化方法，设置输入数据和操作函数
        self.inputs = {
            "in_one": torch.randn(in_one, device=device).to(dtype=dtype),  # 随机生成第一个输入数据
            "in_two": torch.randn(in_two, device=device).to(dtype=dtype),  # 随机生成第二个输入数据
        }
        self.op_func = op_func  # 设置操作函数

    def forward(self, in_one, in_two):
        # 前向方法，执行操作函数
        return self.op_func(in_one, in_two)


# 根据广播的二元操作列表和配置生成 PyTorch 测试
op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)


def copy(in1, in2):
    # 复制函数，用于将 in2 的值复制给 in1
    return in1.copy_(in2)


# 定义不具有广播功能的二元操作列表
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],  # 添加名为 'add' 的操作，使用 torch.add 函数
        ["copy_", copy],  # 添加名为 'copy_' 的操作，使用自定义的 copy 函数
    ],
)

# 短配置列表，包含不同形状的输入数据
binary_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],  # 设备选择为 CPU 或 CUDA
        "dtype_one": [torch.int32],  # 第一个数据类型选择为 torch.int32
        "dtype_two": [torch.int32],  # 第二个数据类型选择为 torch.int32
    },
    tags=["short"],  # 标签为 'short'
)

# 长配置列表，包含更多不同形状和数据类型的输入数据
binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],  # 设备选择为 CPU 或 CUDA
    dtype_one=[torch.int8, torch.int32],  # 第一个数据类型选择为 torch.int8 或 torch.int32
    dtype_two=[torch.int8, torch.int32],  # 第二个数据类型选择为 torch.int8 或 torch.int32
    tags=["long"],  # 标签为 'long'
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    # 二元操作基准测试类继承自 op_bench.TorchBenchmarkBase

    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        # 初始化方法，设置输入数据和操作函数
        self.inputs = {
            "input_one": torch.randn(M, N, K, device=device).to(dtype=dtype_one),  # 随机生成第一个输入数据
            "input_two": torch.randn(M, N, K, device=device).to(dtype=dtype_two),  # 随机生成第二个输入数据
        }
        self.op_func = op_func  # 设置操作函数

    def forward(self, input_one, input_two):
        # 前向方法，执行操作函数
        return self.op_func(input_one, input_two)


# 根据不具有广播功能的二元操作列表和短配置列表生成 PyTorch 测试
op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()  # 如果作为主程序运行，执行基准测试运行器的主函数
```
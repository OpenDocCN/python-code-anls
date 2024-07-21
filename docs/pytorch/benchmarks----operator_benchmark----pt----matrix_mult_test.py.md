# `.\pytorch\benchmarks\operator_benchmark\pt\matrix_mult_test.py`

```py
# 导入 operator_benchmark 模块作为 op_bench
import operator_benchmark as op_bench
# 导入 torch 模块
import torch

"""
使用 einsum 和 torch.bmm 进行批量矩阵乘法的微基准测试。
"""

# 定义短配置列表，包含不同的 B, M, N, K 值组合
batch_mm_configs_short = op_bench.config_list(
    attr_names=["B", "M", "N", "K"],
    attrs=[
        [4, 5, 3, 2],
        [32, 25, 20, 30],
        [128, 100, 120, 110],
    ],
    # 交叉配置设备为 "cpu" 和 "cuda"
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

# 定义长配置列表，包含更大的 B, M, N, K 值组合
batch_mm_configs_long = op_bench.config_list(
    attr_names=["B", "M", "N", "K"],
    attrs=[
        [128, 256, 128, 256],
        [512, 1024, 1024, 512],
    ],
    # 交叉配置设备为 "cpu" 和 "cuda"
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["long"],
)

# 定义操作列表，包含 op_name 和对应的操作函数
batch_mm_op_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["einsum_bmm", torch.einsum],
        ["bmm", torch.bmm],
    ],
)

# 定义批量矩阵乘法基准测试类
class BatchMatrixMultBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, op_func):
        # 初始化输入字典，包含两个随机生成的张量
        self.inputs = {
            "input_one": torch.rand(B, M, N, device=device),
            "input_two": torch.rand(B, N, K, device=device),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        # 根据操作函数选择执行 einsum 或 bmm 操作
        if self.op_func.__name__ == "einsum":
            return torch.einsum("bij,bjk->bik", input_one, input_two)
        else:
            return torch.bmm(input_one, input_two)


"""
使用 einsum 和 torch.mul 进行逐元素矩阵乘法的微基准测试。
"""

# 定义短配置列表，包含不同的 B, M, N 值组合
batch_elementwise_configs_short = op_bench.config_list(
    attr_names=["B", "M", "N"],
    attrs=[
        [4, 5, 3],
        [32, 25, 20],
        [100, 90, 110],
    ],
    # 交叉配置设备为 "cpu" 和 "cuda"
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

# 定义长配置列表，包含更大的 B, M, N 值组合
batch_elementwise_configs_long = op_bench.config_list(
    attr_names=["B", "M", "N"],
    attrs=[
        [128, 128, 128],
        [512, 512, 512],
        [1024, 1024, 1024],
    ],
    # 交叉配置设备为 "cpu" 和 "cuda"
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["long"],
)

# 定义操作列表，包含 op_name 和对应的操作函数
batch_elementwise_op_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["einsum_elementwise", torch.einsum],
        ["mul", torch.mul],
    ],
)

# 定义逐元素矩阵乘法基准测试类
class BatchElementWiseBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, device, op_func):
        # 初始化输入字典，包含两个随机生成的张量
        self.inputs = {
            "input_one": torch.rand(B, M, N, device=device),
            "input_two": torch.rand(B, M, N, device=device),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        # 根据操作函数选择执行 einsum 或 mul 操作
        if self.op_func.__name__ == "einsum":
            return torch.einsum("bij,bij->bij", input_one, input_two)
        else:
            return torch.mul(input_one, input_two)


# 从批量矩阵乘法操作列表和配置列表生成 PyTorch 测试用例
op_bench.generate_pt_tests_from_op_list(
    batch_mm_op_list,
    batch_mm_configs_short + batch_mm_configs_long,
    BatchMatrixMultBenchmark,
)

# 从逐元素矩阵乘法操作列表和配置列表生成 PyTorch 测试用例
op_bench.generate_pt_tests_from_op_list(
    batch_elementwise_op_list,
    batch_elementwise_configs_short + batch_elementwise_configs_long,
    BatchElementWiseBenchmark,
)

# 如果当前脚本作为主程序运行，则执行基准测试运行器
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
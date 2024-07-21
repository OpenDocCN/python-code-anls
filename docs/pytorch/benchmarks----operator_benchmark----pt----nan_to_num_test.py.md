# `.\pytorch\benchmarks\operator_benchmark\pt\nan_to_num_test.py`

```py
# 导入 math 库，用于处理数学运算
import math

# 导入 operator_benchmark 库，用于性能基准测试
import operator_benchmark as op_bench

# 导入 PyTorch 库
import torch


"""Microbenchmarks for torch.nan_to_num / nan_to_num_ operators"""

# PT torch.nan_to_num / nan_to_num_ 操作符的配置

# 定义 nan_to_num 操作符列表，包括操作名和对应的函数
nan_to_num_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["nan_to_num", torch.nan_to_num],
        ["nan_to_num_", torch.nan_to_num_],
    ],
)

# 长尺度配置，用于性能测试，包括不同的尺寸、数据类型和替换无穷大的选项
nan_to_num_long_configs = op_bench.cross_product_configs(
    M=[32, 64, 128],
    N=range(32, 128, 32),
    dtype=[torch.float, torch.double],
    replace_inf=[True, False],
    tags=["long"],
)

# 短尺度配置，用于性能测试，包括较小的尺寸、数据类型和替换无穷大的选项
nan_to_num_short_configs = op_bench.cross_product_configs(
    M=[16, 64],
    N=[64, 64],
    dtype=[torch.float, torch.double],
    replace_inf=[True, False],
    tags=["short"],
)


# ReplaceNaNBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class ReplaceNaNBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dtype, replace_inf, op_func):
        # 初始化函数，生成输入数据
        input = torch.randn(M, N, dtype=dtype)
        # 将第一个元素设置为 NaN
        input[0][0] = float("nan")
        # 设置类的输入数据和替换无穷大标志
        self.inputs = {"input": input, "replace_inf": replace_inf}
        # 设置操作函数
        self.op_func = op_func
        # 设置模块名称为 "nan_to_num"
        self.set_module_name("nan_to_num")

    def forward(self, input, replace_inf: bool):
        # 前向传播函数，执行操作并返回结果

        # 如果替换无穷大为真，则调用操作函数，替换 NaN 为 1.0
        if replace_inf:
            return self.op_func(input, nan=1.0)
        # 否则，调用操作函数，替换 NaN 为 1.0，正无穷大为 math.inf，负无穷大为 -math.inf
        else:
            return self.op_func(input, nan=1.0, posinf=math.inf, neginf=-math.inf)


# 从操作符列表和配置中生成 PyTorch 的性能测试
op_bench.generate_pt_tests_from_op_list(
    nan_to_num_ops_list,
    nan_to_num_long_configs + nan_to_num_short_configs,
    ReplaceNaNBenchmark,
)


if __name__ == "__main__":
    # 如果作为主程序运行，则执行性能基准测试的主函数
    op_bench.benchmark_runner.main()
```
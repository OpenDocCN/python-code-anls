# `.\pytorch\benchmarks\operator_benchmark\pt\hardswish_test.py`

```
# 导入operator_benchmark模块，用于性能基准测试
import operator_benchmark as op_bench
# 导入PyTorch模块
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn

"""
Hardswish运算符的微基准。
"""

# 短配置的Hardswish运算符
hardswish_configs_short = op_bench.config_list(
    attr_names=["N", "C", "H", "W"],  # 定义属性名称
    attrs=[
        [1, 3, 256, 256],  # 第一个属性组合
        [4, 3, 256, 256],  # 第二个属性组合
    ],
    cross_product_configs={
        "device": ["cpu"],  # 跨产品的配置项：设备为CPU
    },
    tags=["short"],  # 标签为short
)

# 长配置的Hardswish运算符
hardswish_configs_long = op_bench.cross_product_configs(
    N=[8, 16], C=[3], H=[256, 512], W=[256, 512], device=["cpu"], tags=["long"]
)

# Hardswish运算符列表
hardswish_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],  # 定义属性名称
    attrs=[
        ["Hardswish", nn.Hardswish],  # 操作名和对应的操作函数
    ],
)

# Hardswish性能基准类
class HardswishBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, op_func):
        self.inputs = {"input_one": torch.rand(N, C, H, W, device=device)}  # 初始化输入张量
        self.op_func = op_func()  # 初始化操作函数对象

    def forward(self, input_one):
        return self.op_func(input_one)  # 执行操作函数

# 从操作列表生成PyTorch的性能测试用例
op_bench.generate_pt_tests_from_op_list(
    hardswish_ops_list,
    hardswish_configs_short + hardswish_configs_long,  # 使用短配置和长配置
    HardswishBenchmark,
)

# 如果当前脚本作为主程序运行，则执行性能基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
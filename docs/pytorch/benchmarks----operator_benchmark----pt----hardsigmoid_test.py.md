# `.\pytorch\benchmarks\operator_benchmark\pt\hardsigmoid_test.py`

```py
# 导入自定义的性能基准模块 `operator_benchmark` 和 PyTorch 库
import operator_benchmark as op_bench
import torch
import torch.nn as nn

"""
hardsigmoid 运算符的微基准。
"""

# 短配置的 hardsigmoid 操作参数列表
hardsigmoid_configs_short = op_bench.config_list(
    # 定义属性名称和属性值的组合
    attr_names=["N", "C", "H", "W"],
    attrs=[
        [1, 3, 256, 256],  # 第一个配置：N=1, C=3, H=256, W=256
        [4, 3, 256, 256],  # 第二个配置：N=4, C=3, H=256, W=256
    ],
    cross_product_configs={
        "device": ["cpu"],  # 跨产品配置：仅在 CPU 上运行
    },
    tags=["short"],  # 标签：短测试
)

# 长配置的 hardsigmoid 操作参数列表
hardsigmoid_configs_long = op_bench.cross_product_configs(
    N=[8, 16],  # N 取值范围为 8 和 16
    C=[3],       # C 固定为 3
    H=[256, 512],  # H 取值范围为 256 和 512
    W=[256, 512],  # W 取值范围为 256 和 512
    device=["cpu"],  # 设备：仅在 CPU 上运行
    tags=["long"]    # 标签：长测试
)

# hardsigmoid 运算符的列表
hardsigmoid_ops_list = op_bench.op_list(
    # 定义属性名称和属性值的组合
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Hardsigmoid", nn.Hardsigmoid],  # op_name: Hardsigmoid, op_func: nn.Hardsigmoid
    ],
)

# HardsigmoidBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class HardsigmoidBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, op_func):
        # 初始化方法，接收参数 N, C, H, W, device, op_func
        self.inputs = {"input_one": torch.rand(N, C, H, W, device=device)}  # 初始化输入张量
        self.op_func = op_func()  # 初始化操作函数实例

    def forward(self, input_one):
        # 前向方法，接收输入 input_one，执行操作函数
        return self.op_func(input_one)

# 从操作列表 hardsigmoid_ops_list 生成 PyTorch 测试用例
op_bench.generate_pt_tests_from_op_list(
    hardsigmoid_ops_list,
    hardsigmoid_configs_short + hardsigmoid_configs_long,  # 合并短和长配置
    HardsigmoidBenchmark,  # 使用 HardsigmoidBenchmark 类进行基准测试
)

# 如果运行为主程序，则执行性能基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
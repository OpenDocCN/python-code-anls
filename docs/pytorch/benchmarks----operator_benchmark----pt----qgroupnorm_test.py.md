# `.\pytorch\benchmarks\operator_benchmark\pt\qgroupnorm_test.py`

```py
# 导入运算基准模块用作操作基准
import operator_benchmark as op_bench
# 导入 PyTorch 库
import torch


# 定义量化组归一化操作的微基准
"""Microbenchmarks for quantized groupnorm operator."""

# 创建一个包含多个配置的交叉产品，用于操作基准
groupnorm_configs_short = op_bench.cross_product_configs(
    # 维度参数的组合
    dims=(
        (32, 8, 16),      # 维度组合 1
        (32, 8, 56, 56),  # 维度组合 2
    ),
    # 分组数量的不同选择
    num_groups=(2, 4),
    # 数据类型为量化整数8位
    dtype=(torch.qint8,),
    # 标记为“short”
    tags=["short"],
)


# 定义量化组归一化的基准类，继承自 TorchBenchmarkBase
class QGroupNormBenchmark(op_bench.TorchBenchmarkBase):

    # 初始化函数，设置输入参数
    def init(self, dims, num_groups, dtype):
        # 创建随机数据张量 X，并量化到指定精度
        X = (torch.rand(*dims) - 0.5) * 256
        num_channels = dims[1]
        scale = 1.0
        zero_point = 0

        # 输入字典，包含量化后的输入张量 qX 和其他参数
        self.inputs = {
            "qX": torch.quantize_per_tensor(
                X, scale=scale, zero_point=zero_point, dtype=dtype
            ),
            "num_groups": num_groups,
            "weight": torch.rand(num_channels, dtype=torch.float),
            "bias": torch.rand(num_channels, dtype=torch.float),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    # 前向函数，执行量化组归一化操作
    def forward(
        self,
        qX,
        num_groups: int,
        weight,
        bias,
        eps: float,
        Y_scale: float,
        Y_zero_point: int,
    ):
        return torch.ops.quantized.group_norm(
            qX,
            num_groups,
            weight=weight,
            bias=bias,
            eps=eps,
            output_scale=Y_scale,
            output_zero_point=Y_zero_point,
        )


# 生成 PyTorch 操作基准测试，并使用之前定义的配置
op_bench.generate_pt_test(groupnorm_configs_short, QGroupNormBenchmark)


# 如果当前脚本作为主程序运行，则执行操作基准测试运行器
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
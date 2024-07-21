# `.\pytorch\benchmarks\operator_benchmark\pt\qlayernorm_test.py`

```
# 导入operator_benchmark模块作为op_bench
import operator_benchmark as op_bench
# 导入torch模块
import torch

# "quantized layernorm operator"的微基准测试

# 定义layernorm_configs_short，包含各种配置的交叉乘积
layernorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (1, 8, 16),
        (8, 8, 16),
        (32, 8, 16),
        (64, 128, 56, 56),
    ),
    dtype=(torch.qint8,),  # 数据类型为torch.qint8
    tags=["short"],  # 标签为"short"
)


# 定义QLayerNormBenchmark类，继承自op_bench.TorchBenchmarkBase
class QLayerNormBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法
    def init(self, dims, dtype):
        # 生成随机张量X，并量化为torch.qint8类型的张量qX
        X = (torch.rand(*dims) - 0.5) * 256
        scale = 1.0
        zero_point = 0
        self.qX = torch.quantize_per_tensor(
            X, scale=scale, zero_point=zero_point, dtype=dtype
        )

        # 初始化输入字典
        self.inputs = {
            "qX": self.qX,
            "weight": torch.rand(*self.qX.size()[1:], dtype=torch.float),
            "bias": torch.rand(*self.qX.size()[1:], dtype=torch.float),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    # 前向方法
    def forward(self, qX, weight, bias, eps: float, Y_scale: float, Y_zero_point: int):
        # 调用torch.ops.quantized.layer_norm进行量化layernorm操作
        return torch.ops.quantized.layer_norm(
            qX,
            qX.size()[1:],  # 传入输入张量的大小作为参数
            weight=weight,  # 权重参数
            bias=bias,      # 偏置参数
            eps=eps,        # epsilon参数
            output_scale=Y_scale,        # 输出的scale
            output_zero_point=Y_zero_point,   # 输出的zero_point
        )


# 生成PyTorch测试，使用layernorm_configs_short和QLayerNormBenchmark
op_bench.generate_pt_test(layernorm_configs_short, QLayerNormBenchmark)

# 如果当前脚本作为主程序运行，则执行benchmark_runner的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
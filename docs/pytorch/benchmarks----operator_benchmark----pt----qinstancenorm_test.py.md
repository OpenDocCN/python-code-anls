# `.\pytorch\benchmarks\operator_benchmark\pt\qinstancenorm_test.py`

```
# 导入名为 operator_benchmark 的模块，作为 op_bench 别名引入
import operator_benchmark as op_bench
# 导入 PyTorch 库
import torch

# 定义 instancenorm_configs_short 变量，包含两个维度组合的配置
instancenorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),        # 维度组合 1
        (32, 8, 56, 56),    # 维度组合 2
    ),
    dtype=(torch.qint8,),  # 数据类型为 torch.qint8
    tags=["short"],         # 标签为 "short"
)

# 定义 QInstanceNormBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class QInstanceNormBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置输入数据和参数
    def init(self, dims, dtype):
        # 创建随机数张量 X，并量化为 dtype 类型
        X = (torch.rand(*dims) - 0.5) * 256
        num_channels = dims[1]
        scale = 1.0
        zero_point = 0

        # 设置输入字典
        self.inputs = {
            "qX": torch.quantize_per_tensor(
                X, scale=scale, zero_point=zero_point, dtype=dtype
            ),
            "weight": torch.rand(num_channels, dtype=torch.float),
            "bias": torch.rand(num_channels, dtype=torch.float),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    # 前向方法，执行量化的实例归一化操作
    def forward(self, qX, weight, bias, eps: float, Y_scale: float, Y_zero_point: int):
        return torch.ops.quantized.instance_norm(
            qX,
            weight=weight,
            bias=bias,
            eps=eps,
            output_scale=Y_scale,
            output_zero_point=Y_zero_point,
        )

# 使用 op_bench 模块生成 PyTorch 的性能测试
op_bench.generate_pt_test(instancenorm_configs_short, QInstanceNormBenchmark)

# 当该脚本作为主程序运行时，执行性能基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
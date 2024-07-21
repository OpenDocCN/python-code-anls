# `.\pytorch\benchmarks\operator_benchmark\pt\linear_prepack_fp16_test.py`

```
# 导入 operator_benchmark 和 torch 库
import operator_benchmark as op_bench
import torch

"""Microbenchmarks for linear_prepack_fp16_ operator. Supports both Caffe2/PyTorch."""

# 针对 PT linear_prepack_fp16 运算符的配置
# 长测试配置，包含多种参数组合
linear_prepack_fp16_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu"], tags=["long"]
)

# 短测试配置，包含特定的参数组合
linear_prepack_fp16_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu"],
    },
    tags=["short"],
)

# 定义 LinearPrepackFP16Benchmark 类，继承自 op_bench.TorchBenchmarkBase
class LinearPrepackFP16Benchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置输入参数和模块名称
    def init(self, M, N, K, device):
        self.inputs = {
            "input_one": torch.rand(
                M, N, K, device=device, requires_grad=False, dtype=torch.float32
            )
        }
        self.set_module_name("linear_prepack_fp16")

    # 前向方法，执行 quantized.linear_prepack_fp16 运算
    def forward(self, input_one):
        return torch.ops.quantized.linear_prepack_fp16(input_one)

# 生成基于 linear_prepack_fp16_short_configs 的测试名称，形如：
# linear_prepack_fp16_M8_N16_K32_devicecpu
op_bench.generate_pt_test(
    linear_prepack_fp16_long_configs + linear_prepack_fp16_short_configs,
    LinearPrepackFP16Benchmark,
)

# 如果该脚本作为主程序运行，则执行 operator benchmark 主程序
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
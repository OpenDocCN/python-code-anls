# `.\pytorch\benchmarks\operator_benchmark\pt\linear_unpack_fp16_test.py`

```py
# 导入名为 operator_benchmark 的模块，命名为 op_bench
# 导入 torch 库
import operator_benchmark as op_bench
import torch

"""Microbenchmarks for linear_unpack_fp16_ operator. Supports both Caffe2/PyTorch."""

# PT linear_unpack_fp16 操作的配置
# 长期配置，包括不同的 M、N、K 值，设备为 CPU，标签为 "long"
linear_unpack_fp16_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu"], tags=["long"]
)

# 短期配置，包括不同的 M、N、K 值，设备为 CPU，标签为 "short"
linear_unpack_fp16_short_configs = op_bench.config_list(
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

# 定义一个名为 LinearUnpackFP16Benchmark 的类，继承自 op_bench.TorchBenchmarkBase
class LinearUnpackFP16Benchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        # 初始化输入字典，其中 "input_one" 键对应的值为通过 quantized.linear_prepack_fp16 操作预打包后的数据
        self.inputs = {
            "input_one": torch.ops.quantized.linear_prepack_fp16(
                torch.rand(
                    M, N, K, device=device, requires_grad=False, dtype=torch.float32
                )
            )
        }
        # 设置模块名称为 "linear_unpack_fp16"
        self.set_module_name("linear_unpack_fp16")

    def forward(self, input_one):
        # 调用 quantized.linear_unpack_fp16 操作，返回其结果
        return torch.ops.quantized.linear_unpack_fp16(input_one)


# 根据 linear_unpack_fp16_short_configs 生成对应的性能测试用例名称
# 生成的测试名称格式为：linear_unpack_fp16_M8_N16_K32_devicecpu
op_bench.generate_pt_test(
    linear_unpack_fp16_long_configs + linear_unpack_fp16_short_configs,
    LinearUnpackFP16Benchmark,
)

# 如果当前脚本被作为主程序运行，则执行 operator_benchmark 的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
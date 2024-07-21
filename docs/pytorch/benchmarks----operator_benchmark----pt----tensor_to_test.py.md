# `.\pytorch\benchmarks\operator_benchmark\pt\tensor_to_test.py`

```py
# 导入 operator_benchmark 库，用作性能测试
import operator_benchmark as op_bench
# 导入 PyTorch 库
import torch

# 创建用于测试的小尺寸张量转换配置列表
tensor_conversion_short_configs = op_bench.cross_product_configs(
    M=(
        8,
        16,
        32,
    ),
    N=(
        16,
        64,
        128,
    ),
    device=["cpu", "cuda"],  # 测试设备包括 CPU 和 CUDA
    tags=["short"],  # 标记为短时测试
)

# 创建用于测试的大尺寸张量转换配置列表
tensor_conversion_long_configs = op_bench.cross_product_configs(
    M=(
        64,
        128,
        256,
        512,
    ),
    N=(
        256,
        512,
        1024,
        2048,
    ),
    device=["cpu", "cuda"],  # 测试设备包括 CPU 和 CUDA
    tags=["long"],  # 标记为长时测试
)

# 定义浮点到半精度张量转换的性能基准测试类
class FloatToHalfTensorConversionBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device):
        # 初始化输入张量字典，随机生成浮点数张量
        self.inputs = {
            "input": torch.rand(
                M, N, device=device, requires_grad=False, dtype=torch.float
            )
        }

    def forward(self, input):
        # 执行张量数据类型转换，将输入张量转换为半精度
        return input.to(torch.half)

# 定义半精度到浮点张量转换的性能基准测试类
class HalfToFloatTensorConversionBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device):
        # 初始化输入张量字典，随机生成半精度张量
        self.inputs = {
            "input": torch.rand(
                M, N, device=device, requires_grad=False, dtype=torch.half
            )
        }

    def forward(self, input):
        # 执行张量数据类型转换，将输入张量转换为单精度
        return input.to(torch.float)

# 使用小尺寸配置生成浮点到半精度张量转换的性能测试
op_bench.generate_pt_test(
    tensor_conversion_short_configs, FloatToHalfTensorConversionBenchmark
)

# 使用大尺寸配置生成浮点到半精度张量转换的性能测试
op_bench.generate_pt_test(
    tensor_conversion_long_configs, FloatToHalfTensorConversionBenchmark
)

# 使用小尺寸配置生成半精度到浮点张量转换的性能测试
op_bench.generate_pt_test(
    tensor_conversion_short_configs, HalfToFloatTensorConversionBenchmark
)

# 使用大尺寸配置生成半精度到浮点张量转换的性能测试
op_bench.generate_pt_test(
    tensor_conversion_long_configs, HalfToFloatTensorConversionBenchmark
)

# 如果该脚本作为主程序运行，则执行基准测试运行器
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
# `.\pytorch\benchmarks\operator_benchmark\pt\clip_ranges_test.py`

```
# 导入 operator_benchmark 库和 torch 库
import operator_benchmark as op_bench
import torch

# 加载名为 sparsenn_operators 的库，用于 C2 ClipRanges 操作符
torch.ops.load_library("//caffe2/torch/fb/sparsenn:sparsenn_operators")

# 针对 C2 ClipRanges 操作符的长配置列表
clip_ranges_long_configs = op_bench.cross_product_configs(
    LENGTH=range(1, 100),
    M=[1],
    N=[2],
    MAX_LENGTH=range(1, 100),
    device=["cpu", "cuda"],
    dtype=[torch.int32],
    tags=["long"],
)

# 针对 C2 ClipRanges 操作符的短配置列表
clip_ranges_short_configs = op_bench.config_list(
    attrs=[
        [6, 1, 2, 1, torch.int32],
        [7, 1, 2, 2, torch.int32],
        [8, 1, 2, 3, torch.int32],
        [9, 1, 2, 4, torch.int32],
        [10, 1, 2, 5, torch.int32],
    ],
    attr_names=["LENGTH", "M", "N", "MAX_LENGTH", "dtype"],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

# 定义 ClipRangesBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class ClipRangesBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置输入参数和模块名
    def init(self, LENGTH, M, N, MAX_LENGTH, device, dtype):
        self.inputs = {
            "input": torch.rand(LENGTH, M, N, device=device).type(dtype),
            "max_length": MAX_LENGTH,
        }
        self.set_module_name("clip_ranges")

    # 前向方法，执行 C2 ClipRanges 操作
    def forward(self, input, max_length: int):
        return torch.ops.fb.clip_ranges(input, max_length)

# 生成 PyTorch 测试用例，结合长配置和短配置，使用 ClipRangesBenchmark 类
op_bench.generate_pt_test(
    clip_ranges_long_configs + clip_ranges_short_configs, ClipRangesBenchmark
)

# 如果作为独立脚本运行，则执行 benchmark_runner 的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
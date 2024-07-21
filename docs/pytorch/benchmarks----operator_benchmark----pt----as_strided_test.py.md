# `.\pytorch\benchmarks\operator_benchmark\pt\as_strided_test.py`

```
# 从 typing 模块导入 List 类型
from typing import List

# 导入 operator_benchmark 库并重命名为 op_bench
import operator_benchmark as op_bench

# 导入 torch 库
import torch


"""Microbenchmarks for as_strided operator"""


# Configs for PT as_strided operator
# 定义短配置列表，包含多个属性：M, N, size, stride, storage_offset
as_strided_configs_short = op_bench.config_list(
    attr_names=["M", "N", "size", "stride", "storage_offset"],
    attrs=[
        [8, 8, (2, 2), (1, 1), 0],
        [256, 256, (32, 32), (1, 1), 0],
        [512, 512, (64, 64), (2, 2), 1],
    ],
    # 交叉生成配置，指定设备为 "cpu" 和 "cuda"
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    # 标记为 "short"
    tags=["short"],
)

# 定义长配置，使用 op_bench.cross_product_configs 生成
as_strided_configs_long = op_bench.cross_product_configs(
    M=[512],
    N=[1024],
    size=[(16, 16), (128, 128)],
    stride=[(1, 1)],
    storage_offset=[0, 1],
    device=["cpu", "cuda"],
    tags=["long"],
)


# 定义 As_stridedBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class As_stridedBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，接受参数 M, N, size, stride, storage_offset, device
    def init(self, M, N, size, stride, storage_offset, device):
        # 初始化输入字典 inputs
        self.inputs = {
            "input_one": torch.rand(M, N, device=device),  # 随机生成 MxN 的张量，使用指定设备
            "size": size,  # 操作的尺寸参数
            "stride": stride,  # 操作的步长参数
            "storage_offset": storage_offset,  # 操作的存储偏移量参数
        }
        # 设置模块名称为 "as_strided"
        self.set_module_name("as_strided")

    # 前向方法，接受参数 input_one, size: List[int], stride: List[int], storage_offset: int
    def forward(
        self, input_one, size: List[int], stride: List[int], storage_offset: int
    ):
        # 调用 torch.as_strided 方法进行操作
        return torch.as_strided(input_one, size, stride, storage_offset)


# 生成 PyTorch 测试，使用 as_strided_configs_short 和 as_strided_configs_long 配置，测试类为 As_stridedBenchmark
op_bench.generate_pt_test(
    as_strided_configs_short + as_strided_configs_long, As_stridedBenchmark
)


# 如果当前脚本作为主程序运行，则执行 operator_benchmark 的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
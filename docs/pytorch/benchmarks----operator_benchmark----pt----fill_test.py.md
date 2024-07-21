# `.\pytorch\benchmarks\operator_benchmark\pt\fill_test.py`

```py
# 导入 operator_benchmark 模块并重命名为 op_bench
import operator_benchmark as op_bench
# 导入 torch 模块
import torch
# 导入 get_all_device_types 函数，用于获取所有设备类型
from torch.testing._internal.common_device_type import get_all_device_types

"""Microbenchmark for Fill_ operator."""
# 定义 fill_short_configs 变量，包含了用于短时运行测试的配置列表
fill_short_configs = op_bench.config_list(
    # 设置属性名为 "N"，对应的属性值为不同的列表
    attr_names=["N"],
    # attrs 包含了多个 N 的取值，用于参数化测试
    attrs=[
        [1],     # N=1
        [1024],  # N=1024
        [2048],  # N=2048
    ],
    # cross_product_configs 用于生成所有可能的组合配置
    cross_product_configs={
        "device": ["cpu", "cuda"],  # 在 cpu 和 cuda 设备上测试
        "dtype": [torch.int32],     # 使用 torch.int32 数据类型
    },
    # 标记这些配置为 "short"
    tags=["short"],
)

# 定义 fill_long_configs 变量，包含了用于长时运行测试的配置列表
fill_long_configs = op_bench.cross_product_configs(
    # N 取值为 10 和 1000，用于参数化测试
    N=[10, 1000],
    # device 使用 get_all_device_types 函数获取所有设备类型进行测试
    device=get_all_device_types(),
    # dtype 包含多种 torch 数据类型，用于测试不同类型的填充操作
    dtype=[
        torch.bool,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.float,
        torch.double,
    ],
    # 标记这些配置为 "long"
    tags=["long"],
)

# 定义 Fill_Benchmark 类，继承自 op_bench.TorchBenchmarkBase
class Fill_Benchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置测试需要的输入参数
    def init(self, N, device, dtype):
        # 初始化输入数据，使用 torch.zeros 创建 N 大小的张量，放在指定设备上，并转换为指定数据类型
        self.inputs = {"input_one": torch.zeros(N, device=device).type(dtype)}
        # 设置测试模块名称为 "fill_"
        self.set_module_name("fill_")

    # 前向方法，执行填充操作并返回结果
    def forward(self, input_one):
        return input_one.fill_(10)

# 生成 PyTorch 测试，使用 fill_short_configs 和 fill_long_configs 进行参数化测试
op_bench.generate_pt_test(fill_short_configs + fill_long_configs, Fill_Benchmark)

# 如果脚本作为主程序运行，则执行 operator_benchmark 的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
# `.\pytorch\benchmarks\operator_benchmark\pt\gather_test.py`

```
# 导入必要的库 numpy
import numpy

# 导入操作基准库 operator_benchmark 的别名为 op_bench
import operator_benchmark as op_bench

# 导入 PyTorch 库
import torch


# 定义 gather 操作的微基准

# 短配置的 gather 操作
gather_configs_short = op_bench.config_list(
    # 定义属性名称和值
    attr_names=["M", "N", "dim"],
    attrs=[
        [256, 512, 0],
        [512, 512, 1],
    ],
    # 交叉组合的配置
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    # 标记为短测试
    tags=["short"],
)

# 长配置的 gather 操作
gather_configs_long = op_bench.cross_product_configs(
    # 定义不同的属性组合
    M=[128, 1024], N=[128, 1024], dim=[0, 1], device=["cpu", "cuda"], tags=["long"]
)


# 定义 GatherBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class GatherBenchmark(op_bench.TorchBenchmarkBase):

    # 初始化方法
    def init(self, M, N, dim, device):
        # 根据维度确定最小值
        min_val = M if dim == 0 else N
        # 设定随机种子
        numpy.random.seed((1 << 32) - 1)
        # 初始化输入数据字典
        self.inputs = {
            "input_one": torch.rand(M, N, device=device),  # 随机生成 MxN 的张量，设备为 device
            "dim": dim,  # 维度参数
            "index": torch.tensor(
                numpy.random.randint(0, min_val, (M, N)), device=device
            ),  # 随机生成大小为 MxN 的索引张量，设备为 device
        }
        # 设置模块名称为 "gather"
        self.set_module_name("gather")

    # 前向方法
    def forward(self, input_one, dim: int, index):
        # 执行 gather 操作
        return torch.gather(input_one, dim, index)


# 生成 PyTorch 的性能测试
op_bench.generate_pt_test(gather_configs_short + gather_configs_long, GatherBenchmark)


# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 运行操作基准测试运行器的主函数
    op_bench.benchmark_runner.main()
```
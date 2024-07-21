# `.\pytorch\benchmarks\tensorexpr\pooling.py`

```
# 从 benchmark 模块中导入 benchmark 类
from . import benchmark

# 定义 PoolingBench 类，继承自 benchmark.Benchmark 类
class PoolingBench(benchmark.Benchmark):
    # 初始化方法，接受多个参数用于配置基准测试
    def __init__(self, case, mode, device, dtype, kernel_size, N, C, H, W):
        # 调用父类的初始化方法，传入 mode 和 device 参数
        super().__init__(mode, device)
        # 设置实例变量 case、kernel_size、N、C、H、W，并初始化 data
        self.case = case
        self.kernel_size = kernel_size
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        # 使用 rand 方法生成随机数据，并存储到 self.data 中
        self.data = self.rand(
            [N, C, H, W], device=device, dtype=dtype, requires_grad=self.requires_grad
        )

    # 前向传播方法
    def forward(self):
        # 根据 case 参数选择不同的池化操作，应用于 self.data
        if self.case == "maxpool":
            y = self.max_pool2d(self.data, self.kernel_size, stride=1)
        elif self.case == "avgpool":
            y = self.avg_pool2d(self.data, self.kernel_size, stride=1)
        return y

    # 返回配置信息的方法
    def config(self):
        return [self.kernel_size, self.N, self.C, self.H, self.W]

    # 计算内存工作量的方法
    def memory_workload(self):
        # 根据 mode 参数选择不同的计算量统计方式
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 1 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (1 + 1) + (2 + 1)

        # 计算缓冲区大小
        buffer_size = self.N * self.C * self.H * self.W
        # 返回计算量估算结果的字典
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    # 静态方法，返回默认配置信息
    @staticmethod
    def default_configs():
        return [[3, 16, 32, 256, 256]]


# 定义 MaxPoolBench 类，继承自 PoolingBench 类
class MaxPoolBench(PoolingBench):
    # 初始化方法，固定 case 参数为 "maxpool"，其余参数传递给父类
    def __init__(self, *args):
        super().__init__("maxpool", *args)

    # 静态方法，返回字符串 "maxpool"
    @staticmethod
    def module():
        return "maxpool"


# 定义 AvgPoolBench 类，继承自 PoolingBench 类
class AvgPoolBench(PoolingBench):
    # 初始化方法，固定 case 参数为 "avgpool"，其余参数传递给父类
    def __init__(self, *args):
        super().__init__("avgpool", *args)

    # 静态方法，返回字符串 "avgpool"
    @staticmethod
    def module():
        return "avgpool"


# 将 MaxPoolBench 类注册到 benchmark 模块中
benchmark.register_benchmark_class(MaxPoolBench)
# 将 AvgPoolBench 类注册到 benchmark 模块中
benchmark.register_benchmark_class(AvgPoolBench)
```
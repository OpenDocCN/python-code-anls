# `.\pytorch\benchmarks\tensorexpr\normalization.py`

```
# 导入 benchmark 和 tensor_engine 模块
from . import benchmark, tensor_engine

# 定义 NormalizationBench 类，继承自 benchmark.Benchmark 类
class NormalizationBench(benchmark.Benchmark):
    # 初始化方法，接受 mode、device、dtype、N、C、H、W 等参数
    def __init__(self, mode, device, dtype, N, C, H, W):
        # 调用父类的初始化方法
        super().__init__(mode, device, dtype)
        # 设置实例变量 N、C、H、W
        self.N = N
        self.C = C
        self.H = H
        self.W = W

        # 使用 nchw_rand 方法生成 NCHW 形状的随机数据，存储在 self.data 中
        self.data = self.nchw_rand(
            [self.N, self.C, self.H, self.W],  # 数据形状
            device=device,  # 使用的设备
            dtype=dtype,    # 数据类型
            requires_grad=self.requires_grad,  # 是否需要梯度
        )
        # 使用 rand 方法生成 C 形状的随机数据，作为 running_mean 和 running_var
        self.running_mean = self.rand([self.C], device=device, dtype=dtype)
        self.running_var = self.rand([self.C], device=device, dtype=dtype)
        # 根据 mode 设置 training 变量
        self.training = self.mode == "both"

    # 返回当前配置的方法，返回 [N, C, H, W] 的列表
    def config(self):
        return [self.N, self.C, self.H, self.W]

    # 计算内存工作量的方法
    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 2 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (2 + 1) + (3 + 1)

        # 计算 buffer_size
        buffer_size = self.N * self.C * self.H * self.W * 4
        # 返回 sol 和 algorithmic 内存工作量的字典
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    # 静态方法，返回默认配置 [[128, 32, 128, 128]]
    @staticmethod
    def default_configs():
        return [[128, 32, 128, 128]]


# 定义 BatchNormBench 类，继承自 NormalizationBench 类
class BatchNormBench(NormalizationBench):
    # 前向传播方法，应用批归一化到 self.data 上，并返回结果 y
    def forward(self):
        y = self.batch_norm(
            self.data, self.running_mean, self.running_var, training=self.training
        )
        return y

    # 静态方法，返回字符串 "batchnorm"
    @staticmethod
    def module():
        return "batchnorm"


# 定义 InstanceNormBench 类，继承自 NormalizationBench 类
class InstanceNormBench(NormalizationBench):
    # 前向传播方法，应用实例归一化到 self.data 上，并返回结果 y
    def forward(self):
        y = self.instance_norm(self.data)
        return y

    # 静态方法，返回字符串 "instance_norm"
    @staticmethod
    def module():
        return "instance_norm"

    # 检查是否支持实例归一化的方法，调用 tensor_engine.is_supported 方法
    def is_supported(self):
        return tensor_engine.is_supported(self.instance_norm)


# 定义 LayerNormBench 类，继承自 NormalizationBench 类
class LayerNormBench(NormalizationBench):
    # 前向传播方法，应用层归一化到 self.data 上，并返回结果 y
    def forward(self):
        y = self.layer_norm(self.data, [self.H, self.W])
        return y

    # 静态方法，返回字符串 "layernorm"
    @staticmethod
    def module():
        return "layernorm"


# 使用 benchmark 模块的 register_benchmark_class 方法注册 BatchNormBench、InstanceNormBench 和 LayerNormBench 类
benchmark.register_benchmark_class(BatchNormBench)
benchmark.register_benchmark_class(InstanceNormBench)
benchmark.register_benchmark_class(LayerNormBench)
```
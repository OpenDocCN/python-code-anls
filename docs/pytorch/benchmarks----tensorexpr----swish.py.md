# `.\pytorch\benchmarks\tensorexpr\swish.py`

```
import torch  # 导入PyTorch库

from . import benchmark  # 从当前包中导入benchmark模块


class SwishBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, M, N):
        super().__init__(mode, device, dtype)  # 调用父类的构造方法
        self.M = M  # 初始化属性 M
        self.N = N  # 初始化属性 N
        self.data = self.rand(  # 生成随机数据张量，并设置为实例属性 data
            [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.data]  # 初始化输入数据列表，包含 self.data
        self.zeros = torch.zeros(M, N, device=device)  # 生成全零张量 self.zeros
        self.six = self.zeros + 6.0  # 创建值为 6.0 的张量 self.six
        self.three = self.zeros + 3.0  # 创建值为 3.0 的张量 self.three
        self.sixth = self.zeros + 1.0 / 6.0  # 创建值为 1/6 的张量 self.sixth

    def forward(self, inp):
        y = inp * (torch.min(torch.relu(inp), self.six) + self.three) * self.sixth  # 执行前向传播计算并返回结果 y
        return y

    def reference(self):
        return self.numpy(self.forward(self.data))  # 返回数据的前向传播结果的 NumPy 格式

    def config(self):
        return [self.M, self.N]  # 返回配置信息，包括 M 和 N 的值

    @staticmethod
    def module():
        return "swish"  # 返回模块名称为 "swish"

    def memory_workload(self):
        if self.mode == "fwd":  # 根据模式设置内存工作负载计算参数
            sol_count = 1 + 1
            algorithmic_count = 3 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (3 + 1) + (3 + 1)

        buffer_size = self.M * self.N  # 计算缓冲区大小
        return {
            "sol": buffer_size * sol_count,  # 返回 SOL 类型内存占用
            "algorithmic": buffer_size * algorithmic_count,  # 返回算法类型内存占用
        }

    @staticmethod
    def default_configs():
        return [[128, 1 << 16]]  # 返回默认配置，包括 M=128, N=65536


benchmark.register_benchmark_class(SwishBench)  # 注册 SwishBench 类到 benchmark 模块中
```
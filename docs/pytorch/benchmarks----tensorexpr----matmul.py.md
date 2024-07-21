# `.\pytorch\benchmarks\tensorexpr\matmul.py`

```
# 导入NumPy库，并使用“np”作为别名
import numpy as np

# 从当前目录下的benchmark模块中导入Benchmark类
from . import benchmark


# 定义MatMulBench类，继承自Benchmark类
class MatMulBench(benchmark.Benchmark):
    
    # 初始化方法，接收mode、device、dtype、B、M、N、K等参数
    def __init__(self, mode, device, dtype, B, M, N, K):
        # 调用父类Benchmark的初始化方法
        super().__init__(mode, device, dtype)
        # 初始化类的属性B、M、N、K
        self.B = B
        self.M = M
        self.N = N
        self.K = K
        # 使用rand方法生成随机数填充的张量d1，形状为[B, M, N]
        self.d1 = self.rand(
            [B, M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 使用rand方法生成随机数填充的张量d2，形状为[B, N, K]
        self.d2 = self.rand(
            [B, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 将d1和d2作为输入列表赋值给类的inputs属性
        self.inputs = [self.d1, self.d2]

    # 前向传播方法，接收d1和d2作为输入，返回它们的矩阵乘积y
    def forward(self, d1, d2):
        y = self.matmul(d1, d2)
        return y

    # 参考方法，使用NumPy的matmul函数计算d1和d2的矩阵乘积，返回结果
    def reference(self):
        return np.matmul(self.numpy(self.d1), self.numpy(self.d2))

    # 配置方法，返回包含B、M、N、K的列表作为配置信息
    def config(self):
        return [self.B, self.M, self.N, self.K]

    # 静态方法，返回字符串"batch_matmul"作为模块名称
    @staticmethod
    def module():
        return "batch_matmul"

    # 内存工作量方法，根据self.mode计算内存使用量
    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = 1 + 1
            algorithmic_count = 1 + (1 + 1)

        # 计算缓冲区大小，分别为两个张量d1和d2的内存大小之和
        buffer_size = (
            self.B * self.M * self.N
            + self.B * self.M * self.N
            + self.B * self.N * self.K
        )
        # 返回内存工作量的字典，包含sol和algorithmic两个键
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    # 计算工作量方法，根据self.mode计算操作次数
    def compute_workload(self):
        if self.mode == "fwd":
            count = 1
        else:
            count = 1 + (1 + 1)

        # 计算乘法操作次数，2乘以张量d1和d2的元素总数B * M * N * K
        op_count = 2 * self.B * self.M * self.N * self.K

        # 返回计算工作量，即操作次数乘以count
        return op_count * count

    # 静态方法，返回默认配置，即包含一个列表[[128, 64, 128, 256]]
    @staticmethod
    def default_configs():
        return [[128, 64, 128, 256]]


# 将MatMulBench类注册到benchmark模块中
benchmark.register_benchmark_class(MatMulBench)
```
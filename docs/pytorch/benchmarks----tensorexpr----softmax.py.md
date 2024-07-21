# `.\pytorch\benchmarks\tensorexpr\softmax.py`

```py
import scipy.special  # 导入科学计算库中的特殊函数模块

from . import benchmark  # 导入当前目录下的 benchmark 模块


class SoftmaxBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, M, N):
        super().__init__(mode, device, dtype)
        self.M = M  # 初始化 M，表示矩阵的行数
        self.N = N  # 初始化 N，表示矩阵的列数
        self.dtype = dtype  # 初始化 dtype，表示数据类型
        self.inputs = [  # 初始化输入数据列表
            self.randn(  # 调用随机生成函数 randn
                [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        ]

    def forward(self, inputs):
        x = self.add(inputs, 0.001)  # 对输入数据 inputs 加上 0.001
        y = self.softmax(x, dim=-1, dtype=self.dtype)  # 对加法结果进行 softmax 处理，dim=-1 表示对最后一个维度进行操作
        return y  # 返回 softmax 处理后的结果

    def reference(self):
        return scipy.special.softmax(self.numpy(self.inputs), axis=-1)  # 返回使用 scipy 中的 softmax 函数处理的 numpy 格式的输入数据

    def config(self):
        return [self.M, self.N]  # 返回配置信息，包括矩阵的行数和列数

    @staticmethod
    def module():
        return "softmax"  # 返回当前模块的名称为 "softmax"

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1  # 前向传播模式下的解法计数
            algorithmic_count = 3 + 1  # 前向传播模式下的算法计数
        else:
            sol_count = (1 + 1) + (1 + 1)  # 反向传播模式下的解法计数
            algorithmic_count = (3 + 1) + (3 + 1)  # 反向传播模式下的算法计数

        buffer_size = self.M * self.N  # 计算缓冲区大小
        return {
            "sol": buffer_size * sol_count,  # 返回解法相关的内存工作量
            "algorithmic": buffer_size * algorithmic_count,  # 返回算法相关的内存工作量
        }

    @staticmethod
    def default_configs():
        return [
            [480, 20],  # 默认配置 1
            [1 << 15, 32],  # 默认配置 2
            [128, 1 << 16],  # 默认配置 3
        ]


benchmark.register_benchmark_class(SoftmaxBench)  # 注册 SoftmaxBench 类到 benchmark 模块中
```
# `.\pytorch\benchmarks\tensorexpr\concat.py`

```
import numpy as np  # 导入 NumPy 库，用于支持数值计算

import torch  # 导入 PyTorch 库，用于深度学习任务

from . import benchmark  # 从当前目录下导入 benchmark 模块


class Concat2D2InputBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, I1_D1, I1_D2, I2_D1, I2_D2, concat_dim):
        super().__init__(mode, device, dtype)
        self.I1_D1 = I1_D1  # 设置第一个输入数据的维度1
        self.I1_D2 = I1_D2  # 设置第一个输入数据的维度2
        self.I2_D1 = I2_D1  # 设置第二个输入数据的维度1
        self.I2_D2 = I2_D2  # 设置第二个输入数据的维度2
        self.concat_dim = concat_dim  # 设置连接的维度
        self.input1 = self.randn(  # 生成指定维度的随机张量作为第一个输入
            [I1_D1, I1_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.input2 = self.randn(  # 生成指定维度的随机张量作为第二个输入
            [I2_D1, I2_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.input1, self.input2]  # 将输入张量组成列表

    def forward(self, input1, input2):
        x1 = self.add(input1, 0.00001)  # 在第一个输入张量上加上一个小值
        x2 = self.add(input2, 0.00001)  # 在第二个输入张量上加上一个小值
        y = self.cat((x1, x2), dim=self.concat_dim)  # 按指定维度连接两个张量
        return y

    def reference(self):
        return np.concatenate(  # 返回 NumPy 数组中按指定维度连接的两个输入张量
            (self.numpy(self.input1), self.numpy(self.input2)),
            axis=self.concat_dim,
        )

    def config(self):
        return [self.I1_D1, self.I1_D2, self.I2_D1, self.I2_D2, self.concat_dim]  # 返回配置信息列表

    @staticmethod
    def module():
        return "concat2d2input"  # 返回模块名称字符串

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1  # 前向传播中的解算操作次数
            algorithmic_count = 3 + 1  # 前向传播中的算法操作次数
        else:
            sol_count = (1 + 1) + (1 + 1)  # 反向传播中的解算操作次数
            algorithmic_count = (3 + 1) + (3 + 1)  # 反向传播中的算法操作次数

        buffer_size = self.I1_D1 * self.I1_D2 + self.I2_D1 * self.I2_D2  # 计算缓冲区大小
        return {
            "sol": buffer_size * sol_count,  # 返回解算操作的内存工作量
            "algorithmic": buffer_size * algorithmic_count,  # 返回算法操作的内存工作量
        }

    @staticmethod
    def default_configs():
        return [
            [1, 160, 1, 14, 1],  # 默认配置1
            [1, 580, 1, 174, 1],  # 默认配置2
            [20, 160, 20, 14, 1],  # 默认配置3
            [20, 580, 20, 174, 1],  # 默认配置4
            [8, 512, 8, 512, 1],  # 默认配置5
            [1 << 13, 1060, 1 << 13, 1040, 1],  # 默认配置6
            [1 << 13, 2000, 1 << 13, 1074, 1],  # 默认配置7
            [1 << 15, 1060, 1 << 15, 2670, 1],  # 默认配置8
            [1 << 15, 5120, 1 << 15, 2512, 1],  # 默认配置9
        ]


benchmark.register_benchmark_class(Concat2D2InputBench)  # 注册基准类到 benchmark 模块中


class ConcatGraphOptBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, I1_D1, I1_D2, I2_D1, I2_D2, concat_dim):
        super().__init__(mode, device, dtype)
        self.I1_D1 = I1_D1  # 设置第一个输入数据的维度1
        self.I1_D2 = I1_D2  # 设置第一个输入数据的维度2
        self.I2_D1 = I2_D1  # 设置第二个输入数据的维度1
        self.I2_D2 = I2_D2  # 设置第二个输入数据的维度2
        self.concat_dim = concat_dim  # 设置连接的维度
        self.input1 = self.randn(  # 生成指定维度的随机张量作为第一个输入
            [I1_D1, I1_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.input2 = self.randn(  # 生成指定维度的随机张量作为第二个输入
            [I2_D1, I2_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.input1, self.input2]  # 将输入张量组成列表
        torch._C._jit_override_can_fuse_on_cpu(True)  # 开启 CPU 下的图优化
        torch._C._jit_cat_wo_conditionals(True)  # 开启不含条件语句的图优化
    # 定义一个方法用于前向传播，接收两个输入参数
    def forward(self, input1, input2):
        # 对输入1和输入2分别加上一个微小的数，以防止除以零的情况
        x1 = self.add(input1, 0.00001)
        x2 = self.add(input2, 0.00001)
        # 将处理后的输入1和输入2在指定维度上进行拼接
        y = self.cat((x1, x2), dim=self.concat_dim)
        # 对拼接后的结果应用ReLU激活函数
        z = self.relu(y)
        # 返回ReLU激活后的结果
        return z

    # 定义一个方法用于生成参考输出，将两个输入数组在指定维度上进行连接
    def reference(self):
        return np.concatenate(
            (self.numpy(self.input1), self.numpy(self.input2)),
            axis=self.concat_dim,
        )

    # 定义一个方法返回当前对象的配置信息，包括各种输入维度和拼接维度
    def config(self):
        return [self.I1_D1, self.I1_D2, self.I2_D1, self.I2_D2, self.concat_dim]

    # 静态方法，返回模块名称字符串
    @staticmethod
    def module():
        return "concatGraphOpt"

    # 定义一个方法用于计算内存工作量，根据模式（前向或反向）计算缓冲区大小和操作数的总数
    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1  # 解算计数
            algorithmic_count = 3 + 1  # 算法计数
        else:
            sol_count = (1 + 1) + (1 + 1)  # 解算计数（反向）
            algorithmic_count = (3 + 1) + (3 + 1)  # 算法计数（反向）

        # 计算缓冲区大小，根据输入维度计算
        buffer_size = self.I1_D1 * self.I1_D2 + self.I2_D1 * self.I2_D2
        # 返回内存工作量的字典，包括解算和算法计数乘以缓冲区大小
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    # 静态方法，返回默认的配置列表
    @staticmethod
    def default_configs():
        return [
            [1 << 13, 1060, 1 << 13, 1040, 1],  # 配置1
            [1 << 13, 2000, 1 << 13, 1074, 1],  # 配置2
            [1 << 15, 1060, 1 << 15, 2670, 1],  # 配置3
            [1 << 15, 5120, 1 << 15, 2512, 1],  # 配置4
        ]
# 注册 ConcatGraphOptBench 类到 benchmark 中，使其可被 benchmark 系统识别和调用
benchmark.register_benchmark_class(ConcatGraphOptBench)
```
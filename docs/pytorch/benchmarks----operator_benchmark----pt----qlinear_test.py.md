# `.\pytorch\benchmarks\operator_benchmark\pt\qlinear_test.py`

```py
# 导入 pt 模块中的 configs 模块
from pt import configs

# 导入 operator_benchmark 模块并命名为 op_bench
import operator_benchmark as op_bench
# 导入 torch 模块
import torch
# 导入 torch.quantized 中的 nnq 模块
import torch.ao.nn.quantized as nnq
# 导入 torch.quantized.dynamic 中的 nnqd 模块
import torch.ao.nn.quantized.dynamic as nnqd

"""
Quantized Linear 操作的微基准。
"""


class _QLinearBenchmarkBase(op_bench.TorchBenchmarkBase):
    def init(self, N, IN, OUT, linear_under_test):
        # 设置量化的比例尺和零点
        scale = torch.tensor(1.0 / 255)
        zero_point = torch.tensor(0)
        # 生成随机输入张量 X，数据类型为 float32
        self.X = torch.randn(N, IN, dtype=torch.float32)
        # 对输入张量 X 进行量化
        self.qX = torch.quantize_per_tensor(
            self.X, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )
        # 生成随机权重张量 W，数据类型为 float32
        W = torch.randn(OUT, IN, dtype=torch.float32)
        # 对权重张量 W 进行量化
        qW = torch.quantize_per_tensor(W, scale=scale, zero_point=0, dtype=torch.qint8)

        # 假设子类中设置了 self.qlinear
        self.qlinear = linear_under_test
        self.qlinear.weight = qW
        self.qlinear.scale = scale
        self.qlinear.zero_point = zero_point

    def forward(self, input):
        # 假设子类中设置了 self.input
        return self.qlinear(input)


class QLinearBenchmark(_QLinearBenchmarkBase):
    def init(self, N, IN, OUT, device):
        super().init(N, IN, OUT, nnq.Linear(IN, OUT))
        self.inputs = {"input": self.qX}
        self.set_module_name("QLinear")


class QDynamicLinearBenchmark(_QLinearBenchmarkBase):
    def init(self, N, IN, OUT, device):
        super().init(N, IN, OUT, nnqd.Linear(IN, OUT))
        self.inputs = {"input": self.X}
        self.set_module_name("QDynamicLinear")


# 生成 QLinearBenchmark 的基准测试
op_bench.generate_pt_test(
    configs.remove_cuda(configs.linear_configs_short + configs.linear_configs_long),
    QLinearBenchmark,
)

# 生成 QDynamicLinearBenchmark 的基准测试
op_bench.generate_pt_test(
    configs.remove_cuda(configs.linear_configs_short + configs.linear_configs_long),
    QDynamicLinearBenchmark,
)

# 如果作为主程序运行，则执行基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```
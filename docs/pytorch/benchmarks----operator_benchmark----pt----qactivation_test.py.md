# `.\pytorch\benchmarks\operator_benchmark\pt\qactivation_test.py`

```py
import operator_benchmark as op_bench
import torch
import torch.ao.nn.quantized.functional as qF

r"""Microbenchmarks for the quantized activations."""

# 创建长配置和短配置的交叉产品，用于量化激活函数的基准测试
qactivation_long_configs = op_bench.cross_product_configs(
    dims=(
        (64, 224, 224),    # VGG-16 relu的原始形状：(-1, 3, 224, 224) 的 ReLU-1
        (128, 112, 112),   # ReLU-6
        (256, 56, 56),     # ReLU-11
        (512, 28, 28),     # ReLU-18
        (512, 14, 14),     # ReLU-25
        (16, 64, 224, 224),# Batch = 16 的 ReLU-1
        (16, 128, 112, 112),# ReLU-6
        (16, 256, 56, 56),  # ReLU-11
        (16, 512, 28, 28),  # ReLU-18
        (16, 512, 14, 14),  # ReLU-25
    ),
    contig=(False, True),  # 是否连续存储的标志
    inplace=(False, True), # 是否原地操作的标志
    dtype=(torch.quint8,), # 数据类型为量化的无符号整数8位
    tags=("long",),        # 标签为"long"
)

# 创建短配置的交叉产品，用于量化激活函数的基准测试
qactivation_short_configs = op_bench.cross_product_configs(
    dims=(
        (3, 4, 5),         # Rank=3
        (2, 3, 4, 5),      # Rank=4
        (512, 512),        # 浮点基准中的维度
        (256, 1024),       # 浮点基准中的维度
    ),
    contig=(False,),       # 是否连续存储的标志
    inplace=(False,),      # 是否原地操作的标志
    dtype=(torch.quint8, torch.qint8, torch.qint32),  # 数据类型可以是量化的无符号整数8位、有符号整数8位或有符号整数32位
    tags=("short",),       # 标签为"short"
)

# 定义量化激活函数的操作列表
qactivation_ops = op_bench.op_list(
    attrs=(
        ("relu", torch.nn.ReLU()),                            # ReLU 激活函数
        ("relu6", torch.ops.quantized.relu6),                 # 量化 ReLU6 激活函数
        ("functional.hardtanh", qF.hardtanh),                 # 量化硬切线激活函数
        ("functional.hardsigmoid", qF.hardsigmoid),           # 量化硬 sigmoid 激活函数
        ("functional.leaky_relu", qF.leaky_relu),             # 量化泄漏 ReLU 激活函数
        ("functional.sigmoid", torch.nn.functional.sigmoid),  # sigmoid 激活函数
        ("functional.tanh", torch.nn.functional.tanh),        # tanh 激活函数
    ),
    attr_names=("op_name", "op_func"),  # 属性名为操作名称和操作函数
)

# 定义量化激活函数基准测试的基类
class QActivationBenchmarkBase(op_bench.TorchBenchmarkBase):
    r"""Base class for all the activations."""

    def _setup(self, dims, contig, dtype):
        # 生成随机输入数据
        f_input = (torch.rand(*dims) - 0.5) * 256
        self.scale = 1.0
        self.zero_point = 0

        # 对张量进行量化
        q_input = torch.quantize_per_tensor(
            f_input, scale=self.scale, zero_point=self.zero_point, dtype=dtype
        )
        if not contig:
            # 如果非连续存储，则进行维度排列变换
            new_shape = list(range(q_input.ndim))[::-1]
            q_input = q_input.permute(new_shape)

        self.inputs = {"q_input": q_input}

    def init(self, dims, contig, inplace, dtype, op_func):
        self._setup(dims, contig, dtype)
        self.qop = op_func

# 定义量化激活函数基准测试类
class QActivationBenchmark(QActivationBenchmarkBase):
    def forward(self, q_input):
        return self.qop(q_input)

# 从操作列表和配置中生成基于 PyTorch 的测试
op_bench.generate_pt_tests_from_op_list(
    qactivation_ops,
    qactivation_short_configs + qactivation_long_configs,
    QActivationBenchmark,
)

# 定义量化激活函数的缩放和零点操作列表
qactivation_scale_zero_point_ops = op_bench.op_list(
    attrs=(
        ("functional.hardswish", qF.hardswish),  # 量化硬 swish 激活函数
        ("functional.elu", qF.elu),              # 量化 ELU 激活函数
        ("functional.celu", qF.celu),            # 量化 CELU 激活函数
    ),
    attr_names=("op_name", "op_func"),           # 属性名为操作名称和操作函数
)
class QActivationScaleZeroPointBenchmark(QActivationBenchmarkBase):
    # 定义一个基于量化激活函数的性能基准类 QActivationBenchmarkBase 的子类
    def forward(self, q_input):
        # 重写父类方法 forward，用于执行量化操作，并传入 scale 和 zero_point 参数
        return self.qop(q_input, scale=self.scale, zero_point=self.zero_point)


op_bench.generate_pt_tests_from_op_list(
    qactivation_scale_zero_point_ops,
    qactivation_short_configs + qactivation_long_configs,
    QActivationScaleZeroPointBenchmark,
)
# 从给定的操作列表 qactivation_scale_zero_point_ops 中生成 PyTorch 的性能测试用例，
# 使用 qactivation_short_configs 和 qactivation_long_configs 的配置
# 并使用 QActivationScaleZeroPointBenchmark 类来实例化每个测试用例

if __name__ == "__main__":
    # 如果当前脚本被直接执行
    op_bench.benchmark_runner.main()
    # 运行性能基准测试的主函数
```
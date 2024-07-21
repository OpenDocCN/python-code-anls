# `.\pytorch\benchmarks\operator_benchmark\pt\quantization_test.py`

```py
# 导入操作基准测试库
import operator_benchmark as op_bench
# 导入 PyTorch 库
import torch
# 导入量化神经网络模块
import torch.ao.nn.quantized as nnq
# 导入量化模块
import torch.ao.quantization as tq
# 导入神经网络模块
import torch.nn as nn

"""Microbenchmarks for general quantization operations."""
# 通用量化操作的微基准测试

# mode 参数用于指示基准测试的方向：
# 如果是 'Q'，则是量化基准测试，否则是反量化基准测试

# 短配置的量化字典
quantize_configs_short_dict = {
    "attr_names": ["C", "M", "N", "dtype", "mode"],
    "attrs": [
        [3, 512, 512, torch.quint8, "Q"],
        [3, 512, 512, torch.quint8, "D"],
    ],
    "tags": ["short"],
}

# 长配置的量化字典
quantize_configs_long_dict = {
    "C": [3, 5, 8],  # 用于通道级别量化：避免单通道测试
    "M": [256, 1024],
    "N": [256, 1024],
    "dtype": [torch.quint8, torch.qint8, torch.qint32],
    "mode": ["D", "Q"],
    "tags": ["long"],
}

# 短配置的张量级别量化配置列表
quantize_per_tensor_configs_short = op_bench.config_list(**quantize_configs_short_dict)

# 长配置的张量级别量化配置列表
quantize_per_tensor_configs_long = op_bench.cross_product_configs(
    **quantize_configs_long_dict
)

# 定义量化和反量化基准类
class QuantizePerTensorBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks both quantization and dequantization."""

    def init(self, C, M, N, dtype, mode):
        assert mode in ("Q", "D")
        # 创建随机张量作为输入
        self.input = torch.rand(C, M, N)
        self.dtype = dtype
        # 初始化量化操作对象
        self.op = nnq.Quantize(scale=1.0, zero_point=0, dtype=dtype)
        self.set_module_name("QuantizePerTensor")

        # 如果 mode 是 "D"，则进行反量化操作
        if mode == "D":
            self.input = self.op(self.input)
            self.op = nnq.DeQuantize()
            self.set_module_name("DequantizePerTensor")

        # 准备输入字典
        self.inputs = {"input": self.input}

    def forward(self, input):
        return self.op(input)

# 生成张量级别量化基准测试
op_bench.generate_pt_test(
    quantize_per_tensor_configs_short + quantize_per_tensor_configs_long,
    QuantizePerTensorBenchmark,
)

# === Per Channel quantization ===

# 短配置的通道级别量化配置列表
quantize_per_channel_configs_short = op_bench.config_list(
    cross_product_configs={"axis": (0,)}, **quantize_configs_short_dict
)

# 长配置的通道级别量化配置列表
quantize_per_channel_configs_long = op_bench.cross_product_configs(
    axis=(0, 1, 2), **quantize_configs_long_dict
)

# 定义通道级别量化基准类
class QuantizePerChannelBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks both quantization and dequantization."""
    # 初始化函数，设置量化相关参数和模式
    def init(self, C, M, N, dtype, axis, mode):
        # 断言模式为 "Q" 或 "D"
        assert mode in ("Q", "D")
        # 创建一个 CxMxN 的随机张量作为输入
        self.input = torch.rand(C, M, N)
        # 设置操作为按通道量化
        self.op = torch.quantize_per_channel

        # 根据指定的 axis 确定通道长度
        channel_len = (C, M, N)[axis]

        # 设置量化相关的参数字典
        self.kwargs = {
            "scales": torch.tensor([1.0] * channel_len),  # 所有通道的量化比例因子为1.0
            "zero_points": torch.tensor([0] * channel_len),  # 所有通道的零点为0
            "dtype": dtype,  # 数据类型
            "axis": axis,  # 量化的轴
        }

        # 设置模块名称为 "QuantizePerChannel"
        self.set_module_name("QuantizePerChannel")

        # 如果模式为 "D"，执行以下操作
        if mode == "D":
            # 对输入数据进行按通道量化
            self.input = self.op(self.input, **self.kwargs)

            # 定义反量化函数，用于反量化输入
            def dequant(input, scales, zero_points, axis: int, dtype: int):
                return input.dequantize()

            # 将操作设置为反量化函数
            self.op = dequant
            # 设置模块名称为 "DequantizePerChannel"
            self.set_module_name("DequantizePerChannel")

        # 设置输入的字典，包含输入数据和量化相关的参数
        self.inputs = {
            "input": self.input,
            "scales": torch.tensor([1.0] * channel_len),  # 所有通道的量化比例因子为1.0
            "zero_points": torch.tensor([0] * channel_len),  # 所有通道的零点为0
            "axis": axis,  # 量化的轴
            "dtype": dtype,  # 数据类型
        }

    # 前向传播函数，执行量化或反量化操作
    def forward(self, input, scales, zero_points, axis: int, dtype: int):
        return self.op(
            input, scales=scales, zero_points=zero_points, axis=axis, dtype=dtype
        )
op_bench.generate_pt_test(
    quantize_per_channel_configs_short + quantize_per_channel_configs_long,
    QuantizePerChannelBenchmark,
)
# 生成基于量化的性能测试，使用短和长的量化通道配置，测试QuantizePerChannelBenchmark类

# === Fake Quantization ===
# 生成假量化的配置字典，包括短和长两种配置
# 短配置包括固定的属性和标签，长配置包括可变的属性和标签
fake_quantize_configs_short_dict = {
    "attr_names": ["N", "C", "H", "W", "zero_point_dtype"],
    "attrs": [
        [1, 3, 512, 512, torch.int32],
    ],
    "tags": ["short"],
}

fake_quantize_configs_long_dict = {
    "N": [1],
    "C": [1, 3, 8, 32],
    "H": [256, 1024],
    "W": [256, 1024],
    "zero_point_dtype": [torch.int32],
    "tags": ["long"],
}

# 使用配置字典生成短和长的假量化配置列表
fake_quantize_configs_short = op_bench.config_list(
    cross_product_configs={
        "device": ("cpu", "cuda"),
    },
    **fake_quantize_configs_short_dict,
)

fake_quantize_configs_long = op_bench.cross_product_configs(
    device=("cpu", "cuda"), **fake_quantize_configs_long_dict
)

# 定义基准类FakeQuantizeBenchmark，用于评估默认参数下的假量化操作性能
class FakeQuantizeBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks fake quantization with default parameters."""

    def init(self, N, C, H, W, zero_point_dtype, device):
        self.inputs = {"input": torch.rand(N, C, H, W).to(device)}
        self.op = tq.FakeQuantize().to(device)
        self.set_module_name("FakeQuantize")

    def forward(self, input):
        return self.op(input)

# 生成基于假量化配置的性能测试，包括短和长的假量化配置
op_bench.generate_pt_test(
    fake_quantize_configs_short + fake_quantize_configs_long, FakeQuantizeBenchmark
)

# op_type用于描述在基准测试中使用的操作类型:
# learnable_kernel表示可以在尺度和零点上进行反向传播的C++内核
# original_kernel表示原始假量化的C++内核

# 定义使用可学习内核的假量化函数
def fakeQuantizePerTensorLearnableKernel(
    input, scale, zero_point, quant_min: int, quant_max: int
):
    return torch._fake_quantize_learnable_per_tensor_affine(
        input, scale, zero_point, quant_min, quant_max
    )

# 定义使用原始内核的假量化函数
def fakeQuantizePerTensorOriginalKernel(
    input, scale, zero_point, quant_min: int, quant_max: int
):
    return torch.fake_quantize_per_tensor_affine(input, 1.0, 0, quant_min, quant_max)

# 生成包含假量化操作的操作列表，包括了操作名称和对应的操作函数
fake_quantize_per_tensor_ops = op_bench.op_list(
    attrs=(
        ("learnable_kernel", fakeQuantizePerTensorLearnableKernel),
        ("original_kernel", fakeQuantizePerTensorOriginalKernel),
    ),
    attr_names=("op_name", "op_func"),
)

# 生成基于假量化操作配置的性能测试配置，包括短和长的假量化操作配置
fake_quantize_operator_configs_short = op_bench.config_list(
    cross_product_configs={
        "nbits": (4, 8),
        "device": ("cpu", "cuda"),
    },
    **fake_quantize_configs_short_dict,
)

fake_quantize_operator_configs_long = op_bench.cross_product_configs(
    nbits=(4, 8), device=("cpu", "cuda"), **fake_quantize_configs_long_dict
)

# TODO(future PR) 合并浮点零点配置到其他配置中，一旦它在所有假量化操作符和设备上完全支持。
# https://github.com/pytorch/pytorch/issues/61866.
# 复制长字典 fake_quantize_configs_long_dict 到新变量 fake_quantize_configs_long_dict_float_zero_point
fake_quantize_configs_long_dict_float_zero_point = (
    fake_quantize_configs_long_dict.copy()
)

# 将新变量 fake_quantize_configs_long_dict_float_zero_point 的 "zero_point_dtype" 键设为包含两种 Torch 浮点类型的列表
fake_quantize_configs_long_dict_float_zero_point["zero_point_dtype"] = [
    torch.float32,
    torch.half,
]

# 使用 op_bench.cross_product_configs() 生成 fake_quantize_operator_configs_long_float_zero_point 对象，
# 结合给定参数和新创建的 fake_quantize_configs_long_dict_float_zero_point 字典
fake_quantize_operator_configs_long_float_zero_point = op_bench.cross_product_configs(
    nbits=(8,),
    device=("cpu", "cuda"),
    **fake_quantize_configs_long_dict_float_zero_point,
)

# 定义 FakeQuantizePerTensorBaseOpBenchmark 类，继承自 op_bench.TorchBenchmarkBase，
# 用于评估三种不同的张量级别的假量化操作符性能
class FakeQuantizePerTensorBaseOpBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks 3 different fake quantize per tensor operators."""

    def init(self, N, C, H, W, zero_point_dtype, nbits, device, op_func):
        # 设置量化的最小和最大值
        self.quant_min = 0
        self.quant_max = 2**nbits - 1
        self.quant_range = 2**nbits
        # 创建一个随机张量作为输入，设备和数据类型由参数决定
        self.input = nn.Parameter(
            torch.rand(N, C, H, W, dtype=torch.float, device=device),
            requires_grad=self.auto_set(),
        )
        # 创建一个张量作为量化尺度，设备由参数决定
        self.scale = nn.Parameter(
            torch.tensor([1.0]).to(device), requires_grad=self.auto_set()
        )
        # 根据 op_func 的名称选择合适的零点张量类型和设备
        if op_func.__name__ == "fakeQuantizePerChannelOriginalKernel":
            self.zero_point = nn.Parameter(
                torch.tensor([0.0]).to(device).to(zero_point_dtype),
                requires_grad=self.auto_set(),
            )
        else:
            self.zero_point = nn.Parameter(
                torch.tensor([0.0]).to(device), requires_grad=self.auto_set()
            )
        
        # 构建输入字典，包含用于操作的各种参数和张量
        self.inputs = {
            "input": self.input,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "quant_min": self.quant_min,
            "quant_max": self.quant_max,
        }
        # 设置操作函数
        self.op_func = op_func

    # 定义前向方法，接受操作所需的输入参数，并返回操作函数的结果
    def forward(self, input, scale, zero_point, quant_min: int, quant_max: int):
        return self.op_func(input, scale, zero_point, quant_min, quant_max)


# 生成基于 fake_quantize_per_tensor_ops 的基准测试，使用 fake_quantize_operator_configs_short 和
# fake_quantize_operator_configs_long 的组合作为配置，基于 FakeQuantizePerTensorBaseOpBenchmark 类执行测试
op_bench.generate_pt_tests_from_op_list(
    fake_quantize_per_tensor_ops,
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerTensorBaseOpBenchmark,
)

# 生成基于 fake_quantize_per_tensor_ops 的梯度测试，使用 fake_quantize_operator_configs_short 和
# fake_quantize_operator_configs_long 的组合作为配置，基于 FakeQuantizePerTensorBaseOpBenchmark 类执行测试
op_bench.generate_pt_gradient_tests_from_op_list(
    fake_quantize_per_tensor_ops,
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerTensorBaseOpBenchmark,
)


# 定义 fakeQuantizePerChannelLearnableKernel 函数，接受输入、量化尺度、零点、轴、量化最小值和最大值，
# 返回使用 Torch 内部函数进行通道级别学习型假量化操作的结果
def fakeQuantizePerChannelLearnableKernel(
    input, scale, zero_point, axis: int, quant_min: int, quant_max: int
):
    return torch._fake_quantize_learnable_per_channel_affine(
        input, scale, zero_point, axis, quant_min, quant_max
    )


# 定义 fakeQuantizePerChannelOriginalKernel 函数，接受输入、量化尺度、零点、轴、量化最小值和最大值，
# 返回使用 Torch 内部函数进行通道级别原始假量化操作的结果
def fakeQuantizePerChannelOriginalKernel(
    input, scale, zero_point, axis: int, quant_min: int, quant_max: int
):
    return torch.fake_quantize_per_channel_affine(
        input, scale, zero_point, axis, quant_min, quant_max
    )


# 生成 fake_quantize_per_channel_ops 操作列表，包含两种假量化操作的定义：
# "learnable_kernel" 使用 learnableKernel 函数，"original_kernel" 使用 originalKernel 函数
fake_quantize_per_channel_ops = op_bench.op_list(
    attrs=(
        ("learnable_kernel", fakeQuantizePerChannelLearnableKernel),
        ("original_kernel", fakeQuantizePerChannelOriginalKernel),
    ),
    attr_names=("op_name", "op_func"),
)
fake_quantize_per_channel_float_zero_point_ops = op_bench.op_list(
    attrs=(("original_kernel", fakeQuantizePerChannelOriginalKernel),),
    attr_names=("op_name", "op_func"),
)
# 创建一个包含单个操作的列表，该操作具有一个名为 "original_kernel" 的属性，其值为 fakeQuantizePerChannelOriginalKernel 函数

class FakeQuantizePerChannelOpBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks 3 different fake quantize per channel operators."""

    def init(self, N, C, H, W, zero_point_dtype, nbits, device, op_func):
        self.quant_min = 0
        self.quant_max = 2**nbits - 1
        self.quant_range = 2**nbits
        # Axis is chosen with respect to the number of channels: C.
        self.axis = 1
        self.input = nn.Parameter(
            torch.rand(
                N,
                C,
                H,
                W,
                dtype=torch.float,
                device=device,
                requires_grad=self.auto_set(),
            )
        )
        # 根据 op_func 的不同初始化 scale 和 zero_point 的值
        if op_func.__name__ == "fakeQuantizePerChannelOriginalKernel":
            self.scale = torch.ones(
                C, device=device, dtype=torch.float32, requires_grad=False
            )
            self.zero_point = torch.zeros(
                C, device=device, dtype=zero_point_dtype, requires_grad=False
            )
        else:
            self.scale = nn.Parameter(
                torch.ones(C, device=device, dtype=torch.float32),
                requires_grad=self.auto_set(),
            )
            self.zero_point = nn.Parameter(
                torch.zeros(C, device=device, dtype=torch.float32),
                requires_grad=self.auto_set(),
            )

        # 将初始化后的参数放入 inputs 字典中
        self.inputs = {
            "input": self.input,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "axis": self.axis,
            "quant_min": self.quant_min,
            "quant_max": self.quant_max,
        }

        self.op_func = op_func

    def forward(
        self, input, scale, zero_point, axis: int, quant_min: int, quant_max: int
    ):
        # 执行具体的操作函数 op_func
        return self.op_func(input, scale, zero_point, axis, quant_min, quant_max)


op_bench.generate_pt_tests_from_op_list(
    fake_quantize_per_channel_ops,
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerChannelOpBenchmark,
)
# 根据操作列表和配置生成 PyTorch 测试，使用 FakeQuantizePerChannelOpBenchmark 类进行测试

op_bench.generate_pt_tests_from_op_list(
    fake_quantize_per_channel_float_zero_point_ops,
    fake_quantize_operator_configs_long_float_zero_point,
    FakeQuantizePerChannelOpBenchmark,
)
# 根据包含浮点数零点的操作列表和配置生成 PyTorch 测试，使用 FakeQuantizePerChannelOpBenchmark 类进行测试

op_bench.generate_pt_gradient_tests_from_op_list(
    fake_quantize_per_channel_ops,
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerChannelOpBenchmark,
)
# 根据操作列表和配置生成 PyTorch 梯度测试，使用 FakeQuantizePerChannelOpBenchmark 类进行测试

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
# 如果当前脚本作为主程序运行，则执行性能基准测试的主程序
```
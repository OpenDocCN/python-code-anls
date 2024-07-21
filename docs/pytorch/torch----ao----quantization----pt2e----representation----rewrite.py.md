# `.\pytorch\torch\ao\quantization\pt2e\representation\rewrite.py`

```py
# 设置 mypy: allow-untyped-defs，允许在类型检查时不对函数进行类型声明
import torch  # 导入 PyTorch 模块
from torch.fx import GraphModule  # 从 torch.fx 模块导入 GraphModule 类
from ..export_utils import _WrapperModule  # 从相对路径的 export_utils 模块导入 _WrapperModule
from ..utils import (  # 从相对路径的 utils 模块导入以下函数
    _get_aten_graph_module_for_pattern,
    remove_tensor_overload_for_qdq_ops,
    _replace_literals_with_new_placeholders,
    _replace_literals_with_existing_placeholders,
)
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # 导入 quantized_decomposed_lib，禁止 F401 错误
from torch.fx.subgraph_rewriter import replace_pattern  # 从 torch.fx.subgraph_rewriter 模块导入 replace_pattern 函数
from torch._higher_order_ops.out_dtype import out_dtype  # 从 torch._higher_order_ops.out_dtype 模块导入 out_dtype
from typing import Optional, Callable, Tuple, Any  # 导入类型提示：Optional、Callable、Tuple、Any
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 类

from functools import partial  # 导入 functools 模块的 partial 函数

__all__ = [  # 将此模块中的公共接口定义为 reference_representation_rewrite
    "reference_representation_rewrite",
]

_QUANTIZED_LINEAR_EXAMPLE_INPUTS = (  # 定义包含量化线性模型示例输入的元组
    torch.randint(-128, 127, (2, 5), dtype=torch.int8),  # 生成随机整数张量
    torch.randn(1, dtype=torch.float),  # 生成标准正态分布的随机数张量
    torch.zeros(1, dtype=torch.int),  # 生成全零张量
    torch.tensor([-128], dtype=torch.int),  # 创建包含单个元素的张量
    torch.tensor([127], dtype=torch.int),  # 创建包含单个元素的张量
    torch.randint(-128, 127, (5, 5), dtype=torch.int8),  # 生成随机整数张量
    torch.randn(1, dtype=torch.float),  # 生成标准正态分布的随机数张量
    torch.zeros(1, dtype=torch.int),  # 生成全零张量
    torch.tensor([-127], dtype=torch.int),  # 创建包含单个元素的张量
    torch.tensor([127], dtype=torch.int),  # 创建包含单个元素的张量
    torch.randn(1, dtype=torch.float),  # 生成标准正态分布的随机数张量
    torch.randn(1, dtype=torch.float),  # 生成标准正态分布的随机数张量
    torch.zeros(1, dtype=torch.int),  # 生成全零张量
    torch.tensor([-128], dtype=torch.int),  # 创建包含单个元素的张量
    torch.tensor([127], dtype=torch.int),  # 创建包含单个元素的张量
)

def _qdq_quantized_linear(
    x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max,
    weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max,
    bias_fp32,
    out_scale, out_zero_point, out_quant_min, out_quant_max
):
    # 对输入、权重进行量化解压缩，返回浮点数张量
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    # 使用 torch.ops.aten.linear.default 进行线性计算，返回浮点数张量
    out_fp32 = torch.ops.aten.linear.default(x_fp32, weight_fp32, bias_fp32)
    # 将浮点数张量重新量化为整数张量，返回整数张量
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8)
    return out_i8

def _reference_quantized_linear(
    x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max,
    weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max,
    bias_fp32,
    out_scale, out_zero_point, out_quant_min, out_quant_max
):
    # 在 clamp 函数中使用 quant_min 和 quant_max 进行张量截断，确保追踪图中包含这些参数
    # 否则将导致匹配失败
    x_i8 = torch.ops.aten.clamp(x_i8, x_quant_min, x_quant_max)
    weight_i8 = torch.ops.aten.clamp(weight_i8, weight_quant_min, weight_quant_max)

    x_i16 = x_i8.to(torch.int16)  # 将输入张量转换为 int16 类型
    weight_i16 = weight_i8.to(torch.int16)  # 将权重张量转换为 int16 类型
    # 将偏置设置为 None，以便在偏置_scale == 输入_scale * 权重_scale 或不等时均能正常工作
    # 执行整数类型输出的线性操作，使用默认的线性函数
    acc_i32 = out_dtype(
        torch.ops.aten.linear.default,
        torch.int32,
        x_i16 - x_zero_point,  # 将输入减去零点偏移量，进行量化操作
        weight_i16 - weight_zero_point,
        None)
    # TODO: 改为使用乘法操作进行缩放
    # 注意：我们在没有用户信号的情况下，使用这些尺度量化偏置可能是可以接受的
    bias_scale = x_scale * weight_scale  # 计算偏置的量化尺度
    bias_i32 = out_dtype(torch.ops.aten.div.Tensor, torch.int32, bias_fp32, bias_scale)  # 使用除法操作对偏置进行量化
    acc_i32 = acc_i32 + bias_i32  # 将偏置加到累加器中
    # TODO: 当我们使用 x_scale/weight_scale 等标量值时，改为使用乘法操作
    acc_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, acc_i32, x_scale * weight_scale / out_scale) + out_zero_point  # 对累加器进行乘法和零点偏移量的操作
    out_i8 = torch.ops.aten.clamp(acc_i32, out_quant_min, out_quant_max).to(torch.int8)  # 对结果进行夹紧操作，并转换为 int8 类型
    return out_i8  # 返回量化后的 int8 输出
# 定义动态量化线性运算的示例输入，包括张量、整数、浮点数和量化参数
_DYNAMIC_QUANTIZED_LINEAR_EXAMPLE_INPUTS = (
    torch.randn((2, 5), dtype=torch.float),  # 生成标准正态分布的张量
    -128,  # 整数 -128
    127,  # 整数 127
    torch.finfo(torch.float32).eps,  # 浮点数类型的最小正数
    torch.randint(-128, 127, (5, 5), dtype=torch.int8),  # 生成指定形状和数据类型的随机整数张量
    torch.randn(1, dtype=torch.float),  # 生成标准正态分布的张量
    torch.zeros(1, dtype=torch.int),  # 形状为(1,)的整数零张量
    torch.tensor([-127], dtype=torch.int),  # 包含单个元素的整数张量
    torch.tensor([127], dtype=torch.int),  # 包含单个元素的整数张量
    torch.randn(1, dtype=torch.float),  # 生成标准正态分布的张量
)

# 定义动态量化线性运算函数，对输入进行动态量化线性操作
def _qdq_dynamic_quantized_linear(
    x_fp32, x_quant_min, x_quant_max, x_eps,
    weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max,
    bias_fp32,
):
    # 根据输入张量的范围和精度，选择量化参数
    x_scale, x_zero_point = torch.ops.quantized_decomposed.choose_qparams(x_fp32, x_quant_min, x_quant_max, x_eps, torch.int8)
    # 将输入张量 x_fp32 量化为 int8 类型
    x_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        x_fp32, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    # 将量化后的输入张量 x_i8 反量化为 float32 类型
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    # 将量化后的权重张量反量化为 float32 类型
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    # 使用默认的线性运算计算输出张量
    out_fp32 = torch.ops.aten.linear.default(x_fp32, weight_fp32, bias_fp32)
    return out_fp32

# 定义参考动态量化线性运算函数，实现更复杂的动态量化线性操作
def _reference_dynamic_quantized_linear(
    x_fp32, x_quant_min, x_quant_max, x_eps,
    weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max,
    bias_fp32,
):
    # 根据输入张量的范围和精度，选择量化参数
    x_scale, x_zero_point = torch.ops.quantized_decomposed.choose_qparams(x_fp32, x_quant_min, x_quant_max, x_eps, torch.int8)
    # 使用分解表示进行量化操作，处理 x_fp32 张量
    # TODO: 当操作准备就绪时，在此处使用 out_dtype(mul, ...) 
    x_fp32 = x_fp32 / x_scale  # fp32
    # 不同的舍入模式可能在此处有所不同
    # PyTorch 使用偶数舍入，这在大多数后端中也很常见
    x_fp32 = torch.round(x_fp32)  # fp32
    x_i32 = x_fp32.to(dtype=torch.int32)  # int32
    x_i32 = x_i32 + x_zero_point  # int32
    # 对 fp32、int32 和 int8 数据类型执行范围限制
    x_i32 = torch.clamp(x_i32, x_quant_min, x_quant_max)  # int32
    x_i8 = x_i32.to(dtype=torch.int8)

    # 对权重张量进行范围限制
    weight_i8 = torch.ops.aten.clamp(weight_i8, weight_quant_min, weight_quant_max)

    # 将 int8 类型转换为 int16 类型
    x_i16 = x_i8.to(torch.int16)
    weight_i16 = weight_i8.to(torch.int16)
    # 始终将偏置设置为 None，以便在不同情况下能够使用相同的表示形式
    acc_i32 = out_dtype(
        torch.ops.aten.linear.default,
        torch.int32,
        x_i16 - x_zero_point,
        weight_i16 - weight_zero_point,
        None)
    # 计算偏置的比例
    bias_scale = x_scale * weight_scale
    bias_i32 = out_dtype(torch.ops.aten.div.Tensor, torch.int32, bias_fp32, bias_scale)
    acc_i32 = acc_i32 + bias_i32
    out_fp32 = acc_i32 * (x_scale * weight_scale)
    return out_fp32

# 定义二维量化卷积的示例输入，包括输入张量和偏置张量
_QUANTIZED_CONV2d_EXAMPLE_INPUTS = (
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),  # 生成指定形状和数据类型的随机整数张量
    torch.randn(1, dtype=torch.float),  # 生成标准正态分布的张量
    # 创建一个大小为 (1,) 的张量，元素类型为整型，所有元素初始化为 0
    torch.zeros(1, dtype=torch.int),
    
    # 创建一个大小为 (1,) 的张量，元素类型为整型，包含一个值为 -128 的元素
    torch.tensor([-128], dtype=torch.int),
    
    # 创建一个大小为 (1,) 的张量，元素类型为整型，包含一个值为 127 的元素
    torch.tensor([127], dtype=torch.int),
    
    # 创建一个大小为 (1, 3, 3, 3) 的张量，元素类型为 8 位整型，元素值在 -128 到 127 之间随机生成
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
    
    # 创建一个大小为 (1,) 的张量，元素类型为单精度浮点数，包含一个从标准正态分布中随机抽取的值
    torch.randn(1, dtype=torch.float),
    
    # 创建一个大小为 (1,) 的张量，元素类型为整型，所有元素初始化为 0
    torch.zeros(1, dtype=torch.int),
    
    # 创建一个大小为 (1,) 的张量，元素类型为整型，包含一个值为 -127 的元素
    torch.tensor([-127], dtype=torch.int),
    
    # 创建一个大小为 (1,) 的张量，元素类型为整型，包含一个值为 127 的元素
    torch.tensor([127], dtype=torch.int),
    
    # 创建一个大小为 (1,) 的张量，元素类型为单精度浮点数，包含一个从标准正态分布中随机抽取的值
    torch.randn(1, dtype=torch.float),
    
    # 创建一个大小为 (1,) 的张量，元素类型为单精度浮点数，包含一个从标准正态分布中随机抽取的值
    torch.randn(1, dtype=torch.float),
    
    # 创建一个大小为 (1,) 的张量，元素类型为整型，所有元素初始化为 0
    torch.zeros(1, dtype=torch.int),
    
    # 创建一个大小为 (1,) 的张量，元素类型为整型，包含一个值为 -128 的元素
    torch.tensor([-128], dtype=torch.int),
    
    # 创建一个大小为 (1,) 的张量，元素类型为整型，包含一个值为 127 的元素
    torch.tensor([127], dtype=torch.int),
# 定义一个名为 _reference_quantized_conv2d 的函数，用于执行参考的量化卷积操作
def _reference_quantized_conv2d(
    x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max,
    weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max,
    bias_fp32,
    out_scale, out_zero_point, out_quant_min, out_quant_max
):
    stride = [1, 1]  # 设置卷积操作的步长
    padding = [0, 0]  # 设置卷积操作的填充大小
    dilation = [1, 1]  # 设置卷积操作的扩张大小
    transposed = False  # 设置是否进行转置卷积的标志
    output_padding = [0, 0]  # 设置输出张量的填充大小
    groups = 1  # 设置卷积操作的分组数

    # 在 x_i8 上使用 clamp 函数，将输入张量限制在 x_quant_min 和 x_quant_max 之间
    x_i8 = torch.ops.aten.clamp(x_i8, x_quant_min, x_quant_max)
    # 在 weight_i8 上使用 clamp 函数，将权重张量限制在 weight_quant_min 和 weight_quant_max 之间
    weight_i8 = torch.ops.aten.clamp(weight_i8, weight_quant_min, weight_quant_max)

    # 将 x_i8 转换为 int16 类型
    x_i16 = x_i8.to(torch.int16)
    # 将 weight_i8 转换为 int16 类型
    weight_i16 = weight_i8.to(torch.int16)

    # 设定 bias 为 None，以确保无论 bias_scale 是否等于 x_scale * weight_scale，表示都能工作
    # 以保持相同的表示
    acc_i32 = out_dtype(
        torch.ops.aten.convolution.default,  # 执行默认的卷积操作
        torch.int32,  # 输出张量的数据类型为 int32
        x_i16 - x_zero_point,  # 对输入张量进行减零点操作
        weight_i16 - weight_zero_point,  # 对权重张量进行减零点操作
        None, stride, padding, dilation, transposed, output_padding, groups
    )

    # 注意：我们在没有用户信号的情况下，使用这些比例对 bias 进行量化，但这可能是可以接受的
    # bias 的量化到 int32 使用了 bias_scale = x_scale * weight_scale 的原因是：
    # 对于线性计算，例如
    # Out_(i, j)_fp32 = Sum_(over k)[X_(i, k)_fp32 * W_(i, k)_fp32] + bias_(i)_fp32
    # 将 X, W fp32 表示为它们的反量化变换
    # A_fp32 = (A_q - A_zero_point)/A_scale
    # Out_(i, j)_fp32 = Sum_(over k)[(X_(i, k)_fp32 - X_zp) * X_scale * (W_(i, k)_fp32 - W_zp) * W_scale] + bias_(i)_fp32
    # 分离出 X_scale 和 W_scale
    # Out_(i, j)_fp32 = ((X_scale * W_scale) * Sum_(over k)[(X_(i, k)_fp32 - X_zp) * (W_(i, k)_fp32 - W_zp)]) + bias_(i)_fp32
    bias_scale = x_scale * weight_scale
    # 将 bias_fp32 使用 bias_scale 缩放，并转换为 int32 类型
    bias_i32 = out_dtype(torch.ops.aten.div.Tensor, torch.int32, bias_fp32, bias_scale)
    
    # 扩展维度以匹配广播维度
    # 由于在图模式替换中存在文字匹配问题，不能简单地使用 bias_i32.unsqueeze(0)
    bias_i32 = bias_i32.unsqueeze(-1)
    bias_i32 = bias_i32.unsqueeze(-1)
    
    # 将 bias_i32 加到 acc_i32 中
    acc_i32 = acc_i32 + bias_i32
    
    # 使用 x_scale * weight_scale / out_scale 缩放 acc_i32，并加上 out_zero_point
    acc_i32 = out_dtype(
        torch.ops.aten.mul.Tensor, torch.int32, acc_i32, x_scale * weight_scale / out_scale) + out_zero_point
    
    # 将 acc_i32 限制在 out_quant_min 和 out_quant_max 之间，并转换为 int8 类型
    out_i8 = torch.ops.aten.clamp(acc_i32, out_quant_min, out_quant_max).to(torch.int8)
    
    # 返回 int8 类型的结果 out_i8
    return out_i8
# 定义一个全局常量，包含多个张量作为量化加法或加ReLU操作的输入示例
_QUANTIZED_ADD_OR_ADD_RELU_EXAMPLE_INPUTS = (
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),  # 生成一个随机的int8张量
    torch.randn(1, dtype=torch.float),  # 生成一个标准正态分布的float张量
    torch.zeros(1, dtype=torch.int),  # 生成一个值全为0的int张量
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),  # 生成一个随机的int8张量
    torch.randn(1, dtype=torch.float),  # 生成一个标准正态分布的float张量
    torch.zeros(1, dtype=torch.int),  # 生成一个值全为0的int张量
    torch.randn(1, dtype=torch.float),  # 生成一个标准正态分布的float张量
    torch.zeros(1, dtype=torch.int),  # 生成一个值全为0的int张量
    torch.tensor([-128], dtype=torch.int),  # 生成一个包含单个元素[-128]的int张量
    torch.tensor([127], dtype=torch.int),  # 生成一个包含单个元素[127]的int张量
)

def _qdq_quantized_add_relu(
    x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point,
    out_scale, out_zero_point, quant_min, quant_max
):
    # 将输入的量化int8张量反量化为float32张量
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, quant_min, quant_max, torch.int8)
    y_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(y_i8, y_scale, y_zero_point, quant_min, quant_max, torch.int8)
    # 执行float32张量的加法操作
    out_fp32 = x_fp32 + y_fp32
    # 对加法结果执行ReLU激活函数
    out_fp32 = torch.ops.aten.relu(out_fp32)
    # 将ReLU后的float32张量重新量化为int8张量
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, quant_min, quant_max, torch.int8
    )
    return out_i8

def _reference_quantized_add_relu(
    x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point,
    out_scale, out_zero_point, quant_min, quant_max
):
    """
    根据`_reference_quantized_add`的注释了解如何基于x_i8和y_i8推导out_i8的公式
    """
    # 将输入的int8张量转换为int32张量
    x_i32 = x_i8.to(torch.int32)
    y_i32 = y_i8.to(torch.int32)
    # 计算乘法操作后的int32张量，使用了out_dtype作为方法
    x_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, (x_i32 - x_zero_point), (x_scale / out_scale))
    y_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, (y_i32 - y_zero_point), (y_scale / out_scale))
    # 计算加法操作后的int32张量
    out_i32 = x_i32 + y_i32 + out_zero_point
    # 对int32张量进行截断并转换为int8张量
    out_i8 = torch.ops.aten.clamp(out_i32, out_zero_point, quant_max).to(torch.int8)
    return out_i8

def _qdq_quantized_add(
    x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point, out_scale, out_zero_point, quant_min, quant_max
):
    # 将输入的量化int8张量反量化为float32张量
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, quant_min, quant_max, torch.int8)
    y_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(y_i8, y_scale, y_zero_point, quant_min, quant_max, torch.int8)
    # 执行float32张量的加法操作
    out_fp32 = x_fp32 + y_fp32
    # 将加法结果重新量化为int8张量
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, quant_min, quant_max, torch.int8
    )
    return out_i8

def _reference_quantized_add(
    x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point,
    out_scale, out_zero_point, quant_min, quant_max
):
    """
    如何根据x_i8和y_i8推导out_i8的公式
    (因为量化加法使用x_i8、y_i8和它们的量化参数，产生out_i8)
    """
    # 将输入的int8张量转换为int32张量
    x_i32 = x_i8.to(torch.int32)
    y_i32 = y_i8.to(torch.int32)
    # 计算加法操作后的int32张量
    x_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, (x_i32 - x_zero_point), (x_scale / out_scale))
    y_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, (y_i32 - y_zero_point), (y_scale / out_scale))
    out_i32 = x_i32 + y_i32 + out_zero_point
    # 截断和限制out_i32，并转换为int8张量
    out_i8 = torch.ops.aten.clamp(out_i32, out_zero_point, quant_max).to(torch.int8)
    return out_i8
out_i8 = out_f32 / out_scale + out_zero_point           # (1)

    # 计算出_f32，它是由 x_f32 + y_f32 计算得来的，其中 x_fp32 和 y_fp32 是 x_i8 和 y_i8 的去量化版本
    out_f32 = x_f32 + y_f32           # (2)
    x_fp32 = (x_i8 - x_zero_point) * x_scale         # (3)
    y_fp32 = (y_i8 - y_zero_point) * y_scale         # (4)

    # 将上述公式应用到 out_i8 的方程中，我们可以得到以下计算过程：
    out_i8 = out_fp32 / out_scale + out_zero_point             # (1)
       = (x_f32 + y_f32) / out_scale + out_zero_point      # 将 (2) 应用到 out_fp32 中，用 x_fp32 + y_fp32 替换 out_fp32
       = ((x_i8 - x_zero_point) * x_scale + (y_i8 - y_zero_point) * y_scale) / out_scale + out_zero_point  # 应用 (3) 和 (4)

    """
    x_i32 = x_i8.to(torch.int32)
    y_i32 = y_i8.to(torch.int32)
    # TODO: 使用 out_dtype 操作
    x_i32 = torch.round((x_scale / out_scale) * (x_i32 - x_zero_point)).to(torch.int32)
    y_i32 = torch.round((y_scale / out_scale) * (y_i32 - y_zero_point)).to(torch.int32)
    out_i32 = x_i32 + y_i32 + out_zero_point
    quant_min = -128
    quant_max = 127
    out_i8 = torch.ops.aten.clamp(out_i32, quant_min, quant_max).to(torch.int8)
    return out_i8

_QUANTIZED_MAX_POOL2D_EXAMPLE_INPUTS = (
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
    torch.randn(1, dtype=torch.float),
    torch.zeros(1, dtype=torch.int),
    torch.tensor([-128], dtype=torch.int),
    torch.tensor([127], dtype=torch.int),
    torch.randn(1, dtype=torch.float),
    torch.zeros(1, dtype=torch.int),
    torch.tensor([-128], dtype=torch.int),
    torch.tensor([127], dtype=torch.int),
)

def _qdq_quantized_max_pool2d(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, out_scale, out_zero_point, out_quant_min, out_quant_max):
    kernel_size = 1
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    out_fp32, _ = torch.ops.aten.max_pool2d_with_indices.default(x_fp32, kernel_size, stride, padding, dilation, ceil_mode)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8)
    return out_i8

def _reference_quantized_max_pool2d(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, out_scale, out_zero_point, out_quant_min, out_quant_max):
    kernel_size = 1
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    # 保留 x_quant_min, x_quant_max 在图形模式匹配中
    x_i8 = torch.clamp(x_i8, x_quant_min, x_quant_max)
    x_i32 = x_i8.to(torch.int32)
    out_i32, _ = torch.ops.aten.max_pool2d_with_indices.default(
        x_i32 - x_zero_point,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode
    )
    out_fp32 = out_i32 * (x_scale / out_scale) + out_zero_point
    # 使用 torch.clamp 函数确保 out_fp32 张量中的值限制在指定的范围内，即在 out_quant_min 和 out_quant_max 之间
    out_fp32 = torch.clamp(out_fp32, out_quant_min, out_quant_max)
    
    # 将 out_fp32 张量转换为 torch.int8 类型，即将浮点数张量转换为 8 位整数类型张量
    out_i8 = out_fp32.to(torch.int8)
    
    # 返回转换后的 8 位整数类型张量 out_i8 作为函数的输出结果
    return out_i8
_QUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS = (
    torch.randn(1, 3, 3, 3, dtype=torch.float),  # 创建一个形状为 (1, 3, 3, 3) 的随机张量，数据类型为 float
    torch.randn(1, dtype=torch.float),  # 创建一个形状为 (1,) 的随机张量，数据类型为 float
    torch.zeros(1, dtype=torch.int),  # 创建一个形状为 (1,) 的全零张量，数据类型为 int
    torch.tensor([-128], dtype=torch.int),  # 创建一个值为 -128 的张量，数据类型为 int
    torch.tensor([127], dtype=torch.int),  # 创建一个值为 127 的张量，数据类型为 int
)

def _quantize_per_tensor_int8(x_fp32, scale, zero_point, quant_min, quant_max):
    x = torch.ops.quantized_decomposed.quantize_per_tensor(x_fp32, scale, zero_point, quant_min, quant_max, torch.int8)
    return x

def _reference_quantize_per_tensor_int8(x_fp32, scale, zero_point, quant_min, quant_max):
    # TODO: use out_dtype(mul, ...) here when the op is ready
    x = x_fp32 / scale  # 将输入张量 x_fp32 按照 scale 缩放，得到 float32 类型
    # round modes might be different here
    # pytorch is rounding to even, which is also common for most of the backends
    x = torch.round(x)  # 对 x 执行四舍五入，得到 float32 类型
    x = x.to(dtype=torch.int32)  # 将 x 转换为 int32 类型
    x = x + zero_point  # 将 x 加上 zero_point，得到 int32 类型
    # clamp works for fp32, int32 and int8 dtypes
    x = torch.clamp(x, quant_min, quant_max)  # 将 x 夹紧到 quant_min 和 quant_max 之间，得到 int32 类型
    x = x.to(dtype=torch.int8)  # 将 x 转换为 int8 类型
    return x

_DEQUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS = (
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),  # 创建一个形状为 (1, 3, 3, 3) 的随机张量，数据类型为 int8
    torch.randn(1, dtype=torch.float),  # 创建一个形状为 (1,) 的随机张量，数据类型为 float
    torch.zeros(1, dtype=torch.int),  # 创建一个形状为 (1,) 的全零张量，数据类型为 int
    torch.tensor([-128], dtype=torch.int),  # 创建一个值为 -128 的张量，数据类型为 int
    torch.tensor([127], dtype=torch.int),  # 创建一个值为 127 的张量，数据类型为 int
)

def _dequantize_per_tensor_int8(x_i8, scale, zero_point, quant_min, quant_max):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, scale, zero_point, quant_min, quant_max, torch.int8)
    return x_fp32

def _reference_dequantize_per_tensor_int8(x_i8, scale, zero_point, quant_min, quant_max):
    # without using quant_min/max in clamp, the traced graph will not have quant_mi/max args.
    # This results in failure to match the pattern.
    # Therefore, we call a torch.ops.aten.clamp here
    x_i8 = torch.ops.aten.clamp(x_i8, quant_min, quant_max)  # 将 x_i8 夹紧到 quant_min 和 quant_max 之间
    # TODO: use out_dtype op
    # note: x_i8.to(torch.int32) does not work here
    # TODO: debug the implementation later when torchdynamo time out issue is resolved
    return ((x_i8.to(torch.float32) - zero_point) * scale).to(dtype=torch.float32)

_QUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS = (
    torch.randn(1, 3, 3, 3, dtype=torch.float),  # 创建一个形状为 (1, 3, 3, 3) 的随机张量，数据类型为 float
    torch.randn(3, dtype=torch.float),  # 创建一个形状为 (3,) 的随机张量，数据类型为 float
    torch.zeros(3, dtype=torch.int),  # 创建一个形状为 (3,) 的全零张量，数据类型为 int
    1,  # 创建一个标量值为 1
    -128,  # 创建一个标量值为 -128
    127,  # 创建一个标量值为 127
)

def _quantize_per_channel_int8(x_fp32, scales, zero_points, ch_axis, quant_min, quant_max):
    out_i8 = torch.ops.quantized_decomposed.quantize_per_channel(
        x_fp32, scales, zero_points, ch_axis, quant_min, quant_max, torch.int8
    )
    return out_i8

def _reference_quantize_per_channel_int8(x_fp32, scales, zero_points, ch_axis, quant_min, quant_max):
    x_fp32 = torch.transpose(x_fp32, ch_axis, -1)  # 将输入张量 x_fp32 沿着 ch_axis 转置
    out_i32 = torch.ops.aten.clamp(torch.round(x_fp32 / scales).to(torch.int32) + zero_points, quant_min, quant_max)  # 对 x_fp32 进行量化操作，返回 int32 类型
    out_i32 = torch.transpose(out_i32, ch_axis, -1)  # 将输出张量 out_i32 沿着 ch_axis 转置
    return out_i32.to(torch.int8)  # 将输出张量 out_i32 转换为 int8 类型
    # 生成一个形状为 (1, 3, 3, 3) 的张量，张量中的元素为从 -128 到 126（不包括127）的随机整数，数据类型为 int8
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
    
    # 生成一个包含三个元素的张量，元素的值服从标准正态分布（均值为0，标准差为1），数据类型为 float
    torch.randn(3, dtype=torch.float),
    
    # 生成一个包含三个元素的张量，元素的值全部为0，数据类型为 int
    torch.zeros(3, dtype=torch.int),
    
    # 整数常量 1
    
    # 整数常量 -128
    
    # 整数常量 127
def _dequantize_per_channel_int8(x_i8, scales, zero_points, ch_axis, quant_min, quant_max):
    # 使用 torch.ops.quantized_decomposed.dequantize_per_channel 函数进行逐通道的整数8位量化解码
    out_fp32 = torch.ops.quantized_decomposed.dequantize_per_channel(
        x_i8, scales, zero_points, ch_axis, quant_min, quant_max, torch.int8
    )
    return out_fp32

def _reference_dequantize_per_channel_int8(x_i8, scales, zero_points, ch_axis, quant_min, quant_max):
    # 使用 torch.ops.aten.clamp 函数对 x_i8 张量进行范围限制，确保在 quant_min 和 quant_max 范围内
    x_i8 = torch.ops.aten.clamp(x_i8, quant_min, quant_max)
    # 将通道轴和最后一个轴进行转置
    x_i8 = torch.transpose(x_i8, ch_axis, -1)
    # 将 x_i8 转换为整数32位类型
    x_i32 = x_i8.to(torch.int32)
    # 对 x_i32 进行零点偏移和缩放操作，转换为浮点32位类型
    out_fp32 = (x_i32 - zero_points).to(torch.float) * scales
    # 再次对通道轴和最后一个轴进行转置，返回解码后的浮点32位张量
    out_fp32 = torch.transpose(out_fp32, ch_axis, -1)
    return out_fp32

def _replace_ph_qdq_per_channel_replacement(gm: torch.fx.GraphModule):
    # 调用 _replace_literals_with_existing_placeholders 函数，替换常量为现有的占位符
    return _replace_literals_with_existing_placeholders(
        gm,
        exclude_literals=[-1],
        literal_to_ph_idx={1: 3, -128: 4, 127: 5}
    )

@dataclass
class _RewriteInfo:
    """用于重写的数据信息，包括示例输入、模式和替换函数，
    以及用于导出模式和替换图模块的后转换函数"""
    
    # 用于导出模式到图模块的示例输入
    example_inputs: Tuple[Any, ...]
    # 模式函数
    pattern: Callable
    # 替换函数
    replacement: Callable
    # 导出模式的后转换函数
    pattern_post_trans: Optional[Callable[[GraphModule], GraphModule]] = None
    # 替换图模块的后转换函数
    replacement_post_trans: Optional[Callable[[GraphModule], GraphModule]] = None

_REWRITE_INFO_LIST = [
    _RewriteInfo(
        _DYNAMIC_QUANTIZED_LINEAR_EXAMPLE_INPUTS,
        _WrapperModule(_qdq_dynamic_quantized_linear),
        _WrapperModule(_reference_dynamic_quantized_linear),
        # 使用 _replace_literals_with_existing_placeholders 函数替换常量为现有的占位符
        partial(
            _replace_literals_with_existing_placeholders,
            literal_to_ph_idx={
                -128: 1,
                127: 2,
                torch.finfo(torch.float32).eps: 3
            }
        ),
        # 使用 _replace_literals_with_existing_placeholders 函数替换常量为现有的占位符
        partial(
            _replace_literals_with_existing_placeholders,
            literal_to_ph_idx={
                -128: 1,
                127: 2,
                torch.finfo(torch.float32).eps: 3
            }
        ),
    ),
    _RewriteInfo(
        _QUANTIZED_LINEAR_EXAMPLE_INPUTS,
        _WrapperModule(_qdq_quantized_linear),
        _WrapperModule(_reference_quantized_linear),
        # 使用 _replace_literals_with_new_placeholders 函数替换常量为新的占位符
        _replace_literals_with_new_placeholders,
        # 使用 _replace_literals_with_new_placeholders 函数替换常量为新的占位符
        _replace_literals_with_new_placeholders,
    ),
    _RewriteInfo(
        _QUANTIZED_CONV2d_EXAMPLE_INPUTS,
        _WrapperModule(_qdq_quantized_conv2d),
        _WrapperModule(_reference_quantized_conv2d),
        partial(_replace_literals_with_new_placeholders, exclude_literals=[-1]),
        partial(_replace_literals_with_new_placeholders, exclude_literals=[-1]),
    ),

创建一个 `_RewriteInfo` 对象，用于重写信息。包括量化卷积的示例输入 `_QUANTIZED_CONV2d_EXAMPLE_INPUTS`，量化卷积的量化函数 `_qdq_quantized_conv2d` 和参考量化卷积函数 `_reference_quantized_conv2d`。还包括两个部分的函数，这些函数将替换具体的字面值为新的占位符，但排除了值为 `-1` 的情况。


    _RewriteInfo(
        _QUANTIZED_ADD_OR_ADD_RELU_EXAMPLE_INPUTS,
        _WrapperModule(_qdq_quantized_add_relu),
        _WrapperModule(_reference_quantized_add_relu),
    ),

创建另一个 `_RewriteInfo` 对象，用于重写信息。包括加法或加法ReLU激活的示例输入 `_QUANTIZED_ADD_OR_ADD_RELU_EXAMPLE_INPUTS`，以及相应的量化函数 `_qdq_quantized_add_relu` 和参考量化函数 `_reference_quantized_add_relu`。


    _RewriteInfo(
        _QUANTIZED_ADD_OR_ADD_RELU_EXAMPLE_INPUTS,
        _WrapperModule(_qdq_quantized_add),
        _WrapperModule(_reference_quantized_add),
    ),

再创建一个 `_RewriteInfo` 对象，用于重写信息。与前一个对象类似，但这次是针对加法的量化函数 `_qdq_quantized_add` 和参考量化函数 `_reference_quantized_add`。


    _RewriteInfo(
        _QUANTIZED_MAX_POOL2D_EXAMPLE_INPUTS,
        _WrapperModule(_qdq_quantized_max_pool2d),
        _WrapperModule(_reference_quantized_max_pool2d),
        _replace_literals_with_new_placeholders,
        _replace_literals_with_new_placeholders
    ),

创建另一个 `_RewriteInfo` 对象，用于重写信息。包括最大池化的示例输入 `_QUANTIZED_MAX_POOL2D_EXAMPLE_INPUTS`，以及相应的量化函数 `_qdq_quantized_max_pool2d` 和参考量化函数 `_reference_quantized_max_pool2d`。还包括两个部分的函数，这些函数将替换具体的字面值为新的占位符。


    _RewriteInfo(
        _QUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS,
        _WrapperModule(_quantize_per_tensor_int8),
        _WrapperModule(_reference_quantize_per_tensor_int8),
    ),

创建另一个 `_RewriteInfo` 对象，用于重写信息。包括每张张量的整数8位量化的示例输入 `_QUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS`，以及相应的量化函数 `_quantize_per_tensor_int8` 和参考量化函数 `_reference_quantize_per_tensor_int8`。


    _RewriteInfo(
        _DEQUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS,
        _WrapperModule(_dequantize_per_tensor_int8),
        _WrapperModule(_reference_dequantize_per_tensor_int8),
    ),

创建另一个 `_RewriteInfo` 对象，用于重写信息。包括每张张量的整数8位反量化的示例输入 `_DEQUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS`，以及相应的反量化函数 `_dequantize_per_tensor_int8` 和参考反量化函数 `_reference_dequantize_per_tensor_int8`。


    _RewriteInfo(
        _QUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS,
        _WrapperModule(_quantize_per_channel_int8),
        _WrapperModule(_reference_quantize_per_channel_int8),
        _replace_ph_qdq_per_channel_replacement,
        _replace_ph_qdq_per_channel_replacement
    ),

创建另一个 `_RewriteInfo` 对象，用于重写信息。包括每通道整数8位量化的示例输入 `_QUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS`，以及相应的量化函数 `_quantize_per_channel_int8` 和参考量化函数 `_reference_quantize_per_channel_int8`。还包括两个部分的函数，这些函数将替换占位符 `_replace_ph_qdq_per_channel_replacement`。


    _RewriteInfo(
        _DEQUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS,
        _WrapperModule(_dequantize_per_channel_int8),
        _WrapperModule(_reference_dequantize_per_channel_int8),
        _replace_ph_qdq_per_channel_replacement,
        _replace_ph_qdq_per_channel_replacement
    ),

最后创建一个 `_RewriteInfo` 对象，用于重写信息。包括每通道整数8位反量化的示例输入 `_DEQUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS`，以及相应的反量化函数 `_dequantize_per_channel_int8` 和参考反量化函数 `_reference_dequantize_per_channel_int8`。还包括两个部分的函数，这些函数将替换占位符 `_replace_ph_qdq_per_channel_replacement`。
# 对给定模型进行参考表示的重写，返回修改后的模型对象
def reference_representation_rewrite(model: GraphModule) -> GraphModule:
    # 移除模型中用于量化-反量化操作的张量重载
    remove_tensor_overload_for_qdq_ops(model)

    # 遍历重写信息列表，依次处理每一个重写信息对象
    for rewrite_info in _REWRITE_INFO_LIST:
        # 获取重写信息对象的示例输入
        example_inputs = rewrite_info.example_inputs
        # 获取当前重写信息对象的模式（待替换的模式）
        pattern = rewrite_info.pattern
        # 获取当前重写信息对象的替换（替换后的模式）
        replacement = rewrite_info.replacement
        # 获取当前重写信息对象的模式后转换函数
        pattern_post_trans = rewrite_info.pattern_post_trans
        # 获取当前重写信息对象的替换后转换函数
        replacement_post_trans = rewrite_info.replacement_post_trans
        
        # 根据示例输入获取模式的图形模块对象
        pattern = _get_aten_graph_module_for_pattern(pattern, example_inputs)  # type: ignore[arg-type, assignment]
        # 移除模式图形模块对象中的量化-反量化操作的张量重载
        remove_tensor_overload_for_qdq_ops(pattern)  # type: ignore[arg-type]
        
        # 根据示例输入获取替换的图形模块对象
        replacement = _get_aten_graph_module_for_pattern(replacement, example_inputs)  # type: ignore[arg-type, assignment]
        # 移除替换图形模块对象中的量化-反量化操作的张量重载
        remove_tensor_overload_for_qdq_ops(replacement)  # type: ignore[arg-type]
        
        # 如果存在模式后转换函数，则对模式进行后转换操作
        if pattern_post_trans:
            pattern = pattern_post_trans(pattern)
        # 如果存在替换后转换函数，则对替换进行后转换操作
        if replacement_post_trans:
            replacement = replacement_post_trans(replacement)
        
        # 重新编译模式的图形模块对象
        pattern.recompile()  # type: ignore[attr-defined]
        # 重新编译替换的图形模块对象
        replacement.recompile()  # type: ignore[attr-defined]
        
        # 使用模式匹配函数替换模型中的模式匹配项，并返回匹配结果
        matches = replace_pattern(model, pattern, replacement)
    
    # 返回经过所有重写操作后的模型对象
    return model
```
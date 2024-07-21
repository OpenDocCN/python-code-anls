# `.\pytorch\torch\onnx\symbolic_caffe2.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import importlib  # 导入模块动态加载库
import inspect    # 导入检查模块信息的库

# 从 torch.onnx 中导入符号操作的辅助函数和 opset9 版本
from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
# 导入内部 JIT 工具和注册函数
from torch.onnx._internal import jit_utils, registration


def register_quantized_ops(domain: str, version: int):
    # 注册所有量化操作
    module = importlib.import_module("torch.onnx.symbolic_caffe2")  # 动态导入符号化 Caffe2 模块
    quant_version_ops = inspect.getmembers(module)  # 获取模块中的成员列表
    aten_q_ops = {  # 定义需要注册的 ATen 量化操作集合
        "relu",
        "_empty_affine_quantized",
        "dequantize",
        "quantize_per_tensor",
        "upsample_nearest2d",
        "avg_pool2d",
        "reshape",
        "slice",
        "cat",
        "max_pool2d",
        "sigmoid",
    }
    for op, func in quant_version_ops:
        name = f"{domain}::{op}"  # 构建完整的操作名称
        if inspect.isfunction(func) and not registration.registry.is_registered_op(
            name, version
        ):
            if op in aten_q_ops:
                # 覆盖内置的 ATen 操作
                registration.registry.register(
                    f"aten::{op}", version, func, custom=True
                )
            registration.registry.register(name, version, func)  # 注册操作到注册表中


def _permute_helper(g: jit_utils.GraphContext, input, axes):
    quant_args = {
        "axes_i": axes,
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),  # 获取输入节点的 Y_scale 属性
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),  # 获取输入节点的 Y_zero_point 属性
    }
    output = g.op("_caffe2::Int8Transpose", input, **quant_args)  # 执行 Int8 转置操作
    symbolic_helper._quantized_ops.add(output)  # 将输出添加到量化操作集合中
    return output  # 返回操作的输出结果


def nchw2nhwc(g: jit_utils.GraphContext, input):
    axes = [0, 2, 3, 1]  # 定义 NCHW 到 NHWC 的轴变换顺序
    return _permute_helper(g, input, axes)  # 调用帮助函数进行转置操作


def nhwc2nchw(g: jit_utils.GraphContext, input):
    axes = [0, 3, 1, 2]  # 定义 NHWC 到 NCHW 的轴变换顺序
    return _permute_helper(g, input, axes)  # 调用帮助函数进行转置操作


def linear_prepack(g: jit_utils.GraphContext, weight, bias):
    # 映射到虚拟的 caffe2 预打包节点
    # 在 onnx -> c2 转换期间，可以从此节点查找原始的权重和偏置
    output = g.op("_caffe2::WeightPrepack", weight, bias)  # 执行权重预打包操作
    symbolic_helper._quantized_ops.add(output)  # 将输出添加到量化操作集合中
    return output  # 返回操作的输出结果


@symbolic_helper.parse_args("v", "v", "v", "f", "i")
def linear(g: jit_utils.GraphContext, input, weight, bias, scale, zero_point):
    kwargs = {
        "Y_scale_f": scale,  # 设置输出的比例因子
        "Y_zero_point_i": zero_point,  # 设置输出的零点
    }
    output = g.op("_caffe2::Int8FC", input, weight, bias, **kwargs)  # 执行 Int8 全连接操作
    symbolic_helper._quantized_ops.add(output)  # 将输出添加到量化操作集合中
    return output  # 返回操作的输出结果


def conv_prepack(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    # 映射到虚拟的 caffe2 预打包节点
    # 在 onnx -> c2 转换期间，可以从此节点查找原始的权重和偏置
    output = g.op("_caffe2::WeightPrepack", input, weight, bias)  # 执行权重预打包操作
    symbolic_helper._quantized_ops.add(output)  # 将输出添加到量化操作集合中
    return output  # 返回操作的输出结果


@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "f", "i")
def conv2d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups
):
    stride,              # 步幅参数，用于定义滑动窗口移动的步长
    padding,             # 填充参数，指定输入数据的边界填充大小
    dilation,            # 扩张参数，用于卷积核中空洞的扩展倍率
    groups,              # 分组参数，定义卷积操作中的通道分组数量
    scale,               # 缩放参数，用于乘以权重或输入以调整数值范围
    zero_point,          # 零点参数，通常用于量化操作中表示零的数值
#`
):
    # 获取权重节点的形状的第二和第三个维度作为卷积核大小
    kernel_size = weight.node()["shape"][1:3]
    # 创建包含卷积操作参数的字典
    kwargs = {
        "strides_i": stride,            # 步幅
        "pads_i": padding + padding,    # 填充，前后两个相同
        "dilations_i": dilation,        # 膨胀率
        "group_i": groups,              # 分组数
        "kernels_i": kernel_size,       # 卷积核大小
        "order_s": "NHWC",              # 输入和输出的维度顺序
        "Y_scale_f": scale,             # 输出的缩放因子
        "Y_zero_point_i": zero_point,   # 输出的零点
    }
    # 执行 _caffe2::Int8Conv 操作
    output = g.op("_caffe2::Int8Conv", input, weight, bias, **kwargs)
    # 将输出添加到量化操作集合中
    symbolic_helper._quantized_ops.add(output)
    return output

@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "f", "i")
def conv2d_relu(
    g: jit_utils.GraphContext,         # 图上下文对象
    input,                             # 输入张量
    weight,                            # 权重张量
    bias,                              # 偏置张量
    stride,                            # 步幅
    padding,                           # 填充
    dilation,                           # 膨胀率
    groups,                             # 分组数
    scale,                              # 输出的缩放因子
    zero_point,                         # 输出的零点
):
    # 获取权重节点的形状的第二和第三个维度作为卷积核大小
    kernel_size = weight.node()["shape"][1:3]
    # 创建包含卷积操作参数的字典
    kwargs = {
        "strides_i": stride,            # 步幅
        "pads_i": padding + padding,    # 填充，前后两个相同
        "dilations_i": dilation,        # 膨胀率
        "group_i": groups,              # 分组数
        "kernels_i": kernel_size,       # 卷积核大小
        "order_s": "NHWC",              # 输入和输出的维度顺序
        "Y_scale_f": scale,             # 输出的缩放因子
        "Y_zero_point_i": zero_point,   # 输出的零点
    }
    # 执行 _caffe2::Int8ConvRelu 操作
    output = g.op("_caffe2::Int8ConvRelu", input, weight, bias, **kwargs)
    # 将输出添加到量化操作集合中
    symbolic_helper._quantized_ops.add(output)
    return output

@symbolic_helper.parse_args("v", "v", "f", "i")
def add(g: jit_utils.GraphContext, input_a, input_b, scale, zero_point):
    # 创建包含加法操作参数的字典
    kwargs = {
        "Y_scale_f": scale,              # 输出的缩放因子
        "Y_zero_point_i": zero_point,    # 输出的零点
    }
    # 执行 _caffe2::Int8Add 操作
    output = g.op("_caffe2::Int8Add", input_a, input_b, **kwargs)
    # 将输出添加到量化操作集合中
    symbolic_helper._quantized_ops.add(output)
    return output

@symbolic_helper.parse_args("v")
def relu(g: jit_utils.GraphContext, input):
    # 检查输入是否在量化操作集合中
    if input not in symbolic_helper._quantized_ops:
        # 如果不在，执行常规的 ReLU 操作
        return opset9.relu(g, input)
    # 创建包含 ReLU 操作参数的字典
    kwargs = {
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),        # 输出的缩放因子
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),  # 输出的零点
    }
    # 执行 _caffe2::Int8Relu 操作
    output = g.op("_caffe2::Int8Relu", input, **kwargs)
    # 将输出添加到量化操作集合中
    symbolic_helper._quantized_ops.add(output)
    return output

@symbolic_helper.parse_args("v", "f", "i", "t")
def quantize_per_tensor(g: jit_utils.GraphContext, input, scale, zero_point, dtype):
    # 创建包含量化操作参数的字典
    kwargs = {
        "Y_scale_f": scale,              # 输出的缩放因子
        "Y_zero_point_i": zero_point,    # 输出的零点
    }
    # 执行 _caffe2::Int8Quantize 操作
    output = g.op("_caffe2::Int8Quantize", input, **kwargs)
    # 将输出添加到量化操作集合中
    symbolic_helper._quantized_ops.add(output)
    return output

@symbolic_helper.parse_args("v")
def dequantize(g: jit_utils.GraphContext, input):
    # 执行 _caffe2::Int8Dequantize 操作
    return g.op("_caffe2::Int8Dequantize", input)

@symbolic_helper.parse_args("v", "t", "t", "t", "t", "t", "t", "t")
def _empty_affine_quantized(
    g: jit_utils.GraphContext,
    input,
    shape,
    scale,
    zero_point,
    dtype,
    pin_memory,
    memory_format,
    layout,
):
    # 返回输入张量
    return input

def upsample_nearest2d(
    g: jit_utils.GraphContext,
    input,
    output_size,
    align_corners=None,
    scales_h=None,
    scales_w=None,
):
    # 代码的实现需要继续完成
    # 检查输入是否不在符号助手的量化操作集合中
    if input not in symbolic_helper._quantized_ops:
        # 如果不在集合中，则调用opset9库中的upsample_nearest2d函数进行最近邻插值操作，并返回结果（类型忽略属性定义）
        return opset9.upsample_nearest2d(g, input, output_size, align_corners)  # type: ignore[attr-defined]

    # 解析输出大小参数output_size，确保其为整数或大小为2的元组
    output_size = symbolic_helper._parse_arg(output_size, "is")

    # 准备kwargs字典，用于传递给_Caffe2插值操作的参数
    kwargs = {
        "output_size_i": output_size,  # 指定输出大小
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),  # 获取输入节点的Y_scale属性
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),  # 获取输入节点的Y_zero_point属性
    }

    # 将输入数据从NCHW格式转换为NHWC格式
    input = nchw2nhwc(g, input)

    # 在图g中调用_Caffe2的Int8ResizeNearest操作，传入输入数据和kwargs参数
    output = g.op("_caffe2::Int8ResizeNearest", input, **kwargs)

    # 将输出数据从NHWC格式转换回NCHW格式
    output = nhwc2nchw(g, output)

    # 将输出添加到符号助手的量化操作集合中
    symbolic_helper._quantized_ops.add(output)

    # 返回最终处理后的输出
    return output
@symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
def max_pool2d(
    g: jit_utils.GraphContext,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    # 如果输入不在量化操作集合中，使用普通的max_pool2d操作
    if input not in symbolic_helper._quantized_ops:
        return opset9.max_pool2d(  # 调用Opset9的max_pool2d函数
            g, input, kernel_size, stride, padding, dilation, ceil_mode
        )
    # 构建用于量化的参数字典
    kwargs = {
        "strides_i": stride,
        "pads_i": padding + padding,
        "kernel_i": kernel_size[0],
        "order_s": "NHWC",
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),
    }
    # 将输入数据从NCHW格式转换为NHWC格式
    input = nchw2nhwc(g, input)
    # 使用Caffe2的Int8MaxPool操作进行量化最大池化
    output = g.op("_caffe2::Int8MaxPool", input, **kwargs)
    # 将输出数据从NHWC格式转换回NCHW格式
    output = nhwc2nchw(g, output)
    # 将输出添加到量化操作集合中
    symbolic_helper._quantized_ops.add(output)
    # 返回量化后的输出
    return output


@symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
def avg_pool2d(
    g: jit_utils.GraphContext,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
):
    # 如果输入不在量化操作集合中，使用普通的avg_pool2d操作
    if input not in symbolic_helper._quantized_ops:
        return opset9.avg_pool2d(  # 调用Opset9的avg_pool2d函数
            g,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    # 构建用于量化的参数字典
    kwargs = {
        "strides_i": stride,
        "pads_i": padding + padding,
        "kernel_i": kernel_size[0],
        "order_s": "NHWC",
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),
    }
    # 将输入数据从NCHW格式转换为NHWC格式
    input = nchw2nhwc(g, input)
    # 使用Caffe2的Int8AveragePool操作进行量化平均池化
    output = g.op("_caffe2::Int8AveragePool", input, **kwargs)
    # 将输出数据从NHWC格式转换回NCHW格式
    output = nhwc2nchw(g, output)
    # 将输出添加到量化操作集合中
    symbolic_helper._quantized_ops.add(output)
    # 返回量化后的输出
    return output


def reshape(g: jit_utils.GraphContext, input, shape):
    # 如果输入不在量化操作集合中，使用普通的reshape操作
    if input not in symbolic_helper._quantized_ops:
        return opset9.reshape(g, input, shape)

    # 构建用于量化的参数字典
    kwargs = {
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),
    }
    # 使用Caffe2的Int8Reshape操作进行量化reshape
    output = g.op("_caffe2::Int8Reshape", input, shape, **kwargs)
    # 将输出添加到量化操作集合中
    symbolic_helper._quantized_ops.add(output)
    # 返回量化后的输出
    return output


@symbolic_helper.parse_args("v", "v", "v", "v", "i")
def slice(g: jit_utils.GraphContext, input, dim, start, end, step):
    # 如果输入不在量化操作集合中，使用普通的slice操作
    if input not in symbolic_helper._quantized_ops:
        return opset9.slice(g, input, dim, start, end, step)

    # 如果步长不为1，则无法导出量化slice
    if step != 1:
        raise RuntimeError("ONNX quantized slice export only works for step 1.")
    # 解析slice的起始、结束和维度参数
    start = symbolic_helper._parse_arg(start, "i")
    end = symbolic_helper._parse_arg(end, "i")
    dim = symbolic_helper._parse_arg(dim, "i")
    # 创建一个包含关键字参数的字典，用于调用 _caffe2::Int8Slice 操作
    kwargs = {
        "start_idx_i": start,  # 指定切片的起始索引
        "end_idx_i": end,      # 指定切片的结束索引
        "dim_i": dim,          # 指定切片的维度
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),        # 获取输入节点的 Y_scale 属性作为 Y_scale_f 参数
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),  # 获取输入节点的 Y_zero_point 属性作为 Y_zero_point_i 参数
    }
    # 调用 _caffe2::Int8Slice 操作，对输入 input 进行切片操作，使用上述关键字参数
    output = g.op("_caffe2::Int8Slice", input, **kwargs)
    # 将输出添加到符号助手的量化操作集合中
    symbolic_helper._quantized_ops.add(output)
    # 返回切片操作的输出结果
    return output
# 定义一个函数 cat，用于在图形上下文 g 中连接张量列表 tensor_list 的数据
# dim 是连接的维度
# scale 和 zero_point 是可选参数，用于量化操作
def cat(g: jit_utils.GraphContext, tensor_list, dim, scale=None, zero_point=None):
    # 将 tensor_list 解包成张量列表 tensors
    tensors = symbolic_helper._unpack_list(tensor_list)
    # 取出第一个张量作为输入
    input = tensors[0]
    # 如果输入不在 _quantized_ops 中，使用 opset9 中的 cat 函数进行连接
    if input not in symbolic_helper._quantized_ops:
        return opset9.cat(g, tensor_list, dim)

    # 解析 dim 参数为整数
    dim = symbolic_helper._parse_arg(dim, "i")
    # 构建 kwargs 字典，包含输出的 scale 和 zero_point
    kwargs = {
        "Y_scale_f": tensors[0].node()["Y_scale"],  # 输出的 scale
        "Y_zero_point_i": tensors[0].node()["Y_zero_point"],  # 输出的 zero_point
    }
    # 使用 "_caffe2::Int8Concat" 操作在 g 上连接张量 tensors，指定 axis_i=dim 和 kwargs
    output = g.op("_caffe2::Int8Concat", *tensors, axis_i=dim, **kwargs)
    # 将 output 添加到 _quantized_ops 集合中
    symbolic_helper._quantized_ops.add(output)
    # 返回操作的输出
    return output


# 使用装饰器 @symbolic_helper.parse_args("v") 定义 sigmoid 函数
# 函数将在图形上下文 g 中应用 sigmoid 操作到输入 input
def sigmoid(g: jit_utils.GraphContext, input):
    # 如果输入不在 _quantized_ops 中，使用 opset9 中的 sigmoid 函数
    if input not in symbolic_helper._quantized_ops:
        return opset9.sigmoid(g, input)
    
    # Caffe2 期望输出的 scale 是 1/2^8，zero_point 是 0（quint8 类型）
    out_scale = 1.0 / 256
    zero_point = 0
    # 构建 kwargs 字典，包含输出的 scale 和 zero_point
    kwargs = {
        "Y_scale_f": out_scale,     # 输出的 scale
        "Y_zero_point_i": zero_point,   # 输出的 zero_point
    }
    # 使用 "_caffe2::Int8Sigmoid" 操作在 g 上应用 sigmoid 到输入 input，指定 kwargs
    output = g.op("_caffe2::Int8Sigmoid", input, **kwargs)
    # 将 output 添加到 _quantized_ops 集合中
    symbolic_helper._quantized_ops.add(output)
    # 返回操作的输出
    return output
```
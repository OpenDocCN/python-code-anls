# `.\pytorch\torch\_inductor\mkldnn_ir.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和类型定义
from typing import Any, List, Optional, Set

import sympy  # 导入 sympy 模块

import torch  # 导入 torch 模块

from torch._prims_common import make_channels_last_strides_for  # 导入 torch._prims_common 模块中的函数

from .ir import (  # 导入自定义模块 .ir 中的多个对象和函数
    ExternKernelAlloc,
    FixedLayout,
    FlexibleLayout,
    ir_node_to_tensor,
    IRNode,
    is_contiguous_storage_and_layout,
    Layout,
    mark_node_as_mutating,
    may_convert_to_optional,
    MultiOutput,
    MultiOutputLayout,
    NoneLayout,
    TensorBox,
)

from .utils import convert_shape_to_inductor, pad_listlike  # 导入自定义模块 .utils 中的函数

from .virtualized import V  # 导入自定义模块 .virtualized 中的对象 V


def _prepare_convolution_fusion_create(
    cls,  # 类型参数 cls，通常表示类本身
    x: "TensorBox",  # 输入张量 x，类型为 TensorBox
    weight: "TensorBox",  # 权重张量 weight，类型为 TensorBox
    bias: "TensorBox",  # 偏置张量 bias，类型为 TensorBox
    padding: List[int],  # 填充参数，列表类型，包含整数
    stride: List[int],  # 步幅参数，列表类型，包含整数
    dilation: List[int],  # 膨胀参数，列表类型，包含整数
    groups: int,  # 分组数，整数类型
    transposed: bool = False,  # 是否转置卷积，默认为 False
    output_padding: Optional[List[int]] = None,  # 输出填充参数，可选的整数列表，默认为 None
):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for convolution post-op fusion's create function, including deciding the output
    layout (channels first or channels last), realizing inputs and make them etc. The
    function only supports the CPU device since conv post-op fusion kernel is only
    supported on CPU right now.
    """
    
    # 从 aten/src/ATen/native/ConvUtils.h 中移植的函数 _conv_input_size
    def _conv_input_size(
        output_size, weight_size, padding, output_padding, stride, dilation, groups
    ):
        """
        计算卷积操作的输入大小。
        
        参数：
        output_size - 输出大小的列表
        weight_size - 权重张量的大小
        padding - 填充参数的列表
        output_padding - 输出填充参数的列表
        stride - 步幅参数的列表
        dilation - 膨胀参数的列表
        groups - 分组数
        
        返回：
        输入大小的列表
        """
        assert len(output_size) == len(weight_size), "Expect input dim == weight dim"
        dim = len(output_size)
        assert dim > 2, "Expect input dim > 2"

        BATCH_DIM = 0
        WEIGHT_INPUT_CHANNELS_DIM = 1
        input_size = []
        input_size.append(output_size[BATCH_DIM])
        input_size.append(weight_size[WEIGHT_INPUT_CHANNELS_DIM] * groups)
        for d in range(2, dim):
            kernel = (weight_size[d] - 1) * dilation[d - 2] + 1
            input_size_d = (
                (output_size[d] - 1) * stride[d - 2]
                - (padding[d - 2] * 2)
                + kernel
                + output_padding[d - 2]
            )
            input_size.append(input_size_d)
        return list(map(int, input_size))

    # 计算原始反卷积权重的大小
    def _original_deconv_weight_size(
        prepacked_weight,
        groups,
    ):
        """
        根据预打包的权重计算原始反卷积权重的大小。
        
        参数：
        prepacked_weight - 预打包的权重张量
        groups - 分组数
        
        返回：
        原始权重的大小列表
        """
        prepacked_weight_size = prepacked_weight.size()
        dim = len(prepacked_weight_size)
        assert dim > 2, "Expect weight dim > 2"
        if groups > 1:
            weight_size = []
            weight_size.append(prepacked_weight_size[1] * groups)
            weight_size.append(prepacked_weight_size[0] / groups)
            for d in range(2, dim):
                weight_size.append(prepacked_weight_size[d])
        else:
            weight_size = prepacked_weight.transpose(0, 1).size()
        return weight_size

    x.realize()  # 实现输入张量 x
    # 实现权重的实现化操作
    weight.realize()
    # 如果存在偏置项，也进行实现化操作
    if bias is not None:
        bias.realize()
    # 进入虚拟图模式，使用虚拟图进行操作
    with V.graph.fake_mode:
        # TODO <Leslie> cleaned up the fake_tensor trace as Linear implementation
        # 将输入节点转换为张量，确保其形状正确
        x_fake = ir_node_to_tensor(x, guard_shape=True)
        # 将权重节点转换为张量，确保其形状正确
        weight_fake = ir_node_to_tensor(weight, guard_shape=True)
        # 计算维度数，排除掉前两个维度（通常是批次和通道）
        dims = len(x_fake.size()) - 2
        # 确保填充参数的长度在有效范围内
        assert 0 < len(padding) <= dims
        # 确保扩张参数的长度在有效范围内
        assert 0 < len(dilation) <= dims
        # 确保步长参数的长度在有效范围内
        assert 0 < len(stride) <= dims
        # 根据维度数对填充参数进行填充
        padding = pad_listlike(padding, dims)
        # 根据维度数对扩张参数进行填充
        dilation = pad_listlike(dilation, dims)
        # 根据维度数对步长参数进行填充
        stride = pad_listlike(stride, dims)
        # 如果输出填充参数为None，则初始化为与维度数对应的填充列表
        if output_padding is None:
            output_padding = pad_listlike([0], dims)
        else:
            # 确保输出填充参数的长度在有效范围内
            assert 0 < len(output_padding) <= dims
            # 根据维度数对输出填充参数进行填充
            output_padding = pad_listlike(output_padding, dims)
        # 确保分组参数为整数类型
        assert isinstance(groups, int)
        # 如果是转置卷积操作
        if transposed:
            # 当转置卷积时，由于预先打包的一维NN权重大小与PyTorch权重不同，无法直接运行aten conv。
            # 在这里我们从输入参数推断输出大小：
            weight_size = _original_deconv_weight_size(weight_fake, groups)
            input_size = x_fake.size()
            output_size = _conv_input_size(
                input_size,
                weight_size,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )
        else:
            # 如果存在偏置项，将其转换为张量，确保其形状正确；否则为None
            bias_fake = (
                ir_node_to_tensor(bias, guard_shape=True) if bias is not None else bias
            )
            # 调用torch.ops.aten.convolution函数进行卷积操作
            output = torch.ops.aten.convolution(
                x_fake,
                weight_fake,
                bias_fake,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            )
            # 获取输出张量的大小
            output_size = output.size()

        # 创建所需的步长顺序列表，逆序排列步长参数的索引
        req_stride_order = [0] + list(reversed(range(1, len(stride) + 1)))
        req_stride_order = [len(req_stride_order)] + req_stride_order

    # 对输入张量应用所需的步长顺序调整
    x = cls.require_stride_order(x, req_stride_order)

    # 如果存在动态形状，不进行权重预打包操作
    # 在静态形状情况下，由于权重已经预打包，我们总是强制输出为卷积内核中的通道顺序。
    # 在动态形状情况下，对于通道为1的输入（如大小为(s0, 1, 28, 28)和步长(784, 784, 28, 1)的张量），
    # x = cls.require_stride_order(x, req_stride_order) 其中 req_stride_order 是按通道逆序排列的顺序
    # 不会改变张量的步长，因为大小为1的维度的步长被忽略。但在卷积内核中，该张量被视为按通道优先，
    # 输出将以连续格式呈现。为了对齐卷积内核的行为，我们在这种情况下将输出步长设置为连续而不是通道顺序。
    dynamic_shapes = not all(isinstance(i, int) for i in (output_size))
    # 如果 dynamic_shapes 为真并且 x 是连续存储和布局的话
    if dynamic_shapes and is_contiguous_storage_and_layout(x):
        # 根据输出大小创建灵活的布局连续步长
        output_stride = FlexibleLayout.contiguous_strides(output_size)
    else:
        # 否则创建适合输出大小的通道最后步长
        output_stride = make_channels_last_strides_for(output_size)

    # 断言输入张量和权重张量都在 CPU 上
    assert x.get_device().type == "cpu" and weight.get_device().type == "cpu"
    # 将输入张量和权重张量放入输入列表中
    inputs = [x, weight]

    # 创建固定布局对象，传入设备、数据类型、输出大小的感应器形状、输出步长的感应器形状
    kernel_layout = FixedLayout(
        x.get_device(),
        x.get_dtype(),
        convert_shape_to_inductor(output_size),
        convert_shape_to_inductor(output_stride),
    )
    # 构建常量参数列表，包括填充、步幅、扩展率、分组
    constant_args = [padding, stride, dilation, groups]
    # 如果是反卷积操作，则在第二个位置插入输出填充
    if transposed:
        constant_args.insert(1, output_padding)

    # 如果存在偏置，则将偏置加入输入列表中；否则将 None 插入常量参数列表的第一个位置
    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    
    # 返回输入列表、常量参数列表、内核布局对象以及请求的步长顺序
    return inputs, constant_args, kernel_layout, req_stride_order
def _prepare_linear_fusion_create(
    cls,
    x: "TensorBox",
    weight: "TensorBox",
    bias: "TensorBox",
):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for linear post-op fusion's create function. The function only supports the CPU device
    since linear post-op fusion kernel is only supported on CPU right now.
    """
    # 实现 TensorBox 对象的实现
    x.realize()
    weight.realize()
    # 如果存在偏置，则实现偏置 TensorBox 对象
    if bias is not None:
        bias.realize()

    # 获取输入张量 x 的尺寸，去除最后一个元素
    *m, _ = x.get_size()
    # 权重在 qlinear weight 预打包过程中已被转置
    # 参考链接：https://github.com/pytorch/pytorch/blob/4979f9c0d72490970e2019bb1d2284f83d93f76b/
    # aten/src/ATen/native/quantized/cpu/qlinear_prepack.cpp#L291
    _, oc = weight.get_size()
    # 计算输出尺寸
    output_size = list(m) + [oc]
    # 请求的步长顺序为输入尺寸的反向顺序
    req_stride_order = list(reversed(range(len(x.get_size()))))

    # 调整输入张量 x 的步长顺序
    x = cls.require_stride_order(x, req_stride_order)
    # 断言输入张量 x 和权重张量都在 CPU 设备上
    assert x.get_device().type == "cpu" and weight.get_device().type == "cpu"
    # 构建输入张量列表
    inputs = [x, weight]

    # 计算输出的步长
    output_stride = FlexibleLayout.contiguous_strides(output_size)
    # 创建固定布局对象 kernel_layout
    kernel_layout = FixedLayout(
        x.get_device(),
        x.get_dtype(),
        output_size,
        output_stride,
    )
    # 初始化常量参数列表
    constant_args: List[Any] = []

    # 如果存在偏置，则添加到 inputs 列表中；否则在常量参数列表中插入空值
    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    # 返回输入列表、常量参数列表、kernel_layout 对象和请求的步长顺序列表
    return inputs, constant_args, kernel_layout, req_stride_order
    # 定义一个类方法用于创建卷积操作对象
    def create(
        cls,
        x: "TensorBox",                     # 输入张量 x，类型为 TensorBox
        weight: "TensorBox",                 # 卷积核权重张量 weight，类型为 TensorBox
        bias: "TensorBox",                   # 偏置张量 bias，类型为 TensorBox
        padding_: List[int],                 # 填充大小列表 padding_
        stride_: List[int],                  # 步幅大小列表 stride_
        dilation_: List[int],                # 膨胀大小列表 dilation_
        groups: int,                         # 分组数
        attr,                                # 属性参数
        scalars: Optional[List[Any]],        # 可选的标量参数列表 scalars
        algorithm,                           # 算法选择参数
    ):
        # 调用内部函数 _prepare_convolution_fusion_create 进行卷积融合操作的准备
        (inputs, constant_args, kernel_layout, _) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        # 将属性参数、可选的标量参数列表以及算法选择参数添加到常量参数列表中
        constant_args = constant_args + [
            attr,
            may_convert_to_optional(scalars),
            algorithm,
        ]
        # 创建并返回一个卷积一元操作对象 ConvolutionUnary
        return ConvolutionUnary(
            layout=kernel_layout,             # 使用的内核布局
            inputs=inputs,                    # 输入列表
            constant_args=constant_args,      # 常量参数列表
        )
# 定义一个名为 ConvolutionBinary 的类，继承自 ExternKernelAlloc 类
class ConvolutionBinary(ExternKernelAlloc):
    # 初始化方法，接受多个参数，包括布局、输入、常量参数等
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        cpp_constant_args=(),
    ):
        # 调用父类 ExternKernelAlloc 的初始化方法，传递布局、输入、常量参数等
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,  # 这里为 None，表示没有传递额外的参数
            python_kernel_name="torch.ops.mkldnn._convolution_pointwise.binary",  # Python 内核名称
            cpp_kernel_name="mkldnn::_convolution_pointwise",  # C++ 内核名称
        )
        # 设置 C++ 内核的重载名称为 "binary"
        self.cpp_kernel_overload_name = "binary"
        # 设置 C++ 内核的键为 "convolution_pointwise_binary"
        self.cpp_kernel_key = "convolution_pointwise_binary"
        # 设置 C++ 操作模式的字符串表示，描述了函数的参数结构
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& other_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view binary_attr,
                c10::optional<at::Scalar> alpha,
                c10::optional<c10::string_view> unary_attr,
                torch::List<c10::optional<at::Scalar>> unary_scalars,
                c10::optional<c10::string_view> unary_algorithm)"""
        # 设置 C++ 常量参数
        self.cpp_constant_args = cpp_constant_args

    # 代码生成方法，生成外部内核分配并根据需要查找模式的包装器
    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )
        # 如果布局是 Layout 类的实例，则生成大小断言
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    # 类方法，用于创建 ConvolutionBinary 类的实例
    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        binary_attr: str,
        binary_alpha: Optional[float],
        unary_attr: Optional[str],
        unary_scalars: Optional[List[Any]],
        unary_algorithm: Optional[str],
    ):
        # 准备卷积融合创建所需的输入、常量参数、内核布局和请求的步长顺序
        inputs, constant_args, kernel_layout, req_stride_order = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        # 确保 other 的步长顺序与请求的步长顺序一致
        other = cls.require_stride_order(other, req_stride_order)
        # 将 other 插入到 inputs 的第二个位置
        inputs.insert(1, other)
        # 将 binary_attr、binary_alpha、unary_attr、unary_scalars、unary_algorithm 添加到常量参数中
        constant_args = constant_args + [
            binary_attr,
            binary_alpha,
            unary_attr,
            may_convert_to_optional(unary_scalars),
            unary_algorithm,
        ]
        # 返回一个新的 ConvolutionBinary 对象，传递布局和输入作为参数
        return ConvolutionBinary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        self,
        kernel_layout,
        inputs,
        constant_args=(),
    ):
        # 由于 op.call 的限制，需要将其他 (Tensor&) 放置在 input[0] 位置
        reordered_inputs = [inputs[1], inputs[0]] + inputs[2:]

        # 调用父类的构造函数，初始化对象
        super().__init__(
            kernel_layout,  # 卷积核布局
            reordered_inputs,  # 重新排序后的输入
            constant_args,  # 常数参数
            None,  # 暂时未指定的参数
            python_kernel_name="torch.ops.mkldnn._convolution_pointwise_.binary",  # Python 内核名称
            cpp_kernel_name="mkldnn::_convolution_pointwise_",  # C++ 内核名称
        )

        # 设置 C++ 内核重载名称
        self.cpp_kernel_overload_name = "binary"
        # 设置 C++ 内核关键字
        self.cpp_kernel_key = "convolution_pointwise_binary_"
        
        # TODO: op.call: input[0] should be at::Tensor&
        # 定义 C++ 操作的 schema，描述操作的输入和输出
        self.cpp_op_schema = """
            at::Tensor&(
                at::Tensor& other_t,
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view binary_attr,
                c10::optional<at::Scalar> alpha,
                c10::optional<c10::string_view> unary_attr,
                torch::List<c10::optional<at::Scalar>> unary_scalars,
                c10::optional<c10::string_view> unary_algorithm)"""

    # 生成代码的函数，用于生成外部调用的内核分配和 schema 查找（如果需要）
    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),  # 获取名称
            self.python_kernel_name,  # Python 内核名称
            self.cpp_kernel_name,  # C++ 内核名称
            self.codegen_args(),  # 生成代码所需的参数
            self.cpp_op_schema,  # C++ 操作的 schema
            self.cpp_kernel_key,  # C++ 内核关键字
            self.cpp_kernel_overload_name,  # C++ 内核重载名称
        )

    # 获取变异名称的函数，返回输入的第一个名称
    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    # 获取未支持符号定义的函数，返回一个空集合
    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    # 类方法，用于创建对象实例
    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        binary_attr: str,
        binary_alpha: Optional[float],
        unary_attr: Optional[str],
        unary_scalars: Optional[List[Any]],
        unary_algorithm: Optional[str],
    ):
        (
            inputs,
            constant_args,
            _,
            req_stride_order,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        # 准备卷积融合操作的输入、常数参数以及要求的步幅顺序
        other = cls.require_stride_order(other, req_stride_order)
        # 对输入的另一个参数进行步幅顺序要求
        inputs.insert(1, other)
        # 将其他参数插入到输入列表的第二个位置
        constant_args = constant_args + [
            binary_attr,
            binary_alpha,
            unary_attr,
            may_convert_to_optional(unary_scalars),
            unary_algorithm,
        ]
        # 将额外的常数参数追加到常数参数列表中
        packed = ConvolutionBinaryInplace(
            kernel_layout=NoneLayout(inputs[1].get_device()),  # type: ignore[arg-type]
            inputs=inputs,
            constant_args=constant_args,
        )
        # 创建就地二进制卷积操作对象
        mark_node_as_mutating(packed, inputs[1])
        # 将操作标记为原地修改，这意味着结果不是目标，而是正在被修改的输入
        # 初始化重新排序输入，因此 inputs[1] 变成了 packed.inputs[0]
        return packed.inputs[0]
        # 返回修改后的输入作为结果
# 定义 ConvolutionTransposeUnary 类，继承自 ExternKernelAlloc 类
class ConvolutionTransposeUnary(ExternKernelAlloc):
    # 初始化方法，接受多个参数：布局(layout)，输入(inputs)，常量参数(constant_args)
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        # 调用父类 ExternKernelAlloc 的初始化方法，传入相应参数
        super().__init__(
            layout,  # 布局参数
            inputs,  # 输入参数
            constant_args,  # 常量参数
            None,  # 不使用外部数据
            python_kernel_name="torch.ops.mkldnn._convolution_transpose_pointwise",  # Python 内核名称
            cpp_kernel_name="mkldnn::_convolution_transpose_pointwise",  # C++ 内核名称
        )
        # 设置 C++ 内核键
        self.cpp_kernel_key = "convolution_transpose_pointwise"
        # 定义 C++ 操作的 schema 字符串
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef output_padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    # 定义 codegen 方法，用于生成外部内核分配和查找 schema，传入 wrapper 对象
    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),  # 获取名称
            self.python_kernel_name,  # Python 内核名称
            self.cpp_kernel_name,  # C++ 内核名称
            self.codegen_args(),  # 代码生成参数
            self.cpp_op_schema,  # C++ 操作 schema
            self.cpp_kernel_key,  # C++ 内核键
        )

    # 类方法 create，用于创建 ConvolutionTransposeUnary 实例
    @classmethod
    def create(
        cls,
        x: "TensorBox",  # 输入张量 x
        weight: "TensorBox",  # 权重张量 weight
        bias: "TensorBox",  # 偏置张量 bias
        padding_: List[int],  # 填充列表
        output_padding_: List[int],  # 输出填充列表
        stride_: List[int],  # 步长列表
        dilation_: List[int],  # 膨胀列表
        groups_: int,  # 组数
        attr,  # 属性
        scalars: Optional[List[Any]],  # 标量列表（可选）
        algorithm,  # 算法
    ):
        transposed = True  # 标志：转置为 True
        # 调用 _prepare_convolution_fusion_create 函数，获取输入、常量参数、内核布局和占位符
        (
            inputs,
            constant_args,
            kernel_layout,
            _,  # 最后一个值不使用
        ) = _prepare_convolution_fusion_create(
            cls,
            x,
            weight,
            bias,
            padding_,
            stride_,
            dilation_,
            groups_,
            transposed,
            output_padding_,
        )
        # 将属性、标量（可能转换为可选类型）、算法添加到常量参数列表中
        constant_args = constant_args + [
            attr,
            may_convert_to_optional(scalars),
            algorithm,
        ]
        # 返回创建的 ConvolutionTransposeUnary 实例
        return ConvolutionTransposeUnary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )


# 定义 QConvPointWisePT2E 类，继承自 ExternKernelAlloc 类
class QConvPointWisePT2E(ExternKernelAlloc):
    # 初始化方法，接受多个参数：布局(layout)，输入(inputs)，常量参数(constant_args)
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        """
        if bias is not None
            - inputs = [x, w, b, weight_scale, weight_zp]
            - const_args is: [stride, padding, dilation, groups, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, weight_scale, weight_zp]
            - const_args is: [bias, stride, padding, dilation, groups, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        """
        # 判断是否有偏置项，确定输入列表和常量参数的内容
        self.has_bias = len(inputs) == 5
        # 调用父类的初始化方法，设置布局、输入列表、常量参数等
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.onednn.qconv2d_pointwise",
            cpp_kernel_name="onednn::qconv2d_pointwise",
        )
        # 设置 C++ 内核的键名
        self.cpp_kernel_key = "qconv2d_pointwise"
        # 设置 C++ 操作的模式字符串
        self.cpp_op_schema = """
            at::Tensor(
                at::Tensor act,
                double act_scale,
                int64_t act_zero_point,
                at::Tensor weight,
                at::Tensor weight_scales,
                at::Tensor weight_zero_points,
                c10::optional<at::Tensor> bias,
                torch::List<int64_t> stride,
                torch::List<int64_t> padding,
                torch::List<int64_t> dilation,
                int64_t groups,
                double output_scale,
                int64_t output_zero_point,
                c10::optional<c10::ScalarType> output_dtype,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""
    def codegen(self, wrapper):
        # 解析输入和常量参数
        args = [x.codegen_reference() for x in self.inputs]  # 生成输入参数的引用代码
        const_args = []
        const_args.extend(self.codegen_const_args())  # 生成常量参数的代码

        x = args[0]  # 提取输入张量 x
        packed_weight = args[1]  # 提取打包的权重张量
        bias = args[2] if self.has_bias else const_args[0]  # 提取偏置张量（如果存在偏置）或者使用常量中的第一个值
        w_scale, w_zp = args[-2], args[-1]  # 提取权重的缩放因子和零点
        (
            stride,
            padding,
            dilation,
            groups,
            x_scale,
            x_zp,
            o_scale,
            o_zp,
            output_dtype,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ) = const_args[-12:]  # 从常量参数中提取各种参数

        codegen_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            bias,
            stride,
            padding,
            dilation,
            groups,
            o_scale,
            o_zp,
            output_dtype,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        )
        # 调用包装器的方法生成外部内核并进行必要的分配和模式查找
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            codegen_args,
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)  # 如果布局是 Layout 类型，则生成大小断言

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: float,
        x_zero_point: int,
        qw: "TensorBox",  # qw
        w_scale: "TensorBox",
        w_zero_point: "TensorBox",
        bias: "TensorBox",
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        groups: int,
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        attr,
        scalars,
        algorithm,
        ):
            transposed = False
            output_padding = None
            (inputs, constant_args, kernel_layout, _) = _prepare_convolution_fusion_create(
                cls,
                qx,
                qw,
                bias,
                padding,
                stride,
                dilation,
                groups,
                transposed,
                output_padding,
            )
            # swap padding and stride to align with functional conv arg order
            # 如果没有偏置，交换常量参数列表中的第二个和第三个参数位置，以符合功能卷积参数顺序
            if bias is None:
                constant_args[1], constant_args[2] = constant_args[2], constant_args[1]
            else:
                # 如果有偏置，交换常量参数列表中的第一个和第二个参数位置
                constant_args[0], constant_args[1] = constant_args[1], constant_args[0]

            # 调用 w_scale 的 realize() 方法
            w_scale.realize()
            # 调用 w_zero_point 的 realize() 方法
            w_zero_point.realize()
            # 将 w_scale 和 w_zero_point 添加到输入列表中
            inputs = inputs + [w_scale, w_zero_point]
            # 将 x_scale, x_zero_point, output_scale, output_zero_point, output_dtype,
            # attr, scalars 经过 may_convert_to_optional 处理后添加到常量参数列表中
            constant_args = constant_args + [
                x_scale,
                x_zero_point,
                output_scale,
                output_zero_point,
                output_dtype,
                attr,
                may_convert_to_optional(scalars),
                algorithm,
            ]

            # 断言输出数据类型不为 None
            assert output_dtype is not None
            # 如果输出数据类型为 torch.float32 或 torch.bfloat16
            if output_dtype in [torch.float32, torch.bfloat16]:
                # 在 _prepare_convolution_fusion_create 中，我们使用 x.dtype（uint8）来创建 kernel_layout
                # 如果设置了 output_dtype 不为 None，则输出缓冲区应该使用 output_dtype 而不是 uint8。
                kernel_layout.dtype = output_dtype

            # 返回 QConvPointWisePT2E 对象，包括 kernel_layout, inputs, constant_args
            return QConvPointWisePT2E(
                layout=kernel_layout,
                inputs=inputs,
                constant_args=constant_args,
            )
class QConvPointWiseBinaryPT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        """
        需要输入/权重/输出的量化参数
        如果存在偏置项
            - inputs = [x, w, b, accum, w_scale, w_zp]
            - const_args = [stride, padding, dilation, groups, x_scale, x_zp, accum_scale, accum_zp, o_scale, o_zp,
            fp32_output, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        否则
            - inputs = [x, w, accum, w_scale, w_zp]
            - const_args = const_args is: [bias, stride, padding, dilation, groups, x_scale, x_zp, accum_scale,
            accum_zp, o_scale, o_zp, fp32_output, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        """
        # 检查是否有偏置项
        self.has_bias = len(inputs) == 6
        # 如果有偏置项，设置就地求和的索引
        self.idx_for_inplace_sum = 3 if self.has_bias else 2
        # 调用父类的初始化方法
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            # 设置 Python 内核名称
            python_kernel_name="torch.ops.onednn.qconv2d_pointwise.binary",
            # 设置 C++ 内核名称
            cpp_kernel_name="onednn::qconv2d_pointwise",
        )
        # 设置 C++ 内核重载名称
        self.cpp_kernel_overload_name = "binary"
        # 设置 C++ 内核关键字
        self.cpp_kernel_key = "qconv2d_pointwise_binary"
        # 设置 C++ 操作模式的架构
        self.cpp_op_schema = """
            at::Tensor(
                at::Tensor act,
                double act_scale,
                int64_t act_zero_point,
                at::Tensor accum,
                double accum_scale,
                int64_t accum_zero_point,
                at::Tensor weight,
                at::Tensor weight_scales,
                at::Tensor weight_zero_points,
                c10::optional<at::Tensor> bias,
                torch::List<int64_t> stride,
                torch::List<int64_t> padding,
                torch::List<int64_t> dilation,
                int64_t groups,
                double output_scale,
                int64_t output_zero_point,
                c10::optional<c10::ScalarType> output_dtype,
                c10::string_view binary_attr,
                c10::optional<at::Scalar> alpha,
                c10::optional<c10::string_view> attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""
    def codegen(self, wrapper):
        # 解析输入和常量参数
        args = [x.codegen_reference() for x in self.inputs]  # 生成输入参数的引用代码
        const_args = []
        const_args.extend(self.codegen_const_args())  # 生成常量参数的代码

        x = args[0]  # 提取输入张量 x
        packed_weight = args[1]  # 提取打包的权重数据
        bias = args[2] if self.has_bias else const_args[0]  # 根据是否有偏置选择合适的偏置值
        accum, w_scale, w_zp = args[-3], args[-2], args[-1]  # 提取累加器、权重的缩放因子和零点

        # 提取常量参数列表中的各种参数
        (
            stride,
            padding,
            dilation,
            groups,
            x_scale,
            x_zp,
            accum_scale,
            accum_zp,
            o_scale,
            o_zp,
            output_dtype,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ) = const_args[-16:]

        conv_args = (
            x,
            x_scale,
            x_zp,
            accum,
            accum_scale,
            accum_zp,
            packed_weight,
            w_scale,
            w_zp,
            bias,
            stride,
            padding,
            dilation,
            groups,
            o_scale,
            o_zp,
            output_dtype,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        )

        # 调用 wrapper 对象的方法，生成外部内核的分配和模式查找
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            conv_args,
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)  # 若布局是 Layout 类型，则生成大小断言代码

    def get_mutation_names(self):
        return [self.inputs[self.idx_for_inplace_sum].get_name()]  # 返回进行原地求和的输入张量名称列表

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()  # 返回空集合，表示没有未支持的符号定义

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale,
        x_zero_point,
        qaccum: "TensorBox",
        accum_scale,
        accum_zero_point,
        qw: "TensorBox",  # packed_weight
        w_scale,
        w_zero_point,
        bias: "TensorBox",
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        groups: int,
        output_scale: "TensorBox",
        output_zero_point: "TensorBox",
        output_dtype,
        binary_attr,
        alpha,
        unary_attr,
        unary_scalars,
        unary_algorithm,
        ):
            transposed = False
            output_padding = None
            (
                inputs,
                constant_args,
                kernel_layout,
                req_stride_order,
            ) = _prepare_convolution_fusion_create(
                cls,
                qx,
                qw,
                bias,
                padding,
                stride,
                dilation,
                groups,
                transposed,
                output_padding,
            )

            # 根据所需的步幅顺序调整累加器张量
            qaccum = cls.require_stride_order(qaccum, req_stride_order)
            inputs.append(qaccum)

            # 如果没有偏置项，交换常量参数中的第二个和第三个参数的位置
            if bias is None:
                constant_args[1], constant_args[2] = constant_args[2], constant_args[1]
            else:
                constant_args[0], constant_args[1] = constant_args[1], constant_args[0]

            # 计算权重和零点
            w_scale.realize()
            w_zero_point.realize()

            # 将权重的缩放因子和零点添加到输入列表中
            inputs = inputs + [w_scale, w_zero_point]
            constant_args = constant_args + [
                x_scale,
                x_zero_point,
                accum_scale,
                accum_zero_point,
                output_scale,
                output_zero_point,
                output_dtype,
                binary_attr,
                alpha,
                unary_attr,
                may_convert_to_optional(unary_scalars),
                unary_algorithm,
            ]

            # 断言二进制属性为"sum"，目前仅支持后处理操作为求和的情况
            assert (
                binary_attr == "sum"
            ), "For now, only post op sum is supported in QConvPointWiseBinaryPT2E."

            # 创建 QConvPointWiseBinaryPT2E 对象，用于量化卷积操作
            packed = QConvPointWiseBinaryPT2E(
                layout=NoneLayout(qaccum.get_device()),
                inputs=inputs,
                constant_args=constant_args,
            )
            # 将该操作标记为对累加器 qaccum 的原位修改
            mark_node_as_mutating(packed, qaccum)

            # 返回 packed 对象中用于原位求和的输入
            return packed.inputs[packed.idx_for_inplace_sum]
class MKLPackedLinear(ExternKernelAlloc):
    # MKL打包线性层的类，继承自ExternKernelAlloc类
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        # 初始化方法，接受布局、输入和常量参数作为参数
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.mkl._mkl_linear",
            cpp_kernel_name="mkl::_mkl_linear",
        )
        # 设置Python和C++内核的名称
        self.cpp_kernel_key = "mkl_linear"
        # 设置C++操作的模式字符串，定义了操作接口
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& self,
                const at::Tensor& mkl_weight_t,
                const at::Tensor& origin_weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                const int64_t prepack_batch_size)"""

    # 生成代码的方法，接受一个包装器对象作为参数
    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )

    @classmethod
    # 创建方法，用于创建MKLPackedLinear实例
    def create(cls, x, packed_w, orig_w, B, batch_size):
        # 确保输入x的步长为1，并实例化它
        x = cls.require_stride1(cls.realize_input(x))
        # 确保输入orig_w的步长为1，并实例化它
        orig_w = cls.require_stride1(cls.realize_input(orig_w))
        *m, _ = x.get_size()
        oc, _ = orig_w.get_size()
        output_size = list(m) + [oc]
        # 计算输出张量的步长
        output_stride = FlexibleLayout.contiguous_strides(output_size)
        inputs = [x, packed_w, orig_w]
        constant_args = [batch_size]
        if B is not None:
            inputs += [B]
        else:
            constant_args.insert(0, None)

        # 返回MKLPackedLinear实例
        return MKLPackedLinear(
            layout=FixedLayout(
                x.get_device(), x.get_dtype(), output_size, output_stride
            ),
            inputs=inputs,
            constant_args=constant_args,
        )


class LinearUnary(ExternKernelAlloc):
    # 线性一元操作类，继承自ExternKernelAlloc类
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        # 初始化方法，接受布局、输入和常量参数作为参数
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.mkldnn._linear_pointwise",
            cpp_kernel_name="mkldnn::_linear_pointwise",
        )
        # 设置C++操作的键名
        self.cpp_kernel_key = "linear_pointwise"
        # 设置C++操作的模式字符串，定义了操作接口
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    # 生成代码的方法，接受一个包装器对象作为参数
    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )

    @classmethod
    # 创建类方法，用于创建线性一元操作的实例
    def create(cls, x, w, B, attr, scalars, algorithm):
        # 确保输入数据是连续的并转换为实际输入
        x = cls.require_contiguous(cls.realize_input(x))
        w = cls.require_contiguous(cls.realize_input(w))

        # 获取输入张量 x 的尺寸（去除最后一个维度的所有维度）和最后一个维度的大小
        *m, ic = x.get_size()
        # 获取权重张量 w 的尺寸中的输出通道数和输入通道数
        oc, ic = w.get_size()
        # 将输入张量 x 和权重张量 w 添加到输入列表中
        inputs = [x, w]
        # 构建常量参数列表，包括属性 attr、标量列表 scalars（如果存在则使用，否则使用默认值[-1]）和算法名称 algorithm
        constant_args = [attr, scalars if scalars else [-1], algorithm]
        
        # 如果偏置 B 不为 None，则将其也转换为连续张量并添加到输入列表中；否则将 None 插入到常量参数列表的开头
        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)
        else:
            constant_args.insert(0, None)

        # 返回一个 LinearUnary 对象的实例，其中包括灵活的布局（包含设备、数据类型和尺寸）以及输入和常量参数
        return LinearUnary(
            layout=FlexibleLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=list(m) + [oc],
            ),
            inputs=inputs,
            constant_args=constant_args,
        )

    # 应用约束的方法，当前为空实现
    def apply_constraint(self):
        pass
# LinearBinary 类，继承自 ExternKernelAlloc 类
class LinearBinary(ExternKernelAlloc):
    # kernel 属性，指定了使用的 Torch 操作的路径
    kernel = "torch.ops.mkldnn._linear_pointwise.binary"

    # 构造方法，初始化对象
    def __init__(
        self,
        layout,  # 布局参数，描述数据的排列方式
        inputs,  # 输入参数，包含输入数据的列表
        constant_args=(),  # 常量参数，默认为空元组
    ):
        # 调用父类构造方法进行初始化
        super().__init__(
            layout,  # 设置布局参数
            inputs,  # 设置输入参数
            constant_args,  # 设置常量参数
            None,  # 未指定输出
            python_kernel_name="torch.ops.mkldnn._linear_pointwise.binary",  # Python 内核名称
            cpp_kernel_name="mkldnn::_linear_pointwise",  # C++ 内核名称
        )
        # 设置 C++ 内核的重载名称为 "binary"
        self.cpp_kernel_overload_name = "binary"
        # 设置 C++ 内核的键为 "linear_pointwise_binary"
        self.cpp_kernel_key = "linear_pointwise_binary"
        # 定义 C++ 操作的模式
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& other_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                c10::string_view attr)
        """

    # 代码生成方法，生成外部内核分配和必要的模式信息
    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),  # 获取对象名称
            self.python_kernel_name,  # Python 内核名称
            self.cpp_kernel_name,  # C++ 内核名称
            self.codegen_args(),  # 生成代码所需的参数
            self.cpp_op_schema,  # C++ 操作的模式信息
            self.cpp_kernel_key,  # C++ 内核的键
            self.cpp_kernel_overload_name,  # C++ 内核的重载名称
        )

    # 类方法，用于创建 LinearBinary 实例
    @classmethod
    def create(cls, x, y, w, B, attr):
        # 确保输入 x 是连续的，并将其实现为输入
        x = cls.require_contiguous(cls.realize_input(x))
        # 确保输入 y 是连续的，并将其实现为输入
        y = cls.require_contiguous(cls.realize_input(y))
        # 确保输入 w 是连续的，并将其实现为输入
        w = cls.require_contiguous(cls.realize_input(w))

        # 解析输入 x 的大小，并取出最后一个维度作为 ic（输入通道数）
        *m, ic = x.get_size()
        # 解析输入 w 的大小，并取出第一个维度作为 oc（输出通道数），ic 保持不变
        oc, ic = w.get_size()

        # 构建输入参数列表
        inputs = [x, y, w]
        # 构建常量参数列表，初始为空列表
        constant_args = [attr]
        # 如果存在偏置 B，则确保其连续，并添加到输入参数列表中；否则将 B 插入常量参数列表的第一个位置
        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)
        else:
            constant_args.insert(0, B)

        # 返回创建的 LinearBinary 实例
        return LinearBinary(
            layout=FlexibleLayout(  # 创建 FlexibleLayout 布局对象
                device=x.get_device(),  # 设备属性从 x 中获取
                dtype=x.get_dtype(),  # 数据类型从 x 中获取
                size=list(m) + [oc],  # 大小为 m + [oc]
            ),
            inputs=inputs,  # 输入参数列表
            constant_args=constant_args,  # 常量参数列表
        )

    # 应用约束的方法，这里什么也不做
    def apply_constraint(self):
        pass


# QLinearPointwisePT2E 类，继承自 ExternKernelAlloc 类
class QLinearPointwisePT2E(ExternKernelAlloc):
    # 构造方法，初始化对象
    def __init__(
        self,
        layout,  # 布局参数，描述数据的排列方式
        inputs,  # 输入参数，包含输入数据的列表
        constant_args=(),  # 常量参数，默认为空元组
        has_bias=True,  # 是否具有偏置，默认为 True
        x_scale_zp_are_tensors=False,  # x_scale_zp_are_tensors 参数，默认为 False
        """
        if bias is not None
            - inputs = [x, w, b, weight_scale, weight_zp]
            - const_args is: [x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, weight_scale, weight_zp]
            - const_args is: [bias, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        """
        # 初始化类的实例属性
        self.has_bias = has_bias
        self.x_scale_zp_are_tensors = x_scale_zp_are_tensors
        # 调用父类构造函数，初始化对象
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name=(
                "torch.ops.onednn.qlinear_pointwise.tensor"
                if x_scale_zp_are_tensors
                else "torch.ops.onednn.qlinear_pointwise.default"
            ),
            cpp_kernel_name="onednn::qlinear_pointwise",
        )
        # 设置 C++ 内核的重载名称
        self.cpp_kernel_overload_name = "tensor" if x_scale_zp_are_tensors else ""
        # 设置 C++ 内核的关键字
        self.cpp_kernel_key = "qlinear_pointwise"
        # 确定 x_scale 和 x_zero_point 的类型字符串
        x_scale_type_str, x_zp_type_str = (
            ("at::Tensor", "at::Tensor")
            if x_scale_zp_are_tensors
            else ("double", "int64_t")
        )
        # 设置 C++ 操作的模式字符串
        self.cpp_op_schema = f"""
            at::Tensor(
                at::Tensor act,
                {x_scale_type_str} act_scale,
                {x_zp_type_str} act_zero_point,
                at::Tensor weight,
                at::Tensor weight_scales,
                at::Tensor weight_zero_points,
                c10::optional<at::Tensor> bias,
                double output_scale,
                int64_t output_zero_point,
                c10::optional<c10::ScalarType> output_dtype,
                c10::string_view post_op_name,
                torch::List<c10::optional<at::Scalar>> post_op_args,
                c10::string_view post_op_algorithm)"""
    # 定义一个静态方法，用于生成代码的核心逻辑
    def codegen(self, wrapper):
        # 解析输入和常量参数
        args = [x.codegen_reference() for x in self.inputs]
        const_args = []
        const_args.extend(self.codegen_const_args())

        # 获取输入参数和常量
        x = args[0]  # 第一个输入参数
        packed_weight = args[1]  # 第二个输入参数（通常是打包的权重）
        bias = args[2] if self.has_bias else const_args[0]  # 如果存在偏置则使用对应的输入参数，否则使用常量参数
        w_scale, w_zp = args[-2], args[-1]  # 权重的缩放因子和零点
        if self.x_scale_zp_are_tensors:
            # 如果输入的缩放因子和零点是张量，则需要更多参数
            assert len(args) >= 4
            x_scale, x_zp = args[-4], args[-3]  # 输入的缩放因子和零点
            (
                o_scale,
                o_zp,
                output_dtype,
                unary_attr,
                unary_scalars,
                unary_algorithm,
            ) = const_args[-6:]  # 输出的缩放因子、零点、输出数据类型及后续的算法参数
        else:
            # 否则使用常量参数来获取相应的值
            assert len(const_args) >= 8
            (
                x_scale,
                x_zp,
                o_scale,
                o_zp,
                output_dtype,
                unary_attr,
                unary_scalars,
                unary_algorithm,
            ) = const_args[-8:]  # 输入和输出的缩放因子、零点，输出数据类型及后续的算法参数

        # 将所有参数组成一个元组
        codegen_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            bias,
            o_scale,
            o_zp,
            output_dtype,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        )

        # 调用外部包装器的方法，生成外部内核分配和查找模式（如果需要）
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            codegen_args,
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )

        # 如果布局是Layout类的实例，则进行大小断言的代码生成
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)
        ):
            # 解构返回值，分别赋给 inputs, constant_args, kernel_layout，忽略第四个返回值
            (inputs, constant_args, kernel_layout, _) = _prepare_linear_fusion_create(
                cls,
                qx,
                qw,
                bias,
            )

            # 如果 x_scale 和 x_zero_point 是 TensorBox 类型
            if isinstance(x_scale, TensorBox) and isinstance(x_zero_point, TensorBox):
                # 确保 x_scale 和 x_zero_point 已实现
                x_scale.realize()
                x_zero_point.realize()
                # 将 x_scale 和 x_zero_point 加入 inputs 列表中
                inputs = inputs + [x_scale, x_zero_point]
                # 设置 x_scale_zp_are_tensors 为 True
                x_scale_zp_are_tensors = True
            else:
                # 否则，确保 x_scale 是 float 类型，x_zero_point 是 int 类型
                assert isinstance(x_scale, float) and isinstance(x_zero_point, int)
                # 将 x_scale 和 x_zero_point 加入 constant_args 列表中
                constant_args = constant_args + [x_scale, x_zero_point]
                # 设置 x_scale_zp_are_tensors 为 False
                x_scale_zp_are_tensors = False
            # 确保 w_scale 和 w_zero_point 已实现
            w_scale.realize()
            w_zero_point.realize()
            # 将 w_scale 和 w_zero_point 加入 inputs 列表中
            inputs = inputs + [w_scale, w_zero_point]
            # 将以下参数加入 constant_args 列表中
            constant_args = constant_args + [
                output_scale,
                output_zero_point,
                output_dtype,
                post_op_name,
                may_convert_to_optional(post_op_args),
                post_op_algorithm,
            ]

            # 确保 output_dtype 不为 None
            assert output_dtype is not None
            # 如果 output_dtype 是 torch.float32 或 torch.bfloat16
            if output_dtype in [torch.float32, torch.bfloat16]:
                # 在 _prepare_linear_fusion_create 中，我们使用 x.dtype (uint8) 来创建 kernel_layout
                # 如果设置了 fp32_output，则输出缓冲区的数据类型应为 float32 而不是 uint8
                kernel_layout.dtype = output_dtype

            # 返回 QLinearPointwisePT2E 对象，传入相应参数
            return QLinearPointwisePT2E(
                layout=kernel_layout,
                inputs=inputs,
                constant_args=constant_args,
                has_bias=(bias is not None),
                x_scale_zp_are_tensors=x_scale_zp_are_tensors,
            )
# 继承自ExternKernelAlloc类，表示QLinearPointwiseBinaryPT2E是一个外部内存分配的特定类型
class QLinearPointwiseBinaryPT2E(ExternKernelAlloc):
    # 初始化函数，接受多个参数用于配置对象的各种属性和行为
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        has_bias=True,
        x_scale_zp_are_tensors=False,
    ):
        """
        根据has_bias参数的不同，inputs和constant_args的内容会有所区别：
        如果has_bias不为None，则inputs包含[x, w, b, weight_scale, weight_zp, x2]，constant_args包含[x_scale, x_zp, o_scale, o_zp, fp32_output, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        如果has_bias为None，则inputs包含[x, w, weight_scale, weight_zp, x2]，constant_args包含[bias, x_scale, x_zp, o_scale, o_zp, fp32_output, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        """
        self.has_bias = has_bias  # 记录是否有偏置项
        self.x_scale_zp_are_tensors = x_scale_zp_are_tensors  # 记录输入的scale和zero point是否是张量
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name=(
                "torch.ops.onednn.qlinear_pointwise.binary_tensor"
                if x_scale_zp_are_tensors
                else "torch.ops.onednn.qlinear_pointwise.binary"
            ),
            cpp_kernel_name="onednn::qlinear_pointwise",
        )
        self.cpp_kernel_overload_name = (
            "binary_tensor" if x_scale_zp_are_tensors else "binary"
        )  # 根据x_scale_zp_are_tensors的值选择不同的CPP内核名称后缀
        self.cpp_kernel_key = "qlinear_pointwise_binary"  # 指定CPP内核的关键字
        # 根据x_scale_zp_are_tensors的值确定输入scale和zero point的类型字符串
        x_scale_type_str, x_zp_type_str = (
            ("at::Tensor", "at::Tensor")
            if x_scale_zp_are_tensors
            else ("double", "int64_t")
        )
        # 定义CPP操作的模式字符串，描述了函数的输入和输出参数
        self.cpp_op_schema = f"""
            at::Tensor(
                at::Tensor act,
                {x_scale_type_str} act_scale,
                {x_zp_type_str} act_zero_point,
                at::Tensor weight,
                at::Tensor weight_scales,
                at::Tensor weight_zero_points,
                c10::optional<at::Tensor> bias,
                double inv_output_scale,
                int64_t output_zero_point,
                c10::optional<c10::ScalarType> output_dtype,
                c10::optional<at::Tensor> other,
                double other_scale,
                int64_t other_zero_point,
                c10::string_view binary_post_op,
                double binary_alpha,
                c10::string_view unary_post_op,
                torch::List<c10::optional<at::Scalar>> unary_post_op_args,
                c10::string_view unary_post_op_algorithm)"""
    def codegen(self, wrapper):
        # 解析输入和常量参数
        args = [x.codegen_reference() for x in self.inputs]  # 解析输入参数列表的引用
        const_args = []
        const_args.extend(self.codegen_const_args())  # 解析常量参数列表

        x = args[0]  # 获取输入的第一个参数
        packed_weight = args[1]  # 获取输入的第二个参数（通常是打包的权重）
        bias = args[2] if self.has_bias else const_args[0]  # 获取偏置参数（如果有偏置则使用输入参数，否则使用常量参数）
        w_scale, w_zp, other = args[-3], args[-2], args[-1]  # 获取权重的缩放因子、零点以及其他参数

        if self.x_scale_zp_are_tensors:
            assert len(args) >= 5
            x_scale, x_zp = args[-5], args[-4]  # 如果输入的缩放因子和零点是张量，则获取它们
            (
                o_scale,
                o_zp,
                output_dtype,
                other_scale,
                other_zp,
                binary_attr,
                alpha,
                unary_attr,
                unary_scalars,
                unary_algorithm,
            ) = const_args[-10:]  # 获取其余的常量参数
        else:
            assert len(const_args) >= 8
            (
                x_scale,
                x_zp,
                o_scale,
                o_zp,
                output_dtype,
                other_scale,
                other_zp,
                binary_attr,
                alpha,
                unary_attr,
                unary_scalars,
                unary_algorithm,
            ) = const_args[-12:]  # 获取所有的常量参数

        codegen_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            bias,
            o_scale,
            o_zp,
            output_dtype,
            other,
            other_scale,
            other_zp,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        )  # 组装成代码生成所需的参数元组

        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            codegen_args,
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )  # 调用包装器的方法生成外部内核并查找架构（如果需要）

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)  # 如果布局是实例，则生成大小断言

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: float,
        x_zero_point: int,
        qw: "TensorBox",  # packed_weight
        w_scale: "TensorBox",
        w_zero_point: "TensorBox",
        bias: "TensorBox",
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        other: "TensorBox",
        other_scale,
        other_zp,
        binary_post_op,
        binary_alpha,
        unary_post_op,
        unary_post_op_args,
        unary_post_op_algorithm,
        (
            inputs,
            constant_args,
            kernel_layout,
            req_stride_order,
        ) = _prepare_linear_fusion_create(
            cls,
            qx,
            qw,
            bias,
        )
        # 调用 _prepare_linear_fusion_create 函数，获取返回的四个变量：inputs, constant_args, kernel_layout, req_stride_order

        if isinstance(x_scale, TensorBox) and isinstance(x_zero_point, TensorBox):
            # 检查 x_scale 和 x_zero_point 是否为 TensorBox 类型
            x_scale.realize()
            x_zero_point.realize()
            # 调用 realize() 方法实现 TensorBox 类型的 x_scale 和 x_zero_point
            inputs = inputs + [x_scale, x_zero_point]
            # 将 x_scale 和 x_zero_point 添加到 inputs 列表中
            x_scale_zp_are_tensors = True
        else:
            assert isinstance(x_scale, float) and isinstance(x_zero_point, int)
            # 断言 x_scale 是 float 类型，x_zero_point 是 int 类型
            constant_args = constant_args + [x_scale, x_zero_point]
            # 将 x_scale 和 x_zero_point 添加到 constant_args 列表中
            x_scale_zp_are_tensors = False

        w_scale.realize()
        w_zero_point.realize()
        # 调用 realize() 方法实现 w_scale 和 w_zero_point
        inputs = inputs + [w_scale, w_zero_point]
        # 将 w_scale 和 w_zero_point 添加到 inputs 列表中

        if binary_post_op == "sum":
            other = cls.require_stride_order(other, req_stride_order)
            # 如果 binary_post_op 为 "sum"，调用 cls.require_stride_order 方法处理 other，使用 req_stride_order 参数

        inputs.append(other)
        # 将 other 添加到 inputs 列表中
        constant_args = constant_args + [
            output_scale,
            output_zero_point,
            output_dtype,
            other_scale,
            other_zp,
            binary_post_op,
            binary_alpha,
            unary_post_op,
            may_convert_to_optional(unary_post_op_args),
            unary_post_op_algorithm,
        ]
        # 将多个参数依次添加到 constant_args 列表中

        if binary_post_op == "sum":
            packed = QLinearPointwiseBinaryPT2E(
                layout=NoneLayout(other.get_device()),
                inputs=inputs,
                constant_args=constant_args,
                has_bias=(bias is not None),
                x_scale_zp_are_tensors=x_scale_zp_are_tensors,
            )
            # 如果 binary_post_op 为 "sum"，创建 QLinearPointwiseBinaryPT2E 对象 packed
            mark_node_as_mutating(packed, other)
            # 调用 mark_node_as_mutating 方法标记 packed 和 other 为 mutating
            # 返回 other，因为它已经被原地修改过
            return packed.inputs[-1]

        assert output_dtype is not None
        # 断言 output_dtype 不为 None
        if output_dtype in [torch.float32, torch.bfloat16]:
            # 如果 output_dtype 是 torch.float32 或 torch.bfloat16
            # 在 _prepare_linear_fusion_create 中，我们使用 x.dtype（uint8）来创建 kernel_layout
            # 如果设置了 fp32_output，输出缓冲区应该是 float32 类型，而不是 uint8 类型。
            kernel_layout.dtype = output_dtype
            # 设置 kernel_layout 的 dtype 为 output_dtype

        return QLinearPointwiseBinaryPT2E(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
            has_bias=(bias is not None),
            x_scale_zp_are_tensors=x_scale_zp_are_tensors,
        )
        # 返回 QLinearPointwiseBinaryPT2E 对象，使用指定的参数
# 定义一个继承自ExternKernelAlloc的MkldnnRnnLayer类
class MkldnnRnnLayer(ExternKernelAlloc):
    # 初始化方法，接受布局、输入、常量参数等参数
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        # 调用父类的初始化方法，传入布局、输入、常量参数等参数
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="aten.mkldnn_rnn_layer",
            cpp_kernel_name="at::mkldnn_rnn_layer",
        )

    # 类方法，用于创建MkldnnRnnLayer对象
    @classmethod
    def create(
        cls,
        x: "TensorBox",
        w0: "TensorBox",
        w1: "TensorBox",
        w2: "TensorBox",
        w3: "TensorBox",
        hx: "TensorBox",
        cx: "TensorBox",
        reverse: bool,
        batch_sizes: List[int],
        mode: int,
        hidden_size: int,
        num_layers: int,
        has_biases: bool,
        bidirectional: bool,
        batch_first: bool,
        train: bool,
        ):
            # 要求输入 x 是步幅为1的，实现输入 x
            x = cls.require_stride1(cls.realize_input(x))
            # 如果是 batch_first，x 在进入 mkldnn_rnn_layer 之前已经在 lstm 中被置换过
            # 确保在 batch_first 情况下 x 是连续的
            x.freeze_layout()
            # 要求输入 w0 是步幅为1的，实现输入 w0
            w0 = cls.require_stride1(cls.realize_input(w0))
            # 要求输入 w1 是步幅为1的，实现输入 w1
            w1 = cls.require_stride1(cls.realize_input(w1))
            # 要求输入 w2 是步幅为1的，实现输入 w2
            w2 = cls.require_stride1(cls.realize_input(w2))
            # 要求输入 w3 是步幅为1的，实现输入 w3
            w3 = cls.require_stride1(cls.realize_input(w3))
            # 要求输入 hx 是步幅为1的，实现输入 hx
            hx = cls.require_stride1(cls.realize_input(hx))
            # 确保 hx 是连续的
            hx.freeze_layout()
            # 要求输入 cx 是步幅为1的，实现输入 cx
            cx = cls.require_stride1(cls.realize_input(cx))
            # 确保 cx 是连续的
            cx.freeze_layout()

            # 获取输入 x 的尺寸
            input_size = x.get_size()
            # 断言输入 x 的维度为 3
            assert len(input_size) == 3, "Expect lstm input to be 3D"
            # 解包输入尺寸
            seq_length, mini_batch, input_size = input_size
            # 设置输出形状为 [序列长度, 小批量大小, 隐藏层大小]
            output_shape = [seq_length, mini_batch, hidden_size]

            # 获取 hx 和 cx 的形状
            hy_shape = hx.get_size()
            cy_shape = cx.get_size()

            # 初始化结果为 IRNode 列表
            res: List[IRNode] = []

            # 构建输入列表
            inputs = [x, w0, w1, w2, w3, hx, cx]
            # 构建常量参数列表
            constant_args = [
                reverse,
                batch_sizes,
                mode,
                hidden_size,
                num_layers,
                has_biases,
                bidirectional,
                batch_first,
                train,
            ]

            # 创建 MkldnnRnnLayer 对象
            packed = MkldnnRnnLayer(
                MultiOutputLayout(x.get_device()),
                inputs=inputs,
                constant_args=constant_args,
            )

            # 定义获取 LSTM 输出步幅的函数
            def get_strides_of_lstm_output(output_shape, batch_first):
                # 断言输出形状为 3D
                assert len(output_shape) == 3, "Expect output_shape to be 3D"
                # 返回灵活布局中连续的步幅
                return FlexibleLayout.contiguous_strides(output_shape)

            # 构建输出尺寸列表
            output_sizes = [output_shape, hy_shape, cy_shape]
            # 构建输出步幅列表
            output_strides = [
                get_strides_of_lstm_output(output_shape, batch_first),
                FlexibleLayout.contiguous_strides(hy_shape),
                FlexibleLayout.contiguous_strides(cy_shape),
            ]
            # 构建输出 IRNode 列表
            output_ir = [
                MultiOutput(
                    FixedLayout(
                        x.get_device(),
                        x.get_dtype(),
                        output_size,
                        output_stride,
                    ),
                    packed,
                    [(tuple, i)],
                )
                for i, (output_size, output_stride) in enumerate(
                    zip(output_sizes, output_strides)
                )
            ]

            # 返回输出 IRNode 列表
            return output_ir
```
# `.\pytorch\test\quantization\jit\test_quantize_jit.py`

```
# Owner(s): ["oncall: quantization"]

# torch 库引入
import io
import itertools
import unittest

# 标准库引入
from typing import List, Tuple

import torch
import torch.jit
import torch.jit.quantized
import torch.nn as nn
import torch.nn.functional as F

# torch.ao.quantization 模块引入
from torch.ao.quantization import (
    default_dynamic_qconfig,
    default_histogram_observer,
    default_observer,
    default_per_channel_weight_observer,
    default_qconfig,
    default_weight_observer,
    float16_dynamic_qconfig,
    fuse_modules,
    get_default_qconfig,
    per_channel_dynamic_qconfig,
    PlaceholderObserver,
    QConfig,
    quantize,
    quantize_dynamic,
    quantize_dynamic_jit,
    quantize_jit,
)

# torch.ao.quantization.quantize_jit 模块引入
from torch.ao.quantization.quantize_jit import (
    convert_dynamic_jit,
    convert_jit,
    fuse_conv_bn_jit,
    prepare_dynamic_jit,
    prepare_jit,
    script_qconfig,
)

from torch.jit._recursive import wrap_cpp_module

from torch.testing import FileCheck

# Annotated models 引入
from torch.testing._internal.common_quantization import (
    AnnotatedConvBnModel,
    AnnotatedConvModel,
    AnnotatedConvTransposeModel,
    AnnotatedNestedModel,
    AnnotatedSingleLayerLinearModel,
    AnnotatedSkipQuantModel,
    ConvBnModel,
    ConvModel,
    ConvTransposeModel,
    default_per_channel_qconfig,
    get_script_module,
    NestedModel,
    QuantizationTestCase,
    SingleLayerLinearModel,
    skipIfNoFBGEMM,
    SkipQuantModel,
    test_only_eval_fn,
)

# Testing utils 引入
from torch.testing._internal.common_quantized import (
    override_qengines,
    qengine_is_fbgemm,
    qengine_is_qnnpack,
)

from torch.testing._internal.common_utils import set_default_dtype
from torch.testing._internal.jit_utils import (
    attrs_with_prefix,
    get_forward,
    get_forward_graph,
)

# QuantizationTestCase 类的子类，用于测试 quantize_jit 使用的图模式量化 passes
class TestQuantizeJitPasses(QuantizationTestCase):
    """Test graph mode quantization passes used by quantize_jit"""
    def test_skip_dequant_constant_prop(self):
        # 定义一个继承自 torch.nn.Module 的模型类 M
        class M(torch.nn.Module):
            # 构造函数，定义一个 3 输入 5 输出的二维卷积层
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3).float()

            # 前向传播函数，返回卷积层的输出
            def forward(self, x):
                return self.conv(x)

        # 使用 Torch Script 将模型 M 脚本化
        m = torch.jit.script(M())
        # 使用默认的逐通道权重观察器创建观察器
        observer = default_per_channel_weight_observer.with_args(ch_axis=1)
        # 定义量化配置字典，包含激活函数的默认观察器和权重的观察器
        qconfig_dict = {"": QConfig(activation=default_observer, weight=observer)}
        # 准备模型 m 进行量化
        m = prepare_jit(m, qconfig_dict)
        # 生成测试数据，形状为 (1, 3, 10, 10)，数据类型为 float
        data = torch.randn(1, 3, 10, 10, dtype=torch.float)

        # 运行量化后的模型 m
        m(data)
        # 将量化后的模型 m 转换为量化模型，开启调试模式
        m = convert_jit(m, debug=True)

        # 冻结量化后的模型，生成冻结模型 freezed
        freezed = torch.jit.freeze(m)
        # 使用测试数据 data 运行冻结模型 freezed
        freezed(data)

        # 在冻结后的模型图中进行检查和验证
        # 检查是否有两个 "aten::quantize_per_tensor" 操作
        FileCheck().check_count("aten::quantize_per_tensor", 2, exactly=True).run(
            freezed.graph
        )
        # 检查是否没有 "aten::quantize_per_channel" 操作
        FileCheck().check_count("aten::quantize_per_channel", 0, exactly=True).run(
            freezed.graph
        )
        # 检查是否有三个 "aten::dequantize" 操作
        FileCheck().check_count("aten::dequantize", 3, exactly=True).run(freezed.graph)
        # 在模型图中进一步检查和验证操作的顺序和出现
        FileCheck().check("aten::quantize_per_tensor").check_next(
            "aten::dequantize"
        ).check_not("aten::quantize_per_channel").check("aten::dequantize").check_next(
            "aten::conv2d"
        ).check_next(
            "aten::quantize_per_tensor"
        ).check_next(
            "aten::dequantize"
        ).run(
            freezed.graph
        )
    # 定义一个测试用例函数，测试 Conv 和 BatchNorm 融合的情况
    def test_foldbn_trivial(self):
        # 定义 BatchNorm 和 Conv 的模块字典，分别适用于二维和三维数据
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        # 测试简单的情况
        class TestModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 创建一个 Conv 层和一个对应维度的 BatchNorm 层
                self.conv = conv_module[dim](1, 20, 5, 1)
                self.bn = bn_module[dim](num_features=20)
                # 设置 BatchNorm 层的 epsilon 值
                self.bn.eps = 0.0023

            def forward(self, x):
                # 执行 Conv 层操作
                x = self.conv(x)
                # 执行 BatchNorm 层操作
                x = self.bn(x)
                return x

        # 使用 itertools 生成两个布尔值和维度的组合
        options = itertools.product([True, False], [2, 3])
        # 创建数据字典，包含二维和三维的随机数据
        data = {2: torch.rand(1, 1, 6, 6), 3: torch.rand(1, 1, 6, 6, 6)}
        
        # 检查转换是否不改变数值
        for tracing, dim in options:
            # 创建并评估一个 TestModule 实例
            eager = TestModule(dim).eval()
            x = data[dim]
            # 获得脚本化或跟踪的模块，并评估它
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            
            # 检查在原始脚本模块的 forward 方法中是否有两个 CallMethod 节点，
            # 分别对应 conv.forward 和 bn.forward。
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 2, exactly=True
            ).run(str(get_forward(scripted_or_traced._c).graph))

            # 运行 FoldConvBatchnorm 优化过程
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            # 检查优化后是否只剩一个 CallMethod 节点（应为 conv.forward）
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 1, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced._c)))

            # 再次检查转换是否不改变数值
            self.assertEqual(eager(x), scripted_or_traced(x))
    def test_foldbn_trivial_nobias(self):
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        # Test trivial case
        class TestModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # Initialize convolutional layer with no bias
                self.conv = conv_module[dim](1, 20, 5, 1, bias=False)
                # Initialize batch normalization layer
                self.bn = bn_module[dim](num_features=20)
                # Set a non-zero epsilon value to avoid division by zero
                self.bn.eps = 0.0027
                # Set a random bias parameter for batch normalization
                self.bn.bias = torch.nn.Parameter(torch.rand([20]))

            def forward(self, x):
                # Apply convolution operation
                x = self.conv(x)
                # Apply batch normalization
                x = self.bn(x)
                return x

        # Generate combinations of tracing and dimensions
        options = itertools.product([True, False], [2, 3])
        # Create input data tensors for dimensions 2 and 3
        data = {2: torch.rand(1, 1, 6, 6), 3: torch.rand(1, 1, 6, 6, 6)}
        # Iterate over all combinations of tracing and dimensions
        for tracing, dim in options:
            # Instantiate TestModule with specified dimension and set to evaluation mode
            eager = TestModule(dim).eval()
            # Get input tensor based on current dimension
            x = data[dim]
            # Convert eager mode module to scripted or traced module
            scripted_or_traced = get_script_module(eager, tracing, x).eval()

            # Check that in the original script module's forward we have two
            # CallMethod nodes. One of them should be for conv.forward and the other
            # for bn.forward.
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 2, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced._c)))

            # Run FoldConvBatchnorm pass.
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            # Check that after the pass one of the CallMethods is gone (supposedly,
            # the bn.forward).
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 1, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced._c)))

            # Check that the transformation doesn't change numerics
            # Assert that outputs from original and transformed modules are equal
            self.assertEqual(eager(x), scripted_or_traced(x))
    # 定义一个测试方法，用于测试在子模块中使用的 BatchNorm 和 Convolution 模块
    def test_foldbn_in_submodule(self):
        # 定义 BatchNorm 模块字典，键为维度，值为对应的 BatchNorm 类
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        # 定义 Convolution 模块字典，键为维度，值为对应的 Convolution 类
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        # 在子模块中测试是否能找到 Conv-BN 模式
        class SubModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化 Convolution 层
                self.conv = conv_module[dim](1, 20, 5, 1)
                # 初始化 BatchNorm 层
                self.bn = bn_module[dim](num_features=20)

            def forward(self, x):
                # 在输入数据上执行 Convolution 操作
                x = self.conv(x)
                # 在输出上执行 BatchNorm 操作
                x = self.bn(x)
                return x

        class TestModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化子模块
                self.sub = SubModule(dim)

            def forward(self, x):
                # 在子模块上执行 forward 方法
                x = self.sub(x)
                return x

        # 创建一个选项的迭代器，用于迭代是否跟踪和维度的所有组合
        options = itertools.product([True, False], [2, 3])
        # 创建数据字典，包含维度为 2 和 3 的随机张量数据
        data = {2: torch.rand(1, 1, 10, 10), 3: torch.rand(1, 1, 10, 10, 10)}
        # 对每一组选项进行迭代
        for tracing, dim in options:
            # 创建一个评估模式下的 TestModule 实例
            eager = TestModule(dim).eval()
            # 获取对应维度的数据张量
            x = data[dim]
            # 获取脚本化或追踪后的模块
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            # 运行 FileCheck，检查 forward 方法调用的数量是否为 2
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 2, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub._c)))

            # 将 Convolution 和 BatchNorm 合并为一个单元
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            # 再次运行 FileCheck，检查 forward 方法调用的数量是否为 1
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 1, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub._c)))

            # 使用断言验证在 eager 和脚本化/追踪后的模块上应用相同输入时的输出是否一致
            self.assertEqual(eager(x), scripted_or_traced(x))
    def test_foldbn_shared_classtype(self):
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class TestModule(torch.nn.Module):
            def __init__(self, dim, bias=False):
                super().__init__()
                # 创建一个卷积层对象，根据维度选择 Conv2d 或 Conv3d
                self.conv1 = conv_module[dim](5, 5, 3, bias=bias)
                # 创建一个批归一化层对象，根据维度选择 BatchNorm2d 或 BatchNorm3d
                self.bn1 = bn_module[dim](num_features=5)
                # 初始化批归一化层的 running_mean 为 -0.2
                self.bn1.running_mean.fill_(-0.2)
                # 设定批归一化层的 bias 参数为随机生成的参数
                self.bn1.bias = torch.nn.Parameter(torch.rand([5]))
                # 设定批归一化层的 eps 参数为 0.0023，用于数值稳定性
                self.bn1.eps = 0.0023
                # 创建第二个卷积层对象，根据维度选择 Conv2d 或 Conv3d
                self.conv2 = conv_module[dim](5, 5, 3, bias=bias)
                # 创建第二个批归一化层对象，根据维度选择 BatchNorm2d 或 BatchNorm3d
                self.bn2 = bn_module[dim](num_features=5)
                # 设定第二个批归一化层的 eps 参数为 0.0029，用于数值稳定性
                self.bn2.eps = 0.0029
                # 创建一个 ReLU 激活函数对象
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # 进行第一次卷积操作
                x = self.conv1(x)
                # 经过第一次批归一化操作
                x = self.bn1(x)
                # 经过 ReLU 激活函数
                x = self.relu(x)
                # 进行第二次卷积操作
                x = self.conv2(x)
                # 经过第二次批归一化操作
                x = self.bn2(x)
                # 再次经过 ReLU 激活函数
                x = self.relu(x)
                return x

        # 生成不同测试选项的组合，包括是否追踪、维度和是否有偏置
        options = itertools.product([True, False], [2, 2], [True, False])
        # 准备测试数据，包括维度为 2 和 3 的随机张量
        data = {2: torch.rand(1, 5, 6, 6), 3: torch.rand(1, 5, 6, 6, 6)}
        # 对每种测试选项进行测试
        for tracing, dim, bias in options:
            # 创建并转换为评估模式的 TestModule 实例
            eager = TestModule(dim, bias).eval()
            # 获取对应维度的测试数据
            x = data[dim]
            # 使用指定的追踪方式获取脚本化或追踪模块
            scripted_or_traced = get_script_module(eager, tracing, x)
            # 尝试融合卷积和批归一化操作
            folded = fuse_conv_bn_jit(scripted_or_traced)
            # 断言评估模式下的输出与脚本化或追踪模块的输出一致
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_no_fusion(self):
        """Test that we don't fuse the cases when module type does not match"""

        # 自定义卷积层，仅传递输入而不做任何处理
        class CustomConv(torch.nn.Module):
            def forward(self, x):
                return x

        # 自定义批归一化层，仅传递输入而不做任何处理
        class CustomBn(torch.nn.Module):
            def forward(self, x):
                return x

        # 自定义模块 M，包含一个自定义卷积层和一个自定义批归一化层
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = CustomConv()
                self.bn = CustomBn()

            def forward(self, x):
                # 先通过自定义卷积层，再通过自定义批归一化层
                return self.bn(self.conv(x))

        # 将模块 M 脚本化
        m = torch.jit.script(M())
        # 尝试融合卷积和批归一化操作，预期不会融合成功
        m = fuse_conv_bn_jit(m)
        # 使用 FileCheck 验证在图中是否存在两次 CallMethod 操作
        FileCheck().check_count("prim::CallMethod", 2, exactly=True).run(m.graph)

    @set_default_dtype(torch.double)
    def test_foldbn_complex_cases(self):
        # 定义一个测试函数，用于测试复杂情况下的 foldbn（可能是某种优化或整合操作）
        # 测试包括不同的 conv2d/conv3d 结合有无偏置的情况，
        # 以及 BatchNorm 有无仿射变换的情况，并且变化层数量。
        # 只有在默认数据类型为双精度时才有效果
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class SubModule(torch.nn.Module):
            def __init__(self, dim, num_blocks, enable_bias, enable_affine):
                super().__init__()
                layers = []
                for i in range(num_blocks):
                    # 添加一个 conv2d 或 conv3d 层，根据参数决定是否使用偏置
                    layers.append(conv_module[dim](20, 20, 5, 1, bias=enable_bias))
                    # 创建一个 BatchNorm 对象，根据参数决定是否启用仿射变换
                    bn_obj = bn_module[dim](num_features=20, affine=enable_affine)
                    if enable_affine:
                        # 如果启用仿射变换，设置权重和偏置为随机张量的参数
                        bn_obj.weight = torch.nn.Parameter(
                            torch.rand_like(bn_obj.weight)
                        )
                        bn_obj.bias = torch.nn.Parameter(torch.rand_like(bn_obj.bias))
                    # 设置运行时均值和方差为随机张量
                    bn_obj.running_mean = torch.rand_like(bn_obj.running_mean)
                    bn_obj.running_var = torch.rand_like(bn_obj.running_var)
                    # 将 BatchNorm 对象添加到层列表中
                    layers.append(bn_obj)
                # 创建一个包含所有层的序列模块
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                # 前向传播函数，对输入 x 执行所有层的前向传播
                return self.layers(x)

        class TestModule(torch.nn.Module):
            def __init__(self, dim, num_blocks, enable_bias, enable_affine):
                super().__init__()
                # 创建子模块，传入维度、块数、是否启用偏置和是否启用仿射变换
                self.sub = SubModule(dim, num_blocks, enable_bias, enable_affine)

            def forward(self, x):
                # 前向传播函数，调用子模块的前向传播
                x = self.sub(x)
                return x

        # 创建所有可能的选项组合
        options = itertools.product(
            [True, False], [2, 3], [True, False], [True, False], [1, 2]
        )
        # 创建数据字典，包含维度为2和3的随机张量数据
        data = {2: torch.rand(1, 20, 10, 10), 3: torch.rand(1, 20, 10, 10, 10)}
        # 遍历所有选项组合
        for tracing, dim, enable_bias, enable_bn_affine, num_layers in options:
            # 创建一个测试模块实例，并设置为评估模式
            eager = TestModule(dim, num_layers, enable_bias, enable_bn_affine).eval()
            x = data[dim]
            # 获取脚本化或追踪模块
            scripted_or_traced = get_script_module(eager, tracing, x).eval()

            # 检查前向图中 'forward' 方法的调用次数是否符合预期
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', num_layers * 2, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub.layers._c)))

            # 将卷积层和 BatchNorm 层融合
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            # 检查前向图中 'forward' 方法的调用次数是否符合预期
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', num_layers, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub.layers._c)))

            # 断言原始模块和融合后的模块在给定输入下的输出是否一致
            self.assertEqual(eager(x), scripted_or_traced(x))
    def test_fuse_linear(self):
        # 定义一个模拟线性层的神经网络模块
        class FunctionalLinear(torch.nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                # 计算输入张量 x 与权重矩阵转置的乘积
                res = torch.matmul(x, self.weight.t())
                # 如果有偏置，则加上偏置
                if self.bias is not None:
                    res.add_(self.bias)
                return res

        # 创建不同形状和参数的输入数据、权重和偏置
        x1 = torch.rand(3)
        w1 = torch.rand(5, 3)
        b1 = torch.rand(5)

        x2 = torch.rand(5, 5)
        w2 = torch.rand(5, 5)
        b2 = torch.rand(5)

        x3 = torch.rand(5, 5, 5)
        w3 = torch.rand(5, 5)
        b3 = torch.rand(5)

        # 对每一种输入数据形状和是否有偏置进行组合，进行模型的融合和检查
        for has_bias, (x, weight, b) in itertools.product(
            [True, False], [(x1, w1, b1), (x2, w2, b2), (x3, w3, b3)]
        ):
            # 根据是否有偏置设置 bias 变量
            bias = b if has_bias else None
            # 使用 torch.jit.trace 对 FunctionalLinear 模块进行跟踪
            model = torch.jit.trace(FunctionalLinear(weight, bias), [x])
            # 获取第一个 matmul 操作的源代码范围
            for node in model.graph.nodes():
                if node.kind() == "aten::matmul":
                    source_range_1 = node.sourceRange()
            # 对模型的图执行线性层融合优化
            torch._C._jit_pass_fuse_linear(model.graph)
            # 获取第二个 linear 操作的源代码范围
            for node in model.graph.nodes():
                if node.kind() == "aten::linear":
                    source_range_2 = node.sourceRange()
            # 使用 FileCheck 检查图中是否包含 "aten::linear" 操作
            FileCheck().check("aten::linear").run(model.graph)
            # 使用 FileCheck 检查图中是否不包含预期不融合的操作
            check_not = ["aten::matmul", "aten::addmm", "aten::add_", "aten::t("]
            for cn in check_not:
                FileCheck().check_not(cn).run(model.graph)
            # 确保两次获取的源代码范围一致
            self.assertTrue(source_range_1 == source_range_2)
            # 运行模型，确保没有错误
            model(x)

        # 检查 matmul 操作未被融合
        class Matmul(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return torch.matmul(x, self.weight)

        # 创建不同形状的输入数据和权重，对 3D matmul 进行检查
        x = torch.rand(5, 6, 5)
        w = torch.rand(5, 5, 100)
        model = torch.jit.trace(Matmul(w), [x])
        # 对模型的图执行线性层融合优化
        torch._C._jit_pass_fuse_linear(model.graph)
        # 使用 FileCheck 检查图中是否包含 "aten::matmul" 操作
        FileCheck().check("aten::matmul").run(model.graph)
        # 使用 FileCheck 检查图中是否不包含 "aten::linear" 操作
        FileCheck().check_not("aten::linear").run(model.graph)
        # 确保模型运行没有问题
        model(x)

    def test_insert_observers(self):
        # 定义一个包含卷积层的神经网络模块
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                return self.conv(x)

        # 使用 torch.jit.script 将 M 模块转换为 Torch 脚本
        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        # 准备 Torch 脚本模块，应用量化配置
        m = prepare_jit(m, qconfig_dict)
        # 检查是否为卷积层的输入和输出插入了观察器
        assert len(attrs_with_prefix(m, "_observer_")) == 2
        # 检查卷积层的权重是否插入了观察器
        assert len(attrs_with_prefix(m.conv, "_observer_")) == 1
    def test_insert_observers_interface(self):
        @torch.jit.interface
        # 定义一个 TorchScript 接口 SubInterface
        class SubInterface(torch.nn.Module):
            # 接口方法定义，接受一个输入，返回一个 Tensor 对象
            def addOne(self, inp) -> torch.Tensor:
                pass

        # 实现 SubInterface 接口的具体类 Sub
        class Sub(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入和输出维度为 5
                self.fc = torch.nn.Linear(5, 5)

            # 实现接口方法，对输入的 Tensor 执行线性层操作并加一
            def addOne(self, inp):
                return self.fc(inp) + 1

            # 前向传播方法调用接口方法
            def forward(self, x):
                return self.addOne(x)

        # 主模块 M，包含一个卷积层和一个 Sub 类的实例
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个卷积层，输入通道数 3，输出通道数 5，卷积核大小 3x3
                self.conv = torch.nn.Conv2d(3, 5, 3)
                # 创建一个 Sub 类的实例
                self.sub = Sub()

            # 前向传播方法，先通过卷积层处理输入，然后调用 Sub 实例的前向方法
            def forward(self, x):
                return self.sub(self.conv(x))

        # 将模型 M 转换为 TorchScript
        m = torch.jit.script(M())
        # 定义配置字典，指定子模块 "sub" 的量化配置
        qconfig_dict = {"sub.conv": default_qconfig}
        # 准备模型 m，应用量化配置
        m = prepare_jit(m, qconfig_dict)

    def test_insert_observers_interface_unshare_type(self):
        @torch.jit.interface
        # 定义一个 TorchScript 接口 OperatorIf
        class OperatorIf(nn.Module):
            # 接口方法定义，接受一个输入 Tensor，返回一个 Tensor
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass

        # 实现 OperatorIf 接口的具体类 Operator
        class Operator(nn.Module):
            # 初始化方法，接受一个参数 a
            def __init__(self, a):
                super().__init__()
                self.a = a

            # 实现接口方法，对输入的 Tensor 执行加权操作
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                return self.a * (inp + self.a)

        # 内部模块 Inner，包含一个 OperatorIf 类型的成员 op
        class Inner(nn.Module):
            # op 属性的类型是 OperatorIf
            op: OperatorIf

            # 初始化方法，接受一个 OperatorIf 类型的参数 op
            def __init__(self, op):
                super().__init__()
                self.op = op

            # 前向传播方法调用 op 的 forward 方法
            def forward(self, inp):
                return self.op(inp)

        # 外部模块 Outer，包含两个 Inner 模块实例
        class Outer(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 Inner 实例 inner_a，传入一个 Operator 实例
                self.inner_a = Inner(Operator(1))
                # 创建一个 Inner 实例 inner_b，传入一个 Operator 实例
                self.inner_b = Inner(Operator(3.0))

            # 前向传播方法，调用两个 Inner 实例的前向传播方法并求和
            def forward(self, inp):
                return self.inner_a(inp) + self.inner_b(inp)

        # 配置字典，指定两个子模块 "inner_a" 和 "inner_b" 的量化配置
        qconfig_dict = {"inner_a": default_qconfig, "inner_b": default_qconfig}

        # 创建一个 Outer 类的实例
        eager_model = Outer()
        # 迭代两次，分别用 True 和 False 进行追踪
        for tracing in [True, False]:
            # 创建一个随机输入张量 x
            x = torch.rand(3)
            # 获取 Outer 模型的 TorchScript 表示
            script_model = get_script_module(eager_model, tracing, x)
            # 确保模型能够成功运行
            prepare_jit(script_model, qconfig_dict)
    def test_insert_observers_child_qconfig(self):
        # 定义包含子模块的 PyTorch 模型类 Sub
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(x)

        # 定义包含子模块和卷积层的 PyTorch 模型类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                return self.sub(self.conv(x))

        # 对模型进行 TorchScript 脚本化
        m = torch.jit.script(M())
        # 定义量化配置字典，指定子模块中的线性层使用默认量化配置
        qconfig_dict = {"sub.fc": default_qconfig}
        # 准备 TorchScript 模型，应用量化配置
        m = prepare_jit(m, qconfig_dict)
        # 断言：子模块的输入和输出处均插入了观察器
        assert len(attrs_with_prefix(m, "_observer_")) == 2
        # 断言：卷积层未被量化，因此没有观察器
        assert len(attrs_with_prefix(m.conv, "_observer_")) == 0
        # 断言：子模块本身没有观察器，因为观察器在最外层调用处插入
        assert len(attrs_with_prefix(m.sub, "_observer_")) == 0
        # 断言：子模块中的线性层的权重有一个观察器
        assert len(attrs_with_prefix(m.sub.fc, "_observer_")) == 1

    @unittest.skipUnless(
        "fbgemm" in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_insert_observers_weight_dtype(self):
        # 定义包含卷积层的 PyTorch 模型类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                return F.relu(self.conv(x))

        # 对模型进行 TorchScript 脚本化
        m = torch.jit.script(M())
        # 定义空字符串键的量化配置字典，应用默认量化配置
        qconfig_dict = {"": default_qconfig}
        # 准备 TorchScript 模型，应用量化配置
        m = prepare_jit(m, qconfig_dict)
        # 获取激活的数据类型
        activation_dtypes = {
            obs.getattr("dtype")
            for x, obs in m._modules._c.items()
            if x.startswith("_observer_")
        }
        # 获取权重的数据类型
        weight_dtypes = {
            obs.getattr("dtype")
            for x, obs in m.conv._modules._c.items()
            if x.startswith("_observer_")
        }
        # 断言：期望只有一个激活数据类型
        assert len(activation_dtypes) == 1, "Expected to have 1 activation dtype"
        # 断言：期望只有一个权重数据类型
        assert len(weight_dtypes) == 1, "Expected to have 1 weight dtype"
        # 断言：期望激活数据类型与权重数据类型不同
        assert next(iter(activation_dtypes)) != next(
            iter(weight_dtypes)
        ), "Expected activation dtype to be different from weight dtype"

    def test_insert_observers_for_reused_weight(self):
        # 定义包含卷积操作的 PyTorch 模型类 M
        class M(torch.nn.Module):
            def forward(self, x, y, weight):
                x = F.conv2d(x, weight)
                y = F.conv2d(y, weight)
                return x + y

        # 对模型进行 TorchScript 脚本化并设置为评估模式
        m = torch.jit.script(M()).eval()
        # 准备 TorchScript 模型，应用默认量化配置
        m = prepare_jit(m, {"": default_qconfig})
        # 断言：期望有6个观察器，分别用于 x、y、weight的输入，两个 F.conv2d 的输出以及 add 的输出
        assert len(attrs_with_prefix(m, "_observer")) == 6
    def test_insert_observers_shared_class_type(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 构造函数，初始化模块的两个卷积层 conv1 和 conv2
            def __init__(self):
                super().__init__()
                # 创建一个输入通道为 3，输出通道为 5，卷积核大小为 3 的卷积层 conv1
                self.conv1 = torch.nn.Conv2d(3, 5, 3).float()
                # 创建另一个输入通道为 3，输出通道为 5，卷积核大小为 3 的卷积层 conv2
                self.conv2 = torch.nn.Conv2d(3, 5, 3).float()

            # 前向传播函数，对输入 x 执行 conv1 和 conv2 的连续卷积操作
            def forward(self, x):
                return self.conv2(self.conv1(x))

        # 将类 M 实例化并进行脚本化，得到 torch.jit.ScriptModule
        m = torch.jit.script(M())
        # 定义一个包含默认量化配置的 qconfig 字典
        qconfig_dict = {"": default_qconfig}
        # 对模型 m 应用量化准备函数，返回量化后的模型 m
        m = prepare_jit(m, qconfig_dict)
        
        # 获取 conv1 和 conv2 的观察器属性列表
        conv1_observers = attrs_with_prefix(m.conv1, "_observer_")
        conv2_observers = attrs_with_prefix(m.conv2, "_observer_")
        
        # 断言确保每个卷积层只有一个观察器子模块
        assert len(conv1_observers) == 1, "Expected to have 1 observer submodules"
        assert len(conv2_observers) == 1, "Expected to have 1 observer submodules"
        
        # 断言确保 conv1 和 conv2 共享相同的观察器，因为它们属于同一个类类型
        assert (
            conv1_observers == conv2_observers
        ), "Expect conv1 and conv2 to have same observers since the class type is shared"

    def test_insert_observers_for_general_ops(self):
        """Make sure we skip observers for ops that doesn't require
        observation, e.g. flatten
        """
        
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 构造函数，初始化一个卷积层 conv
            def __init__(self):
                super().__init__()
                # 创建一个输入通道为 3，输出通道为 3，卷积核大小为 3 的卷积层 conv
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            # 前向传播函数，对输入 x 执行卷积操作后进行 flatten 操作
            def forward(self, x):
                x = self.conv(x)
                x = torch.flatten(x)
                return x

        # 将类 M 实例化并进行脚本化，得到 torch.jit.ScriptModule
        m = torch.jit.script(M())
        # 定义一个包含默认量化配置的 qconfig 字典
        qconfig_dict = {"": default_qconfig}
        # 对模型 m 应用量化准备函数，返回量化后的模型 m
        m = prepare_jit(m, qconfig_dict)
        
        # 断言确保卷积层 conv 的输入和输出各有一个观察器
        assert len(attrs_with_prefix(m, "_observer_")) == 2
        
        # 使用 FileCheck 检查图中的观察器情况，确保在 conv 操作后有观察器，但在 flatten 操作后没有
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check(
            'prim::GetAttr[name="conv"]'
        ).check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check(
            "aten::flatten"
        ).check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(
            m.graph
        )

    # TODO: this is too long, split this to test_insert_observers.py and remove
    # insrt_observers prefix
    def test_insert_observers_propagate_observed(self):
        """Make sure we propagate observed property through general ops"""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()  # 定义第一个卷积层
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()  # 定义第二个卷积层

            def forward(self, x):
                x = self.conv1(x)  # 对输入应用第一个卷积层
                x = torch.flatten(x)  # 将张量展平
                # 我们不希望为 self.conv2 的输入插入观察器，因为 self.conv1 的输出已经被观察过了
                x = self.conv2(x)  # 对展平后的张量应用第二个卷积层
                return x

        m = torch.jit.script(M())  # 将模型 M 脚本化
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)  # 准备模型以进行量化
        # 检查模型中以 "_observer_" 开头的属性数量是否为3
        assert len(attrs_with_prefix(m, "_observer_")) == 3
        # 使用 FileCheck 检查模型的图结构，确保观察器的正确插入和未插入
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check(
            'prim::GetAttr[name="conv1"]'
        ).check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check(
            "aten::flatten"
        ).check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).check(
            'prim::GetAttr[name="conv2"]'
        ).check(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(
            m.graph
        )

    def test_insert_observers_propagate_observed_in_submodule(self):
        """Make sure we propagate observed property through general ops"""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()  # 定义第一个卷积层
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()  # 定义第二个卷积层
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层

            def forward(self, x):
                x = self.conv1(x)  # 对输入应用第一个卷积层
                x = self.avgpool(x)  # 对应用第一个卷积层后的结果应用自适应平均池化
                # 我们不希望为 self.conv2 的输入插入观察器，因为 self.conv1 的输出已经被观察过了
                x = self.conv2(x)  # 对自适应平均池化后的结果应用第二个卷积层
                return x

        m = torch.jit.script(M())  # 将模型 M 脚本化
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)  # 准备模型以进行量化
        # 检查模型中以 "_observer_" 开头的属性数量是否为3
        assert len(attrs_with_prefix(m, "_observer_")) == 3
        # 使用 FileCheck 检查模型的图结构，确保观察器的正确插入和未插入
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check(
            'prim::GetAttr[name="conv1"]'
        ).check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check(
            "prim::CallMethod"
        ).check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).check(
            'prim::GetAttr[name="conv2"]'
        ).check(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(
            m.graph
        )
    def test_insert_observers_propagate_observed_for_function(self):
        # 定义一个用于对通道进行重新排序的函数
        def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
            # 获取输入张量 x 的形状信息
            batchsize, num_channels, height, width = x.data.size()
            # 计算每组通道数
            channels_per_group = num_channels // groups
            # 重新形状化张量 x
            x = x.view(batchsize, groups, channels_per_group, height, width)
            # 转置张量 x，交换指定维度
            x = torch.transpose(x, 1, 2).contiguous()
            # 展平张量 x
            x = x.view(batchsize, -1, height, width)
            return x

        # 定义一个简单的神经网络模型 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加两个卷积层
                self.conv1 = torch.nn.Conv2d(3, 3, 1).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 1).float()

            def forward(self, x):
                # 前向传播函数，使用 conv1 对输入 x 进行卷积
                x = self.conv1(x)
                # 对卷积后的结果使用 channel_shuffle 函数进行通道重排
                x = channel_shuffle(x, 1)
                # 继续使用 conv2 对处理后的 x 进行卷积
                x = self.conv2(x)
                return x

        # 准备测试数据，包含两个样本
        data = [
            (
                torch.rand((1, 3, 10, 10), dtype=torch.float),
                torch.randint(0, 1, (1,), dtype=torch.long),
            )
            for _ in range(2)
        ]
        # 对模型 M 进行 JIT 编译并设置默认的量化配置
        m = torch.jit.script(M()).eval()
        m = prepare_jit(m, {"": default_qconfig})
        # 断言检查模型中观察属性的数量，预期为3
        assert (
            len(
                attrs_with_prefix(
                    m,
                    "_observer_",
                )
            )
            == 3
        )
    # 定义一个测试方法，用于测试插入观察者是否正常工作
    def test_insert_observers_for_if(self):
        # 定义一个继承自 torch.nn.Module 的类 QuantProp
        class QuantProp(torch.nn.Module):
            # 初始化方法，接受一个布尔类型参数 use_skip
            def __init__(self, use_skip):
                super().__init__()
                # 定义一个 3 输入、3 输出的 1x1 卷积层，并转换为浮点数
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                # 设置是否使用 skip 的标志
                self.use_skip = use_skip

            # 前向传播方法
            def forward(self, x):
                # 如果使用 skip
                if self.use_skip:
                    # 对输入 x 进行卷积操作
                    x = self.conv(x)
                    # 将 x 重塑为和 x.shape 相同的张量，并返回
                    return torch.reshape(x, x.shape)
                else:
                    # 否则，对输入 x 进行卷积操作，并直接返回结果
                    x = self.conv(x)
                    return torch.reshape(x, x.shape)

        # 定义一个继承自 torch.nn.Module 的类 Res
        class Res(torch.nn.Module):
            # 初始化方法，接受一个布尔类型参数 use_skip
            def __init__(self, use_skip):
                super().__init__()
                # 定义一个 3 输入、3 输出的 1x1 卷积层，并转换为浮点数
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                # 设置是否使用 skip 的标志
                self.use_skip = use_skip

            # 前向传播方法
            def forward(self, x):
                # 如果使用 skip，直接返回对输入 x 的卷积结果
                if self.use_skip:
                    return self.conv(x)
                else:
                    # 否则，也返回对输入 x 的卷积结果
                    return self.conv(x)

        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 QuantProp 实例，并设置 use_skip 为 True
                self.quant_prop = QuantProp(True)
                # 创建一个 Res 实例，并设置 use_skip 为 False
                self.res = Res(False)

            # 前向传播方法
            def forward(self, x):
                # 调用 quant_prop 对象的前向传播方法处理输入 x
                x = self.quant_prop(x)
                # 调用 res 对象的前向传播方法处理上一步的输出 x
                x = self.res(x)
                # 返回处理后的结果 x
                return x

        # 创建一个包含一个随机张量的数据列表
        data = [torch.rand(1, 3, 10, 10, dtype=torch.float)]
        # 定义一个包含 True 和 False 的列表，用于追踪模型的追踪和脚本模式
        result = {False: [1, 2, 2], True: [2, 1, 0]}
        # 遍历追踪列表中的每一个元素
        for tracing in [True, False]:
            # 如果当前追踪为 True
            if tracing:
                # 使用 torch.jit.trace 方法追踪 M 类的实例，并转换为评估模式
                m = torch.jit.trace(M(), data).eval()
            else:
                # 否则，使用 torch.jit.script 方法将 M 类的实例转换为脚本模式并转换为评估模式
                m = torch.jit.script(M()).eval()
            # 对追踪后的模型 m 进行准备，使用默认的量化配置
            m = prepare_jit(m, {"": default_qconfig})
            # 断言检查 m 模型中以 "_observer_" 前缀开头的属性数量是否等于 result[tracing][0]
            assert (
                len(
                    attrs_with_prefix(
                        m,
                        "_observer_",
                    )
                )
                == result[tracing][0]
            )
            # 断言检查 m.quant_prop 模型中以 "_observer_" 前缀开头的属性数量是否等于 result[tracing][1]
            assert (
                len(
                    attrs_with_prefix(
                        m.quant_prop,
                        "_observer_",
                    )
                )
                == result[tracing][1]
            )
            # 断言检查 m.res 模型中以 "_observer_" 前缀开头的属性数量是否等于 result[tracing][2]
            assert (
                len(
                    attrs_with_prefix(
                        m.res,
                        "_observer_",
                    )
                )
                == result[tracing][2]
            )
    def test_insert_observers_for_nested_if(self):
        # 定义一个测试函数，用于测试插入观察器是否正确处理嵌套的条件语句

        class Res(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的类 Res
            def __init__(self, use_skip):
                super().__init__()
                # 调用父类的初始化方法
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                # 创建一个 2D 卷积层对象 self.conv，输入通道数为 3，输出通道数为 3，卷积核大小为 1
                self.cond = use_skip
                # 设置实例变量 self.cond 为传入的 use_skip 参数
                self.use_skip = use_skip
                # 设置实例变量 self.use_skip 为传入的 use_skip 参数

            def forward(self, x):
                # 定义该模块的前向传播方法 forward，参数 x 为输入数据
                if self.use_skip:
                    # 如果 self.use_skip 为 True
                    if self.cond:
                        # 如果 self.cond 为 True
                        return self.conv(x)
                        # 返回卷积层对象 self.conv 对输入 x 的计算结果
                    else:
                        # 如果 self.cond 为 False
                        return self.conv(x)
                        # 返回卷积层对象 self.conv 对输入 x 的计算结果
                else:
                    # 如果 self.use_skip 为 False
                    return self.conv(x)
                    # 返回卷积层对象 self.conv 对输入 x 的计算结果

        class M(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的类 M
            def __init__(self):
                super().__init__()
                # 调用父类的初始化方法
                self.res1 = Res(True)
                # 创建 Res 类的实例 self.res1，并传入参数 True
                self.res2 = Res(False)
                # 创建 Res 类的实例 self.res2，并传入参数 False

            def forward(self, x):
                # 定义该模块的前向传播方法 forward，参数 x 为输入数据
                x = self.res1(x)
                # 对输入 x 应用实例 self.res1 的 forward 方法
                x = self.res2(x)
                # 对输入 x 应用实例 self.res2 的 forward 方法
                return x
                # 返回结果 x

        data = torch.rand((1, 3, 10, 10), dtype=torch.float)
        # 生成一个形状为 (1, 3, 10, 10) 的随机张量数据 data，数据类型为 torch.float
        result = {True: 3, False: 1}
        # 创建一个字典 result，用于存储预期的结果数量，True 对应 3，False 对应 1
        for tracing in [True, False]:
            # 遍历 tracing 取值为 True 和 False 的情况
            if tracing:
                # 如果 tracing 为 True
                m = torch.jit.trace(M(), data).eval()
                # 对 M 类的实例进行追踪编译为 TorchScript，然后设为评估模式
            else:
                # 如果 tracing 为 False
                m = torch.jit.script(M()).eval()
                # 对 M 类的实例进行脚本编译为 TorchScript，然后设为评估模式
            m = prepare_jit(m, {"": default_qconfig})
            # 对模型 m 应用 prepare_jit 函数，设置量化配置为 default_qconfig
            assert len(attrs_with_prefix(m, "_observer_")) == result[tracing]
            # 断言模型 m 中带有前缀 "_observer_" 的属性数量与 result 中对应 tracing 值的预期数量相等
    def test_insert_observers_for_if_consistent_observation(self):
        """检查在条件分支中观察是否一致的情况下，对 if 语句的量化是否有效"""

        class M(torch.nn.Module):
            def __init__(self, cond):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()  # 创建一个3通道的2D卷积层
                self.cond = cond  # 初始化条件参数

            def forward(self, x):
                x = self.conv(x)  # 对输入 x 进行卷积操作
                # x 已经被观察
                if self.cond:
                    x = torch.flatten(x)  # 如果条件为真，对 x 进行展平操作
                return x  # 返回处理后的 x

        class M2(torch.nn.Module):
            def __init__(self, cond):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()  # 创建第一个3通道的2D卷积层
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()  # 创建第二个3通道的2D卷积层
                self.cond = cond  # 初始化条件参数

            def forward(self, x):
                x = self.conv1(x)  # 对输入 x 进行第一个卷积操作
                if self.cond:
                    x = self.conv2(x)  # 如果条件为真，对 x 进行第二个卷积操作
                    # x 将在此分支中被观察
                else:
                    x = torch.flatten(x)  # 如果条件为假，对 x 进行展平操作
                # 由于两个分支的输出都被量化，if 节点的量化是一致的
                return x  # 返回处理后的 x

        data = torch.rand((1, 3, 5, 5), dtype=torch.float)  # 创建一个随机数据张量
        options = list(itertools.product([True, False], [True, False]))  # 创建条件和跟踪选项的组合列表
        for cond, tracing in options:
            if tracing:
                m = torch.jit.trace(M(cond), data)  # 如果跟踪为真，对模型 M 进行追踪
            else:
                m = torch.jit.script(M(cond))  # 如果跟踪为假，对模型 M 进行脚本化
            m = prepare_jit(m, {"": default_qconfig})  # 准备 JIT 模型
            assert len(attrs_with_prefix(m, "_observer_")) == 2  # 断言观察器的数量为2个

        for cond, tracing in options:
            if tracing:
                m = torch.jit.trace(M2(cond), data)  # 如果跟踪为真，对模型 M2 进行追踪
            else:
                m = torch.jit.script(M2(cond))  # 如果跟踪为假，对模型 M2 进行脚本化
            m = prepare_jit(m, {"": default_qconfig})  # 准备 JIT 模型
            num_observers = 2 if tracing and not cond else 3  # 计算预期的观察器数量
            assert len(attrs_with_prefix(m, "_observer_")) == num_observers  # 断言观察器的数量与预期相符
    # 定义一个测试方法，用于测试插入量化和反量化操作
    def test_insert_quant_dequant(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个输入通道为3，输出通道为5，卷积核大小为3的二维卷积层
                self.conv = torch.nn.Conv2d(3, 5, 3).float()

            # 前向传播方法，将输入数据通过卷积层处理并返回
            def forward(self, x):
                return self.conv(x)

        # 遍历两种情况：是否按通道量化
        for is_per_channel in [True, False]:
            # 使用 JIT 脚本化模型 M
            m = torch.jit.script(M())
            # 根据是否按通道量化选择不同的观察器
            observer = (
                default_per_channel_weight_observer.with_args(ch_axis=1)
                if is_per_channel
                else default_observer
            )
            # 构建量化配置字典，对激活和权重应用选择的观察器
            qconfig_dict = {"": QConfig(activation=observer, weight=observer)}
            # 准备 JIT 模型，应用量化配置
            m = prepare_jit(m, qconfig_dict)
            # 生成随机数据作为输入
            data = torch.randn(1, 3, 10, 10, dtype=torch.float)

            # 执行一次模型前向传播
            m(data)
            # 转换 JIT 模型为量化版本，开启调试模式
            m = convert_jit(m, debug=True)
            # 断言量化后模型的子模块数为1，即只有一个卷积层
            assert (
                len(m._modules._c.items()) == 1
            ), "Expected to have single submodule of conv"
            # 确保量化后的模型可以执行
            m(data)
            # 根据是否按通道量化选择相应的量化函数名
            quant_func = (
                "aten::quantize_per_channel"
                if is_per_channel
                else "aten::quantize_per_tensor"
            )
            # 使用 FileCheck 来检查量化后模型图中量化操作的数量是否符合预期
            FileCheck().check_count(quant_func, 3, exactly=True).run(m.graph)
    def test_insert_quant_dequant_shared_class_type(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加两个 Conv2d 层，输入通道数、输出通道数、卷积核大小均为 3
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            # 前向传播方法
            def forward(self, x):
                return self.conv2(self.conv1(x))

        # 遍历两种量化模式：按通道和按张量
        for is_per_channel in [True, False]:
            # 使用 torch.jit.script 将 M 类型实例化为 TorchScript 模型
            m = torch.jit.script(M())
            # 根据量化模式选择观察器
            observer = (
                default_per_channel_weight_observer.with_args(ch_axis=1)
                if is_per_channel
                else default_observer
            )
            # 创建量化配置对象
            qconfig = QConfig(activation=observer, weight=observer)
            qconfig_dict = {"": qconfig}
            # 准备 TorchScript 模型进行量化
            m = prepare_jit(m, qconfig_dict)

            # 断言：期望有 3 个观察器（输入、输出以及 conv1/conv2 之间的值）
            assert (
                len(attrs_with_prefix(m, "_observer_")) == 3
            ), "Expected to have 3 obervers"
            # 断言：期望 conv1 有 1 个观察器（权重）
            assert (
                len(attrs_with_prefix(m.conv1, "_observer_")) == 1
            ), "Expected to have 1 obervers"
            # 断言：期望 conv2 有 1 个观察器（权重）
            assert (
                len(attrs_with_prefix(m.conv2, "_observer_")) == 1
            ), "Expected to have 1 obervers"

            # 创建测试数据
            data = torch.randn(1, 3, 10, 10, dtype=torch.float)
            # 执行模型推理
            m(data)
            # 将模型转换为量化后的 TorchScript 模型，启用调试模式
            m = convert_jit(m, debug=True)
            # 再次执行模型推理
            m(data)
            # 断言：期望所有观察器已被移除
            assert (
                len(attrs_with_prefix(m, "_observer_")) == 0
            ), "Expected to have 0 obervers"
            # 断言：期望 conv1 的观察器已被移除
            assert (
                len(attrs_with_prefix(m.conv1, "_observer_")) == 0
            ), "Expected to have 0 obervers"
            # 断言：期望 conv2 的观察器已被移除
            assert (
                len(attrs_with_prefix(m.conv2, "_observer_")) == 0
            ), "Expected to have 0 obervers"

            # 根据量化模式选择相应的量化函数
            quant_func = (
                "aten::quantize_per_channel"
                if is_per_channel
                else "aten::quantize_per_tensor"
            )
            # 遍历模块列表，例如 ["conv1", "conv2"]
            for module in ["conv1", "conv2"]:
                # 获取指定模块的 TorchScript 表示
                conv = m._c.getattr(module)
                # 检查权重量化的节点序列
                FileCheck().check(quant_func).check_next("aten::dequantize").check(
                    'prim::CallMethod[name="_conv_forward"]'
                ).check("return").run(get_forward_graph(conv))
                # 检查 _conv_forward 方法中是否不存在量化节点
                FileCheck().check_not(quant_func).check("aten::conv2d").check_not(
                    quant_func
                ).check("return").run(conv._get_method("_conv_forward").graph)
    def test_dedup_module_uses(self):
        # 定义一个名为 M 的内部类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在构造函数中初始化一个 ReLU 激活函数
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # 前向传播函数，应用 ReLU 激活函数
                x = self.relu(x)
                # 减去 0.5 的操作
                x -= 0.5
                # 返回再次应用 ReLU 激活函数后的结果
                return self.relu(x)

        # 生成一个形状为 (2, 2) 的随机张量数据
        data = torch.randn((2, 2))
        # 使用 torch.jit.script 将模型 M 实例化为脚本模块
        m = torch.jit.script(M())
        # 记录模型在给定数据上的前向传播结果
        ref_res = m(data)
        # 断言语句，检查模块中以 "relu" 开头的子模块数量为 1
        assert (
            len([x for x, _ in m._modules._c.items() if x.startswith("relu")]) == 1
        ), "Expected to have 1 relu modules after dedup module uses"
        # 调用 Torch 的模块使用去重的 JIT 传递
        torch._C._jit_pass_dedup_module_uses(m._c)
        # 递归包装 C++ 模块为 Python 模块
        m = torch.jit._recursive.wrap_cpp_module(m._c)
        # 获取模型在数据上的新的前向传播结果
        res = m(data)
        # 断言语句，检查模块中以 "relu" 开头的子模块数量为 2
        assert (
            len([x for x, _ in m._modules._c.items() if x.startswith("relu")]) == 2
        ), "Expected to have 2 relu modules after dedup module uses"
        # 使用 self.assertEqual 检查 res 和 ref_res 的相等性
        self.assertEqual(res, ref_res)

    def test_replicate_dequantize(self):
        # 定义一个名为 M 的内部类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在构造函数中初始化一个 3x3 的 Conv2d 模块并转换为 float 类型
                self.conv = torch.nn.Conv2d(3, 3, 1).float()

            def forward(self, x):
                # 前向传播函数，对输入张量 x 进行去量化操作
                x = torch.dequantize(x)
                # 使用 self.conv 处理去量化后的张量 x
                r = self.conv(x)
                # 将处理结果与 x 相加
                r += x
                # 返回加和结果
                return r

        # 生成一个形状为 [1, 3, 10, 10] 的随机张量数据，并进行量化
        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        # 使用 torch.jit.script 将模型 M 实例化为脚本模块
        m = torch.jit.script(M())
        # 记录模型在量化后的数据上的前向传播结果
        ref_res = m(x)
        # 使用 FileCheck 检查模型的图中 "aten::dequantize" 操作出现的次数为 1
        FileCheck().check_count("aten::dequantize", 1, exactly=True).run(m.graph)
        # 调用 Torch 的去量化复制 JIT 传递
        torch._C._jit_pass_replicate_dequantize(m.graph)
        # 使用 FileCheck 检查模型的图中 "aten::dequantize" 操作出现的次数为 2
        FileCheck().check_count("aten::dequantize", 2, exactly=True).run(m.graph)
        # 获取模型前向传播函数的引用，并在量化后的数据上执行
        res = get_forward(m._c)(x)
        # 使用 self.assertEqual 检查 res 和 ref_res 的相等性
        self.assertEqual(res, ref_res)
    def test_replicate_dequantize_in_block(self):
        # 定义一个测试方法，用于验证在模块中复制 dequantize 操作
        class M(torch.nn.Module):
            def __init__(self, cond):
                super().__init__()
                # 创建一个输入通道为3，输出通道为3的1x1卷积层
                self.conv = torch.nn.Conv2d(3, 3, 1).float()

                # 条件参数
                self.cond = cond

            def forward(self, x):
                # 对输入数据进行 dequantize 操作
                x = torch.dequantize(x)
                if self.cond:
                    # 如果条件为真，应用卷积操作
                    x = self.conv(x)
                else:
                    # 否则，执行 x + 3 的操作
                    x = x + 3
                return x

        # 创建一个形状为[1, 3, 10, 10]的随机张量，并进行量化
        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        # 对模块 M 进行脚本化
        m = torch.jit.script(M(True))
        # 计算参考结果
        ref_res = m(x)
        # 使用 FileCheck 验证图中的 dequantize 操作数量
        FileCheck().check_count("aten::dequantize", 1, exactly=True).run(m.graph)
        # 在图中复制 dequantize 操作
        torch._C._jit_pass_replicate_dequantize(m.graph)
        # 验证复制后图中 dequantize 操作的数量
        FileCheck().check_count("aten::dequantize", 2, exactly=True).run(m.graph)
        # 验证 dequantize 操作紧跟在 conv 方法调用前
        FileCheck().check("aten::dequantize").check_next("CallMethod").run(m.graph)
        # 验证 dequantize 操作在 add 方法调用前
        FileCheck().check("aten::dequantize").check("aten::dequantize").check_next(
            "aten::add"
        ).run(m.graph)
        # 获取正向传播的结果
        res = get_forward(m._c)(x)
        # 断言结果与参考结果相等
        self.assertEqual(res, ref_res)

    def test_swap_functional_linear(self):
        # TODO: This pass replaces any function called "linear" with "aten::linear"
        # No longer necessary, and also quite surprising
        # 定义一个测试方法，用于验证替换 functional 中的 linear 方法为 aten::linear
        def linear(input, weight, bias):
            return torch.nn.functional.linear(input, weight, bias)

        class M(torch.nn.Module):
            def forward(self, x, weight, bias):
                # 对输入数据进行 dequantize 操作
                x = torch.dequantize(x)
                # 对权重进行 dequantize 操作
                weight = torch.dequantize(weight)
                # 调用 linear 方法，传入输入、权重和偏置
                x = linear(x, weight, bias)
                # 对输出进行量化
                x = torch.quantize_per_tensor(
                    x, scale=1.0, zero_point=0, dtype=torch.quint8
                )
                return x

        # 创建一个形状为[10, 5]的随机张量，并进行量化
        x = torch.rand((10, 5), dtype=torch.float)
        x = torch.quantize_per_tensor(x, scale=0.5, zero_point=1, dtype=torch.quint8)
        # 创建一个形状为[5, 5]的随机张量作为权重，并进行量化
        weight = torch.rand((5, 5), dtype=torch.float)
        weight = torch.quantize_per_tensor(
            weight, scale=0.5, zero_point=1, dtype=torch.qint8
        )
        # 创建一个形状为[5]的随机张量作为偏置
        bias = torch.rand((5), dtype=torch.float)
        # 对模块 M 进行脚本化
        m = torch.jit.script(M())
        # 计算参考结果
        ref_res = m(x, weight, bias)
        # 使用 FileCheck 验证图中的 CallFunction 调用
        FileCheck().check("CallFunction").run(m.graph)
        # 在图中替换 functional.linear 方法为 aten::linear
        torch._C._jit_pass_swap_functional_linear(m.graph)
        # 验证图中不再存在 CallFunction 调用
        FileCheck().check("aten::linear").check_not("CallFunction").run(m.graph)
        # 获取重新计算后的结果
        res = m(x, weight, bias)
        # 断言结果与参考结果相等
        self.assertEqual(res, ref_res)
    def test_replicate_quantize_for_if(self):
        """We want to move quantize nodes for output of prim::If
        inside the prim::If blocks so that we can match quantization
        patterns.
        """
        # 定义一个测试函数，用于验证在 prim::If 的输出中移动量化节点，
        # 以便可以匹配量化模式。

        class Res(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = True

            def forward(self, x: torch.Tensor, cond: bool) -> torch.Tensor:
                # to avoid being frozen
                self.use_skip = cond
                # 如果条件 self.use_skip 为真，则执行以下操作
                if self.use_skip:
                    return self.conv(x)
                # 否则执行以下操作
                else:
                    return self.conv2(x)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.res1 = Res()
                self.res2 = Res()

            def forward(self, x):
                x = self.res1(x, True)
                x = self.res2(x, False)
                return x

        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        qconfig_dict = {"": default_qconfig}
        m = torch.jit.script(M()).eval()
        m = quantize_jit(m, qconfig_dict, test_only_eval_fn, [data])
        # make sure patterns in both branches are fused
        # 确保两个分支中的模式被融合
        FileCheck().check_count("quantized::conv2d(", 4, exactly=True).run(m.graph)

    def test_finalize_for_linear(self):
        # 定义一个测试函数，用于验证线性层的量化最终状态

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                return self.fc(x)

        data = [[torch.rand((1, 5), dtype=torch.float)]]
        qconfig_dict = {"": default_qconfig}
        model = torch.jit.script(M()).eval()
        model = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data])
        # make sure there is only one quantize_per_tensor for input
        # and linear_prepack is folded
        # 确保输入只有一个 quantize_per_tensor，
        # 并且 linear_prepack 已经被折叠
        FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True).check_not(
            "quantized::linear_prepack"
        ).check("quantized::linear").run(model.graph)

    def test_inplace_option(self):
        # 定义一个测试函数，用于验证 inplace 选项

        for tracing in [True, False]:
            model = get_script_module(
                torch.nn.Conv2d(3, 3, 3).float(), tracing, self.img_data_2d[0][0]
            )
            qconfig_dict = {"": default_qconfig}
            quantize_jit(
                model, qconfig_dict, test_only_eval_fn, [self.img_data_2d], inplace=True
            )
            FileCheck().check("quantized::conv2d").run(model.graph)

            FileCheck().check_not("aten::conv2d").run(model.graph)
    def test_finalize_debug(self):
        # 定义一个简单的神经网络模型类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个 3 输入通道、3 输出通道、3x3 卷积层和一个平均池化层
                self.conv = torch.nn.Conv2d(3, 3, 3).float()
                self.avgpool = torch.nn.AvgPool2d(3)

            def forward(self, x):
                # 前向传播函数，首先使用卷积层处理输入 x
                x = self.conv(x)
                # 然后通过平均池化层处理结果 x
                x = self.avgpool(x)
                return x

        # 创建一个包含单个样本的数据列表
        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        # 创建一个空字符串键对应默认量化配置的字典
        qconfig_dict = {"": default_qconfig}
        # 使用 torch.jit.script 将模型 M 实例化为 TorchScript，并设为评估模式
        model = torch.jit.script(M()).eval()
        # 对模型进行量化，使用自定义的量化配置字典 qconfig_dict 和评估函数 test_only_eval_fn，开启调试模式
        model = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data], debug=True)
        # 使用 FileCheck 验证模型图中不包含 "quantized::conv2d"，但包含 "aten::conv2d"、"aten::avg_pool2d"、
        # "aten::q_scale"、"aten::q_zero_point"、"prim::dtype"、"aten::quantize_per_tensor" 和 "aten::dequantize"
        FileCheck().check_not("quantized::conv2d").check("aten::conv2d").check(
            "aten::avg_pool2d"
        ).check("aten::q_scale").check_next("aten::q_zero_point").check_next(
            "prim::dtype"
        ).check_next(
            "aten::quantize_per_tensor"
        ).check(
            "aten::dequantize"
        ).run(
            model.graph
        )

    def test_module_list(self):
        # 定义一个简单的全连接层模型类 SimpleLinearLayer，继承自 torch.nn.Module
        class SimpleLinearLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个输入和输出大小都为 5 的线性层
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                # 前向传播函数，直接通过线性层处理输入 x
                return self.fc(x)

        # 定义一个复杂模型类 ComplexModel，继承自 torch.nn.Module
        class ComplexModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 ModuleList 创建包含两个 SimpleLinearLayer 实例的层列表
                self.layers = torch.nn.ModuleList(
                    [SimpleLinearLayer() for i in range(2)]
                )

            def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
                # 前向传播函数，逐层处理输入 x 并将每层处理结果保存在 states 列表中返回
                states = []
                for layer in self.layers:
                    val = layer(x)
                    states.append(val)
                return states

        # 创建一个大小为 (1, 5) 的随机张量作为输入数据
        data = torch.rand((1, 5), dtype=torch.float)
        # 创建一个空字符串键对应默认量化配置的字典
        qconfig_dict = {"": default_qconfig}
        # 使用 torch.jit.script 将模型 ComplexModel 实例化为 TorchScript，并设为评估模式
        model = torch.jit.script(ComplexModel()).eval()
        # 对模型进行准备，使用自定义的量化配置字典 qconfig_dict
        model = prepare_jit(model, qconfig_dict)
        # 使用 attrs_with_prefix 函数检查模型中以 "_observer" 开头的属性数量是否为 3
        assert len(attrs_with_prefix(model, "_observer")) == 3
        # 执行模型前向传播，使用输入数据 data
        model(data)
        # 将模型转换为量化后的模型，关闭调试模式
        model = convert_jit(model, debug=False)
        # 使用 FileCheck 验证模型图中包含 "quantized::linear"
        FileCheck().check("quantized::linear").check("quantized::linear").run(
            model.graph
        )
    # 定义一个测试函数，用于测试卷积层的跟踪
    def test_conv_trace(self):
        # 定义一个内部的神经网络模型类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个一维卷积层，输入通道数为 3，输出通道数为 3，卷积核大小为 3
                self.conv1d = torch.nn.Conv1d(3, 3, 3).float()
                # 创建一个二维卷积层，输入通道数为 3，输出通道数为 3，卷积核大小为 3x3
                self.conv2d = torch.nn.Conv2d(3, 3, 3).float()
                # 创建一个三维卷积层，输入通道数为 3，输出通道数为 3，卷积核大小为 3x3x3
                self.conv3d = torch.nn.Conv3d(3, 3, 3).float()

            # 前向传播函数
            def forward(self, x, y, z):
                # 对输入 x 进行一维卷积操作
                a = self.conv1d(x)
                # 对输入 y 进行二维卷积操作
                b = self.conv2d(y)
                # 对输入 z 进行三维卷积操作
                c = self.conv3d(z)
                # 返回卷积结果元组 (a, b, c)
                return (a, b, c)

        # 定义量化配置字典
        qconfig_dict = {"": default_qconfig}
        # 定义输入数据元组，包含一维、二维和三维的随机张量数据
        inputs = (
            torch.rand((1, 3, 10), dtype=torch.float),
            torch.rand((1, 3, 10, 10), dtype=torch.float),
            torch.rand((1, 3, 10, 10, 10), dtype=torch.float),
        )
        # 使用 torch.jit.trace 方法对 M 类型的模型进行跟踪，并转换为评估模式
        model = torch.jit.trace(M(), inputs).eval()
        # 调用 prepare_jit 函数，对量化配置字典中的模型进行预处理
        m = prepare_jit(model, qconfig_dict)
        # 使用 FileCheck 对象检查一维卷积层前向图中的操作，确保不包含低阶卷积操作
        FileCheck().check("aten::conv1d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.conv1d._c))
        )
        # 使用 FileCheck 对象检查二维卷积层前向图中的操作，确保不包含低阶卷积操作
        FileCheck().check("aten::conv2d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.conv2d._c))
        )
        # 使用 FileCheck 对象检查三维卷积层前向图中的操作，确保不包含低阶卷积操作
        FileCheck().check("aten::conv3d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.conv3d._c))
        )

    # 定义一个测试函数，用于测试转置卷积层的跟踪
    def test_convtranspose_trace(self):
        # 定义一个内部的神经网络模型类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个一维转置卷积层，输入通道数为 3，输出通道数为 3，卷积核大小为 3
                self.convtranspose1d = torch.nn.ConvTranspose1d(3, 3, 3).float()
                # 创建一个二维转置卷积层，输入通道数为 3，输出通道数为 3，卷积核大小为 3x3
                self.convtranspose2d = torch.nn.ConvTranspose2d(3, 3, 3).float()
                # 创建一个三维转置卷积层，输入通道数为 3，输出通道数为 3，卷积核大小为 3x3x3
                self.convtranspose3d = torch.nn.ConvTranspose3d(3, 3, 3).float()

            # 前向传播函数
            def forward(self, x, y, z):
                # 对输入 x 进行一维转置卷积操作
                a = self.convtranspose1d(x)
                # 对输入 y 进行二维转置卷积操作
                b = self.convtranspose2d(y)
                # 对输入 z 进行三维转置卷积操作
                c = self.convtranspose3d(z)
                # 返回转置卷积结果元组 (a, b, c)
                return (a, b, c)

        # 定义量化配置字典
        qconfig_dict = {"": default_qconfig}
        # 定义输入数据元组，包含一维、二维和三维的随机张量数据
        inputs = (
            torch.rand((1, 3, 10), dtype=torch.float),
            torch.rand((1, 3, 10, 10), dtype=torch.float),
            torch.rand((1, 3, 10, 10, 10), dtype=torch.float),
        )
        # 使用 torch.jit.trace 方法对 M 类型的模型进行跟踪，并转换为评估模式
        model = torch.jit.trace(M(), inputs).eval()
        # 调用 prepare_jit 函数，对量化配置字典中的模型进行预处理
        m = prepare_jit(model, qconfig_dict)
        # 使用 FileCheck 对象检查一维转置卷积层前向图中的操作，确保不包含低阶卷积操作
        FileCheck().check("aten::conv_transpose1d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.convtranspose1d._c))
        )
        # 使用 FileCheck 对象检查二维转置卷积层前向图中的操作，确保不包含低阶卷积操作
        FileCheck().check("aten::conv_transpose2d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.convtranspose2d._c))
        )
        # 使用 FileCheck 对象检查三维转置卷积层前向图中的操作，确保不包含低阶卷积操作
        FileCheck().check("aten::conv_transpose3d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.convtranspose3d._c))
        )
    def test_replicate_dequant_same_value(self):
        class Mul(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个包含3个输入通道、3个输出通道和3x3卷积核的卷积层
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                # 将输入张量 x 经过卷积操作
                x = self.conv(x)
                # 返回经过两次乘法操作的张量
                return x * x

        # 创建包含一个随机数据张量的列表
        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        # 定义一个空字符串为键，对应默认量化配置为值的字典
        qconfig_dict = {"": default_qconfig}
        # 使用 Torch Script 对 Mul 类进行脚本化，并转为评估模式
        model = torch.jit.script(Mul()).eval()
        # 对模型进行量化，使用给定的量化配置字典和测试评估函数
        m = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data])
        # 在量化后的模型图中运行 FileCheck，检查是否包含 quantized::mul，但不包含 aten::mul
        FileCheck().check("quantized::mul(").check_not("aten::mul").run(m.graph)

    @skipIfNoFBGEMM
    def test_quantize_fork_wait(self):
        """Tests the case where fork and wait calls are in different subgraphs
        Calling inline fork-wait only removes the fork call and leaves aten::wait
        calls in the graph, with Tensor as input (instead of Future[Tensor])
        """

        class MainModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 ForkModule 的实例
                self.fork_ops = ForkModule()

            def init_values(self, x):
                # 调用 ForkModule 的 forward 方法，获取共享模块
                shared_module = self.fork_ops(x)
                # 将共享模块保存在 self.fork_dict 中
                self.fork_dict = shared_module

            def forward(self, x):
                # 调用 torch.jit._wait 等待 fork_ops 的输出
                val = torch.jit._wait(self.fork_ops(x))
                return val

        class TestModule(torch.nn.Module):
            def forward(self, x):
                # 创建权重张量 w 和偏置张量 b
                w = torch.ones(5, 5)
                b = torch.zeros(5)
                # 使用线性函数计算并返回输出张量
                return torch.nn.functional.linear(x, w, b)

        class ForkModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 TestModule 的实例
                self.test = TestModule()

            def forward(self, x):
                # 调用 torch.jit._fork 方法，将 test 的 forward 方法作为子图进行分叉
                fut = torch.jit._fork(self.test.forward, x)
                return fut

        # 创建 MainModule 的实例并设置为评估模式
        model = MainModule().eval()
        # 对 MainModule 进行动态追踪，并传入输入张量的示例
        traced = torch.jit.trace(model, (torch.randn(5, 5),))
        # 准备动态 JIT 模型，使用默认量化配置
        model = prepare_dynamic_jit(traced, {"": default_qconfig})
        # 将动态 JIT 模型转换为量化模型
        model = convert_dynamic_jit(model)
        # 在量化后的模型图中运行 FileCheck，检查是否包含 quantized::linear_dynamic
        FileCheck().check("quantized::linear_dynamic").run(model.graph)
        # 确保模型保存操作正常执行
        b = io.BytesIO()
        torch.jit.save(model, b)
    @skipIfNoFBGEMM
    # 跳过测试，如果系统不支持 FBGEMM，FBGEMM 是用于量化的低精度运算库
    def test_linear(self):
        # 定义 ModuleLinear 类，继承自 torch.nn.Module，用于定义一个线性层模型
        class ModuleLinear(torch.nn.Module):
            def __init__(self, has_relu=False, f_relu=False):
                super().__init__()
                # 创建一个包含 30 个输入和 4 个输出的线性层，数据类型为 float
                self.linear = torch.nn.Linear(30, 4).float()
                # 根据条件选择是否添加 ReLU 激活函数
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                # 在前向传播中应用选择的激活函数和线性层
                return self.relu(self.linear(x))

        # 定义 FuncLinear 类，继承自 torch.nn.Module，用于定义一个函数式的线性层模型
        class FuncLinear(torch.nn.Module):
            def __init__(self, has_relu=False, f_relu=False):
                super().__init__()
                # 随机初始化权重矩阵 w，形状为 (4, 30)
                self.w = torch.randn(4, 30)
                # 随机初始化偏置向量 b，形状为 (4,)
                self.b = torch.randn(4)
                # 根据条件选择是否添加 ReLU 激活函数
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                # 在前向传播中应用选择的激活函数和函数式线性层
                return self.relu(F.linear(x, self.w, self.b))

        # 准备数据，包含一个形状为 (1, 30) 的随机张量的列表
        data = [[torch.rand((1, 30), dtype=torch.float)]]
        # 遍历模型类和追踪选项的组合，使用 itertools.product 生成所有组合
        for model, tracing in itertools.product(
            [ModuleLinear(has_relu=False), FuncLinear(has_relu=False)], [True, False]
        ):
            # 检查图模式下的操作，量化后的线性操作，返回修改后的模型
            model = self.checkGraphModeOp(model, data, "quantized::linear", tracing)
            # 使用 FileCheck 检查图中量化的操作是否正确
            FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True).run(
                model.graph
            )
            # 使用 FileCheck 确保图中没有量化前打包的线性操作
            FileCheck().check_not("quantized::linear_prepack").run(model.graph)

        # 遍历 f_relu 和追踪选项的组合，使用 itertools.product 生成所有组合
        for f_relu, tracing in itertools.product([True, False], [True, False]):
            # 遍历模型类的列表
            for model in [
                ModuleLinear(has_relu=True, f_relu=f_relu),
                FuncLinear(has_relu=True, f_relu=f_relu),
            ]:
                # 检查图模式下的操作，量化后的线性和 ReLU 操作，返回修改后的模型
                model = self.checkGraphModeOp(
                    model, data, "quantized::linear_relu", tracing
                )
                # 使用 FileCheck 检查图中没有原始的线性和 ReLU 操作
                checker = (
                    FileCheck()
                    .check_not("aten::linear")
                    .check_not("aten::relu")
                    .check_not("quantized::linear(")
                    .check_not("quantized::relu(")
                    .run(model.graph)
                )
    def test_quantized_conv(self):
        # 定义不同维度的卷积模块
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class Conv(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 根据给定维度创建对应的卷积层，并转换为浮点数
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                # 执行卷积操作
                return self.conv(x)

        # 生成不同维度和跟踪选项的所有组合
        options = itertools.product([1, 2, 3], [True, False])
        for dim, tracing in options:
            # 调用 self.checkGraphModeOp 方法，检查模型操作
            model = self.checkGraphModeOp(
                Conv(dim),
                self.img_data_dict[dim],
                f"quantized::conv{dim}d",  # 期望的量化卷积操作名称
                tracing,  # 是否启用跟踪
            )
            # 确保输入中只有一个量化的 per-tensor 操作，并且 conv2d_prepack 被折叠
            FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True).run(
                model.graph
            )

            # 确保图中没有量化卷积预打包的操作
            FileCheck().check_not(f"quantized::conv{dim}d_prepack").run(model.graph)

    @skipIfNoFBGEMM
    def test_quantized_conv_relu(self):
        """用于测试 conv1d_relu/conv2d_relu/conv3d_relu 的功能"""
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class ConvNdRelu(torch.nn.Module):
            def __init__(self, dim, inplace):
                super().__init__()
                # 根据给定维度创建卷积和ReLU激活层，并转换为浮点数
                self.conv = conv_module[dim](3, 3, 3).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                # 执行卷积和ReLU操作
                return self.relu(self.conv(x))

        class ConvNdFunctionalRelu(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 根据给定维度创建卷积层，并转换为浮点数
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                # 执行卷积操作，然后应用函数式ReLU
                return F.relu(self.conv(x))

        class ConvNdInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 根据给定维度创建卷积层，并转换为浮点数

                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                # 执行卷积操作，然后应用就地函数式ReLU
                return F.relu(self.conv(x), True)

        # 生成不同维度和跟踪选项的所有组合
        options = itertools.product([1, 2, 3], [True, False])
        for dim, tracing in options:
            # 对于每个原始模型，检查图模式操作
            for orig_m in [
                ConvNdRelu(dim, True),
                ConvNdRelu(dim, False),
                ConvNdFunctionalRelu(dim),
                ConvNdInplaceFunctionalRelu(dim),
            ]:
                conv_name = f"conv{dim}d"
                m = self.checkGraphModeOp(
                    orig_m,
                    self.img_data_dict[dim],
                    f"quantized::conv{dim}d_relu(",  # 期望的量化卷积ReLU操作名称
                    tracing=tracing,
                )

                # 确保图中不存在普通卷积、ReLU、量化卷积或量化ReLU操作
                FileCheck().check_not(f"aten::conv{dim}d(").check_not(
                    "aten::relu"
                ).check_not(f"quantized::conv{dim}d(").check_not(
                    "quantized::relu("
                ).run(
                    m.graph
                )

    @skipIfNoFBGEMM
    def test_quantized_add_alpha(self):
        """Test quant fusion for multiple aten::add using same
        constant alpha as the third argument
        """

        # 定义一个名为 test_quantized_add_alpha 的测试方法，用于验证对多个 aten::add 操作进行量化融合，其中第三个参数是相同的常量 alpha
        
        class QuantizedAdd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # 模型的前向传播方法
                x = self.conv1(x)
                y = self.conv2(y)
                z = x + y  # 执行第一个 aten::add 操作
                w = y + z  # 执行第二个 aten::add 操作
                return z + w  # 执行第三个 aten::add 操作

        # 准备测试数据
        data = [
            [
                torch.randn(1, 2, 5, 5, dtype=torch.float),
                torch.randn(1, 2, 5, 5, dtype=torch.float),
            ]
        ]

        # 针对图模式和非图模式进行追踪
        for tracing in [True, False]:
            # 创建 QuantizedAdd 实例，并使用自定义方法 checkGraphModeOp 进行操作检查
            m = self.checkGraphModeOp(QuantizedAdd(), data, "quantized::add", tracing)
            # 使用 FileCheck 检查图中 quantized::add 的数量，确保有三次出现
            FileCheck().check_count("quantized::add", 3, exactly=True).run(m.graph)
            # 使用 FileCheck 检查图中不应出现 aten::add 或 aten::add_ 操作
            FileCheck().check_not("aten::add").check_not("aten::add_").run(m.graph)

    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    # 定义一个测试方法，用于测试量化和非量化加法操作的模型
    def test_quantized_add(self):
        # 定义量化加法模型类
        class QuantizedAdd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建两个二维卷积层，输入通道和输出通道均为2，卷积核大小为2x2
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # 对输入 x 进行第一层卷积操作
                x = self.conv1(x)
                # 对输入 y 进行第二层卷积操作
                y = self.conv2(y)
                # 返回卷积结果的加法运算
                return x + y

        # 定义量化原地加法模型类
        class QuantizedInplaceAdd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建两个二维卷积层，输入通道和输出通道均为2，卷积核大小为2x2
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # 对输入 x 进行第一层卷积操作
                x = self.conv1(x)
                # 对输入 y 进行第二层卷积操作
                y = self.conv2(y)
                # 将第二层卷积结果原地加到第一层卷积结果上
                x += y
                # 返回原地加法后的结果
                return x

        # 定义非量化加法模型类
        class NonQuantizedAdd(torch.nn.Module):
            def forward(self, x, y):
                # 返回输入 x 和 y 的加法结果
                return x + y

        # 定义非量化原地加法模型类
        class NonQuantizedInplaceAdd(torch.nn.Module):
            def forward(self, x, y):
                # 将输入 y 原地加到输入 x 上
                x += y
                # 返回原地加法后的结果
                return x

        # 定义测试数据，包含两个随机张量作为输入
        data = [
            [
                torch.randn(1, 2, 3, 3, dtype=torch.float),
                torch.randn(1, 2, 3, 3, dtype=torch.float),
            ]
        ]
        
        # 遍历不同的模型和量化标志进行测试
        for m, quantized in [
            (QuantizedAdd(), True),                  # 测试量化加法模型
            (QuantizedInplaceAdd(), True),           # 测试量化原地加法模型
            (NonQuantizedAdd(), False),              # 测试非量化加法模型
            (NonQuantizedInplaceAdd(), False),       # 测试非量化原地加法模型
        ]:
            # 遍历不同的跟踪模式进行测试
            for tracing in [True, False]:
                # 根据量化标志选择相应的操作名称
                op = "quantized::add" if quantized else "aten::add"
                # 调用辅助方法检查图模式的操作
                m = self.checkGraphModeOp(m, data, op, tracing)
                # 如果是量化模型，执行以下操作（TODO: 在重构 checkGraphModeOp 后移除）
                if quantized:
                    # 使用 FileCheck 检查图中不包含 aten::add 或 aten::add_ 操作
                    FileCheck().check_not("aten::add").check_not("aten::add_").run(
                        m.graph
                    )
                else:
                    # 使用 FileCheck 检查图中不包含 quantized::add 操作
                    FileCheck().check_not("quantized::add").run(m.graph)
    # 定义一个测试方法，用于测试加量化标量的操作
    def test_quantized_add_scalar(self):
        # 定义一个量化加法操作的模块
        class QuantizedAddScalar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用2x2的卷积层作为成员变量
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 返回加3后的结果
                return x + 3

        # 定义一个原位量化加法操作的模块
        class QuantizedInplaceAddScalar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用2x2的卷积层作为成员变量
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 原位加3
                x += 3
                # 返回结果
                return x

        # 定义一个非量化加法操作的模块
        class NonQuantizedAddScalar(torch.nn.Module):
            def forward(self, x):
                # 返回加3后的结果
                return x + 3

        # 定义一个非量化原位加法操作的模块
        class NonQuantizedInplaceAddScalar(torch.nn.Module):
            def forward(self, x):
                # 原位加3
                x += 3
                # 返回结果
                return x

        # 定义测试数据
        data = [[torch.randn(1, 2, 3, 3, dtype=torch.float)]]
        
        # 针对每个模块和量化标志进行测试
        for m, quantized in [
            (QuantizedAddScalar(), True),
            (QuantizedInplaceAddScalar(), True),
            (NonQuantizedAddScalar(), False),
            (NonQuantizedInplaceAddScalar(), False),
        ]:
            # 针对图模式和非图模式进行测试
            for tracing in [True, False]:
                # 根据量化标志选择操作名称
                op = "quantized::add_scalar" if quantized else "aten::add"
                # 调用测试辅助函数检查图模式操作
                # 不检查数值一致性，因为对 add_scalar 操作不支持
                m = self.checkGraphModeOp(m, data, op, tracing, check=False)
                # TODO: 在 checkGraphModeOp 重构后删除以下代码
                # 如果是量化操作，检查图中不应包含 aten::add 或 aten::add_
                if quantized:
                    FileCheck().check_not("aten::add").check_not("aten::add_").run(
                        m.graph
                    )
                else:
                    # 如果是非量化操作，检查图中不应包含 quantized::add_scalar
                    FileCheck().check_not("quantized::add_scalar").run(m.graph)

    # 使用 FBGEMM 不可用时跳过测试
    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    def test_quantized_cat(self):
        """测试量化的 torch.cat 操作。

        量化的 cat 操作的输出取决于输入，只有在输入被量化时，才会量化其输出。
        """

        class QuantizedCat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                return torch.cat([x, y], 1)

        class NonQuantizedCat(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat([x, y], 1)

        # 准备测试数据
        data = [
            [
                torch.randn(1, 2, 5, 5, dtype=torch.float),
                torch.randn(1, 2, 5, 5, dtype=torch.float),
            ]
        ]

        # 针对跟踪和非跟踪两种模式进行测试
        for tracing in [True, False]:
            # 检查量化的 QuantizedCat 模块的图中是否包含 "quantized::cat"，不应包含 "aten::cat"
            m = self.checkGraphModeOp(QuantizedCat(), data, "quantized::cat", tracing)
            FileCheck().check_not("aten::cat").run(m.graph)

            # 检查非量化的 NonQuantizedCat 模块的图中是否包含 "aten::cat"，不应包含 "quantized::cat"
            m = self.checkGraphModeOp(NonQuantizedCat(), data, "aten::cat", tracing)
            FileCheck().check_not("quantized::cat").run(m.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm(self):
        """测试量化的批量归一化操作。

        针对不同维度的 BatchNorm 模块进行量化，检查图中是否包含 "quantized::batch_norm"，不应包含 "aten::batch_norm"。
        """

        bn_module = {
            1: torch.nn.BatchNorm1d,
            2: torch.nn.BatchNorm2d,
            3: torch.nn.BatchNorm3d,
        }

        class M(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.bn = bn_module[dim](3to(torch.float)

            def forward(self, x):
                return self.bn(x)

        # 针对跟踪和非跟踪两种模式以及不同维度进行测试
        options = itertools.product([True, False], [1, 2, 3])
        for tracing, dim in options:
            model = self.checkGraphModeOp(
                M(dim), self.img_data_dict[dim], "quantized::batch_norm", tracing
            )

            FileCheck().check_not("aten::batch_norm").run(model.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm_relu_BNRelu(self):
        """测试量化的批量归一化和 ReLU 结合的操作。

        针对不同维度的 BNRelu 模块进行量化，检查图中是否包含 "quantized::batch_norm_relu"，不应包含 "aten::batch_norm" 或 "aten::relu"。
        """

        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}

        class BNRelu(torch.nn.Module):
            def __init__(self, dim, inplace):
                super().__init__()
                self.bn = bn_module[dim](3to(torch.float)
                self.relu = torch.nn.ReLU(inplace=inplace)

            def forward(self, x):
                return self.relu(self.bn(x))

        # 针对跟踪和非跟踪两种模式以及不同维度和不同的 inplace 参数进行测试
        options = itertools.product([True, False], [2, 3])
        for tracing, dim in options:
            for instance in [BNRelu(dim, True), BNRelu(dim, False)]:
                # 检查模型图中是否包含 "quantized::batch_norm_relu"，不应包含 "aten::batch_norm" 或 "aten::relu"
                model = self.checkGraphModeOp(
                    instance,
                    self.img_data_dict[dim],
                    "quantized::batch_norm_relu",
                    tracing,
                )
                FileCheck().check_not("aten::batch_norm").check_not(
                    "aten::relu"
                ).check_not("aten::relu_").run(model.graph)
    # 定义一个测试方法，用于测试量化后的批量归一化和ReLU激活功能
    def test_qbatch_norm_relu_BNFuncRelu(self):
        # 定义批量归一化模块的字典，包括二维和三维情况
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}

        # 定义一个自定义的批量归一化函数与ReLU激活结合的类
        class BNFuncRelu(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 根据维度选择对应的批量归一化模块，设置通道数为3，并转换为浮点数
                self.bn = bn_module[dim](3to(torch.float)

            # 前向传播方法，对输入数据进行批量归一化后，应用ReLU激活函数
            def forward(self, x):
                return F.relu(self.bn(x), False)

        # 定义测试选项，包括是否追踪图模式和维度信息
        options = itertools.product([True, False], [2, 3])
        # 遍历所有测试选项
        for tracing, dim in options:
            # 创建BNFuncRelu类的实例
            instance = BNFuncRelu(dim)
            # 调用检查图模式操作的方法，验证模型的行为是否符合预期
            model = self.checkGraphModeOp(
                instance, self.img_data_dict[dim], "quantized::batch_norm_relu", tracing
            )
            # 使用FileCheck检查模型图中不应包含的操作，如标准化和ReLU激活
            FileCheck().check_not("aten::batch_norm").check_not("aten::relu").check_not(
                "aten::relu_"
            ).run(model.graph)

    # 跳过FBGEMM不支持的情况下定义的测试方法
    @skipIfNoFBGEMM
    def test_qbatch_norm_relu_BNFuncInplaceRelu(self):
        # 定义批量归一化模块的字典，包括二维和三维情况
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}

        # 定义一个自定义的批量归一化函数与原地ReLU激活结合的类
        class BNFuncInplaceRelu(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 根据维度选择对应的批量归一化模块，设置通道数为3，并转换为浮点数
                self.bn = bn_module[dim](3to(torch.float)

            # 前向传播方法，对输入数据进行批量归一化后，应用原地ReLU激活函数
            def forward(self, x):
                return F.relu(self.bn(x), True)

        # 定义测试选项，包括是否追踪图模式和维度信息
        options = itertools.product([True, False], [2, 3])
        # 遍历所有测试选项
        for tracing, dim in options:
            # 创建BNFuncInplaceRelu类的实例
            instance = BNFuncInplaceRelu(dim)
            # 调用检查图模式操作的方法，验证模型的行为是否符合预期
            model = self.checkGraphModeOp(
                instance, self.img_data_dict[dim], "quantized::batch_norm_relu", tracing
            )
            # 使用FileCheck检查模型图中不应包含的操作，如标准化和ReLU激活
            FileCheck().check_not("aten::batch_norm").check_not("aten::relu").check_not(
                "aten::relu_"
            ).run(model.graph)
    def test_quantized_mul(self):
        # 定义一个测试函数，用于测试量化乘法操作

        class QuantizedMul(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()  # 创建一个二维卷积层
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()  # 创建另一个二维卷积层

            def forward(self, x, y):
                x = self.conv1(x)  # 对输入 x 进行卷积操作
                y = self.conv2(y)  # 对输入 y 进行卷积操作
                return x * y  # 返回 x 和 y 的乘积

        class QuantizedInplaceMul(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()  # 创建一个二维卷积层
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()  # 创建另一个二维卷积层

            def forward(self, x, y):
                x = self.conv1(x)  # 对输入 x 进行卷积操作
                y = self.conv2(y)  # 对输入 y 进行卷积操作
                x *= y  # 就地修改 x，使其等于 x 和 y 的乘积
                return x

        class NonQuantizedMul(torch.nn.Module):
            def forward(self, x, y):
                return x * y  # 返回 x 和 y 的乘积

        class NonQuantizedInplaceMul(torch.nn.Module):
            def forward(self, x, y):
                x *= y  # 就地修改 x，使其等于 x 和 y 的乘积
                return x

        data = [
            [
                torch.randn(1, 2, 10, 10, dtype=torch.float),  # 创建一个随机张量作为输入 x
                torch.randn(1, 2, 10, 10, dtype=torch.float),  # 创建一个随机张量作为输入 y
            ]
        ]
        for m, quantized in [
            (QuantizedMul(), True),  # 使用量化乘法模型
            (QuantizedInplaceMul(), True),  # 使用就地量化乘法模型
            (NonQuantizedMul(), False),  # 使用非量化乘法模型
            (NonQuantizedInplaceMul(), False),  # 使用就地非量化乘法模型
        ]:
            for tracing in [True, False]:
                op = "quantized::mul" if quantized else "aten::mul"
                # 调用函数检查图模式的操作，并返回修改后的模型 m
                m = self.checkGraphModeOp(m, data, op, tracing)
                # TODO: 在 checkGraphModeOp 重构之后移除该行
                if quantized:
                    FileCheck().check_not("aten::mul").check_not("aten::mul_").run(
                        m.graph
                    )
                else:
                    FileCheck().check_not("quantized::mul").run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul_scalar(self):
        # 定义一个测试函数，用于测试量化乘法操作
        class QuantizedMulScalar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 返回乘以常数3后的结果
                return x * 3

        class QuantizedInplaceMulScalar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 将张量本地地乘以常数3
                x *= 3
                return x

        class NonQuantizedMulScalar(torch.nn.Module):
            def forward(self, x):
                # 返回乘以常数3后的结果
                return x * 3

        class NonQuantizedInplaceMulScalar(torch.nn.Module):
            def forward(self, x):
                # 将张量本地地乘以常数3
                x *= 3
                return x

        # 准备测试数据
        data = [[torch.randn(1, 2, 5, 5, dtype=torch.float)]]
        # 对每个模型和量化标志进行迭代测试
        for m, quantized in [
            (QuantizedMulScalar(), True),
            (QuantizedInplaceMulScalar(), True),
            (NonQuantizedMulScalar(), False),
            (NonQuantizedInplaceMulScalar(), False),
        ]:
            # 对跟踪模式进行迭代测试
            for tracing in [True, False]:
                # 根据量化标志选择操作名称
                op = "quantized::mul_scalar" if quantized else "aten::mul"
                # 调用函数检查图模式操作，不检查数值一致性
                m = self.checkGraphModeOp(m, data, op, tracing, check=False)
                # TODO: 在重构 checkGraphModeOp 后移除此部分
                # 如果是量化操作，则确保图中不包含标准乘法操作
                if quantized:
                    FileCheck().check_not("aten::mul").check_not("aten::mul_").run(
                        m.graph
                    )
                else:
                    # 如果是非量化操作，则确保图中不包含量化乘法操作
                    FileCheck().check_not("quantized::mul_scalar").run(m.graph)

    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    @override_qengines
    def test_hardswish(self):
        # 定义一个测试函数，用于测试 Hardswish 函数
        class FunctionalHardswish(torch.nn.Module):
            def __init__(self, inplace):
                super().__init__()
                self.inplace = inplace

            def forward(self, input):
                # 调用 PyTorch 的 functional.hardswish 函数
                return torch.nn.functional.hardswish(input, inplace=self.inplace)

        # 准备测试模块列表
        modules = [
            torch.nn.Hardswish(),
            FunctionalHardswish(True),
            FunctionalHardswish(False),
        ]

        # 对每种测试用例进行迭代测试
        for test_case in itertools.product([True, False], modules):
            tracing, m = test_case
            # 调用函数检查图模式操作，测试量化 Hardswish 函数
            m = self.checkGraphModeOp(
                m, self.img_data_2d, "quantized::hardswish", tracing
            )
            # 确保图中不包含标准 Hardswish 操作
            FileCheck().check_not("aten::hardswish").check_not("aten::hardswish_").run(
                m.graph
            )

    @override_qengines
    # 定义一个测试函数 test_elu，用于测试 ELU 激活函数的功能
    def test_elu(self):
        # 定义一个内部类 FunctionalELU，继承自 torch.nn.Module，实现 ELU 激活函数的功能
        class FunctionalELU(torch.nn.Module):
            # 初始化函数，设置是否原地操作
            def __init__(self, inplace=False):
                super().__init__()
                self.inplace = inplace

            # 前向传播函数，应用 ELU 激活函数
            def forward(self, input):
                return torch.nn.functional.elu(input, inplace=self.inplace)

        # 模块列表包括 torch.nn.ELU 和 FunctionalELU
        modules = [torch.nn.ELU, FunctionalELU]
        # 使用 itertools.product 生成所有可能的测试用例
        for test_case in itertools.product([True, False], [True, False], modules):
            # 获取测试用例中的跟踪、原地操作标志和模块类别
            tracing, inplace, mod_class = test_case
            # 创建模块对象
            m = mod_class(inplace=inplace)
            # 调用 self.checkGraphModeOp 方法检查图模式操作，传入模块对象、图像数据、操作名称和跟踪标志
            m = self.checkGraphModeOp(m, self.img_data_2d, "quantized::elu", tracing)
            # 使用 FileCheck 检查生成的图中是否不包含标准的 ELU 操作
            FileCheck().check_not("aten::elu").check_not("aten::elu_").run(m.graph)

    # 装饰器，覆盖量化引擎，用于测试层归一化功能
    @override_qengines
    def test_layer_norm(self):
        # 创建包含随机数据的列表
        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float)] for _ in range(2)]
        # 创建 LayerNorm 层对象，指定归一化维度
        layer_norm = torch.nn.LayerNorm([2, 5, 5])
        # 对于每种跟踪模式
        for tracing in [True, False]:
            # 调用 self.checkGraphModeOp 方法检查图模式操作，传入归一化层对象、数据、操作名称和跟踪标志
            m = self.checkGraphModeOp(
                layer_norm, data, "quantized::layer_norm", tracing
            )
            # 使用 FileCheck 检查生成的图中是否不包含标准的 layer_norm 操作
            FileCheck().check_not("aten::layer_norm").run(m.graph)

    # 装饰器，覆盖量化引擎，用于测试分组归一化功能
    @override_qengines
    def test_group_norm(self):
        # 创建包含随机数据的列表
        data = [[torch.rand((1, 4, 5, 5), dtype=torch.float)] for _ in range(2)]
        # 创建 GroupNorm 层对象，指定组数和通道数
        group_norm = torch.nn.GroupNorm(2, 4)
        # 对于每种跟踪模式
        for tracing in [True, False]:
            # 调用 self.checkGraphModeOp 方法检查图模式操作，传入分组归一化层对象、数据、操作名称和跟踪标志
            m = self.checkGraphModeOp(
                group_norm, data, "quantized::group_norm", tracing
            )
            # 使用 FileCheck 检查生成的图中是否不包含标准的 group_norm 操作
            FileCheck().check_not("aten::group_norm").run(m.graph)

    # 装饰器，覆盖量化引擎，用于测试实例归一化功能
    @override_qengines
    def test_instance_norm(self):
        # 创建不同维度下的随机数据列表
        data_1d = [[torch.rand((1, 4, 5), dtype=torch.float)] for _ in range(2)]
        data_2d = [[torch.rand((1, 4, 5, 1), dtype=torch.float)] for _ in range(2)]
        data_3d = [[torch.rand((1, 4, 5, 1, 1), dtype=torch.float)] for _ in range(2)]
        # 组合不同维度的数据
        data = {1: data_1d, 2: data_2d, 3: data_3d}
        # 实例归一化模块字典，包含不同维度的实例归一化类
        instance_norm_modules = {
            1: torch.nn.InstanceNorm1d,
            2: torch.nn.InstanceNorm2d,
            3: torch.nn.InstanceNorm3d,
        }

        # 针对每种维度和跟踪模式的组合，生成测试选项
        options = itertools.product([1, 2, 3], [True, False])
        for dim, tracing in options:
            # 创建相应维度的实例归一化对象
            instance_norm = instance_norm_modules[dim](4            # 调用 self.checkGraphModeOp 方法检查图模式操作，传入实例归一化对象、数据、操作名称和跟踪标志
            m = self.checkGraphModeOp(
                instance_norm, data[dim], "quantized::instance_norm", tracing
            )
            # 使用 FileCheck 检查生成的图中是否不包含标准的 instance_norm 操作
            FileCheck().check_not("aten::instance_norm").run(m.graph)

    # 装饰器，如果没有 FBGEMM，则跳过测试
    @skipIfNoFBGEMM
    def test_dequantize_tuple(self):
        """Make sure dequantize can support Tuple of tensor"""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()  # 定义第一个卷积层
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()  # 定义第二个卷积层

            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                x1 = self.conv1(x)  # 执行第一个卷积操作
                x2 = self.conv2(x)  # 执行第二个卷积操作
                return x1, x2  # 返回两个卷积操作的结果作为元组

        for tracing in [True, False]:
            self.checkGraphModeOp(M(), self.img_data_2d, "quantized::conv2d", tracing)  # 检查图模式操作

    @skipIfNoFBGEMM
    def test_clamp(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()  # 定义卷积层
                self.relu6 = torch.nn.ReLU6()  # 定义 ReLU6 激活函数
                self.relu6_ = torch.nn.ReLU6(True)  # 定义带 inplace 参数的 ReLU6 激活函数
                self.hardtanh = torch.nn.Hardtanh()  # 定义 Hardtanh 激活函数
                self.hardtanh_ = torch.nn.Hardtanh(inplace=True)  # 定义带 inplace 参数的 Hardtanh 激活函数

            def forward(self, x):
                x = self.conv(x)  # 执行卷积操作
                x = self.relu6(x)  # 执行 ReLU6 激活操作
                self.relu6_(x)  # 执行带 inplace 参数的 ReLU6 激活操作
                x = F.relu6(x)  # 使用函数库 F 中的 ReLU6 激活函数
                x = torch.clamp(x, -3, 3)  # 对张量 x 进行范围截断操作
                x = x.clamp(-2.5, 2.5)  # 对张量 x 进行范围截断操作
                # x = x.clamp_(-2, 2)  # 当量化后的 `clamp_` 操作准备就绪时启用
                x = self.hardtanh(x)  # 执行 Hardtanh 激活操作
                self.hardtanh_(x)  # 执行带 inplace 参数的 Hardtanh 激活操作
                x = F.hardtanh(x)  # 使用函数库 F 中的 Hardtanh 激活函数
                F.hardtanh_(x)  # 使用函数库 F 中的带 inplace 参数的 Hardtanh 激活函数
                return x  # 返回处理后的张量 x

        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float)]]  # 定义测试数据
        options = itertools.product(
            ["aten::clamp", "aten::hardtanh", "aten::hardtanh_"], [True, False]
        )  # 创建所有操作和追踪选项的组合
        for op, tracing in options:
            m = self.checkGraphModeOp(M(), data, op, tracing)  # 检查图模式下的操作
            FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True).run(
                m.graph
            )  # 检查图中量化操作的数量

            FileCheck().check_count("aten::dequantize", 1, exactly=True).run(m.graph)  # 检查图中反量化操作的数量

    @override_qengines
    def test_conv_with_benchmark_flag(self):
        r"""Verifies that convolutions get quantized when
        torch.backends.cudnn.benchmark is enabled
        """
        if not qengine_is_qnnpack():  # 如果不是 QNNPACK 引擎则跳过测试
            return
        with torch.backends.cudnn.flags(enabled=True):  # 启用 cuDNN 标志
            m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))  # 创建包含单个卷积层的模型
            m.eval()  # 设置模型为评估模式
            m = torch.jit.trace(m, torch.rand(4, 1, 4, 4))  # 对模型进行追踪
            qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")  # 获取 QNNPACK 的默认量化配置
            prepared_model = torch.ao.quantization.prepare_jit(m, {"": qconfig})  # 准备 JIT 模型进行量化
            prepared_model(torch.rand(4, 1, 4, 4))  # 对模型进行前向传播
            converted_model = torch.ao.quantization.convert_jit(prepared_model)  # 将 JIT 模型转换为量化模型
            FileCheck().check("quantized::conv2d").run(converted_model.graph)  # 检查转换后的模型图中的量化卷积操作

    @skipIfNoFBGEMM
    # 定义一个测试方法，测试线性模型的功能
    def test_cat_linear(self):
        # 定义一个继承自torch.nn.Module的线性模型类
        class LinearModel(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 初始化权重为一个5x5的随机张量
                self.weight = torch.randn(5, 5)

            # 前向传播方法
            def forward(self, x, y):
                # 将输入张量x和y连接起来，形成一个新的张量a
                a = torch.cat([x, y])
                # 对连接后的张量a进行线性变换，使用预先定义的权重self.weight，得到张量b
                b = F.linear(a, self.weight)
                # 再次对张量b进行线性变换，使用相同的权重self.weight，得到张量c
                c = F.linear(b, self.weight)
                # 返回两个线性变换后的张量b和c
                return b, c

        # 创建一个LinearModel类的实例，并设为评估模式
        model = LinearModel().eval()
        # 定义量化配置，使用默认的量化配置
        qconfig = {"": default_qconfig}
        # 将模型转换为脚本模型（JIT编译）
        float_model = torch.jit.script(model)
        # 准备JIT编译后的模型，应用量化配置
        prepared_model = prepare_jit(float_model, qconfig)
        # 在随机生成的5x5张量上运行准备好的模型
        prepared_model(torch.rand(5, 5), torch.rand(5, 5))
        # 将准备好的模型转换为量化后的模型
        converted_model = convert_jit(prepared_model)
        # 使用FileCheck检查转换后的模型的图中是否包含"quantized::linear"两次出现的情况
        FileCheck().check("quantized::linear").check("quantized::linear").run(
            converted_model.graph
        )
class TestQuantizeDynamicJitPasses(QuantizationTestCase):
    def test_prepare_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(x)

        # 将模型转换为 TorchScript，以便进行动态量化准备
        model = torch.jit.script(M())
        
        # 对于每个量化配置进行准备动态量化操作
        for qconfig in [float16_dynamic_qconfig, default_dynamic_qconfig]:
            m = prepare_dynamic_jit(model, {"": qconfig})

            # 检查权重的观察器是否添加
            assert len(attrs_with_prefix(m.fc, "_observer_")) == 1

            if qconfig == float16_dynamic_qconfig:
                # 针对 float16 动态量化配置，检查权重观察器是否正确添加
                observer_name = 'PlaceholderObserver = prim::GetAttr[name="_observer_'
                FileCheck().check(observer_name).run(m.fc.graph)
            else:
                # 对于其他动态量化配置，检查整个模型的观察器是否正确添加
                assert len(attrs_with_prefix(m, "_observer_")) == 1
                observer_name = 'Observer = prim::GetAttr[name="_observer_'
                FileCheck().check(observer_name).check(
                    'prim::GetAttr[name="fc"]'
                ).check("prim::CallMethod").check_not(observer_name).run(m.graph)

    def test_prepare_dynamic_child_qconfig(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(x)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                return self.sub(self.conv(x))

        # 将模型转换为 TorchScript，以便进行动态量化准备
        m = torch.jit.script(M())
        
        # 仅对子模块进行动态量化准备
        m = prepare_dynamic_jit(m, {"sub.fc": default_dynamic_qconfig})

        # 检查子模块的输入是否添加了观察器
        assert len(attrs_with_prefix(m, "_observer_")) == 1
        # 检查卷积层是否未被量化
        assert len(attrs_with_prefix(m.conv, "_observer_")) == 0
        # 由于在最外层调用处观察，检查子模块是否未添加观察器
        assert len(attrs_with_prefix(m.sub, "_observer_")) == 0
        # 检查线性层的权重观察器是否正确添加
        assert len(attrs_with_prefix(m.sub.fc, "_observer_")) == 1
        FileCheck().check('prim::GetAttr[name="sub').check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("prim::CallMethod").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(
            m.graph
        )
    @override_qengines
    # 使用装饰器覆盖量化引擎的设置
    def test_dynamic_multi_op(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个输入输出维度为 (5, 5) 的线性层，并将其转换为 float 类型
                self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

            def forward(self, x):
                # 对输入 x 进行加法操作
                x = x + 5
                # 返回经过线性层 fc1 处理后的结果
                return self.fc1(x)

        # 创建一个形状为 (5, 5) 的随机输入张量 x
        x = torch.randn(5, 5)
        # 遍历两种追踪模式：跟踪和非跟踪
        for tracing in [True, False]:
            # 调用 checkGraphModeOp 方法，验证模型 M 的图模式操作
            model = self.checkGraphModeOp(
                M(), x, "quantized::linear_dynamic", tracing=tracing, dynamic=True
            )
            # 断言 add 操作不是动态量化的
            FileCheck().check("aten::add").run(model.graph)
    @override_qengines
    def test_dynamic_shared_weights(self):
        # 定义一个包含参数的自定义模块
        class myMod(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                # 创建一个线性层并设置权重
                self.linear = nn.Linear(5, 5)
                self.linear.weight = weight

            def forward(self, x):
                # 在前向传播中使用线性层
                return self.linear(x)

        # 定义一个动态模型
        class DynamicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个包含参数的张量
                self.weight = torch.nn.Parameter(torch.ones(5, 5))
                # 使用自定义模块myMod，并传入参数张量
                self.mod1 = myMod(self.weight)

            def forward(self, x):
                # 在前向传播中使用自定义模块mod1
                y = self.mod1(x)
                # 使用动态量化操作，调用torch.nn.functional.linear函数
                z = torch.nn.functional.linear(y, self.weight)
                return z

        # 使用Torch的脚本化机制将DynamicModel转换为脚本模式，并设置为eval模式
        model = torch.jit.script(DynamicModel()).eval()
        # 生成一个随机的5x5浮点数张量作为输入数据
        data = torch.randn(5, 5, dtype=torch.float)
        # 定义要量化的操作和计数
        quant_ops = ["mod1", ""]
        counts = [1, 2]
        # 遍历操作和计数，对模型进行动态量化
        for op, count in zip(quant_ops, counts):
            # 设置量化配置字典，使用默认的动态量化配置
            qconfig_dict = {op: default_dynamic_qconfig}
            # 对模型进行动态量化
            m1 = quantize_dynamic_jit(model, qconfig_dict)
            # 在量化后的图中运行数据，检查量化线性动态操作的数量是否符合预期，同时不包含某些特定操作
            FileCheck().check_count(
                "quantized::linear_dynamic(", count, exactly=True
            ).check_not("aten::_choose_qparams_per_tensor").run(m1.graph)

            # 在转换前显式调用模型的前向传播
            m2 = prepare_dynamic_jit(model, qconfig_dict)
            m2(data)
            # 将准备好的动态模型转换为脚本模式，关闭调试模式
            m2 = convert_dynamic_jit(m2, debug=False)
            # 在转换后的模型上运行数据，比较量化前后的输出结果是否一致
            out_ref = m2(data)
            self.assertEqual(out_graph, out_ref)

    @override_qengines
    def test_dynamic_with_if(self):
        # 定义一个继承自torch.nn.Module的类Res，表示一个简单的神经网络模块
        class Res(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(5, 5))  # 初始化权重矩阵为5x5的全1张量

            def forward(self, x: torch.Tensor, cond: bool) -> torch.Tensor:
                # 根据条件cond选择不同的线性函数进行前向传播计算
                if cond:
                    return torch.nn.functional.linear(x, self.weight)
                else:
                    return torch.nn.functional.linear(x, self.weight)

        # 定义一个继承自torch.nn.Module的类M，包含两个Res实例作为其属性
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.res1 = Res()  # 初始化第一个Res实例
                self.res2 = Res()  # 初始化第二个Res实例

            def forward(self, x):
                x = self.res1(x, True)   # 使用第一个Res实例进行前向传播计算
                x = self.res2(x, False)  # 使用第二个Res实例进行前向传播计算
                return x

        model = torch.jit.script(M()).eval()  # 将M模型转换为torchscript并设为评估模式
        data = torch.randn(5, 5, dtype=torch.float)  # 创建一个5x5的随机张量
        qconfig_dict = {"": default_dynamic_qconfig}  # 定义一个量化配置字典

        # 遍历两次，一次跟踪计算图，一次不跟踪计算图
        for tracing in [True, False]:
            # 检查并返回模型的操作模式，检查是否为动态量化线性操作
            m1 = self.checkGraphModeOp(
                M(), data, "quantized::linear_dynamic", tracing=tracing, dynamic=True
            )
            # 使用FileCheck检查m1的计算图确保正确运行，并检查是否有两个quantized::linear_dynamic操作
            FileCheck().check_count(
                "quantized::linear_dynamic(", 2, exactly=True
            ).check_not("aten::_choose_qparams_per_tensor").run(m1.graph)

        # 检查权重观察器是否正确运行，并将参考的量化参数存入ref_qparams列表中
        ref_qparams = []
        qconfig = script_qconfig(default_dynamic_qconfig)
        wt_module = wrap_cpp_module(qconfig.weight)
        for wt in [model.res1.weight, model.res2.weight]:
            wt_module(wt)  # 将权重数据传递给权重模块
            qparams = wt_module.calculate_qparams()  # 计算权重的量化参数
            ref_qparams.append((qparams[0].item(), qparams[1].item()))  # 将量化参数添加到ref_qparams列表中

        # 对模型进行动态量化，并启用调试模式
        m2 = quantize_dynamic_jit(model, qconfig_dict, debug=True)
        graph_params = []
        # 遍历模型的所有模块和观察器，获取量化后的参数并存入graph_params列表中
        for x, obs in m2._modules._c.items():
            if x == "res1":
                graph_params.append(
                    (
                        obs.getattr("weight.2_scale_0"),
                        obs.getattr("weight.2_zero_point_0"),
                    )
                )
            elif x == "res2":
                graph_params.append(
                    (
                        obs.getattr("weight.4_scale_0"),
                        obs.getattr("weight.4_zero_point_0"),
                    )
                )
        
        # 使用断言检查量化后的参数graph_params与参考的ref_qparams是否一致
        self.assertEqual(ref_qparams, graph_params)
    def test_dynamic_weight_observer(self):
        # 定义一个简单的神经网络模型 M，包含两个线性层
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()
                self.fc2 = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                # 前向传播函数，先通过 self.fc 进行线性变换，然后通过 self.fc2 进行线性变换
                x = self.fc(x)
                return self.fc2(x)

        # 定义量化配置字典，使用默认的动态量化配置
        qconfig_dict = {"": default_dynamic_qconfig}
        # 创建一个评估模式下的实例化模型 eager_model
        eager_model = M().eval()
        # 对于每种追踪情况（tracing=True 和 tracing=False）
        for tracing in [True, False]:
            # 生成一个随机输入 x
            x = torch.rand(5, 5)
            # 获取脚本模块化的模型，根据追踪情况选择是否追踪 x
            model = get_script_module(eager_model, tracing, x)
            ref_qparams = []
            # 遍历模型的权重列表，计算参考量化参数 ref_qparams
            for wt in [model.fc.weight, model.fc2.weight]:
                wt_module = default_dynamic_qconfig.weight()
                wt_module(wt)
                qparams = wt_module.calculate_qparams()
                ref_qparams.append((qparams[0].item(), qparams[1].item()))
            # 对模型进行动态量化
            model = quantize_dynamic_jit(model, qconfig_dict, debug=True)
            graph_qparams = []
            # 遍历模型的模块，获取量化后的参数 graph_qparams
            for x, obs in model._modules._c.items():
                n = 2 if x == "fc" and tracing else 1
                graph_qparams.append(
                    (
                        obs.getattr(f"weight.{n}_scale_0"),
                        obs.getattr(f"weight.{n}_zero_point_0"),
                    )
                )
            # 断言参考量化参数与图中量化参数是否相等
            self.assertEqual(ref_qparams, graph_qparams)

    def test_convert_dynamic_fp16(self):
        # 定义一个简单的神经网络模型 M，包含一个线性层
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                # 前向传播函数，对输入 x 执行线性变换
                return self.fc(x)

        # 使用 TorchScript 对模型 M 进行脚本化
        m = torch.jit.script(M())
        # 对模型进行动态 FP16 量化
        m = quantize_dynamic_jit(m, {"": float16_dynamic_qconfig}, debug=True)
        # 使用 FileCheck 验证图中是否包含特定的操作顺序和缺失的操作
        FileCheck().check("aten::_saturate_weight_to_fp16").check(
            "aten::linear"
        ).check_not("aten::dequantize").check_not("aten::quantize").run(m.graph)

    def test_quantize_dynamic_fp16(self):
        # 定义一个简单的神经网络模型 M，包含一个线性层
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                # 前向传播函数，对输入 x 执行线性变换
                return self.fc(x)

        # 使用 TorchScript 对模型 M 进行脚本化
        m = torch.jit.script(M())
        # 对模型进行动态 FP16 量化
        m = quantize_dynamic_jit(m, {"": float16_dynamic_qconfig})

        # 使用 FileCheck 验证图中是否包含特定的操作顺序和缺失的操作
        FileCheck().check("quantized::linear_dynamic_fp16").check_not(
            "aten::linear"
        ).check_not("aten::dequantize").check_not("aten::quantize").run(m.graph)
    class TestQuantizeDynamicJitOps(QuantizationTestCase):
        """Test graph mode post training dynamic quantization works
        for individual ops end to end.
        """

        @override_qengines
        def test_linear(self):
            # 定义一个简单的线性模型
            class FunctionalLinear(torch.nn.Module):
                def __init__(self, weight, bias):
                    super().__init__()
                    self.weight = weight  # 初始化权重
                    self.bias = bias  # 初始化偏置

                def forward(self, x):
                    # 调用 torch.nn.functional 的线性函数进行前向传播
                    return F.linear(x, self.weight, self.bias)

            # 创建一个随机张量作为输入数据
            x = torch.rand(5, 5)
            # 循环进行跟踪和非跟踪两种情况的测试
            for tracing in [True, False]:
                # 检查图模式操作，期望得到动态量化后的线性操作
                model = self.checkGraphModeOp(
                    torch.nn.Linear(5, 5),  # 创建一个标准的线性层
                    x,
                    "quantized::linear_dynamic",  # 期望的量化后操作的名称
                    tracing=tracing,  # 是否进行跟踪
                    dynamic=True,  # 动态量化模式
                )

            # 创建一个随机权重张量
            weight = torch.rand(5, 5)
            b = torch.rand(5)
            # 使用 itertools 生成跟踪和是否有偏置的所有组合
            for tracing, has_bias in itertools.product([True, False], [True, False]):
                bias = b if has_bias else None  # 如果有偏置则设置偏置，否则为 None
                # 检查自定义的 FunctionalLinear 模型的图模式操作
                model = self.checkGraphModeOp(
                    FunctionalLinear(weight, bias),  # 创建自定义的线性模型实例
                    x,
                    "quantized::linear_dynamic",  # 期望的量化后操作的名称
                    tracing=tracing,  # 是否进行跟踪
                    dynamic=True,  # 动态量化模式
                )

        @skipIfNoFBGEMM
        # 确保在尝试量化 EmbeddingBag 时，如果 padding_idx 不为 None，会抛出错误
        @skipIfNoFBGEMM
    def test_embedding_bag_padding_idx_error(self):
        # 定义一个测试函数，用于测试 embedding_bag 的 padding_idx 参数错误情况
        class M(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                # 初始化模型，包括一个 EmbeddingBag 层
                self.embedding = torch.nn.EmbeddingBag(
                    num_embeddings=10,            # 嵌入的数量
                    embedding_dim=12,             # 嵌入的维度
                    include_last_offset=True,      # 是否包括最后一个偏移量
                    sparse=True,                  # 是否使用稀疏张量
                    _weight=weights,              # 嵌入的权重
                    mode="sum",                   # 汇总方式
                    padding_idx=0,                # 填充索引
                )

            def forward(self, indices, offsets):
                # 前向传播函数，计算嵌入向量
                e = self.embedding(indices, offsets)
                return e

        # 创建模型实例
        weights = torch.randn(10, 12, dtype=torch.float32)
        module = M(weights)

        # 定义输入数据
        indices = torch.tensor([0, 1, 2, 3, 4])   # 索引张量
        offsets = torch.tensor([0, 2, 5])         # 偏移量张量
        dummy_inputs = (indices, offsets)

        # 定义量化配置
        int4_qconfig = QConfig(
            activation=PlaceholderObserver.with_args(
                dtype=torch.float, custom_op_name="embedding_bag_4bit"
            ),
            weight=PlaceholderObserver.with_args(custom_op_name="embedding_bag_4bit"),
        )
        int8_qconfig = QConfig(
            activation=PlaceholderObserver.with_args(
                dtype=torch.float, custom_op_name="embedding_bag_byte"
            ),
            weight=PlaceholderObserver.with_args(custom_op_name="embedding_bag_byte"),
        )

        # 错误消息字符串
        error_msg = r"Expected aten::embedding_bag padding_idx input to be None"

        # 遍历追踪和配置的组合，进行量化测试
        for trace, qconfig in itertools.product(
            [True, False], [int4_qconfig, int8_qconfig]
        ):
            if trace:
                # 如果进行追踪，则使用 torch.jit.trace 追踪模型
                m = torch.jit.trace(module, dummy_inputs)
            else:
                # 否则，使用 torch.jit.script 进行脚本化
                m = torch.jit.script(module)
            # 准备模型以进行 JIT 量化
            m = prepare_jit(m, {"embedding": qconfig})
            # 断言期望的运行时错误消息
            with self.assertRaisesRegex(RuntimeError, error_msg):
                m = convert_jit(m)
# 定义一个测试类 TestQuantizeJit，继承自 QuantizationTestCase，用于测试量化功能
class TestQuantizeJit(QuantizationTestCase):
    
    # 装饰器：重写量化引擎，应用于下面的测试方法
    @override_qengines
    def test_single_linear(self):
        r"""Compare the result of quantizing single linear layer in
        eager mode and graph mode
        """
        # 在 eager 模式下量化线性层的结果，并与图模式进行比较
        # 创建带注释的单层线性模型，使用 torch.backends.quantized.engine 引擎，设为评估模式
        annotated_linear_model = AnnotatedSingleLayerLinearModel(
            torch.backends.quantized.engine
        ).eval()
        
        # 创建未注释的单层线性模型，设为评估模式
        linear_model = SingleLayerLinearModel().eval()
        
        # 从 eager 模式复制权重，以便稍后比较两个量化模型的结果
        linear_model.fc1.weight = torch.nn.Parameter(
            annotated_linear_model.fc1.module.weight.detach()
        )
        linear_model.fc1.bias = torch.nn.Parameter(
            annotated_linear_model.fc1.module.bias.detach()
        )
        
        # 使用 quantize 函数对带注释的线性模型进行量化，只测试评估函数，传入 self.calib_data 作为参数
        model_eager = quantize(
            annotated_linear_model, test_only_eval_fn, [self.calib_data]
        )

        # 创建量化配置字典，使用 torch.backends.quantized.engine 的默认配置
        qconfig_dict = {"": get_default_qconfig(torch.backends.quantized.engine)}
        
        # 使用 torch.jit.trace 对 linear_model 进行跟踪，使用 self.calib_data[0][0] 作为输入
        model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
        
        # 使用 torch.jit.script 对 linear_model 进行脚本化
        model_script = torch.jit.script(linear_model)
        
        # 在 model_eager 上进行推断，得到结果 result_eager
        result_eager = model_eager(self.calib_data[0][0])
        
        # 遍历需要测试的模型列表，对每个模型进行量化
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(
                model_under_test,
                qconfig_dict,
                test_only_eval_fn,
                [self.calib_data],
                inplace=False,
            )
            # 断言量化后的模型在给定输入上的输出与 eager 模式的结果一致
            self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    # 如果没有 FBGEMM，跳过此测试
    @skipIfNoFBGEMM
    def test_observer_with_ignored_function(self):
        r"""Test observers with ignored function and make sure it works in
        graph mode
        """
        # 在 eager 模式下创建被注解的单层线性模型，并设置为评估模式
        annotated_linear_model = AnnotatedSingleLayerLinearModel("fbgemm").eval()
        # 遍历不同的 QConfig 对象
        for qconfig in [
            QConfig(activation=default_observer, weight=default_weight_observer),
            QConfig(
                activation=default_histogram_observer, weight=default_weight_observer
            ),
            QConfig(
                activation=default_observer, weight=default_per_channel_weight_observer
            ),
        ]:
            # 将当前 qconfig 应用到注解后的线性模型中
            annotated_linear_model.qconfig = qconfig
            # 创建单层线性模型并设置为评估模式
            linear_model = SingleLayerLinearModel().eval()
            # 从 eager 模式中复制权重以便稍后比较两个量化模型的结果
            linear_model.fc1.weight = torch.nn.Parameter(
                annotated_linear_model.fc1.module.weight.detach()
            )
            linear_model.fc1.bias = torch.nn.Parameter(
                annotated_linear_model.fc1.module.bias.detach()
            )
            # 在 eager 模式下量化注解后的线性模型
            model_eager = quantize(
                annotated_linear_model, test_only_eval_fn, [self.calib_data]
            )

            # 创建 qconfig 字典以便将其应用到模型中
            qconfig_dict = {"": qconfig}
            # 对线性模型进行追踪以便量化
            model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
            # 对线性模型进行脚本化以便量化
            model_script = torch.jit.script(linear_model)
            # 在 eager 模式下得到量化模型的结果
            result_eager = model_eager(self.calib_data[0][0])
            # 遍历需要测试的模型（追踪模型和脚本化模型）
            for model_under_test in [model_traced, model_script]:
                # 对模型进行即时量化
                model_quantized = quantize_jit(
                    model_under_test,
                    qconfig_dict,
                    test_only_eval_fn,
                    [self.calib_data],
                    inplace=False,
                )
                # 断言量化后模型的结果与 eager 模式下量化的结果相等
                self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @override_qengines
    def test_conv(self):
        r"""Compare the result of quantizing conv layer in
        eager mode and graph mode
        """
        # eager mode
        # 创建并评估一个被注释的卷积模型对象
        annotated_conv_model = AnnotatedConvModel(
            torch.backends.quantized.engine
        ).eval()
        # 创建并评估一个普通的卷积模型对象
        conv_model = ConvModel().eval()
        # 从 eager 模式中复制权重，以便稍后比较两个量化模型的结果
        conv_model.conv.weight = torch.nn.Parameter(
            annotated_conv_model.conv.weight.detach()
        )
        # 在 eager 模式下对注释的卷积模型进行量化
        model_eager = quantize(
            annotated_conv_model, test_only_eval_fn, [self.img_data_2d]
        )
        # 获取默认的量化配置信息
        qconfig_dict = {"": get_default_qconfig(torch.backends.quantized.engine)}
        # 对卷积模型进行追踪（tracing）
        model_traced = torch.jit.trace(conv_model, self.img_data_2d[0][0])
        # 对卷积模型进行脚本化（scripting）
        model_script = torch.jit.script(conv_model)
        # 计算在 eager 模式下模型的结果
        result_eager = model_eager(self.img_data_2d[0][0])
        # 对于追踪和脚本化的模型，进行 JIT 量化
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(
                model_under_test,
                qconfig_dict,
                test_only_eval_fn,
                [self.img_data_2d],
                inplace=False,
            )
            # 断言 JIT 量化模型的输出与 eager 模式下的结果相等
            self.assertEqual(model_quantized(self.img_data_2d[0][0]), result_eager)

    @override_qengines
    def test_conv_transpose(self):
        r"""Compare the result of quantizing conv_transpose layer in
        eager mode and graph mode
        """
        if not qengine_is_qnnpack():
            return  # Currently only qnnpack is supported
        # eager mode
        # 创建并评估一个被注释的转置卷积模型对象
        annotated_conv_model = AnnotatedConvTransposeModel(
            torch.backends.quantized.engine
        ).eval()
        # 创建并评估一个转置卷积模型对象
        conv_model = ConvTransposeModel().eval()
        # 从 eager 模式中复制权重，以便稍后比较两个量化模型的结果
        conv_model.conv.weight = torch.nn.Parameter(
            annotated_conv_model.conv.weight.detach()
        )
        # 在 eager 模式下对注释的转置卷积模型进行量化
        model_eager = quantize(
            annotated_conv_model, test_only_eval_fn, [self.img_data_2d]
        )
        # 获取默认的量化配置信息
        qconfig_dict = {"": get_default_qconfig(torch.backends.quantized.engine)}
        # 对转置卷积模型进行追踪（tracing）
        model_traced = torch.jit.trace(conv_model, self.img_data_2d[0][0])
        # 对转置卷积模型进行脚本化（scripting）
        model_script = torch.jit.script(conv_model)
        # 计算在 eager 模式下模型的结果
        result_eager = model_eager(self.img_data_2d[0][0])
        # 对于追踪和脚本化的模型，进行 JIT 量化
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(
                model_under_test,
                qconfig_dict,
                test_only_eval_fn,
                [self.img_data_2d],
                inplace=False,
            )
            # 断言 JIT 量化模型的输出与 eager 模式下的结果相等
            self.assertEqual(model_quantized(self.img_data_2d[0][0]), result_eager)

    @override_qengines
    def test_conv_bn(self):
        r"""Compare the result of quantizing conv + bn layer in
        eager mode and graph mode
        """
        # 在 eager 模式下
        conv_model = AnnotatedConvBnModel().eval()
        # 创建一个与 conv_model 共享权重的脚本化模型以便后续比较量化模型的结果
        conv_model_to_script = ConvBnModel().eval()
        conv_model_to_script.conv.weight = torch.nn.Parameter(
            conv_model.conv.weight.detach()
        )
        # 将 conv + bn 层融合成一个单独的量化模块，替换原模型中的 conv + bn
        fuse_modules(conv_model, ["conv", "bn"], inplace=True)
        # 对 eager 模式下的模型进行量化
        model_eager = quantize(conv_model, test_only_eval_fn, [self.img_data_2d])
        qconfig_dict = {"": default_qconfig}
        # 对脚本化模型进行 JIT 编译并量化
        model_script = quantize_jit(
            torch.jit.script(conv_model_to_script),
            qconfig_dict,
            test_only_eval_fn,
            [self.img_data_2d],
            inplace=False,
        )
        # 在输入数据上分别评估两个量化模型并比较结果
        result_eager = model_eager(self.img_data_2d[0][0])
        result_script = model_script(self.img_data_2d[0][0])
        # 断言两个模型的输出结果应该相等
        self.assertEqual(result_eager, result_script)

    @override_qengines
    def test_nested(self):
        # Eager mode
        # 创建一个 eager_model 对象，使用 AnnotatedNestedModel 类，并设置为评估模式
        eager_model = AnnotatedNestedModel(torch.backends.quantized.engine).eval()

        # Graph mode
        # 创建一个 script_model 对象，使用 NestedModel 类，并设置为评估模式
        script_model = NestedModel().eval()

        # Copy weights for eager_model
        # 复制 eager_model 的权重到 script_model 的相应层
        script_model.sub1.fc.weight = torch.nn.Parameter(
            eager_model.sub1.fc.weight.detach()
        )
        script_model.sub1.fc.bias = torch.nn.Parameter(
            eager_model.sub1.fc.bias.detach()
        )
        script_model.sub2.fc1.weight = torch.nn.Parameter(
            eager_model.sub2.fc1.module.weight.detach()
        )
        script_model.sub2.fc1.bias = torch.nn.Parameter(
            eager_model.sub2.fc1.module.bias.detach()
        )
        script_model.sub2.fc2.weight = torch.nn.Parameter(
            eager_model.sub2.fc2.weight.detach()
        )
        script_model.sub2.fc2.bias = torch.nn.Parameter(
            eager_model.sub2.fc2.bias.detach()
        )
        script_model.fc3.weight = torch.nn.Parameter(
            eager_model.fc3.module.weight.detach()
        )
        script_model.fc3.bias = torch.nn.Parameter(eager_model.fc3.module.bias.detach())

        # 对 eager_model 进行量化
        model_eager = quantize(eager_model, test_only_eval_fn, [self.calib_data])

        # 设置量化配置字典
        qconfig_dict = {
            "sub2.fc1": default_per_channel_qconfig
            if qengine_is_fbgemm()
            else default_qconfig,
            "fc3": default_qconfig,
        }

        # 对 script_model 进行跟踪量化
        model_traced = torch.jit.trace(script_model, self.calib_data[0][0])

        # 对 script_model 进行脚本量化
        model_script = torch.jit.script(script_model)

        # 对 model_eager 进行评估
        result_eager = model_eager(self.calib_data[0][0])

        # 遍历需要测试的模型列表，对每个模型进行量化，并进行断言测试
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(
                model_under_test,
                qconfig_dict,
                test_only_eval_fn,
                [self.calib_data],
                inplace=False,
            )
            self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @override_qengines
    def test_skip_quant(self):
        """Test None qconfig"""
        # 创建一个处于 eager 模式下的 AnnotatedSkipQuantModel 对象，并设置为评估模式
        eager_model = AnnotatedSkipQuantModel(torch.backends.quantized.engine).eval()

        # 创建一个处于 graph 模式下的 SkipQuantModel 对象，并设置为评估模式
        script_model = SkipQuantModel().eval()
        
        # 为 script_model 复制 eager_model 的权重
        script_model.sub.fc1.weight = torch.nn.Parameter(
            eager_model.sub.module.fc1.weight.detach()
        )
        script_model.sub.fc1.bias = torch.nn.Parameter(
            eager_model.sub.module.fc1.bias.detach()
        )
        script_model.sub.fc2.weight = torch.nn.Parameter(
            eager_model.sub.module.fc2.weight.detach()
        )
        script_model.sub.fc2.bias = torch.nn.Parameter(
            eager_model.sub.module.fc2.bias.detach()
        )
        script_model.fc.weight = torch.nn.Parameter(eager_model.fc.weight.detach())
        script_model.fc.bias = torch.nn.Parameter(eager_model.fc.bias.detach())

        # 对 eager_model 进行模块融合优化
        eager_model.fuse_modules()

        # 对 eager_model 进行量化，使用 test_only_eval_fn 函数进行评估，并传入 self.calib_data 作为校准数据
        model_eager = quantize(eager_model, test_only_eval_fn, [self.calib_data])

        # 定义量化配置字典 qconfig_dict
        qconfig_dict = {
            "": get_default_qconfig(torch.backends.quantized.engine),
            "fc": None,
        }

        # 对 script_model 进行 TorchScript 追踪，使用 self.calib_data[0][0] 作为输入
        model_traced = torch.jit.trace(script_model, self.calib_data[0][0])

        # 对 script_model 进行 TorchScript 脚本化
        model_script = torch.jit.script(script_model)

        # 使用 model_eager 对 self.calib_data[0][0] 进行推理，得到结果 result_eager
        result_eager = model_eager(self.calib_data[0][0])

        # 遍历模型列表 [model_traced, model_script]，对每个模型进行量化
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(
                model_under_test,
                qconfig_dict,
                test_only_eval_fn,
                [self.calib_data],
                inplace=False,
            )
            # 断言量化后的模型对 self.calib_data[0][0] 的推理结果与 result_eager 相等
            self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @override_qengines


注释完毕。
    def test_single_linear_dynamic(self):
        r"""Compare the result of dynamic quantization of single linear layer in
        eager mode and graph mode.
        """
        if qengine_is_qnnpack():
            # 如果 QNNPACK 引擎可用，使用 eager 模式进行量化
            
            # 创建一个已注释的单层线性模型对象，并设为评估模式
            annotated_linear_model = AnnotatedSingleLayerLinearModel("qnnpack").eval()
            # 创建一个未注释的单层线性模型对象，并设为评估模式
            linear_model = SingleLayerLinearModel().eval()
            
            # 从 eager 模式中复制权重，以便稍后比较两个量化模型的结果
            linear_model.fc1.weight = torch.nn.Parameter(
                annotated_linear_model.fc1.module.weight.detach()
            )
            linear_model.fc1.bias = torch.nn.Parameter(
                annotated_linear_model.fc1.module.bias.detach()
            )
            
            # 定义动态量化配置字典
            qconfig_dict = {"": default_dynamic_qconfig}
            # 使用动态量化对 annotated_linear_model 进行量化
            model_eager = quantize_dynamic(annotated_linear_model, qconfig_dict)

            # 对 linear_model 进行跟踪编译
            model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
            # 对 linear_model 进行脚本化
            model_script = torch.jit.script(linear_model)
            # 在 eager 模式下执行 annotated_linear_model，得到结果 result_eager
            result_eager = model_eager(self.calib_data[0][0])

            # 遍历跟踪编译和脚本化后的模型，进行动态量化
            for model_under_test in [model_traced, model_script]:
                model_quantized = quantize_dynamic_jit(model_under_test, qconfig_dict)
                
                # 断言量化后的模型对于相同输入等于 eager 模式下的结果
                self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

                # 检查确保 choose_qparams->quant->dequant->linear 在数值上等价于最终的量化模型
                model_fake_quantized = quantize_dynamic_jit(
                    model_under_test, qconfig_dict, debug=True
                )
                self.assertEqual(
                    model_fake_quantized(self.calib_data[0][0]), result_eager
                )

    @skipIfNoFBGEMM
    def test_linear_dynamic_fp16(self):
        # 创建一个未注释的单层线性模型对象，并设为评估模式
        linear_model = SingleLayerLinearModel().eval()
        
        # 创建一个超过 fp16 最大值的权重张量
        x = torch.ones(5, 5) * 65532
        linear_model.fc1.weight = torch.nn.Parameter(x)
        
        import warnings
        
        # 使用 fp16 动态量化对 linear_model 进行量化
        model_eager = quantize_dynamic(linear_model, dtype=torch.float16)
        # 在 eager 模式下执行 linear_model，得到结果 result_eager
        result_eager = model_eager(self.calib_data[0][0])
        
        # 遍历跟踪模式
        for trace in [True]:
            with warnings.catch_warnings(record=True) as w:
                # 使用 graph mode 对 linear_model 进行量化，检查线性动态 fp16 操作
                quantized_model = self.checkGraphModeOp(
                    linear_model,
                    self.calib_data[0][0],
                    "quantized::linear_dynamic_fp16",
                    tracing=trace,
                    dynamic=True,
                    qconfig=float16_dynamic_qconfig,
                )
            # 断言量化后的模型对于相同输入等于 eager 模式下的结果
            self.assertEqual(quantized_model(self.calib_data[0][0]), result_eager)
```
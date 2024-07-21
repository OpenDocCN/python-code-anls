# `.\pytorch\test\test_mkldnn_fusion.py`

```
# 导入必要的模块和类
# Owner(s): ["module: mkldnn"]
import itertools  # 导入 itertools 模块，用于迭代操作
import unittest  # 导入 unittest 模块，用于编写和运行测试
from typing import NamedTuple, List  # 导入 NamedTuple 和 List 类型提示

import torch  # 导入 PyTorch 深度学习库
from torch import nn  # 导入神经网络模块

from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo  # 导入测试工具函数和装饰器
from torch.testing._internal.jit_utils import JitTestCase  # 导入用于 JIT 测试的基类

from test_tensorexpr import warmup_and_run_forward  # 导入测试相关的函数

FUSION_GROUP = 'prim::TensorExprGroup'  # 定义融合组的名称常量

class PointwisePostOp(NamedTuple):
    attr : str
    pointwise_module : nn.Module
    scalars : List = []
    algorithm : str = ""

CONV_MODULES = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}  # 定义卷积模块字典
CONV_TRANSPOSE_MODULES = {2: torch.nn.ConvTranspose2d}  # 定义转置卷积模块字典

@skipIfTorchDynamo("too slow")  # 根据条件跳过测试，条件为 Torch Dynamo 太慢
@unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled")  # 根据条件跳过测试，条件为 MKL-DNN 构建被禁用
class TestMkldnnFusion(JitTestCase):
    def assertFused(self, graph, fused_patterns):
        # 验证图中确实包含指定的融合模式
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)

    def _check_model(self, m, x, trace=False):
        # 保存并设置调试参数，控制融合组的内联行为
        old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)

        # 保存并设置 JIT 可否在 CPU 上进行融合的状态
        old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_override_can_fuse_on_cpu(True)

        # 保存并设置 TE 是否必须使用 LLVM CPU 的状态
        old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

        # 将模型设置为评估模式，并生成脚本化的模型
        m.eval()
        with torch.no_grad():
            if trace:
                script = torch.jit.trace(m, x)
            else:
                script = torch.jit.script(m)
        script = torch.jit.freeze(script)

        # 运行预热和前向传播，获取输出并验证
        with torch.no_grad():
            y = warmup_and_run_forward(script, x)
            y = script(x)
            y_ref = m(x)

            graph = script.graph_for(*x)
            self.assertEqual(y, y_ref)

        # 恢复调试参数的原始设置
        torch._C._debug_set_fusion_group_inlining(old_fusion_inlining)
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuser_state)
        torch._C._jit_set_te_must_use_llvm_cpu(old_te_must_use_llvm_cpu)

        return graph
    # 定义一个测试函数，用于测试单个卷积操作
    def test_single_conv(self):
        # 定义一个继承自 nn.Module 的类 M，用于创建包含卷积层的模型
        class M(nn.Module):
            # 初始化函数，接受输入通道数、输出通道数、是否使用偏置以及其他参数
            def __init__(self, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                # 创建一个二维卷积层对象，根据传入参数设置输入通道数、输出通道数、是否使用偏置以及其他参数
                self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)

            # 前向传播函数，接受输入张量 x，对其进行卷积操作并返回结果
            def forward(self, x):
                # 执行卷积操作并返回结果张量
                res = self.conv(x)
                return res

        # 针对内存格式和是否启用通道优先的情况进行迭代测试
        for memory_format, enabled in [
            [torch.contiguous_format, False],  # 内存格式为通用格式，不启用通道优先
            [torch.channels_last, True],       # 内存格式为通道优先
        ]:
            # 针对是否跟踪图形进行迭代测试
            for trace in [True, False]:
                # 设置输入图像大小、批量大小、卷积核大小
                input_size = 224
                batch_size = 1
                kernel_size = 3
                # 针对是否使用偏置、扩展率、组数的组合进行迭代测试
                options = itertools.product([True, False], [1, 2], [1, 4])
                for bias, dilation, groups in options:
                    # 计算输入通道数和输出通道数
                    iC = 3 * groups
                    oC = 10 * groups
                    # 创建模型对象 m，设置输入通道数、输出通道数、卷积核大小、步长、填充、扩展率和组数，并指定内存格式
                    m = M(iC,
                          oC,
                          bias,
                          kernel_size=(kernel_size, kernel_size),
                          stride=2,
                          padding=1,
                          dilation=dilation,
                          groups=groups).to(memory_format=memory_format)
                    # 创建输入张量 x，形状为（批量大小，输入通道数，图像大小，图像大小），并指定内存格式
                    x = torch.randn(batch_size, iC, input_size, input_size).to(memory_format=memory_format)
                    # 检查模型 m 在输入 x 上的图形表示，返回图形对象
                    graph = self._check_model(m, x, trace)
                    # 根据是否启用融合操作，选择不同的卷积节点名称
                    conv_node_name = 'aten::_convolution' if trace else 'aten::conv2d'
                    # 如果启用融合操作，断言图形包含融合组合，并且仅包含一个融合组合
                    if enabled:
                        self.assertFused(graph, [conv_node_name])
                        self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
                    # 如果未启用融合操作，断言图形包含指定类型的卷积节点
                    else:
                        self.assertGraphContains(graph, kind=conv_node_name)
    def test_conv_unary_fusion_nnc(self):
        # 定义测试函数，用于测试卷积与一元操作融合的情况
        class M(nn.Module):
            def __init__(self, unary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                # 初始化卷积层，包括输入通道数、输出通道数、是否使用偏置以及其他参数
                self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
                # 初始化一元操作函数
                self.unary = unary_fn

            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 执行一元操作
                x = self.unary(x)
                return x

        # 遍历不同的内存格式与启用状态组合
        for memory_format, enabled in [
            [torch.contiguous_format, False],
            [torch.channels_last, True],
        ]:
            # 遍历不同的一元操作函数
            for unary_fn in [torch.relu]:
                # 遍历是否使用偏置
                for bias in [True, False]:
                    # 遍历不同的输出通道数
                    for oC in [1, 10]:
                        # 创建模型实例并指定内存格式
                        m = M(unary_fn, 3, oC, bias, kernel_size=(3, 3)).to(memory_format=memory_format)
                        # 创建随机输入张量并指定内存格式
                        x = torch.randn(1, 3, 224, 224).to(memory_format=memory_format)

                        # 对模型进行图分析
                        graph = self._check_model(m, x)
                        # 如果启用融合，则断言融合组存在
                        if enabled:
                            self.assertFused(graph, ['aten::conv2d', 'aten::' + unary_fn.__name__])
                            self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
                        else:
                            # 否则断言图中包含卷积操作
                            self.assertGraphContains(graph, kind='aten::conv2d')

    def test_unsupported_conv(self):
        # 定义测试函数，用于测试不支持的卷积操作
        class M(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                # 初始化使用给定模块的卷积层
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                # 执行卷积操作
                res = self.conv(x)
                return res

        # 遍历不同的模块、维度和内存格式组合
        for module, dim, memory_format in [
            [nn.Conv3d, 3, torch.contiguous_format],
            [nn.Conv3d, 3, torch.channels_last_3d],
            [nn.ConvTranspose2d, 2, torch.contiguous_format],
            [nn.ConvTranspose2d, 2, torch.channels_last],
        ]:
            trace = True
            input_size = 224
            batch_size = 1
            kernel_size = 3
            groups = 2
            bias = True
            iC = 3 * groups
            oC = 10 * groups
            dilation = 2
            # 创建模型实例并指定内存格式
            m = M(module,
                  iC,
                  oC,
                  bias,
                  kernel_size=kernel_size,
                  stride=2,
                  padding=1,
                  dilation=dilation,
                  groups=groups).to(memory_format=memory_format)
            # 根据维度添加输入大小
            input_sizes = [batch_size, iC, input_size, input_size]
            if dim == 3:
                input_sizes.append(input_size)
            # 创建随机输入张量并指定内存格式
            x = torch.randn(input_sizes).to(memory_format=memory_format)
            # 对模型进行图分析
            graph = self._check_model(m, x, trace)
            # 断言图中包含特定的卷积操作
            self.assertGraphContains(graph, kind='aten::_convolution')
    def test_linear_unary_fusion_ops(self):
        # 定义一个名为 test_linear_unary_fusion_ops 的测试方法
        class M(nn.Module):
            # 定义一个名为 M 的内部类，继承自 nn.Module
            def __init__(self, unary_fn, in_channels, out_channels, bias, **kwargs):
                # M 类的初始化方法，接受一元函数、输入通道数、输出通道数、偏置等参数
                super().__init__()
                # 调用父类 nn.Module 的初始化方法
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                # 初始化一个线性层，输入通道数、输出通道数、是否使用偏置及其他关键字参数
                self.unary = unary_fn
                # 保存传入的一元函数作为实例变量

            def forward(self, x):
                # 定义前向传播方法，接收输入 x
                x = self.linear(x)
                # 将输入 x 经过线性层 self.linear 处理
                x = self.unary(x)
                # 将线性层输出的 x 再经过保存的一元函数处理
                return x
                # 返回处理后的输出 x

        for pointwise_info in self._unary_list().values():
            # 遍历保存在 self._unary_list() 返回值中的所有一元操作信息
            options = itertools.product([[[2, 3, 10], None], [[2, 10], None], [[1, 10], [0, 1]]], [True, False])
            # 对输入形状和步幅的组合进行排列组合，以及是否使用偏置的两种选项
            for (input_shape, input_stride), bias in options:
                # 遍历所有排列组合的输入形状、步幅和是否使用偏置
                with torch.no_grad():
                    # 在上下文中不追踪梯度的操作
                    mod = M(pointwise_info.pointwise_module, input_shape[-1], 10, bias).eval()
                    # 创建 M 类的实例 mod，传入一元操作的模块、输入形状的最后一个维度作为输入通道数、输出通道数 10、是否使用偏置的值，并设置为评估模式
                    v = torch.randn(input_shape)
                    # 生成一个符合输入形状要求的随机张量 v
                    if input_stride is not None:
                        v = v.as_strided(input_shape, input_stride)
                        # 如果定义了输入步幅，将随机张量 v 按照指定形状和步幅重新排列
                    ref = mod(v)
                    # 使用 mod 对 v 进行前向传播，得到参考输出 ref
                    attr = pointwise_info.attr
                    # 获取一元操作的属性 attr
                    scalars = pointwise_info.scalars
                    # 获取一元操作的标量参数 scalars
                    algorithm = pointwise_info.algorithm
                    # 获取一元操作的算法 algorithm
                    fused = torch.ops.mkldnn._linear_pointwise(
                        v, mod.linear.weight, mod.linear.bias, attr, scalars, algorithm
                    )
                    # 调用 MKL-DNN 提供的线性一元融合操作，输入 v、线性层权重和偏置、属性 attr、标量参数 scalars、算法 algorithm
                    self.assertEqual(ref, fused)
                    # 断言参考输出 ref 与融合操作的输出 fused 相等
    def test_conv_unary_fusion_ops(self):
        # 定义测试用例类
        class M(nn.Module):
            # M 类的构造函数，接受一元函数、维度、输入通道数、输出通道数、膨胀、分组、偏置等参数
            def __init__(self, unary_fn, dim, in_channels, out_channels, dilation, groups, bias, **kwargs):
                super().__init__()
                # 创建卷积层对象，根据不同维度选择合适的卷积操作
                self.conv = CONV_MODULES[dim](in_channels, out_channels, dilation=dilation, groups=groups, bias=bias, **kwargs)
                # 设置一元操作函数
                self.unary = unary_fn

            # 前向传播函数
            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 执行一元操作
                x = self.unary(x)
                return x

        # 定义输入形状的字典
        input_shapes = {2: (112, 112), 3: (55, 55, 55)}
        # 遍历一元操作列表中的每个元素
        for pointwise_info in self._unary_list().values():
            # 遍历维度列表 [2, 3]
            for dim in [2, 3]:
                # 根据维度选择通道优先或者通道优先三维格式
                channels_last = torch.channels_last if dim == 2 else torch.channels_last_3d
                # 构建选项迭代器，包括偏置、膨胀、分组、存储格式等参数组合
                options = itertools.product([True, False], [1, 2], [1, 4], [torch.contiguous_format, channels_last])
                # 遍历所有参数组合
                for bias, dilation, groups, memory_format in options:
                    # 计算输出通道数和输入通道数
                    oC = 32 * groups
                    iC = 3 * groups
                    # 构建输入张量形状
                    x_shape = (1, iC) + input_shapes[dim]
                    # 创建随机张量 x，并指定存储格式
                    x = torch.randn(x_shape, dtype=torch.float32).to(memory_format=memory_format)
                    # 创建 M 类的实例 mod
                    mod = M(pointwise_info.pointwise_module, dim, iC, oC, dilation, groups, bias, kernel_size=3)
                    # 将 mod 实例转移到指定的存储格式，并设置为评估模式
                    mod = mod.to(memory_format=memory_format).eval()
                    # 关闭梯度计算
                    with torch.no_grad():
                        # 计算参考结果 ref
                        ref = mod(x)
                        # 获取一元操作的属性、标量和算法
                        attr = pointwise_info.attr
                        scalars = pointwise_info.scalars
                        algorithm = pointwise_info.algorithm
                        # 调用 MKL-DNN 库的点卷积融合操作
                        fused = torch.ops.mkldnn._convolution_pointwise(
                            x, mod.conv.weight, mod.conv.bias, mod.conv.padding, mod.conv.stride, mod.conv.dilation,
                            mod.conv.groups, attr, scalars, algorithm
                        )
                    # 断言参考结果和融合结果相等
                    self.assertEqual(ref, fused)
    # 定义测试函数 test_conv_binary_fusion_ops，用于测试二进制融合操作
    def test_conv_binary_fusion_ops(self):
        # 定义内部类 M，继承自 nn.Module，用于模拟包含卷积和二进制操作的神经网络模块
        class M(nn.Module):
            def __init__(self, binary_fn, dim, in_channels, out_channels, dilation, groups, bias, **kwargs):
                super().__init__()
                # 创建卷积层，使用指定的维度、输入输出通道数、扩张、分组、偏置等参数
                self.conv = CONV_MODULES[dim](in_channels, out_channels, dilation=dilation, groups=groups, bias=bias, **kwargs)
                # 设置二进制操作的函数
                self.binary = binary_fn

            def forward(self, x, other):
                # 前向传播函数，先通过卷积层处理输入 x
                x = self.conv(x)
                # 然后通过二进制操作处理 x 和其他输入 other
                x = self.binary(x, other)
                return x

        # 定义输入的形状字典
        input_shapes = {2: (112, 112), 3: (22, 22, 22)}
        # 遍历 _binary_list() 返回的二进制函数字典
        for pointwise_name, pointwise_fn in self._binary_list().items():
            # 遍历维度为 2 和 3 的情况
            for dim in [2, 3]:
                # 根据维度不同选择合适的通道排序方式
                channels_last = torch.channels_last if dim == 2 else torch.channels_last_3d
                # 使用 itertools 生成各种参数组合
                options = itertools.product([False, True], [True, False], [1, 2], [1, 4], [torch.contiguous_format, channels_last])
                # 遍历参数组合
                for fuse_relu, bias, dilation, groups, memory_format in options:
                    # 计算输出通道数和输入通道数
                    oC = 32 * groups
                    iC = 3 * groups
                    # 创建输入 x 的形状
                    x_shape = (1, iC) + input_shapes[dim]
                    # 生成随机输入 x，并转换为指定的内存格式
                    x = torch.randn(x_shape, dtype=torch.float32).to(memory_format=memory_format)
                    # 创建 M 类的实例 mod，传入二进制函数、维度、输入输出通道数、扩张、分组、偏置等参数
                    mod = M(pointwise_fn, dim, iC, oC, dilation, groups, bias, kernel_size=3)
                    # 将 mod 转换为指定的内存格式，并设置为评估模式
                    mod = mod.to(memory_format=memory_format).eval()
                    # 创建其他输入 other，形状与 mod.conv(x) 相同
                    other = torch.randn_like(mod.conv(x))
                    # 禁止梯度计算
                    with torch.no_grad():
                        # 计算模型的输出 ref
                        ref = mod(x, other)
                        # 初始化 unary_attr 为 None
                        unary_attr = None
                        # 如果设置了 fuse_relu，则对 ref 应用 ReLU 激活函数
                        if fuse_relu:
                            ref.relu_()
                            unary_attr = "relu"
                        # 设置属性 attr 为 pointwise_name
                        attr = pointwise_name
                        # 调用底层的 MKL-DNN 操作 _convolution_pointwise，进行卷积和二进制操作融合
                        fused = torch.ops.mkldnn._convolution_pointwise(
                            x, other, mod.conv.weight, mod.conv.bias, mod.conv.padding, mod.conv.stride, mod.conv.dilation,
                            mod.conv.groups, attr, None, unary_attr, [], None
                        )
                        # 对于属性为 "add" 的二进制加法操作，支持原位版本
                        if attr == "add":
                            fused_inplace = torch.ops.mkldnn._convolution_pointwise_(
                                other, x, mod.conv.weight, mod.conv.bias, mod.conv.padding, mod.conv.stride, mod.conv.dilation,
                                mod.conv.groups, attr, None, unary_attr, [], None
                            )
                            # 断言 ref 与原位操作 fused_inplace 相等
                            self.assertEqual(ref, fused_inplace)
                        # 断言 ref 与融合操作 fused 相等，设置容差为 5e-4
                        self.assertEqual(ref, fused, atol=5e-4, rtol=5e-4)
    def test_linear_binary_fusion_ops(self):
        # 定义一个测试函数，用于测试线性和二进制融合操作

        class M(nn.Module):
            # 定义一个简单的神经网络模型类
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                # 初始化线性层，设置输入通道数、输出通道数、是否包含偏置等参数
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                # 设置二进制函数
                self.binary = binary_fn

            def forward(self, x, other):
                # 模型的前向传播函数
                x = self.linear(x)  # 对输入 x 进行线性变换
                x = self.binary(x, other)  # 对线性变换后的结果和其他输入进行二进制操作
                return x

        out_feature = 20
        # 设置输出特征的维度为 20

        for pointwise_name, pointwise_fn in self._binary_list().items():
            # 遍历二进制函数列表中的每个函数及其名称
            options = itertools.product([[2, 3, 10], [2, 10]], [True, False])
            # 生成输入形状和是否包含偏置的所有可能组合
            for input_shape, bias in options:
                # 遍历所有可能的输入形状和偏置设置
                with torch.no_grad():
                    # 使用 torch.no_grad() 上下文，确保在评估模式下运行模型
                    mod = M(pointwise_fn, input_shape[-1], out_feature, bias).eval()
                    # 实例化模型，设置输入形状的最后一维作为输入通道数，20 为输出通道数，bias 为偏置设置
                    v = torch.randn(input_shape)
                    # 创建一个随机输入张量 v
                    other = torch.randn(input_shape[:-1] + [out_feature])
                    # 创建另一个随机张量 other，形状与 v 除最后一维外相同，最后一维为 20
                    ref = mod(v, other)
                    # 使用模型计算 v 和 other 的融合结果 ref
                    attr = pointwise_name
                    # 设置属性名称为 pointwise_name
                    fused = torch.ops.mkldnn._linear_pointwise(
                        v, other, mod.linear.weight, mod.linear.bias, attr
                    )
                    # 调用底层线性和逐点操作的融合函数，得到融合后的结果 fused
                    self.assertEqual(ref, fused)
                    # 使用断言检查模型计算结果 ref 和底层融合函数计算结果 fused 是否相等
# 如果当前模块被直接运行（而不是被导入到其他模块），则执行以下代码块
if __name__ == "__main__":
    # 调用 run_tests() 函数，用于执行程序的测试
    run_tests()
```
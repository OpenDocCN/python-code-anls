# `.\pytorch\test\test_xnnpack_integration.py`

```
# Owner(s): ["oncall: mobile"]

# 引入所需的模块和库
import io
import itertools
import unittest

# 引入 Hypothesis 相关模块
from hypothesis import assume, given, strategies as st

# 引入 PyTorch 及其相关模块
import torch
import torch.backends.xnnpack
import torch.testing._internal.hypothesis_utils as hu
from torch.nn import functional as F
from torch.testing import FileCheck

# 引入 PyTorch 测试和工具相关的公共函数和变量
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    run_tests,
    slowTest,
    TEST_WITH_TSAN,
    TestCase,
)

# 引入移动端优化相关模块
from torch.utils.mobile_optimizer import optimize_for_mobile


# 定义测试类 TestXNNPACKOps，继承自 unittest.TestCase
@unittest.skipUnless(
    torch.backends.xnnpack.enabled,
    " XNNPACK must be enabled for these tests." " Please build with USE_XNNPACK=1.",
)
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN fails with XNNPACK. Does not seem to have a good reason for failures.",
)
class TestXNNPACKOps(TestCase):
    # 跳过某些平台上失败的测试用例，具体信息见 GitHub issue
    @unittest.skip(
        "Fails on some platforms, see https://github.com/pytorch/pytorch/issues/73488"
    )
    @given(
        batch_size=st.integers(0, 3),
        data_shape=hu.array_shapes(1, 3, 2, 64),
        weight_output_dim=st.integers(2, 64),
        use_bias=st.booleans(),
    )
    # 定义测试线性操作的方法
    def test_linear(self, batch_size, data_shape, weight_output_dim, use_bias):
        data_shape = [batch_size] + list(data_shape)
        input_data = torch.rand(data_shape)
        weight = torch.rand((weight_output_dim, data_shape[-1]))
        if use_bias:
            bias = torch.rand(weight_output_dim)
        else:
            bias = None
        # 计算参考结果
        ref_result = F.linear(input_data, weight, bias)
        # 对权重和偏置进行打包预处理
        packed_weight_bias = torch.ops.prepacked.linear_clamp_prepack(weight, bias)
        # 使用预打包的权重和偏置执行线性操作
        output_linearprepacked = torch.ops.prepacked.linear_clamp_run(
            input_data, packed_weight_bias
        )
        # 断言参考结果和优化后的结果之间的接近程度
        torch.testing.assert_close(
            ref_result, output_linearprepacked, rtol=1e-2, atol=1e-3
        )

    @given(
        input_size=st.integers(2, 32),
        weight_output_dim=st.integers(2, 64),
        use_bias=st.booleans(),
    )
    # 定义测试一维输入线性操作的方法
    def test_linear_1d_input(self, input_size, weight_output_dim, use_bias):
        input_data = torch.rand(input_size)
        weight = torch.rand((weight_output_dim, input_data.shape[-1]))
        if use_bias:
            bias = torch.rand(weight_output_dim)
        else:
            bias = None
        # 计算参考结果
        ref_result = F.linear(input_data, weight, bias)
        # 对权重和偏置进行打包预处理
        packed_weight_bias = torch.ops.prepacked.linear_clamp_prepack(weight, bias)
        # 使用预打包的权重和偏置执行线性操作
        output_linearprepacked = torch.ops.prepacked.linear_clamp_run(
            input_data, packed_weight_bias
        )
        # 断言参考结果和优化后的结果之间的接近程度
        torch.testing.assert_close(
            ref_result, output_linearprepacked, rtol=1e-2, atol=1e-3
        )
    @given(
        batch_size=st.integers(0, 3),  # 随机生成批量大小在0到3之间的整数
        input_channels_per_group=st.integers(1, 32),  # 随机生成每组输入通道数在1到32之间的整数
        height=st.integers(5, 64),  # 随机生成高度在5到64之间的整数
        width=st.integers(5, 64),  # 随机生成宽度在5到64之间的整数
        output_channels_per_group=st.integers(1, 32),  # 随机生成每组输出通道数在1到32之间的整数
        groups=st.integers(1, 16),  # 随机生成组数在1到16之间的整数
        kernel_h=st.integers(1, 7),  # 随机生成卷积核高度在1到7之间的整数
        kernel_w=st.integers(1, 7),  # 随机生成卷积核宽度在1到7之间的整数
        stride_h=st.integers(1, 2),  # 随机生成垂直步长在1到2之间的整数
        stride_w=st.integers(1, 2),  # 随机生成水平步长在1到2之间的整数
        pad_h=st.integers(0, 2),  # 随机生成垂直填充在0到2之间的整数
        pad_w=st.integers(0, 2),  # 随机生成水平填充在0到2之间的整数
        dilation=st.integers(1, 2),  # 随机生成膨胀率在1到2之间的整数
        use_bias=st.booleans(),  # 随机生成是否使用偏置的布尔值
        format=st.sampled_from(  # 随机选择存储格式，可以是None或指定的torch存储格式之一
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    def test_conv2d(
        self,
        batch_size,
        input_channels_per_group,
        height,
        width,
        output_channels_per_group,
        groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation,
        use_bias,
        format,
    ):
        input_channels = input_channels_per_group * groups  # 计算总输入通道数
        output_channels = output_channels_per_group * groups  # 计算总输出通道数
        kernels = (kernel_h, kernel_w)  # 定义卷积核大小的元组
        strides = (stride_h, stride_w)  # 定义步长的元组
        paddings = (pad_h, pad_w)  # 定义填充的元组
        dilations = (dilation, dilation)  # 定义膨胀率的元组
        assume(height + 2 * paddings[0] >= dilations[0] * (kernels[0] - 1) + 1)  # 假设输入高度加上两倍垂直填充大于等于卷积核高度乘以膨胀率减1再加1
        assume(width + 2 * paddings[1] >= dilations[1] * (kernels[1] - 1) + 1)  # 假设输入宽度加上两倍水平填充大于等于卷积核宽度乘以膨胀率减1再加1

        input_data = torch.rand((batch_size, input_channels, height, width))  # 生成指定形状的随机输入数据张量
        if format is not None:  # 如果指定了存储格式
            input_data = input_data.contiguous(memory_format=format)  # 将输入数据按指定格式连续化

        weight = torch.rand(  # 生成指定形状的随机权重张量
            (output_channels, input_channels_per_group, kernel_h, kernel_w)
        )
        bias = None  # 初始化偏置为None
        if use_bias:  # 如果使用偏置
            bias = torch.rand(output_channels)  # 生成指定形状的随机偏置张量

        ref_result = F.conv2d(  # 使用PyTorch的函数进行普通卷积操作
            input_data, weight, bias, strides, paddings, dilations, groups
        )

        packed_weight_bias = torch.ops.prepacked.conv2d_clamp_prepack(  # 使用预打包的权重和偏置进行卷积操作
            weight, bias, strides, paddings, dilations, groups
        )
        xnnpack_result = torch.ops.prepacked.conv2d_clamp_run(  # 运行预打包的卷积操作
            input_data, packed_weight_bias
        )

        torch.testing.assert_close(  # 断言两个张量的值在指定的相对和绝对容差范围内接近
            ref_result, xnnpack_result, rtol=1e-2, atol=1e-3
        )
    @given(
        batch_size=st.integers(1, 3),  # 生成批量大小在1到3之间的随机整数
        input_channels_per_group=st.integers(1, 32),  # 生成每组输入通道数在1到32之间的随机整数
        height=st.integers(5, 64),  # 生成高度在5到64之间的随机整数
        width=st.integers(5, 64),  # 生成宽度在5到64之间的随机整数
        output_channels_per_group=st.integers(1, 32),  # 生成每组输出通道数在1到32之间的随机整数
        groups=st.integers(1, 16),  # 生成分组数在1到16之间的随机整数
        kernel_h=st.integers(1, 7),  # 生成卷积核高度在1到7之间的随机整数
        kernel_w=st.integers(1, 7),  # 生成卷积核宽度在1到7之间的随机整数
        stride_h=st.integers(1, 2),  # 生成高度方向步长在1到2之间的随机整数
        stride_w=st.integers(1, 2),  # 生成宽度方向步长在1到2之间的随机整数
        pad_h=st.integers(0, 2),  # 生成高度方向填充在0到2之间的随机整数
        pad_w=st.integers(0, 2),  # 生成宽度方向填充在0到2之间的随机整数
        output_pad_h=st.integers(0, 2),  # 生成输出高度填充在0到2之间的随机整数
        output_pad_w=st.integers(0, 2),  # 生成输出宽度填充在0到2之间的随机整数
        dilation=st.integers(1, 2),  # 生成扩展率在1到2之间的随机整数
        use_bias=st.booleans(),  # 随机生成布尔值，表示是否使用偏置
        format=st.sampled_from(  # 从给定列表中随机选择一个元素，表示数据格式
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    def test_conv2d_transpose(
        self,
        batch_size,  # 批量大小
        input_channels_per_group,  # 每组输入通道数
        height,  # 高度
        width,  # 宽度
        output_channels_per_group,  # 每组输出通道数
        groups,  # 分组数
        kernel_h,  # 卷积核高度
        kernel_w,  # 卷积核宽度
        stride_h,  # 高度方向步长
        stride_w,  # 宽度方向步长
        pad_h,  # 高度方向填充
        pad_w,  # 宽度方向填充
        output_pad_h,  # 输出高度填充
        output_pad_w,  # 输出宽度填充
        dilation,  # 扩展率
        use_bias,  # 是否使用偏置
        format,  # 数据格式
    ):
        input_channels = input_channels_per_group * groups  # 计算总输入通道数
        output_channels = output_channels_per_group * groups  # 计算总输出通道数
        kernels = (kernel_h, kernel_w)  # 卷积核尺寸
        strides = (stride_h, stride_w)  # 步长
        paddings = (pad_h, pad_w)  # 填充
        output_paddings = (output_pad_h, output_pad_w)  # 输出填充
        dilations = (dilation, dilation)  # 扩展率

        assume(height + 2 * paddings[0] >= dilations[0] * (kernels[0] - 1) + 1)  # 假设输入高度满足卷积运算条件
        assume(width + 2 * paddings[1] >= dilations[1] * (kernels[1] - 1) + 1)  # 假设输入宽度满足卷积运算条件
        assume((output_pad_h < stride_h) and (output_pad_h < dilation))  # 假设输出高度填充满足条件
        assume((output_pad_w < stride_w) and (output_pad_w < dilation))  # 假设输出宽度填充满足条件

        input_data = torch.rand((batch_size, input_channels, height, width))  # 生成随机输入数据
        if format is not None:
            input_data = input_data.contiguous(memory_format=format)  # 如果指定了格式，则转换数据格式

        weight = torch.rand(  # 生成随机权重张量
            (input_channels, output_channels_per_group, kernel_h, kernel_w)
        )
        bias = None
        if use_bias:
            bias = torch.rand(output_channels)  # 如果使用偏置，则生成随机偏置向量

        # Note that groups/dilation is in reverse order from conv2d
        ref_result = F.conv_transpose2d(  # 使用PyTorch的反卷积函数计算参考结果
            input_data,
            weight,
            bias,
            strides,
            paddings,
            output_paddings,
            groups,
            dilation,
        )

        packed_weight_bias = torch.ops.prepacked.conv2d_transpose_clamp_prepack(  # 使用预打包函数生成权重和偏置
            weight, bias, strides, paddings, output_paddings, dilations, groups
        )

        xnnpack_result = torch.ops.prepacked.conv2d_transpose_clamp_run(  # 运行预打包的反卷积函数
            input_data, packed_weight_bias
        )

        torch.testing.assert_close(  # 使用PyTorch测试工具，断言两个结果的近似程度
            ref_result.contiguous(), xnnpack_result.contiguous(), rtol=1e-2, atol=1e-3
        )
@unittest.skipUnless(
    torch.backends.xnnpack.enabled,
    " XNNPACK must be enabled for these tests." " Please build with USE_XNNPACK=1.",
)
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN fails with XNNPACK. Does not seem to have a good reason for failures.",
)
class TestXNNPACKSerDes(TestCase):
    @unittest.skip(
        "Fails on some platforms, see https://github.com/pytorch/pytorch/issues/73488"
    )
    @given(
        batch_size=st.integers(0, 3),
        data_shape=hu.array_shapes(1, 3, 2, 64),
        weight_output_dim=st.integers(2, 64),
        use_bias=st.booleans(),
    )
    def test_linear(self, batch_size, data_shape, weight_output_dim, use_bias):
        # 定义一个简单的线性层模型
        class Linear(torch.nn.Module):
            def __init__(self, weight, bias=None):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return F.linear(x, self.weight, self.bias)

        # 定义一个使用预打包权重的线性层模型
        class LinearPrePacked(torch.nn.Module):
            def __init__(self, weight, bias=None):
                super().__init__()
                # 使用预打包操作来封装权重和偏置
                self.packed_weight_bias = torch.ops.prepacked.linear_clamp_prepack(
                    weight, bias
                )

            def forward(self, x):
                # 使用预打包操作后的权重和偏置来运行线性层
                return torch.ops.prepacked.linear_clamp_run(x, self.packed_weight_bias)

        # 将 batch_size 添加到 data_shape 列表中
        data_shape = [batch_size] + list(data_shape)
        # 随机生成权重矩阵，形状为 (weight_output_dim, data_shape[-1])
        weight = torch.rand((weight_output_dim, data_shape[-1]))
        # 如果使用偏置，则随机生成偏置向量，否则设为 None
        if use_bias:
            bias = torch.rand(weight_output_dim)
        else:
            bias = None
        # 使用权重和偏置创建脚本化的线性层模型
        scripted_linear = torch.jit.script(Linear(weight, bias))
        # 使用权重和偏置创建预打包的脚本化线性层模型
        scripted_linear_clamp_prepacked = torch.jit.script(
            LinearPrePacked(weight, bias)
        )
        # 生成随机输入数据
        input_data = torch.rand(data_shape)
        # 计算未打包的线性层的参考结果
        ref_result = scripted_linear(input_data)
        # 计算预打包线性层的输出结果
        output_linearprepacked = scripted_linear_clamp_prepacked(input_data)
        # 断言两种模型的输出结果接近
        torch.testing.assert_close(
            ref_result, output_linearprepacked, rtol=1e-2, atol=1e-3
        )

        # 序列化模型并反序列化
        input_data = torch.rand(data_shape)
        buffer = io.BytesIO()
        torch.jit.save(scripted_linear, buffer)
        buffer.seek(0)
        deserialized_linear = torch.jit.load(buffer)
        buffer = io.BytesIO()
        torch.jit.save(scripted_linear_clamp_prepacked, buffer)
        buffer.seek(0)
        deserialized_linear_clamp_prepacked = torch.jit.load(buffer)
        # 使用反序列化后的模型计算输出结果
        ref_result = deserialized_linear(input_data)
        output_linearprepacked = deserialized_linear_clamp_prepacked(input_data)
        # 断言反序列化后的模型输出结果接近
        torch.testing.assert_close(
            ref_result, output_linearprepacked, rtol=1e-2, atol=1e-3
        )
    @given(
        batch_size=st.integers(0, 3),
        input_channels_per_group=st.integers(1, 32),
        height=st.integers(5, 64),
        width=st.integers(5, 64),
        output_channels_per_group=st.integers(1, 32),
        groups=st.integers(1, 16),
        kernel_h=st.integers(1, 7),
        kernel_w=st.integers(1, 7),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.integers(0, 2),
        pad_w=st.integers(0, 2),
        dilation=st.integers(1, 2),
        use_bias=st.booleans(),
        format=st.sampled_from(
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    # 定义测试方法test_conv2d，用于测试2D卷积操作
    def test_conv2d(
        self,
        batch_size,
        input_channels_per_group,
        height,
        width,
        output_channels_per_group,
        groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation,
        use_bias,
        format,
    )

    @given(
        batch_size=st.integers(0, 3),
        input_channels_per_group=st.integers(1, 32),
        height=st.integers(5, 64),
        width=st.integers(5, 64),
        output_channels_per_group=st.integers(1, 32),
        groups=st.integers(1, 16),
        kernel_h=st.integers(1, 7),
        kernel_w=st.integers(1, 7),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.integers(0, 2),
        pad_w=st.integers(0, 2),
        output_pad_h=st.integers(0, 2),
        output_pad_w=st.integers(0, 2),
        dilation=st.integers(1, 2),
        use_bias=st.booleans(),
        format=st.sampled_from(
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    # 定义测试方法test_conv2d_transpose，用于测试2D转置卷积操作
    def test_conv2d_transpose(
        self,
        batch_size,
        input_channels_per_group,
        height,
        width,
        output_channels_per_group,
        groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        output_pad_h,
        output_pad_w,
        dilation,
        use_bias,
        format,
    )

    @unittest.skip(
        "Fails on some platforms, see https://github.com/pytorch/pytorch/issues/73488"
    )
    @given(
        batch_size=st.integers(0, 3),
        input_channels_per_group=st.integers(1, 32),
        height=st.integers(5, 64),
        width=st.integers(5, 64),
        output_channels_per_group=st.integers(1, 32),
        groups=st.integers(1, 16),
        kernel_h=st.integers(1, 7),
        kernel_w=st.integers(1, 7),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.integers(0, 2),
        pad_w=st.integers(0, 2),
        dilation=st.integers(1, 2),
        linear_weight_output_dim=st.integers(2, 64),
        use_bias=st.booleans(),
        format=st.sampled_from(
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    # 跳过某些平台上的测试，参见GitHub上的问题73488
    # 定义测试组合模型的方法，用于评估模型的性能和正确性
    def test_combined_model(
        # 定义方法参数：批量大小
        self,
        batch_size,
        # 输入通道数（每组）
        input_channels_per_group,
        # 图像高度
        height,
        # 图像宽度
        width,
        # 输出通道数（每组）
        output_channels_per_group,
        # 组数（指定模型中的组数）
        groups,
        # 卷积核高度
        kernel_h,
        # 卷积核宽度
        kernel_w,
        # 垂直方向的步长
        stride_h,
        # 水平方向的步长
        stride_w,
        # 垂直方向的填充数
        pad_h,
        # 水平方向的填充数
        pad_w,
        # 卷积核的膨胀率
        dilation,
        # 线性层的权重输出维度
        linear_weight_output_dim,
        # 是否使用偏置项
        use_bias,
        # 格式（例如输入输出的数据格式）
        format,
# 根据条件决定是否跳过这些测试，需要 XNNPACK 功能开启
@unittest.skipUnless(
    torch.backends.xnnpack.enabled,
    " XNNPACK must be enabled for these tests." " Please build with USE_XNNPACK=1.",
)
# 根据条件决定是否跳过这些测试，与 TSAN 有关
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN fails with XNNPACK. Does not seem to have a good reason for failures.",
)
class TestXNNPACKRewritePass(TestCase):
    @staticmethod
    # 验证经过转换的模块
    def validate_transformed_module(
        # 用于满足 flake
        self,
        pattern_count_map,
        data_shape,
        prepack_removal=False,
        fuse_clamping_ops=False,
    ):
        # 创建符合正态分布的输入数据
        input_data = torch.normal(1, 20, size=data_shape)

        # 对于两种 JIT 方法：脚本化和追踪
        for jit_method in ["script", "trace"]:
            # 获取模块实例
            module_instance = self
            if jit_method == "script":
                # 将模块脚本化
                scripted_model = torch.jit.script(module_instance)
            else:
                # 进行模块追踪
                scripted_model = torch.jit.trace(module_instance, input_data)
            
            # 设置模型为评估模式
            scripted_model.eval()
            # 计算基准结果
            ref_result = scripted_model(input_data)
            # 在 JIT 编译的模型中插入预打包操作
            torch._C._jit_pass_insert_prepacked_ops(scripted_model._c)
            # 如果需要融合夹紧操作或预打包操作移除
            if fuse_clamping_ops or prepack_removal:
                scripted_model._c = torch._C._freeze_module(scripted_model._c)
            # 如果需要融合夹紧操作
            if fuse_clamping_ops:
                torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv(scripted_model._c)
            # 如果需要移除预打包操作
            if prepack_removal:
                torch._C._jit_pass_fold_prepacking_ops(scripted_model._c)

            # 创建字节流缓冲区
            buffer = io.BytesIO()
            # 将脚本化的模型保存到缓冲区
            torch.jit.save(scripted_model, buffer)
            buffer.seek(0)
            # 从缓冲区加载脚本化模型
            deserialized_scripted_model = torch.jit.load(buffer)
            
            # 遍历模式-计数映射中的每个模式及其对应的计数值
            for pattern, v in pattern_count_map.items():
                if v == 0:
                    # 检查模式在反序列化后的图中出现
                    FileCheck().check(pattern).run(deserialized_scripted_model.graph)
                elif v == -1:
                    # 检查模式在反序列化后的图中不出现
                    FileCheck().check_not(pattern).run(
                        deserialized_scripted_model.graph
                    )
                else:
                    # 检查模式在反序列化后的图中出现的次数是否符合预期
                    FileCheck().check_count(pattern, v, exactly=True).run(
                        deserialized_scripted_model.graph
                    )
            
            # 使用 XNNPACK 优化后的模型计算结果
            xnnpack_result = deserialized_scripted_model(input_data)
            # 断言基准结果与 XNNPACK 优化结果的接近程度
            torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)
    def test_decomposed_linear(self):
        # 定义数据形状为 [2, 32]
        data_shape = [2, 32]
        # 设置权重输出维度为 24，权重形状为 (24, 32)
        weight_output_dim = 24
        weight_shape = (weight_output_dim, data_shape[-1])

        # 定义一个继承自 torch.nn.Module 的类，实现加法和矩阵乘法的分解线性层
        class DecomposedLinearAddmm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化权重为随机数，并设置不需要梯度
                self.weight = torch.nn.Parameter(
                    torch.rand(weight_shape), requires_grad=False
                )
                # 初始化偏置为随机数，并设置不需要梯度
                self.bias = torch.nn.Parameter(
                    torch.rand(weight_output_dim), requires_grad=False
                )

            def forward(self, x):
                # 计算权重的转置
                weight_t = self.weight.t()
                # 返回偏置与输入 x 乘以权重转置的结果
                return torch.addmm(self.bias, x, weight_t)

        # 定义另一个继承自 torch.nn.Module 的类，实现矩阵乘法和加法的分解线性层
        class DecomposedLinearMatmulAdd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化权重为随机数，并设置不需要梯度
                self.weight = torch.nn.Parameter(
                    torch.rand(weight_shape), requires_grad=False
                )
                # 初始化偏置为随机数，并设置不需要梯度
                self.bias = torch.nn.Parameter(
                    torch.rand(weight_output_dim), requires_grad=False
                )

            def forward(self, x):
                # 计算权重的转置
                weight_t = self.weight.t()
                # 计算输入 x 与权重转置的矩阵乘法
                y = torch.matmul(x, weight_t)
                # 将偏置加到结果上并返回
                res = y.add_(self.bias)
                return res

        # 定义另一个继承自 torch.nn.Module 的类，实现仅矩阵乘法的分解线性层
        class DecomposedLinearMatmul(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化权重为随机数，并设置不需要梯度
                self.weight = torch.nn.Parameter(
                    torch.rand(weight_shape), requires_grad=False
                )
                # 初始化偏置为随机数，并设置不需要梯度
                self.bias = torch.nn.Parameter(
                    torch.rand(weight_output_dim), requires_grad=False
                )

            def forward(self, x):
                # 计算权重的转置
                weight_t = self.weight.t()
                # 返回输入 x 与权重转置的矩阵乘法结果
                res = torch.matmul(x, weight_t)
                return res

        # 定义一个字典，描述一种模式及其出现次数的映射关系
        pattern_count_map = {
            "Tensor = prim::CallFunction": -1,
            "prepacked::linear_clamp_prepack": 1,
            "prepacked::linear_clamp_run": 1,
        }
        # 使用测试方法验证分解线性层的转换
        TestXNNPACKRewritePass.validate_transformed_module(
            DecomposedLinearAddmm(), pattern_count_map, data_shape
        )
        TestXNNPACKRewritePass.validate_transformed_module(
            DecomposedLinearMatmulAdd(), pattern_count_map, data_shape
        )
        TestXNNPACKRewritePass.validate_transformed_module(
            DecomposedLinearMatmul(), pattern_count_map, data_shape
        )
# 使用装饰器 unittest.skipUnless，根据 torch.backends.xnnpack.enabled 的值决定是否跳过测试
# 如果 xnnpack 已启用，则不跳过；否则输出指定的跳过消息
@unittest.skipUnless(
    torch.backends.xnnpack.enabled,
    " XNNPACK must be enabled for these tests." " Please build with USE_XNNPACK=1.",
)
# 使用装饰器 unittest.skipIf，根据 TEST_WITH_TSAN 的值决定是否跳过测试
# 如果 TEST_WITH_TSAN 为真，输出指定的跳过消息，因为 TSAN 在多线程环境中无法安全地进行 fork
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
# 定义一个测试类 TestXNNPACKConv1dTransformPass，继承自 unittest.TestCase
class TestXNNPACKConv1dTransformPass(TestCase):
    # 静态方法 validate_transform_conv1d_to_conv2d，用于验证 conv1d 转换为 conv2d 的功能
    @staticmethod
    def validate_transform_conv1d_to_conv2d(
        self, pattern_count_transformed_map, pattern_count_optimized_map, data_shape
    ):
        # 生成指定形状的正态分布随机数据，作为输入数据
        input_data = torch.normal(1, 20, size=data_shape)

        # 遍历两种 JIT 方法："script" 和 "trace"
        for jit_method in ["script", "trace"]:
            # 将当前测试类实例化为 module_instance
            module_instance = self
            # 根据 JIT 方法选择性地创建脚本化模型或跟踪模型
            if jit_method == "script":
                scripted_model = torch.jit.script(module_instance)
            else:
                scripted_model = torch.jit.trace(module_instance, input_data)
            # 将模型设置为评估模式
            scripted_model.eval()
            # 使用 JIT pass 将 conv1d 转换为 conv2d
            torch._C._jit_pass_transform_conv1d_to_conv2d(scripted_model._c)
            # 对脚本化模型进行移动优化
            optimized_scripted_model = optimize_for_mobile(scripted_model)

            # 创建一个字节流缓冲区，并将脚本化模型保存到其中
            buffer = io.BytesIO()
            torch.jit.save(scripted_model, buffer)
            buffer.seek(0)
            # 从字节流缓冲区加载反序列化的脚本化模型
            deserialized_scripted_model = torch.jit.load(buffer)

            # 遍历转换后的模式计数映射，对模型图执行相应的文件检查
            for pattern, v in pattern_count_transformed_map.items():
                if v == 0:
                    FileCheck().check(pattern).run(deserialized_scripted_model.graph)
                elif v == -1:
                    FileCheck().check_not(pattern).run(
                        deserialized_scripted_model.graph
                    )
                else:
                    FileCheck().check_count(pattern, v, exactly=True).run(
                        deserialized_scripted_model.graph
                    )
            # 使用转换后的模型对输入数据进行推断
            transformed_result = deserialized_scripted_model(input_data)
            # 断言转换后的结果与参考结果的接近程度
            torch.testing.assert_close(
                ref_result, transformed_result, rtol=1e-2, atol=1e-3
            )

            # 创建一个字节流缓冲区，并将优化后的脚本化模型保存到其中
            optimized_buffer = io.BytesIO()
            torch.jit.save(optimized_scripted_model, optimized_buffer)
            optimized_buffer.seek(0)
            # 从字节流缓冲区加载反序列化的优化后的脚本化模型
            deserialized_optimized_scripted_model = torch.jit.load(optimized_buffer)

            # 遍历优化后模式计数映射，对优化后的模型图执行相应的文件检查
            for pattern, v in pattern_count_optimized_map.items():
                if v == 0:
                    FileCheck().check(pattern).run(
                        deserialized_optimized_scripted_model.graph
                    )
                elif v == -1:
                    FileCheck().check_not(pattern).run(
                        deserialized_optimized_scripted_model.graph
                    )
                else:
                    FileCheck().check_count(pattern, v, exactly=True).run(
                        deserialized_optimized_scripted_model.graph
                    )
            # 使用优化后的模型对输入数据进行推断，得到 XNNPACK 结果
            xnnpack_result = deserialized_optimized_scripted_model(input_data)
            # 断言参考结果与 XNNPACK 结果的接近程度
            torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)
    # 如果在 FBCODE 环境下，跳过此测试用例，原因是 T137513244
    @unittest.skipIf(IS_FBCODE, "T137513244")
    def test_conv1d_basic(self):
        # 定义测试用例的参数范围
        batch_size_list = range(1, 3)  # 批大小范围为1到2
        input_channels_per_group_list = range(10, 12)  # 每组输入通道数范围为10到11
        width_list = range(10, 12)  # 输入宽度范围为10到11
        output_channels_per_group_list = range(10, 12)  # 每组输出通道数范围为10到11
        groups_list = range(1, 3)  # 组数范围为1到2
        kernel_list = range(1, 4)  # 卷积核大小范围为1到3
        stride_list = range(1, 3)  # 步长范围为1到2
        padding_list = range(0, 3)  # 填充范围为0到2
        dilation_list = range(1, 3)  # 膨胀率范围为1到2
    
        # 遍历所有参数组合
        for hparams in itertools.product(
            batch_size_list,
            input_channels_per_group_list,
            width_list,
            output_channels_per_group_list,
            groups_list,
            kernel_list,
            stride_list,
            padding_list,
            dilation_list,
        ):
            (
                batch_size,
                input_channels_per_group,
                width,
                output_channels_per_group,
                groups,
                kernel,
                stride,
                padding,
                dilation,
            ) = hparams
    
            # 计算总的输入通道数和输出通道数
            input_channels = input_channels_per_group * groups
            output_channels = output_channels_per_group * groups
    
            # 定义卷积层的权重和偏置的形状
            conv_weight_shape = (output_channels, input_channels_per_group, kernel)
            conv_bias_shape = output_channels
    
            # 定义 Conv1D 类，继承自 torch.nn.Module
            class Conv1D(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 初始化权重和偏置，并设置为不可训练
                    self.weight = torch.nn.Parameter(
                        torch.rand(conv_weight_shape), requires_grad=False
                    )
                    self.bias = torch.nn.Parameter(
                        torch.rand(conv_bias_shape), requires_grad=False
                    )
                    self.stride = stride  # 设置卷积层的步长
                    self.padding = padding  # 设置卷积层的填充
                    self.dilation = dilation  # 设置卷积层的膨胀率
                    self.groups = groups  # 设置卷积层的分组数
    
                # 定义前向传播函数
                def forward(self, x):
                    return F.conv1d(
                        x,
                        self.weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )
    
            # 定义数据的形状
            data_shape = (batch_size, input_channels, width)
    
            # 定义转换后的卷积模式和优化后的卷积模式的映射字典
            pattern_count_transformed_map = {
                "Tensor = aten::conv1d": -1,
                "Tensor = aten::conv2d": 1,
            }
            pattern_count_optimized_map = {
                "Tensor = aten::conv1d": -1,
                "Tensor = aten::conv2d": -1,
                "prepacked::conv2d_clamp_prepack": -1,
                "prepacked::conv2d_clamp_run": 1,
            }
    
            # 调用静态方法 validate_transform_conv1d_to_conv2d 进行验证转换
            TestXNNPACKConv1dTransformPass.validate_transform_conv1d_to_conv2d(
                Conv1D(),
                pattern_count_transformed_map,
                pattern_count_optimized_map,
                data_shape,
            )
    # 将该测试标记为慢速测试，参考链接 https://github.com/pytorch/pytorch/issues/46066
    @slowTest
# 如果当前脚本作为主程序执行，则运行测试函数
if __name__ == "__main__":
    run_tests()
```
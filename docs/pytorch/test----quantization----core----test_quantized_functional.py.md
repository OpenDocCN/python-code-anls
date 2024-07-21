# `.\pytorch\test\quantization\core\test_quantized_functional.py`

```
# Owner(s): ["oncall: quantization"]

# Torch库的引入，包括量化相关的功能模块和常规的神经网络功能模块
import torch
import torch.ao.nn.quantized.functional as qF  # 引入量化相关的函数
import torch.nn.functional as F  # 引入常规的神经网络函数

# 标准库的引入
import numpy as np  # 引入NumPy库

# 测试工具的引入
from hypothesis import assume, given  # 引入假设和给定函数
from hypothesis import strategies as st  # 引入策略模块
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,  # 引入量化测试用例基类
    _make_conv_test_input,  # 引入创建卷积测试输入的函数
)
from torch.testing._internal.common_quantized import override_quantized_engine  # 引入覆盖量化引擎的函数
from torch.testing._internal.common_utils import (
    IS_PPC,  # 引入是否为PPC架构的常量
    TEST_WITH_UBSAN,  # 引入是否使用UBSAN进行测试的常量
)

# 定义测试类，继承自QuantizationTestCase
class TestQuantizedFunctionalOps(QuantizationTestCase):
    
    # 测试torch.relu函数的量化版本和非量化版本的API一致性
    def test_relu_api(self):
        X = torch.arange(-5, 5, dtype=torch.float)  # 创建一个浮点数序列
        scale = 2.0  # 设置量化的缩放因子
        zero_point = 1  # 设置量化的零点
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)  # 对输入张量X进行量化
        qY = torch.relu(qX)  # 对量化后的张量qX应用ReLU函数
        qY_hat = F.relu(qX)  # 对未量化的张量X应用ReLU函数
        self.assertEqual(qY, qY_hat)  # 断言量化后的结果qY与未量化的结果qY_hat相等

    # 内部实现函数：测试卷积操作的API一致性和功能正确性
    def _test_conv_api_impl(
        self, qconv_fn, conv_fn, batch_size, in_channels_per_group,
        input_feature_map_size, out_channels_per_group, groups, kernel_size,
        stride, padding, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
        Y_scale, Y_zero_point, use_bias, use_channelwise,
    ):
        for i in range(len(kernel_size)):
            assume(input_feature_map_size[i] + 2 * padding[i]
                   >= dilation[i] * (kernel_size[i] - 1) + 1)  # 假设输入特征图大小满足卷积核的合理性条件
        (X, X_q, W, W_q, b) = _make_conv_test_input(
            batch_size, in_channels_per_group, input_feature_map_size,
            out_channels_per_group, groups, kernel_size, X_scale,
            X_zero_point, W_scale, W_zero_point, use_bias, use_channelwise)  # 生成卷积操作的测试输入

        Y_exp = conv_fn(X, W, b, stride, padding, dilation, groups)  # 执行非量化的卷积操作，得到期望输出
        Y_exp = torch.quantize_per_tensor(
            Y_exp, scale=Y_scale, zero_point=Y_zero_point, dtype=torch.quint8)  # 对期望输出进行量化
        Y_act = qconv_fn(
            X_q, W_q, b, stride, padding, dilation, groups,
            padding_mode="zeros", scale=Y_scale, zero_point=Y_zero_point)  # 执行量化的卷积操作，得到实际输出

        # 确保结果一致性，使用np.testing.assert_array_almost_equal函数进行比较
        # assert_array_almost_equal函数使用decimal=0参数来忽略参考和测试之间的偏差
        np.testing.assert_array_almost_equal(
            Y_exp.int_repr().numpy(), Y_act.int_repr().numpy(), decimal=0)  # 比较量化后的整数表示的结果
    # 使用 `hypothesis` 提供的 `given` 装饰器，定义测试函数 `test_conv1d_api`，接受多个参数用于生成测试数据
    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           L=st.integers(4, 16),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel=st.integers(1, 7),
           stride=st.integers(1, 2),
           pad=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    # 定义 `test_conv1d_api` 测试方法，用于验证 `conv1d` 函数的正确性
    def test_conv1d_api(
        self, batch_size, in_channels_per_group, L, out_channels_per_group,
        groups, kernel, stride, pad, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_channelwise, qengine,
    ):
        # 如果指定的量化引擎不在支持的列表中，则退出测试
        if qengine not in torch.backends.quantized.supported_engines:
            return
        # 如果量化引擎为 'qnnpack'，且运行环境为 PPC 或者使用 UBSAN，则退出测试
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return
            # 强制禁用通道级量化
            use_channelwise = False

        # 定义输入特征图的大小为 L 组成的元组
        input_feature_map_size = (L, )
        # 定义卷积核大小为 kernel 组成的元组
        kernel_size = (kernel, )
        # 定义步长为 stride 组成的元组
        stride = (stride, )
        # 定义填充为 pad 组成的元组
        padding = (pad, )
        # 定义扩展率为 dilation 组成的元组
        dilation = (dilation, )

        # 使用指定的量化引擎上下文，设置量化卷积函数为 qF.conv1d
        with override_quantized_engine(qengine):
            qconv_fn = qF.conv1d
            # 设置标准卷积函数为 F.conv1d
            conv_fn = F.conv1d
            # 调用内部方法 _test_conv_api_impl，传入所有参数以测试卷积 API 的实现
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)
    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           H=st.integers(4, 16),
           W=st.integers(4, 16),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    # 定义一个参数化测试函数，用来测试 conv2d 函数的正确性
    def test_conv2d_api(
        self, batch_size, in_channels_per_group, H, W, out_channels_per_group,
        groups, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_channelwise, qengine,
    ):
        # 检查所选的量化引擎是否被 PyTorch 支持，如果不支持则退出测试
        if qengine not in torch.backends.quantized.supported_engines:
            return
        # 如果量化引擎为 'qnnpack'，并且在 PPC 架构下或者使用 UBSAN 进行测试，则退出测试
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return

        # 定义输入特征图的大小
        input_feature_map_size = (H, W)
        # 定义卷积核的大小
        kernel_size = (kernel_h, kernel_w)
        # 定义步长
        stride = (stride_h, stride_w)
        # 定义填充
        padding = (pad_h, pad_w)
        # 定义扩张率
        dilation = (dilation, dilation)

        # 使用指定的量化引擎覆盖当前的量化引擎设置
        with override_quantized_engine(qengine):
            # 获取量化卷积函数和普通卷积函数的引用
            qconv_fn = qF.conv2d
            conv_fn = F.conv2d
            # 调用内部实现的卷积 API 测试函数
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)
    # 使用 hypothesis 库的 given 装饰器定义测试用例，测试卷积运算的 API
    @given(batch_size=st.integers(1, 3),  # 批量大小在 1 到 3 之间的整数
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),  # 输入每组通道数从给定列表中随机选择
           D=st.integers(4, 8),  # 输入特征图的深度在 4 到 8 之间的整数
           H=st.integers(4, 8),  # 输入特征图的高度在 4 到 8 之间的整数
           W=st.integers(4, 8),  # 输入特征图的宽度在 4 到 8 之间的整数
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),  # 输出每组通道数从给定列表中随机选择
           groups=st.integers(1, 4),  # 组数在 1 到 4 之间的整数
           kernel_d=st.integers(1, 4),  # 卷积核的深度在 1 到 4 之间的整数
           kernel_h=st.integers(1, 4),  # 卷积核的高度在 1 到 4 之间的整数
           kernel_w=st.integers(1, 4),  # 卷积核的宽度在 1 到 4 之间的整数
           stride_d=st.integers(1, 2),  # 深度方向的步长在 1 到 2 之间的整数
           stride_h=st.integers(1, 2),  # 高度方向的步长在 1 到 2 之间的整数
           stride_w=st.integers(1, 2),  # 宽度方向的步长在 1 到 2 之间的整数
           pad_d=st.integers(0, 2),  # 深度方向的填充在 0 到 2 之间的整数
           pad_h=st.integers(0, 2),  # 高度方向的填充在 0 到 2 之间的整数
           pad_w=st.integers(0, 2),  # 宽度方向的填充在 0 到 2 之间的整数
           dilation=st.integers(1, 2),  # 膨胀率在 1 到 2 之间的整数
           X_scale=st.floats(1.2, 1.6),  # 输入特征图的缩放因子在 1.2 到 1.6 之间的浮点数
           X_zero_point=st.integers(0, 4),  # 输入特征图的零点在 0 到 4 之间的整数
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),  # 权重的缩放因子列表，每个元素在 0.2 到 1.6 之间的浮点数
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),  # 权重的零点列表，每个元素在 -5 到 5 之间的整数
           Y_scale=st.floats(4.2, 5.6),  # 输出特征图的缩放因子在 4.2 到 5.6 之间的浮点数
           Y_zero_point=st.integers(0, 4),  # 输出特征图的零点在 0 到 4 之间的整数
           use_bias=st.booleans(),  # 是否使用偏置的布尔值
           use_channelwise=st.booleans(),  # 是否使用通道级量化的布尔值
           qengine=st.sampled_from(("fbgemm",)))  # 量化引擎从给定列表中随机选择为 "fbgemm"
    def test_conv3d_api(
        self, batch_size, in_channels_per_group, D, H, W,
        out_channels_per_group, groups, kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation, X_scale,
        X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
        use_channelwise, qengine,
    ):
        # 测试 conv3d 函数的正确性
        # 目前 conv3d 只支持 FbGemm 引擎
    
        if qengine not in torch.backends.quantized.supported_engines:
            return
    
        input_feature_map_size = (D, H, W)
        kernel_size = (kernel_d, kernel_h, kernel_w)
        stride = (stride_d, stride_h, stride_w)
        padding = (pad_d, pad_h, pad_w)
        dilation = (dilation, dilation, dilation)
    
        # 使用指定的量化引擎覆盖当前引擎
        with override_quantized_engine(qengine):
            qconv_fn = qF.conv3d
            conv_fn = F.conv3d
            # 调用内部实现的 conv3d API 测试函数
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)
    
    @given(N=st.integers(1, 10),  # 样本数量在 1 到 10 之间的整数
           C=st.integers(1, 10),  # 通道数在 1 到 10 之间的整数
           H=st.integers(4, 8),  # 输入特征图高度在 4 到 8 之间的整数
           H_out=st.integers(4, 8),  # 输出特征图高度在 4 到 8 之间的整数
           W=st.integers(4, 8),  # 输入特征图宽度在 4 到 8 之间的整数
           W_out=st.integers(4, 8),  # 输出特征图宽度在 4 到 8 之间的整数
           scale=st.floats(.1, 2),  # 缩放因子在 0.1 到 2 之间的浮点数
           zero_point=st.integers(0, 4))  # 零点在 0 到 4 之间的整数
    # 定义一个测试方法，用于测试 grid_sample 函数的行为
    def test_grid_sample(self, N, C, H, H_out, W, W_out, scale, zero_point):
        # 创建一个随机张量 X，形状为 (N, C, H, W)
        X = torch.rand(N, C, H, W)
        # 对张量 X 进行量化，使用给定的缩放因子 scale 和零点 zero_point，数据类型为 torch.quint8
        X_q = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        # 创建一个随机网格 grid，形状为 (N, H_out, W_out, 2)
        grid = torch.rand(N, H_out, W_out, 2)

        # 对量化后的张量 X_q 应用 grid_sample 函数，得到输出 out
        out = F.grid_sample(X_q, grid)
        # 对未量化的张量 X 应用 grid_sample 函数并量化，使用相同的 scale 和 zero_point，得到期望输出 out_exp
        out_exp = torch.quantize_per_tensor(F.grid_sample(X, grid), scale=scale, zero_point=zero_point, dtype=torch.quint8)
        
        # 使用 numpy.testing.assert_array_almost_equal 函数比较 out 和 out_exp 的整数表示，精度为小数点后 0 位
        np.testing.assert_array_almost_equal(
            out.int_repr().numpy(), out_exp.int_repr().numpy(), decimal=0)
```
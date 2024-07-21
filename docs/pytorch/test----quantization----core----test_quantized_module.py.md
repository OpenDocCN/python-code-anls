# `.\pytorch\test\quantization\core\test_quantized_module.py`

```py
# Owner(s): ["oncall: quantization"]

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.ao.nn.intrinsic as nni  # 导入AO子模块中的内置运算加速器接口
import torch.ao.nn.intrinsic.quantized as nniq  # 导入AO子模块中的量化内置运算加速器接口
import torch.ao.nn.quantized.reference as nnqr  # 导入AO子模块中的参考量化运算接口
import torch.ao.quantization  # 导入AO模块中的量化操作接口
import torch.ao.nn.quantized as nnq  # 导入AO子模块中的量化神经网络模块
import torch.ao.nn.quantized.dynamic as nnqd  # 导入AO子模块中的动态量化神经网络模块

from torch.ao.quantization import (  # 导入AO量化模块中的指定子模块和函数
    get_default_static_quant_module_mappings,  # 获取默认的静态量化模块映射
    default_float_qparams_observer,  # 默认的浮点量化参数观察器
    PerChannelMinMaxObserver,  # 按通道计算的最小最大值观察器
)
from torch.package import PackageExporter, PackageImporter  # 导入打包和导入相关模块
from torch.testing._internal.common_quantization import (  # 导入内部量化测试的通用模块
    QuantizationTestCase,  # 量化测试用例基类
    prepare_dynamic,  # 准备动态量化
    _make_conv_test_input,  # 创建卷积测试输入
    skipIfNoFBGEMM,  # 如果没有FBGEMM，则跳过测试
    lengths_to_offsets,  # 长度转换为偏移量
    skipIfNoONEDNN,  # 如果没有ONEDNN，则跳过测试
    _make_conv_add_extra_input_tensor,  # 创建卷积添加额外输入张量
)
from torch.testing._internal.common_quantized import (  # 导入内部量化测试的通用量化模块
    _calculate_dynamic_qparams,  # 计算动态量化参数
    override_quantized_engine,  # 覆盖量化引擎
    override_qengines,  # 覆盖量化引擎
    qengine_is_qnnpack,  # 判断量化引擎是否为QNNPACK
    qengine_is_onednn,  # 判断量化引擎是否为ONEDNN
)
import torch.fx  # 导入Torch的FX模块
from hypothesis import assume, given  # 导入假设和给定模块
from hypothesis import strategies as st  # 导入假设的策略模块
import torch.testing._internal.hypothesis_utils as hu  # 导入内部测试的假设工具
hu.assert_deadline_disabled()  # 禁用测试的截止时间限制

import copy  # 导入复制模块
import io  # 导入IO操作模块
import numpy as np  # 导入NumPy库
import itertools  # 导入迭代工具

"""
Note that tests in this file are just API test, to make sure we wrapped the
quantized operator implementations correctly in the user facing APIs, these are
not correctness test for the underlying quantized operators. For correctness
test please see `test/quantization/test_quantized_op.py`.
"""

class TestStaticQuantizedModule(QuantizationTestCase):
    def test_relu(self):
        relu_module = nn.ReLU()  # 创建ReLU非量化模块
        relu6_module = nnq.ReLU6()  # 创建ReLU6量化模块

        x = torch.arange(-10, 10, dtype=torch.float)  # 创建一个浮点数张量x
        y_ref = torch.relu(x)  # 计算ReLU的参考输出
        y6_ref = torch.nn.modules.ReLU6()(x)  # 计算ReLU6的参考输出

        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.qint32)  # 对张量x进行量化
        qy = relu_module(qx)  # 使用ReLU模块对量化张量进行计算
        qy6 = relu6_module(qx)  # 使用ReLU6模块对量化张量进行计算

        self.assertEqual(y_ref, qy.dequantize(),  # 检查ReLU模块API是否失败
                         msg="ReLU module API failed")
        self.assertEqual(y6_ref, qy6.dequantize(),  # 检查ReLU6模块API是否失败
                         msg="ReLU6 module API failed")

    @override_qengines
    def test_linear(self):
        """test API functionality for nn.quantized.linear"""
        options = itertools.product(  # 使用itertools生成参数组合
            [1, 5],  # 批大小
            [16, 32],  # 输入特征数
            [4, 8],  # 输出特征数
            [True, False],  # 是否使用偏置
            [True, False])  # 是否使用按通道量化
        for (batch_size, in_features, out_features, use_bias, per_channel) in options:
            self._test_linear_api_impl(  # 调用测试线性API的实现函数
                nnq.Linear, 'QuantizedLinear', torch.ops.quantized.linear, batch_size,
                in_features, out_features, use_bias, per_channel)

    @override_qengines
    def test_linear_relu(self):
        """test API functionality for nn.intrinsic.quantized.linear_relu"""
        # 生成测试选项的所有组合：批大小、输入特征数、输出特征数、是否使用偏置、是否逐通道量化
        options = itertools.product(
            [1, 5],
            [16, 32],
            [4, 8],
            [True, False],
            [True, False])
        # 遍历每个测试选项组合
        for (batch_size, in_features, out_features, use_bias, per_channel) in options:
            # 调用测试线性层 API 的实现方法，针对量化的线性ReLU
            self._test_linear_api_impl(
                nniq.LinearReLU, 'QuantizedLinearReLU', torch.ops.quantized.linear_relu,
                batch_size, in_features, out_features, use_bias, per_channel)

    def test_quant_dequant_api(self):
        # 创建一个浮点数张量
        r = torch.tensor([[1., -1.], [1., -1.]], dtype=torch.float)
        scale, zero_point, dtype = 1.0, 2, torch.qint8
        # 测试量化 API
        qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
        # 创建量化对象
        quant_m = nnq.Quantize(scale, zero_point, dtype)
        qr2 = quant_m(r)
        # 断言量化结果是否一致
        self.assertEqual(qr, qr2)
        # 测试反量化 API
        rqr = qr.dequantize()
        # 创建反量化对象
        dequant_m = nnq.DeQuantize()
        rqr2 = dequant_m(qr2)
        # 断言反量化结果是否一致
        self.assertEqual(rqr, rqr2)

    @override_qengines
    # 定义一个测试 Conv1d API 的方法
    def test_conv1d_api(self):
        # 生成所有可能的选项组合
        options = itertools.product(
            ["zeros", "reflect"],  # pad_mode 填充模式可以是 "zeros" 或者 "reflect"
            [True, False],  # use_bias 是否使用偏置
            [True, False],  # use_channelwise 是否使用通道级量化
        )
        # 遍历每一种选项组合
        for pad_mode, use_bias, use_channelwise in options:
            # 如果当前使用的量化引擎是 qnnpack，则强制关闭 use_channelwise
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False
            
            # 定义一些测试用的参数
            batch_size = 2
            in_channels_per_group = 2
            length = 8
            out_channels_per_group = 2
            groups = 3
            kernel = 3
            stride = 2
            pad = 1
            dilation = 1
            
            # 计算完整的输入和输出通道数
            in_channels = in_channels_per_group * groups
            out_channels = out_channels_per_group * groups
            
            # 定义输入特征图的大小
            input_feature_map_size = (length,)
            
            # 定义卷积核大小、步长、填充和扩展率
            kernel_size = (kernel, )
            stride = (stride, )
            pad = (pad, )
            dilation = (dilation, )
            
            # 设置输入 X 的量化参数
            X_scale = 1.3
            X_zero_point = 2
            
            # 设置权重 W 的量化参数
            W_scale = [0.5]
            W_zero_point = [0] if qengine_is_onednn() else [3]
            
            # 设置输出 Y 的量化参数
            Y_scale = 5.0
            Y_zero_point = 4
            
            # 在 qnnpack 引擎下，强制关闭 use_channelwise
            if torch.backends.quantized.engine == 'qnnpack':
                use_channelwise = False
            
            # 使用 nnq.Conv1d 创建量化卷积模块
            qconv_cls = nnq.Conv1d
            module_name = "QuantizedConv1d"
            qconv_module = qconv_cls(
                in_channels, out_channels, kernel, stride, pad,
                dilation, groups, use_bias, padding_mode=pad_mode
            )
            
            # 使用 nn.Conv1d 创建标准卷积模块，并将其转换为 float 类型
            conv_module = nn.Conv1d(
                in_channels, out_channels, kernel, stride, pad,
                dilation, groups, use_bias, padding_mode=pad_mode)
            conv_module = conv_module.float()
            
            # 调用测试卷积 API 的实现方法
            self._test_conv_api_impl(
                module_name, qconv_module, conv_module, batch_size,
                in_channels_per_group, input_feature_map_size,
                out_channels_per_group, groups, kernel_size, stride, pad, pad_mode,
                dilation, X_scale, X_zero_point, W_scale, W_zero_point, Y_scale,
                Y_zero_point, use_bias, "none", use_channelwise)

    # 使用装饰器 override_qengines 重写量化引擎
    @override_qengines
    def test_conv1d_relu_api(self):
        # 生成所有可能的参数组合，用于测试
        options = itertools.product(
            ["zeros", "reflect"],  # pad_mode
            [True, False],  # use_bias
            [True, False],  # use_channelwise
        )
        batch_size = 2  # 定义批处理大小
        in_channels_per_group = 2  # 每组内的输入通道数
        length = 8  # 输入特征图的长度
        out_channels_per_group = 2  # 每组内的输出通道数
        groups = 3  # 组数
        kernel = 3  # 卷积核大小
        stride = 2  # 步幅
        pad = 1  # 填充
        dilation = 1  # 空洞卷积的扩张率

        # 设置输出通道数和输入通道数
        in_channels = in_channels_per_group * groups
        out_channels = out_channels_per_group * groups
        # 输入特征图大小
        input_feature_map_size = (length,)
        # 卷积核大小
        kernel_size = (kernel, )
        # 步幅大小
        stride = (stride, )
        # 填充大小
        pad = (pad, )
        # 空洞卷积的扩张率
        dilation = (dilation, )
        
        X_scale = 1.3  # 输入张量的缩放因子
        X_zero_point = 2  # 输入张量的零点
        W_scale = [0.5]  # 卷积核的缩放因子
        # 根据量化引擎选择卷积核的零点
        W_zero_point = [0] if qengine_is_onednn() else [3]
        Y_scale = 5.0  # 输出张量的缩放因子
        Y_zero_point = 4  # 输出张量的零点

        qconv_cls = nniq.ConvReLU1d  # 使用量化卷积ReLU1d类
        module_name = "QuantizedConvReLU1d"  # 模块名称

        # 遍历所有参数组合进行测试
        for pad_mode, use_bias, use_channelwise in options:
            # 根据量化引擎选择是否启用通道智能
            if torch.backends.quantized.engine == 'qnnpack':
                use_channelwise = False
            
            # 初始化量化ReLU1d模块
            qconv_module = qconv_cls(
                in_channels, out_channels, kernel, stride, pad,
                dilation, groups, use_bias, padding_mode=pad_mode
            )

            # 初始化普通的Conv1d模块
            conv_module = nn.Conv1d(
                in_channels, out_channels, kernel, stride, pad,
                dilation, groups, use_bias, padding_mode=pad_mode)
            relu_module = nn.ReLU()
            # 将Conv1d和ReLU组合成ConvReLU1d
            conv_module = nni.ConvReLU1d(conv_module, relu_module)
            conv_module = conv_module.float()  # 转换为float类型

            # 调用测试函数进行API测试
            self._test_conv_api_impl(
                module_name, qconv_module, conv_module, batch_size,
                in_channels_per_group, input_feature_map_size,
                out_channels_per_group, groups, kernel_size, stride, pad, pad_mode,
                dilation, X_scale, X_zero_point, W_scale, W_zero_point, Y_scale,
                Y_zero_point, use_bias, "relu", use_channelwise)

    @override_qengines
    # 定义测试 Conv2d API 的方法
    def test_conv2d_api(self):
        # 生成所有可能的参数组合
        options = itertools.product(
            ["zeros", "reflect"],  # pad_mode，填充模式选项
            [True, False],  # use_bias，是否使用偏置项
            [True, False],  # use_channelwise，是否使用通道级量化
        )
        # 遍历每种参数组合进行测试
        for pad_mode, use_bias, use_channelwise in options:
            # 如果当前使用的量化引擎是 qnnpack，则强制禁用通道级量化
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False
            
            # 设定测试所需的常量参数
            batch_size = 2
            in_channels_per_group = 2
            H = 8
            W = 8
            out_channels_per_group = 2
            groups = 3
            kernel_h = 3
            kernel_w = 3
            stride_h = 2
            stride_w = 2
            pad_h = 1
            pad_w = 1
            dilation = 1
            
            # 测试 Conv2d 模块的正确性
            in_channels = in_channels_per_group * groups
            out_channels = out_channels_per_group * groups
            input_feature_map_size = (H, W)
            kernel_size = (kernel_h, kernel_w)
            stride = (stride_h, stride_w)
            padding = (pad_h, pad_w)
            dilation = (dilation, dilation)
            X_scale = 1.3
            X_zero_point = 2
            W_scale = [0.5]
            # 根据量化引擎判断权重的零点
            W_zero_point = [0] if qengine_is_onednn() else [3]
            Y_scale = 5.0
            Y_zero_point = 4
            qconv_cls = nnq.Conv2d
            module_name = "QuantizedConv2d"
            
            # 创建量化 Conv2d 模块
            qconv_module = qconv_cls(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, use_bias, padding_mode=pad_mode
            )
            
            # 创建普通 Conv2d 模块并转换为 float 类型
            conv_module = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, use_bias, padding_mode=pad_mode)
            conv_module = conv_module.float()
            
            # 调用内部方法测试 Conv2d API 的实现
            self._test_conv_api_impl(
                module_name, qconv_module, conv_module, batch_size,
                in_channels_per_group, input_feature_map_size,
                out_channels_per_group, groups, kernel_size, stride, padding,
                pad_mode, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, "none", use_channelwise)

    # 使用装饰器重写量化引擎的设置
    @override_qengines
    # 定义测试卷积层和ReLU激活函数接口的方法
    def test_conv2d_relu_api(self):
        # 定义各种参数组合
        options = itertools.product(
            ["zeros", "reflect"],  # pad_mode，填充模式选项
            [True, False],  # use_bias，是否使用偏置
            [True, False],  # use_channelwise，是否使用通道级量化
        )
        batch_size = 2  # 批量大小
        in_channels_per_group = 2  # 每组输入通道数
        H = 8  # 输入特征图的高度
        W = 8  # 输入特征图的宽度
        out_channels_per_group = 2  # 每组输出通道数
        groups = 3  # 组数
        kernel_h = 3  # 卷积核高度
        kernel_w = 3  # 卷积核宽度
        stride_h = 2  # 卷积步幅高度
        stride_w = 2  # 卷积步幅宽度
        pad_h = 1  # 垂直方向填充数
        pad_w = 1  # 水平方向填充数
        dilation = 1  # 膨胀系数
        # 测试 conv2d 模块的正确性
        in_channels = in_channels_per_group * groups  # 总输入通道数
        out_channels = out_channels_per_group * groups  # 总输出通道数
        input_feature_map_size = (H, W)  # 输入特征图尺寸
        kernel_size = (kernel_h, kernel_w)  # 卷积核尺寸
        stride = (stride_h, stride_w)  # 卷积步幅
        padding = (pad_h, pad_w)  # 填充大小
        dilation = (dilation, dilation)  # 膨胀系数
        X_scale = 1.3  # 输入数据的缩放因子
        X_zero_point = 2  # 输入数据的零点
        W_scale = [0.5]  # 权重的缩放因子列表
        W_zero_point = [0] if qengine_is_onednn() else [3]  # 权重的零点列表
        Y_scale = 5.0  # 输出数据的缩放因子
        Y_zero_point = 4  # 输出数据的零点
        qconv_cls = nniq.ConvReLU2d  # 使用量化的ConvReLU2d类
        module_name = "QuantizedConvReLU2d"  # 模块名称
        # 遍历所有选项组合进行测试
        for pad_mode, use_bias, use_channelwise in options:
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False  # 如果使用的是qnnpack引擎，强制不使用通道级量化
            # 创建量化的卷积ReLU模块
            qconv_module = qconv_cls(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, use_bias, padding_mode=pad_mode
            )
            # 创建标准的卷积模块
            conv_module = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, use_bias, padding_mode=pad_mode)
            relu_module = nn.ReLU()  # 创建ReLU激活函数模块
            conv_module = nni.ConvReLU2d(conv_module, relu_module)  # 将卷积和ReLU模块结合
            conv_module = conv_module.float()  # 转换为float类型

            # 调用内部实现的卷积API测试方法
            self._test_conv_api_impl(
                module_name, qconv_module, conv_module, batch_size,
                in_channels_per_group, input_feature_map_size,
                out_channels_per_group, groups, kernel_size, stride, padding,
                pad_mode, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, "relu", use_channelwise)
    # 定义测试 Conv3d API 的方法
    def test_conv3d_api(self):
        # 生成所有可能的选项组合
        options = itertools.product(
            [True, False],  # use_bias 是否使用偏置
            [True, False],  # use_channelwise 是否使用通道级量化
        )
        # 定义批处理大小
        batch_size = 2
        # 每组的输入通道数
        in_channels_per_group = 2
        # 输入特征图的高度、宽度、深度
        H = 8
        W = 8
        D = 8
        # 每组的输出通道数
        out_channels_per_group = 2
        # 分组数
        groups = 3
        # 卷积核的高度、宽度、深度
        kernel_h = 3
        kernel_w = 3
        kernel_d = 3
        # 步长的设置
        stride_h = 2
        stride_w = 2
        stride_d = 2
        # 填充模式（3D 不支持 reflect 填充）
        pad_mode = "zeros"
        # 填充的高度、宽度、深度
        pad_h = 1
        pad_w = 1
        pad_d = 1
        # 膨胀率
        dilation = 1
        # 计算实际输入通道数
        in_channels = in_channels_per_group * groups
        # 计算实际输出通道数
        out_channels = out_channels_per_group * groups
        # 输入特征图的尺寸
        input_feature_map_size = (D, H, W)
        # 卷积核的尺寸
        kernel_size = (kernel_d, kernel_h, kernel_w)
        # 步长的设置
        stride = (stride_d, stride_h, stride_w)
        # 填充的设置
        padding = (pad_d, pad_h, pad_w)
        # 膨胀率的设置
        dilation = (dilation, dilation, dilation)
        # 输入张量的缩放因子和零点
        X_scale = 1.3
        X_zero_point = 2
        # 权重的缩放因子和零点
        W_scale = [0.5]
        W_zero_point = [0] if qengine_is_onednn() else [3]
        # 输出张量的缩放因子和零点
        Y_scale = 5.0
        Y_zero_point = 4
        # 量化卷积类
        qconv_cls = nnq.Conv3d
        # 模块名称
        module_name = "QuantizedConv3d"
        
        # 遍历所有选项组合进行测试
        for use_bias, use_channelwise in options:
            # 如果使用 QNNPACK 引擎，则强制关闭通道级量化
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False
            # 使用 fbgemm 引擎执行以下代码块
            with override_quantized_engine('fbgemm'):
                # 创建量化卷积模块
                qconv_module = qconv_cls(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode
                )
                # 创建浮点数卷积模块
                conv_module = nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode)
                # 将卷积模块转换为浮点数
                conv_module = conv_module.float()

                # 调用私有方法 _test_conv_api_impl，测试卷积 API
                self._test_conv_api_impl(
                    module_name, qconv_module, conv_module, batch_size,
                    in_channels_per_group, input_feature_map_size,
                    out_channels_per_group, groups, kernel_size, stride, padding,
                    pad_mode, dilation, X_scale, X_zero_point, W_scale,
                    W_zero_point, Y_scale, Y_zero_point, use_bias, "none",
                    use_channelwise)

    @skipIfNoFBGEMM
    # 定义测试函数 test_conv3d_relu_api，用于测试 Conv3d 和 ReLU 模块的 API 准确性
    def test_conv3d_relu_api(self):
        # 枚举所有可能的选项组合，包括是否使用偏置和是否使用通道级别的量化
        options = itertools.product(
            [True, False],  # use_bias 是否使用偏置
            [True, False],  # use_channelwise 是否使用通道级别的量化
        )
        # 定义一些测试中需要使用的参数和变量
        batch_size = 2
        in_channels_per_group = 2
        H = 8
        W = 8
        D = 8
        out_channels_per_group = 2
        groups = 3
        kernel_h = 3
        kernel_w = 3
        kernel_d = 3
        stride_h = 2
        stride_w = 2
        stride_d = 2
        pad_mode = "zeros"  # 3d 不支持反射填充
        pad_h = 1
        pad_w = 1
        pad_d = 1
        dilation = 1
        
        # 计算总的输入通道数和输出通道数
        in_channels = in_channels_per_group * groups
        out_channels = out_channels_per_group * groups
        
        # 定义输入特征图大小、卷积核大小、步长、填充、扩张率等参数
        input_feature_map_size = (D, H, W)
        kernel_size = (kernel_d, kernel_h, kernel_w)
        stride = (stride_d, stride_h, stride_w)
        padding = (pad_d, pad_h, pad_w)
        dilation = (dilation, dilation, dilation)
        
        # 定义输入和权重的缩放因子以及零点
        X_scale = 1.3
        X_zero_point = 2
        W_scale = [0.5]
        W_zero_point = [0] if qengine_is_onednn() else [3]
        Y_scale = 5.0
        Y_zero_point = 4
        
        # 使用的量化卷积类和模块名称
        qconv_cls = nniq.ConvReLU3d
        module_name = "QuantizedConvReLU3d"
        
        # 遍历所有选项组合，执行测试
        for use_bias, use_channelwise in options:
            # 如果量化引擎是 qnnpack，则强制不使用通道级别量化
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False
            
            # 使用 fbgemm 作为量化引擎的上下文
            with override_quantized_engine('fbgemm'):
                # 创建量化 ConvReLU3d 模块
                qconv_module = qconv_cls(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode
                )
                
                # 创建普通的 Conv3d 模块和 ReLU 模块
                conv_module = nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode)
                relu_module = nn.ReLU()
                conv_module = nni.ConvReLU3d(conv_module, relu_module)
                conv_module = conv_module.float()  # 转换为 float 类型
                
                # 执行具体的 ConvAPI 测试实现
                self._test_conv_api_impl(
                    module_name, qconv_module, conv_module, batch_size,
                    in_channels_per_group, input_feature_map_size,
                    out_channels_per_group, groups, kernel_size, stride, padding,
                    pad_mode, dilation, X_scale, X_zero_point, W_scale,
                    W_zero_point, Y_scale, Y_zero_point, use_bias, "relu",
                    use_channelwise)
    def test_conv2d_add(self):
        """测试 nn.intrinsic.quantized.ConvAdd2d 的 API 功能"""

        # 使用 onednn 引擎进行量化引擎的覆盖
        with override_quantized_engine('onednn'):
            # 定义测试选项的组合
            options = itertools.product(
                ["zeros", "reflect"],  # pad_mode 填充模式选项
                [True, False],  # use_bias 是否使用偏置选项
                [True, False],  # use_channelwise 是否使用通道级量化选项
            )

            # 定义常量参数
            batch_size = 2
            in_channels_per_group = 2
            H = 8
            W = 8
            out_channels_per_group = 2
            groups = 3
            kernel_h = 3
            kernel_w = 3
            stride_h = 2
            stride_w = 2
            pad_h = 1
            pad_w = 1
            dilation = 1

            # 测试卷积模块的正确性
            in_channels = in_channels_per_group * groups  # 输入通道数
            out_channels = out_channels_per_group * groups  # 输出通道数
            input_feature_map_size = (H, W)  # 输入特征图大小
            kernel_size = (kernel_h, kernel_w)  # 卷积核大小
            stride = (stride_h, stride_w)  # 步幅
            padding = (pad_h, pad_w)  # 填充
            dilation = (dilation, dilation)  # 膨胀率

            X_scale = 1.3  # 输入张量的比例因子
            X_zero_point = 2  # 输入张量的零点
            X2_scale = 1.2  # 第二个输入张量的比例因子
            X2_zero_point = 1  # 第二个输入张量的零点
            W_scale = [0.5]  # 权重的比例因子
            W_zero_point = [0] if qengine_is_onednn() else [3]  # 权重的零点
            Y_scale = 5.0  # 输出张量的比例因子
            Y_zero_point = 4  # 输出张量的零点

            qconv_cls = nniq.ConvAdd2d  # 量化卷积类
            module_name = "QuantizedConvAdd2d"  # 模块名称

            # 遍历测试选项进行测试
            for pad_mode, use_bias, use_channelwise in options:
                # 实例化量化卷积模块
                qconv_module = qconv_cls(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode
                )

                # 实例化标准卷积模块
                conv_module = nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode)

                # 使用 torch.ao.nn.intrinsic.ConvAdd2d 封装 conv_module
                conv_module = torch.ao.nn.intrinsic.ConvAdd2d(conv_module, torch.add)

                # 将卷积模块转换为 float 类型
                conv_module = conv_module.float()

                # 调用私有方法测试卷积 API 实现
                self._test_conv_api_impl(
                    module_name, qconv_module, conv_module, batch_size,
                    in_channels_per_group, input_feature_map_size,
                    out_channels_per_group, groups, kernel_size, stride, padding,
                    pad_mode, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
                    Y_scale, Y_zero_point, use_bias, "add", use_channelwise, X2_scale, X2_zero_point)

    @skipIfNoONEDNN  # 如果没有 ONEDNN，则跳过测试
    def test_conv2d_add_relu(self):
        """测试 nn.intrinsic.quantized.ConvAdd2d 的 API 功能"""
        # 使用 'onednn' 引擎覆盖量化引擎
        with override_quantized_engine('onednn'):
            # 使用 itertools 的 product 方法生成参数组合
            options = itertools.product(
                ["zeros", "reflect"],  # pad_mode 填充模式
                [True, False],  # use_bias 是否使用偏置
                [True, False],  # use_channelwise 是否使用通道级量化
            )
            # 定义测试用例的固定参数
            batch_size = 2
            in_channels_per_group = 2
            H = 8
            W = 8
            out_channels_per_group = 2
            groups = 3
            kernel_h = 3
            kernel_w = 3
            stride_h = 2
            stride_w = 2
            pad_h = 1
            pad_w = 1
            dilation = 1
            # 测试卷积模块的正确性
            in_channels = in_channels_per_group * groups
            out_channels = out_channels_per_group * groups
            input_feature_map_size = (H, W)
            kernel_size = (kernel_h, kernel_w)
            stride = (stride_h, stride_w)
            padding = (pad_h, pad_w)
            dilation = (dilation, dilation)
            X_scale = 1.3
            X_zero_point = 2
            X2_scale = 1.2
            X2_zero_point = 1
            W_scale = [0.5]
            W_zero_point = [0] if qengine_is_onednn() else [3]
            Y_scale = 5.0
            Y_zero_point = 4
            qconv_cls = nniq.ConvAddReLU2d
            module_name = "QuantizedConvAddReLU2d"
            # 遍历所有选项组合进行测试
            for pad_mode, use_bias, use_channelwise in options:
                # 初始化量化卷积模块
                qconv_module = qconv_cls(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode
                )

                # 初始化标准卷积模块
                conv_module = nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode)
                # 转换为 ConvAddReLU2d 模块
                conv_module = torch.ao.nn.intrinsic.ConvAddReLU2d(conv_module, torch.add, nn.ReLU())
                # 转换为浮点数类型
                conv_module = conv_module.float()

                # 执行卷积 API 的具体测试实现
                self._test_conv_api_impl(
                    module_name, qconv_module, conv_module, batch_size,
                    in_channels_per_group, input_feature_map_size,
                    out_channels_per_group, groups, kernel_size, stride, padding,
                    pad_mode, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
                    Y_scale, Y_zero_point, use_bias, "add_relu", use_channelwise, X2_scale, X2_zero_point)
    # 定义一个测试函数，用于测试池化模块的正确性
    def test_pool_api(self):
        """Tests the correctness of the pool module.
        The correctness is defined against the functional implementation.
        """
        # 定义输入张量的维度大小
        N, C, H, W = 10, 10, 10, 3
        # 定义池化操作的参数
        kwargs = {
            'kernel_size': 2,
            'stride': None,
            'padding': 0,
            'dilation': 1
        }

        # 定义量化所需的比例和零点
        scale, zero_point = 1.0 / 255, 128

        # 创建随机张量 X
        X = torch.randn(N, C, H, W, dtype=torch.float32)
        # 对 X 进行量化，得到量化张量 qX
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch.quint8)
        # 使用 torch.nn.functional.max_pool2d 对 qX 进行最大池化操作，作为预期结果
        qX_expect = torch.nn.functional.max_pool2d(qX, **kwargs)

        # 创建待测试的 MaxPool2d 对象
        pool_under_test = torch.ao.nn.quantized.MaxPool2d(**kwargs)
        # 对 qX 使用待测试的池化对象，得到池化后的结果 qX_hat
        qX_hat = pool_under_test(qX)
        # 断言池化后的结果与预期结果相等
        self.assertEqual(qX_expect, qX_hat)

        # JIT 测试
        self.checkScriptable(pool_under_test, [[X]])

    # 测试 dropout 模块的正确性
    def test_dropout(self):
        """Tests the correctness of the dropout module.
        The correctness is defined against the functional implementation.
        """
        # 创建一个随机张量 x
        x = torch.randn((2, 4, 6, 8), dtype=torch.float)
        # 创建一个概率为 0.5 的浮点数 dropout 模块
        float_mod = torch.nn.Dropout(p=0.5)
        float_mod.training = False

        # 对 x 应用浮点数 dropout 模块，得到参考结果 y_ref
        y_ref = float_mod(x)
        # 将 y_ref 进行量化，得到量化参考结果 quant_ref
        quant_ref = torch.quantize_per_tensor(y_ref, 1.0, 0, dtype=torch.quint8)

        # 创建概率为 0.5 的量化 dropout 模块
        quant_mod = nnq.Dropout(p=0.5)
        # 对 x 进行量化，得到量化输入 qx
        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.quint8)
        # 使用量化 dropout 模块对 qx 进行处理，得到量化结果 qy
        qy = quant_mod(qx)

        # 断言量化结果的整数表示与量化参考结果的整数表示相等，用于检验 dropout 模块的 API 是否失败
        self.assertEqual(quant_ref.int_repr().numpy(), qy.int_repr().numpy(),
                         msg="Dropout module API failed")

    # 测试 dropout 模块的序列化
    def _test_dropout_serialization(self, get_model, data1, data2):
        # 获取模型 m1
        m1 = get_model()
        m1.qconfig = torch.ao.quantization.default_qconfig
        mp1 = torch.ao.quantization.prepare(m1)
        mp1(data1)
        mq1 = torch.ao.quantization.convert(mp1)
        ref1 = mq1(data2)

        # 获取模型 m2
        m2 = get_model()
        m2.qconfig = torch.ao.quantization.default_qconfig
        mp2 = torch.ao.quantization.prepare(m2)
        mq2 = torch.ao.quantization.convert(mp2)

        # 加载模型 m2 的状态字典，并得到序列化后的结果 ref2
        mq2.load_state_dict(mq1.state_dict())
        ref2 = mq2(data2)

        # 断言 ref1 和 ref2 的所有元素是否近似相等
        self.assertTrue(torch.allclose(ref1, ref2))

    # 测试 dropout 模块的序列化
    def test_dropout_serialization(self):
        # 创建两个随机数据集 data1 和 data2
        data1 = torch.randn(2, 4, 6, 8)
        data2 = torch.randn(2, 4, 6, 8)

        # 定义一个获取模型的函数 _get_model
        def _get_model():
            return nn.Sequential(
                torch.ao.quantization.QuantStub(),
                nn.Dropout(p=0.5),
                torch.ao.quantization.DeQuantStub()
            ).eval()

        # 调用 _test_dropout_serialization 进行测试
        self._test_dropout_serialization(_get_model, data1, data2)
    def test_batch_norm2d(self):
        """Tests the correctness of the batchnorm2d module.
        The correctness is defined against the functional implementation.
        """
        # 创建一个形状为 (2, 4, 6, 8) 的随机张量 x，数据类型为 float
        x = torch.randn((2, 4, 6, 8), dtype=torch.float)
        # 创建一个 nn.BatchNorm2d 模块的实例 float_mod
        float_mod = torch.nn.BatchNorm2d(4)
        # 将 float_mod 设置为推断模式
        float_mod.training = False

        # 对 x 应用 float_mod 模块，得到参考输出 y_ref
        y_ref = float_mod(x)
        # 对 y_ref 进行量化转换，得到量化的参考输出 quant_ref
        quant_ref = torch.quantize_per_tensor(y_ref, 1.0, 0, dtype=torch.quint8)

        # 创建一个 nnq.BatchNorm2d 模块的实例 quant_mod
        quant_mod = nnq.BatchNorm2d(4)
        # 对输入 x 进行量化转换，得到量化的输入 qx
        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.quint8)
        # 对 qx 应用 quant_mod 模块，得到量化输出 qy
        qy = quant_mod(qx)

        # 断言量化参考输出与量化输出 qy 的整数表示是否相等，否则输出错误信息
        self.assertEqual(quant_ref.int_repr().numpy(), qy.int_repr().numpy(),
                         msg="BatchNorm2d module API failed")

    def test_batch_norm3d(self):
        """Tests the correctness of the batchnorm3d module.
        The correctness is defined against the functional implementation.
        """
        # 创建一个形状为 (2, 4, 6, 8, 10) 的随机张量 x，数据类型为 float
        x = torch.randn((2, 4, 6, 8, 10), dtype=torch.float)
        # 创建一个 nn.BatchNorm3d 模块的实例 float_mod
        float_mod = torch.nn.BatchNorm3d(4)
        # 将 float_mod 设置为推断模式
        float_mod.training = False

        # 对 x 应用 float_mod 模块，得到参考输出 y_ref
        y_ref = float_mod(x)
        # 对 y_ref 进行量化转换，得到量化的参考输出 quant_ref
        quant_ref = torch.quantize_per_tensor(y_ref, 1.0, 0, dtype=torch.quint8)

        # 创建一个 nnq.BatchNorm3d 模块的实例 quant_mod
        quant_mod = nnq.BatchNorm3d(4)
        # 对输入 x 进行量化转换，得到量化的输入 qx
        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.quint8)
        # 对 qx 应用 quant_mod 模块，得到量化输出 qy
        qy = quant_mod(qx)

        # 断言量化参考输出与量化输出 qy 的整数表示是否相等，否则输出错误信息
        self.assertEqual(quant_ref.int_repr().numpy(), qy.int_repr().numpy(),
                         msg="BatchNorm3d module API failed")

    def _test_batch_norm_serialization(self, get_model, data1, data2):
        # 获取一个模型实例 m1
        m1 = get_model()
        # 设置 m1 的量化配置为默认配置
        m1.qconfig = torch.ao.quantization.default_qconfig
        # 准备模型 m1 进行量化
        mp1 = torch.ao.quantization.prepare(m1)
        # 应用准备后的模型 mp1 到数据 data1
        mp1(data1)
        # 将准备后的模型 mp1 转换为量化模型 mq1
        mq1 = torch.ao.quantization.convert(mp1)
        # 对数据 data2 应用量化模型 mq1，得到参考输出 ref1
        ref1 = mq1(data2)

        # 获取一个模型实例 m2
        m2 = get_model()
        # 设置 m2 的量化配置为默认配置
        m2.qconfig = torch.ao.quantization.default_qconfig
        # 准备模型 m2 进行量化
        mp2 = torch.ao.quantization.prepare(m2)
        # 将准备后的模型 mp2 转换为量化模型 mq2
        mq2 = torch.ao.quantization.convert(mp2)

        # 加载 mq1 的状态字典到 mq2
        mq2.load_state_dict(mq1.state_dict())
        # 对数据 data2 应用量化模型 mq2，得到参考输出 ref2
        ref2 = mq2(data2)

        # 断言 ref1 和 ref2 是否在误差允许范围内相等，否则输出错误信息
        self.assertTrue(torch.allclose(ref1, ref2))

    def test_batch_norm2d_serialization(self):
        # 创建形状为 (2, 4, 6, 8) 的随机数据 data1 和 data2
        data1 = torch.randn(2, 4, 6, 8)
        data2 = torch.randn(2, 4, 6, 8)

        # 定义一个函数 _get_model，返回一个量化后的 nn.Sequential 模型
        def _get_model():
            return nn.Sequential(
                torch.ao.quantization.QuantStub(),
                nn.BatchNorm2d(4),
                torch.ao.quantization.DeQuantStub()
            ).eval()

        # 调用 _test_batch_norm_serialization 测试量化序列化
        self._test_batch_norm_serialization(_get_model, data1, data2)

    def test_batch_norm3d_serialization(self):
        # 创建形状为 (2, 4, 6, 8, 1) 的随机数据 data1 和 data2
        data1 = torch.randn(2, 4, 6, 8, 1)
        data2 = torch.randn(2, 4, 6, 8, 1)

        # 定义一个函数 _get_model，返回一个量化后的 nn.Sequential 模型
        def _get_model():
            return nn.Sequential(
                torch.ao.quantization.QuantStub(),
                nn.BatchNorm3d(4),
                torch.ao.quantization.DeQuantStub()
            ).eval()

        # 调用 _test_batch_norm_serialization 测试量化序列化
        self._test_batch_norm_serialization(_get_model, data1, data2)
    def test_layer_norm(self):
        """Tests the correctness of the layernorm module.
        The correctness is defined against the functional implementation.
        """
        # 定义输入量化参数
        x_scale = 10.0 / 256
        x_zero_point = 0
        # 定义输出量化参数
        y_scale = 5.0 / 256
        y_zero_point = 127

        # 定义输入张量的维度
        dims = (1, 4, 8)

        # 生成随机的输入张量 X，并将其量化为 torch.quint8 类型的张量 qX
        X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.quint8)
        # 对量化后的张量 qX 进行反量化得到 dqX
        dqX = qX.dequantize()

        # 创建一个对 dqX 进行 LayerNorm 的 float 模块
        float_mod = torch.nn.LayerNorm(dqX.size()[1:]).float()
        # 随机初始化 float 模块的权重和偏置
        float_mod.weight = torch.nn.Parameter(torch.rand(*dims[1:]))
        float_mod.bias = torch.nn.Parameter(torch.rand(*dims[1:]))

        # 对 dqX 进行 LayerNorm 操作得到 dqY_ref
        dqY_ref = float_mod(dqX)
        # 将 dqY_ref 量化为 torch.quint8 类型的张量 qY_ref
        qY_ref = torch.quantize_per_tensor(
            dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

        # 创建一个 quantized 的 LayerNorm 模块 quant_mod
        quant_mod = nnq.LayerNorm(
            qX.size()[1:], float_mod.weight, float_mod.bias, y_scale, y_zero_point)
        # 对 qX 应用 quant_mod 得到 qY
        qY = quant_mod(qX)

        # 检查 quantized 后的输出 qY 是否与参考输出 qY_ref 相等
        self.assertEqual(qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                         msg=f"LayerNorm module API failed, qY_ref\n{qY_ref} vs qY\n{qY}")

    def test_group_norm(self):
        """Tests the correctness of the groupnorm module.
        The correctness is defined against the functional implementation.
        """
        # 定义输入量化参数
        x_scale = 10.0 / 256
        x_zero_point = 0
        # 定义输出量化参数
        y_scale = 5.0 / 256
        y_zero_point = 127

        # 定义输入张量的维度
        dims = (1, 4, 8)

        # 生成随机的输入张量 X，并将其量化为 torch.quint8 类型的张量 qX
        X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.quint8)
        # 对量化后的张量 qX 进行反量化得到 dqX
        dqX = qX.dequantize()

        # 创建一个对 dqX 进行 GroupNorm 的 float 模块
        float_mod = torch.nn.GroupNorm(2, 4).float()
        # 随机初始化 float 模块的权重和偏置
        float_mod.weight = torch.nn.Parameter(torch.rand(dims[1]))
        float_mod.bias = torch.nn.Parameter(torch.rand(dims[1]))

        # 对 dqX 进行 GroupNorm 操作得到 dqY_ref
        dqY_ref = float_mod(dqX)
        # 将 dqY_ref 量化为 torch.quint8 类型的张量 qY_ref
        qY_ref = torch.quantize_per_tensor(
            dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

        # 创建一个 quantized 的 GroupNorm 模块 quant_mod
        quant_mod = nnq.GroupNorm(
            2, 2, float_mod.weight, float_mod.bias, y_scale, y_zero_point)
        # 对 qX 应用 quant_mod 得到 qY
        qY = quant_mod(qX)

        # 检查 quantized 后的输出 qY 是否与参考输出 qY_ref 相等
        self.assertEqual(qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                         msg=f"GroupNorm module API failed, qY_ref\n{qY_ref} vs qY\n{qY}")
    def test_instance_norm(self):
        """Tests the correctness of the instancenorm{n}d modules.
        The correctness is defined against the functional implementation.
        """
        # 定义输入张量的缩放因子和零点偏移
        x_scale = 10.0 / 256
        x_zero_point = 0
        # 定义输出张量的缩放因子和零点偏移
        y_scale = 5.0 / 256
        y_zero_point = 127

        # 定义不同维度对应的模块类型的元组列表
        dims_to_modules = [
            ((1, 4, 8), torch.nn.InstanceNorm1d, nnq.InstanceNorm1d),
            ((1, 4, 8, 1), torch.nn.InstanceNorm2d, nnq.InstanceNorm2d),
            ((1, 4, 8, 1, 1), torch.nn.InstanceNorm3d, nnq.InstanceNorm3d),
        ]

        # 对于每个维度对应的模块类型元组
        for dim_to_modules in dims_to_modules:
            dims, float_cls, q_cls = dim_to_modules

            # 创建随机输入张量 X，并量化为 qX
            X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
            qX = torch.quantize_per_tensor(
                X, x_scale, x_zero_point, dtype=torch.quint8)
            dqX = qX.dequantize()

            # 创建浮点数模块并设置权重和偏置
            float_mod = float_cls(dims[1]).float()
            float_mod.weight = torch.nn.Parameter(torch.rand(dims[1]))
            float_mod.bias = torch.nn.Parameter(torch.rand(dims[1]))

            # 计算浮点数模块的输出 dqY_ref
            dqY_ref = float_mod(dqX)
            # 将 dqY_ref 量化为 qY_ref
            qY_ref = torch.quantize_per_tensor(
                dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

            # 创建量化模块并应用于 qX，得到 qY
            quant_mod = q_cls(
                dims[1], float_mod.weight, float_mod.bias, y_scale,
                y_zero_point)
            qY = quant_mod(qX)

            # 断言量化输出 qY 与参考量化输出 qY_ref 相等
            self.assertEqual(
                qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                msg=f"InstanceNorm module API failed, qY_ref\n{qY_ref} vs qY\n{qY}")

    def _test_activation_module_impl(self, name, float_module_class, quantized_module_class, extra_kwargs):
        """Tests the correctness of the ELU module.
        The correctness is defined against the functional implementation.
        """
        # 定义输入张量的缩放因子和零点偏移
        x_scale = 10.0 / 256
        x_zero_point = 0
        # 定义输出张量的缩放因子和零点偏移
        y_scale = 5.0 / 256
        y_zero_point = 127
        # 定义 ELU 模块的 alpha 参数
        alpha = 1.5

        # 定义输入张量的维度
        dims = (1, 4, 8)

        # 创建随机输入张量 X，并量化为 qX
        X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.quint8)
        dqX = qX.dequantize()

        # 创建浮点数模块并根据额外的参数创建实例
        float_mod = float_module_class(**extra_kwargs).float()

        # 计算浮点数模块的输出 dqY_ref
        dqY_ref = float_mod(dqX)
        # 将 dqY_ref 量化为 qY_ref
        qY_ref = torch.quantize_per_tensor(
            dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

        # 创建量化模块并应用于 qX，得到 qY
        quant_mod = quantized_module_class(y_scale, y_zero_point, **extra_kwargs)
        qY = quant_mod(qX)

        # 断言量化输出 qY 与参考量化输出 qY_ref 相等
        self.assertEqual(qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                         msg=f"{name} module API failed, qY_ref\n{qY_ref} vs qY\n{qY}")
    def _test_leaky_relu_serialization(self):
        # 设置原始的缩放因子和零点
        scale_original = 10.0 / 256
        zero_point_original = 1.0

        # 创建一个 LeakyReLU 量化模块对象并获取其状态字典
        quant_mod_original = nnq.LeakyReLU(scale_original, zero_point_original)
        state_dict = quant_mod_original.state_dict()

        # 更新缩放因子和零点，创建一个新的 LeakyReLU 量化模块对象，并加载之前获取的状态字典
        scale_new = 5.0 / 256
        zero_point_new = 2.0
        quant_mod_new = nnq.LeakyReLU(scale_new, zero_point_new)
        quant_mod_new.load_state_dict(state_dict)

        # 断言新旧模块对象的缩放因子和零点是否相等
        self.assertEqual(quant_mod_original.scale, quant_mod_new.scale)
        self.assertEqual(quant_mod_original.zero_point, quant_mod_new.zero_point)

    def test_elu(self):
        """Tests the correctness of the ELU module.
        The correctness is defined against the functional implementation.
        """
        # 调用 _test_activation_module_impl 方法，测试 ELU 模块的正确性
        self._test_activation_module_impl("ELU", nn.ELU, nnq.ELU, {"alpha": 1.5})

    def test_leaky_relu(self):
        # 调用 _test_activation_module_impl 方法，测试 LeakyReLU 模块的正确性
        self._test_activation_module_impl("LeakyReLU", nn.LeakyReLU, nnq.LeakyReLU, {"negative_slope": 0.2})
        # 调用 _test_leaky_relu_serialization 方法，测试 LeakyReLU 模块的序列化和反序列化功能
        self._test_leaky_relu_serialization()

    def test_sigmoid(self):
        # 调用 _test_activation_module_impl 方法，测试 Sigmoid 模块的正确性
        self._test_activation_module_impl("Sigmoid", nn.Sigmoid, nnq.Sigmoid, {})

    def _test_hard_swish_serialization(self):
        # 设置原始的缩放因子和零点
        scale_original = 10.0 / 256
        zero_point_original = 1.0

        # 创建一个 Hardswish 量化模块对象并获取其状态字典
        quant_mod_original = nnq.Hardswish(scale_original, zero_point_original)
        state_dict = quant_mod_original.state_dict()

        # 更新缩放因子和零点，创建一个新的 Hardswish 量化模块对象，并加载之前获取的状态字典
        scale_new = 5.0 / 256
        zero_point_new = 2.0
        quant_mod_new = nnq.Hardswish(scale_new, zero_point_new)
        quant_mod_new.load_state_dict(state_dict)

        # 断言新旧模块对象的缩放因子和零点是否相等
        self.assertEqual(quant_mod_original.scale, quant_mod_new.scale)
        self.assertEqual(quant_mod_original.zero_point, quant_mod_new.zero_point)

    def test_hard_swish(self):
        # 调用 _test_activation_module_impl 方法，测试 Hardswish 模块的正确性
        self._test_activation_module_impl("Hardswish", nn.Hardswish, nnq.Hardswish, {})
        # 调用 _test_hard_swish_serialization 方法，测试 Hardswish 模块的序列化和反序列化功能
        self._test_hard_swish_serialization()

    @given(
        num_embeddings=st.integers(10, 50),
        embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
        set_qconfig=st.booleans(),
    )
    @skipIfNoFBGEMM
    # 定义测试嵌入式 API 的函数，用于测试量化嵌入操作
    def test_embedding_api(self, num_embeddings, embedding_dim, set_qconfig):
        # 随机生成要生成的长度数量
        num_lengths = np.random.randint(1, 6)
        # 随机生成长度数组，作为索引长度，类型为 int32
        lengths = np.random.randint(0, 21, size=num_lengths).astype(np.int32)
        # 计算总的索引数量
        num_indices = np.sum(lengths)
        # 生成随机的索引数组，范围在 [0, num_embeddings) 之间，类型为 int64
        indices = torch.from_numpy(np.random.randint(low=0, high=num_embeddings, size=num_indices, dtype=np.int64))
        # 生成随机的权重数组，形状为 (num_embeddings, embedding_dim)，数据类型为 float32
        weights = torch.from_numpy((np.random.random_sample((num_embeddings, embedding_dim)) + 1).astype(np.float32))

        # 创建默认的浮点量化参数观察器对象
        obs = default_float_qparams_observer()
        # 应用观察器于权重数据
        obs(weights)
        # 计算量化参数
        qparams = obs.calculate_qparams()

        # 定义量化的数据类型列表
        dtypes = [torch.quint4x2, torch.quint8]
        # 定义量化嵌入操作函数列表
        embedding_funcs = [torch.ops.quantized.embedding_4bit, torch.ops.quantized.embedding_byte]

        # 遍历数据类型和量化嵌入函数
        for dtype, embedding_func in zip(dtypes, embedding_funcs):
            # 对权重进行通道量化
            qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=dtype)
            # 创建量化嵌入层对象
            qemb = nnq.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, dtype=dtype)
            # 设置量化权重
            qemb.set_weight(qweight)
            # 应用索引到量化嵌入层
            qemb(indices)

            # 断言量化权重与量化嵌入层的权重相等
            self.assertEqual(qweight, qemb.weight())
            # 获取紧凑参数的权重
            w_packed = qemb._packed_params._packed_weight
            # 调用位量化嵌入操作符
            module_out = qemb(indices)

            # 直接调用位量化嵌入操作符，得到参考输出
            ref = embedding_func(w_packed, indices, pruned_weights=False)
            # 断言量化嵌入层输出与参考输出相等
            self.assertEqual(module_out, ref)
            # 检查嵌入层的序列化
            self.checkEmbeddingSerialization(qemb, num_embeddings, embedding_dim, indices, None, set_qconfig=False,
                                             is_emb_bag=False, dtype=dtype)

    # 使用假设条件定义测试参数
    @given(
        num_embeddings=st.integers(10, 50),
        embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
        num_offsets=st.integers(1, 20),
        set_qconfig=st.booleans(),
    )
    # 如果没有 FBGEMM 库，则跳过该测试
    @skipIfNoFBGEMM
    # 定义一个测试方法，用于测试动态量化的 embedding_bag 模块在 int8 上的执行和序列化
    def test_embedding_bag_api(self, num_embeddings, embedding_dim, num_offsets, set_qconfig):
        r"""Test execution and serialization for dynamic quantized embedding_bag modules on int8
        """

        # 随机生成一个长度在 1 到 5 之间的整数，作为 lengths 数组的长度
        num_lengths = np.random.randint(1, 6)
        # 随机生成 num_lengths 个在 0 到 20 之间的整数，作为长度数组 lengths
        lengths = np.random.randint(0, 21, size=num_lengths).astype(np.int32)
        # 计算出所有 indices 的总数
        num_indices = np.sum(lengths)
        # 从 numpy 数组转换为 PyTorch 的 tensor，生成 num_indices 个在 0 到 num_embeddings 之间的随机整数 indices
        indices = torch.from_numpy(np.random.randint(low=0, high=num_embeddings, size=num_indices, dtype=np.int64))

        # 根据 lengths 计算出 offsets 数组
        offsets = lengths_to_offsets(lengths)
        # 将最后一个 offset 加入 offsets 数组中
        offsets = torch.cat((offsets, torch.tensor([indices.size(0)], dtype=torch.long)), 0)
        # 随机生成一个形状为 (num_embeddings, embedding_dim) 的浮点数数组，作为权重 weights
        weights = torch.from_numpy((np.random.random_sample((num_embeddings, embedding_dim)) + 1).astype(np.float32))

        # 对于每种量化类型进行以下操作
        for qdtype in [torch.quint8, torch.quint4x2]:
            # 创建一个 PerChannelMinMaxObserver 对象，用于观察权重 weights 的量化参数
            obs = PerChannelMinMaxObserver(dtype=qdtype, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
            obs(weights)
            # 计算权重 weights 的量化参数，得到量化的缩放因子和零点
            qparams = obs.calculate_qparams()
            # 将权重 weights 按照计算得到的量化参数 qparams 量化为 8 位
            qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=qdtype)
            # 创建一个动态量化的 EmbeddingBag 模块 qemb
            qemb = nnq.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                    include_last_offset=True, mode='sum', _weight=qweight, dtype=qdtype)
            # 对 qemb 模块进行前向计算，传入 indices 和 offsets
            qemb(indices, offsets)

            # 断言模块的权重是否正确
            self.assertEqual(qweight, qemb.weight())

            # 获取 qemb 模块的 packed weight，通常在量化模块中用于加速计算
            w_packed = qemb._packed_params._packed_weight
            # 对 qemb 进行前向计算，传入 indices 和 offsets
            module_out = qemb(indices, offsets)

            # 直接调用 quantized.embedding_bag_byte 或 quantized.embedding_bag_4bit 运算符
            if qdtype == torch.quint8:
                ref = torch.ops.quantized.embedding_bag_byte(w_packed, indices, offsets, mode=0,
                                                             per_sample_weights=None,
                                                             include_last_offset=True)
            else:
                ref = torch.ops.quantized.embedding_bag_4bit(w_packed, indices, offsets, mode=0,
                                                             per_sample_weights=None,
                                                             include_last_offset=True)

            # 断言模块计算结果与直接调用运算符的结果是否一致
            self.assertEqual(module_out, ref)
            # 检查 embedding 序列化的正确性
            self.checkEmbeddingSerialization(qemb, num_embeddings, embedding_dim, indices,
                                             offsets, set_qconfig, is_emb_bag=True, dtype=qdtype)
    # 定义名为 test_prelu 的测试方法
    def test_prelu(self):
        # 循环遍历参数数量范围为 1 到 9
        for num_parameters in range(1, 10):
            # 创建一个形状为 (4, num_parameters, 4) 的随机张量 x
            x = torch.randn(4, num_parameters, 4)
            # 对张量 x 进行动态量化为 torch.quint8 类型，保持范围不减小
            qx = torch.quantize_per_tensor_dynamic(x, dtype=torch.quint8, reduce_range=False)

            # 创建一个 PReLU 激活函数对象，设置 num_parameters 参数数量
            f_prelu = torch.nn.PReLU(num_parameters=num_parameters)
            # 设置 PReLU 的权重为随机生成的绝对值张量
            f_prelu.weight = torch.nn.Parameter(torch.randn(num_parameters).abs())
            # 配置 PReLU 的量化配置，包括激活函数和权重的默认观察者
            f_prelu.qconfig = torch.ao.quantization.QConfig(
                activation=torch.ao.quantization.default_observer,
                weight=torch.ao.quantization.default_observer,)
            # 通过量化配置设置激活后处理函数
            f_prelu.activation_post_process = f_prelu.qconfig.activation()
            # 对输入 x 应用 PReLU 激活函数并进行激活后处理
            f_prelu.activation_post_process(f_prelu(x))
            # 将浮点类型的 PReLU 转换为量化后的 PReLU
            q_prelu = nnq.PReLU.from_float(f_prelu)
            # 获取权重的观察者对象
            w_obs = f_prelu.qconfig.weight()
            # 应用权重的观察者到 PReLU 的权重
            w_obs(f_prelu.weight)
            # 计算权重的量化参数：缩放因子和零点
            w_scale, w_zp = w_obs.calculate_qparams()
            # 将 PReLU 的权重量化为 torch.quint8 类型
            q_prelu_weight = torch.quantize_per_tensor(
                f_prelu.weight,
                dtype=torch.quint8,
                scale=w_scale,
                zero_point=w_zp
            ).dequantize()

            # 检查权重是否合理
            self.assertEqual(q_prelu.weight.dequantize(), q_prelu_weight)
            # 更新 PReLU 的权重为量化后的值
            f_prelu.weight = torch.nn.Parameter(q_prelu.weight.dequantize())
            # 对输入 qx 进行量化 PReLU 操作
            qy = q_prelu(qx)
            # 对 qx 进行反量化后再次应用 PReLU，得到参考量化结果 qY_ref
            qy_ref = torch.quantize_per_tensor(
                f_prelu(qx.dequantize()), q_prelu.scale, q_prelu.zero_point, dtype=torch.quint8
            )
            # 检查输出是否合理，允许的绝对误差为 0.1，相对误差为 0.1
            self.assertEqual(qy, qy_ref, atol=.1, rtol=.1)

    # 定义名为 test_channel_shuffle 的测试方法
    def test_channel_shuffle(self):
        """Tests the correctness of the ChannelShuffle module.
        """
        # 设置输入张量的量化参数：缩放因子和零点
        x_scale = 10.0 / 256
        x_zero_point = 1
        # 设置输出张量的量化参数与输入相同
        y_scale = x_scale
        y_zero_point = x_zero_point

        # 设置输入张量的维度为 (1, 4, 4, 8)，分组数为 2
        dims = (1, 4, 4, 8)
        groups = 2

        # 创建一个形状为 dims 的随机浮点张量 X
        X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
        # 对张量 X 进行量化为 torch.quint8 类型
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.quint8)
        # 对量化后的张量 qX 进行反量化
        dqX = qX.dequantize()

        # 创建一个浮点类型的 ChannelShuffle 模块
        float_mod = torch.nn.ChannelShuffle(groups).float()
        # 对反量化后的输入 dqX 应用浮点类型的 ChannelShuffle 模块，得到参考反量化输出 dqY_ref
        dqY_ref = float_mod(dqX)
        # 对参考反量化输出 dqY_ref 进行量化为 torch.quint8 类型
        qY_ref = torch.quantize_per_tensor(
            dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

        # 创建一个量化类型的 ChannelShuffle 模块
        quant_mod = torch.nn.ChannelShuffle(groups)
        # 对输入 qX 应用量化类型的 ChannelShuffle 模块
        qY = quant_mod(qX)

        # 检查量化输出 qY 是否与参考量化输出 qY_ref 相等
        self.assertEqual(qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                         msg=f"ChannelShuffle module API failed, qY_ref\n{qY_ref} vs qY\n{qY}")

    @skipIfNoONEDNN
    # 定义一个测试方法，用于测试 nn.intrinsic.quantized.linear_leaky_relu 的 API 功能
    def test_linear_leaky_relu(self):
        """test API functionality for nn.intrinsic.quantized.linear_leaky_relu"""
        # 使用 'onednn' 引擎覆盖当前的量化引擎
        with override_quantized_engine('onednn'):
            # 生成测试参数的所有可能组合
            options = itertools.product(
                [1, 5],        # 批量大小
                [16, 32],      # 输入特征数
                [4, 8],        # 输出特征数
                [True, False], # 是否使用偏置
                [True, False], # 是否逐通道量化
                [0.01, 0.05])  # 负斜率参数
            # 遍历所有参数组合
            for (batch_size, in_features, out_features, use_bias,
                 per_channel, neg_slope) in options:
                # 调用内部方法 _test_linear_api_impl 进行线性 Leaky ReLU 操作的测试
                self._test_linear_api_impl(
                    nniq.LinearLeakyReLU, 'QuantizedLinearLeakyReLU',
                    torch.ops.quantized.linear_leaky_relu,
                    batch_size, in_features, out_features, use_bias,
                    per_channel, negative_slope=neg_slope)

    # 如果没有 ONEDNN 支持，则跳过该测试
    @skipIfNoONEDNN
    def test_linear_tanh(self):
        """test API functionality for nn.intrinsic.quantized.linear_tanh"""
        # 使用 'onednn' 引擎覆盖当前的量化引擎
        with override_quantized_engine('onednn'):
            # 生成测试参数的所有可能组合
            options = itertools.product(
                [1, 5],        # 批量大小
                [16, 32],      # 输入特征数
                [4, 8],        # 输出特征数
                [True, False], # 是否使用偏置
                [True, False]) # 是否逐通道量化
            # 遍历所有参数组合
            for (batch_size, in_features, out_features, use_bias,
                 per_channel) in options:
                # 调用内部方法 _test_linear_api_impl 进行线性 Tanh 操作的测试
                self._test_linear_api_impl(
                    nniq.LinearTanh, 'QuantizedLinearTanh',
                    torch.ops.quantized.linear_tanh,
                    batch_size, in_features, out_features, use_bias,
                    per_channel)
# 定义一个名为 TestDynamicQuantizedModule 的测试类，继承自 QuantizationTestCase
class TestDynamicQuantizedModule(QuantizationTestCase):

    # 覆盖装饰器，指示以下函数覆盖父类中相同名称的方法
    @override_qengines
    # 定义测试动态 Conv1d 的方法
    def test_dynamic_conv1d(self):
        # 获取量化后的 Conv1d 模块
        q_mod = torch.ao.nn.quantized.Conv1d
        # 获取动态量化的 Conv1d 模块
        dq_mod = torch.ao.nn.quantized.dynamic.Conv1d
        # 设置维度为 3
        dim = 3
        # 设置数据类型为 quint8
        dtype = torch.quint8

        # 针对是否包含偏置项进行遍历测试
        for bias in [True, False]:
            # 调用内部方法 _test_qconv_impl，传入相关参数进行测试
            self._test_qconv_impl(q_mod, dq_mod, dim, dtype, bias)

    @override_qengines
    # 定义测试动态 Conv2d 的方法
    def test_dynamic_conv2d(self):
        # 获取量化后的 Conv2d 模块
        q_mod = torch.ao.nn.quantized.Conv2d
        # 获取动态量化的 Conv2d 模块
        dq_mod = torch.ao.nn.quantized.dynamic.Conv2d
        # 设置维度为 4
        dim = 4
        # 设置数据类型为 quint8
        dtype = torch.quint8

        # 针对是否包含偏置项进行遍历测试
        for bias in [True, False]:
            # 调用内部方法 _test_qconv_impl，传入相关参数进行测试
            self._test_qconv_impl(q_mod, dq_mod, dim, dtype, bias)

    @override_qengines
    # 定义测试动态 Conv3d 的方法
    def test_dynamic_conv3d(self):
        # 获取量化后的 Conv3d 模块
        q_mod = torch.ao.nn.quantized.Conv3d
        # 获取动态量化的 Conv3d 模块
        dq_mod = torch.ao.nn.quantized.dynamic.Conv3d
        # 设置维度为 5
        dim = 5
        # 设置数据类型为 quint8
        dtype = torch.quint8

        # 如果当前量化引擎是 qnnpack，则直接返回，因为 qnnpack 不支持解包 Conv3d
        if qengine_is_qnnpack():
            return

        # 针对是否包含偏置项进行遍历测试
        for bias in [True, False]:
            # 调用内部方法 _test_qconv_impl，传入相关参数进行测试
            self._test_qconv_impl(q_mod, dq_mod, dim, dtype, bias)

    @override_qengines
    # 定义测试动态 ConvTranspose1d 的方法
    def test_dynamic_convtranspose1d(self):
        # 获取量化后的 ConvTranspose1d 模块
        q_mod = torch.ao.nn.quantized.ConvTranspose1d
        # 获取动态量化的 ConvTranspose1d 模块
        dq_mod = torch.ao.nn.quantized.dynamic.ConvTranspose1d
        # 设置维度为 3
        dim = 3
        # 设置数据类型为 quint8
        dtype = torch.quint8

        # 针对是否包含偏置项进行遍历测试
        for bias in [True, False]:
            # 调用内部方法 _test_qconv_impl，传入相关参数进行测试
            self._test_qconv_impl(q_mod, dq_mod, dim, dtype, bias)

    @override_qengines
    # 定义测试动态 ConvTranspose2d 的方法
    def test_dynamic_convtranspose2d(self):
        # 获取量化后的 ConvTranspose2d 模块
        q_mod = torch.ao.nn.quantized.ConvTranspose2d
        # 获取动态量化的 ConvTranspose2d 模块
        dq_mod = torch.ao.nn.quantized.dynamic.ConvTranspose2d
        # 设置维度为 4
        dim = 4
        # 设置数据类型为 quint8
        dtype = torch.quint8

        # 针对是否包含偏置项进行遍历测试
        for bias in [True, False]:
            # 调用内部方法 _test_qconv_impl，传入相关参数进行测试
            self._test_qconv_impl(q_mod, dq_mod, dim, dtype, bias)

    @override_qengines
    # 定义测试动态 ConvTranspose3d 的方法
    def test_dynamic_convtranspose3d(self):
        # 获取量化后的 ConvTranspose3d 模块
        q_mod = torch.ao.nn.quantized.ConvTranspose3d
        # 获取动态量化的 ConvTranspose3d 模块
        dq_mod = torch.ao.nn.quantized.dynamic.ConvTranspose3d
        # 设置维度为 5
        dim = 5
        # 设置数据类型为 quint8
        dtype = torch.quint8

        # 如果当前量化引擎是 qnnpack，则直接返回，因为 qnnpack 不支持解包 Conv3d
        if qengine_is_qnnpack():
            return

        # 针对是否包含偏置项进行遍历测试
        for bias in [True, False]:
            # 调用内部方法 _test_qconv_impl，传入相关参数进行测试
            self._test_qconv_impl(q_mod, dq_mod, dim, dtype, bias)

    # 给定装饰器，定义参数化测试用例
    @given(
        batch_size=st.integers(1, 5),  # 批量大小在 1 到 5 之间取整数
        in_features=st.integers(16, 32),  # 输入特征在 16 到 32 之间取整数
        out_features=st.integers(4, 8),  # 输出特征在 4 到 8 之间取整数
        use_bias=st.booleans(),  # 是否使用偏置项
        use_default_observer=st.booleans(),  # 是否使用默认观察器
    )
    # 覆盖装饰器，指示以下函数覆盖父类中相同名称的方法
    @override_qengines
    # 给定装饰器，定义参数化测试用例
    @given(
        dtype=st.sampled_from([torch.qint8, torch.float16]),  # 数据类型从 qint8 和 float16 中抽样
        bidirectional=st.booleans(),  # 是否双向
    )
    # 覆盖装饰器，指示以下函数覆盖父类中相同名称的方法
    @override_qengines
    # 覆盖装饰器，指示以下函数覆盖父类中相同名称的方法
    @override_qengines
    def test_gru_api(self):
        r"""Test execution and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        # 检查模块是否与操作的数值匹配，并确保模块可以用于所有引擎和数据类型的实例化

        for dtype in [torch.qint8, torch.float16]:
            if dtype == torch.float16 and torch.backends.quantized.engine in ("qnnpack", "onednn"):
                # 对于 qnnpack 或 onednn 引擎，不支持 fp16 动态量化
                continue

            # 测试默认实例化
            seq_len = 4
            batch = 2
            input_size = 3
            hidden_size = 7
            num_layers = 2
            bias = True
            bidirectional = False

            # 创建随机输入张量 x 和隐藏状态张量 h
            x = torch.rand(seq_len, batch, input_size)
            h = torch.rand(num_layers * (bidirectional + 1), batch, hidden_size)

            # 使用动态量化的 GRU 模型创建 cell_dq 对象
            cell_dq = torch.ao.nn.quantized.dynamic.GRU(input_size=input_size,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers,
                                                        bias=bias,
                                                        batch_first=False,
                                                        dropout=0.0,
                                                        bidirectional=bidirectional,
                                                        dtype=dtype)

            # 获取所有权重参数
            _all_params = ([m.param for m in cell_dq._all_weight_values])

            # 调用 torch.quantized_gru 进行前向计算
            result = torch.quantized_gru(x,
                                         h,
                                         _all_params,
                                         cell_dq.bias,
                                         cell_dq.num_layers,
                                         float(cell_dq.dropout),
                                         False,
                                         bidirectional,
                                         False)

            # 调用 cell_dq 进行前向计算
            y, h = cell_dq(x, h)

            # 断言检查结果
            self.assertEqual(result[0], y, msg="GRU module API failed")
            self.assertEqual(result[1], h, msg="GRU module API failed")

    @given(
        dtype=st.sampled_from([torch.qint8, torch.float16]),
    )
    @override_qengines
    def test_cell_api(self, dtype):
        r"""Test execution and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        # 检查模块是否与操作的数值匹配，并确保可以为所有引擎和数据类型实例化模块
        batch = 7  # 定义批量大小为7
        input_size = 3  # 定义输入大小为3
        hidden_size = 7  # 定义隐藏状态大小为7
        bias = True  # 启用偏置项

        x = torch.rand(batch, input_size)  # 创建随机输入张量 x
        h = torch.rand(batch, hidden_size)  # 创建随机隐藏状态张量 h

        # 定义不同类型的动态量化 LSTM/GRU/RNNCell 及其对应初始状态
        cell_dict = {'LSTMCell': torch.ao.nn.quantized.dynamic.LSTMCell,
                     'GRUCell': torch.ao.nn.quantized.dynamic.GRUCell,
                     'RNNTanh': torch.ao.nn.quantized.dynamic.RNNCell,
                     'RNNReLU': torch.ao.nn.quantized.dynamic.RNNCell
                     }
        state = {'LSTMCell': (h, h),
                 'GRUCell': h,
                 'RNNTanh': h,
                 'RNNReLU': h}

        # 定义不同类型的动态量化函数及其对应的操作
        qfn_dict = {'LSTMCell': torch.ops.quantized.quantized_lstm_cell_dynamic,
                    'GRUCell': torch.ops.quantized.quantized_gru_cell_dynamic,
                    'RNNTanh': torch.ops.quantized.quantized_rnn_tanh_cell_dynamic,
                    'RNNReLU': torch.ops.quantized.quantized_rnn_relu_cell_dynamic}

        # 遍历所有的 RNN 类型
        for rnn_type in cell_dict.keys():
            if not (dtype == torch.float16 and torch.backends.quantized.engine in ("qnnpack", "onednn")):
                # 如果数据类型不是 torch.float16 或者量化引擎不是 qnnpack 或 onednn，则跳过
                # 因为 fp16 动态量化不支持 qnnpack 或 onednn 引擎
                kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'bias': bias, 'dtype': dtype}
                if rnn_type == 'RNNReLU':
                    kwargs['nonlinearity'] = "relu"
                elif rnn_type == 'RNNTanh':
                    kwargs['nonlinearity'] = "tanh"

                # 根据参数实例化对应类型的动态量化单元
                cell_dq = cell_dict[rnn_type](**kwargs)

                # 调用对应的动态量化函数处理输入和状态，得到结果
                result = qfn_dict[rnn_type](x, state[rnn_type],
                                            cell_dq._packed_weight_ih, cell_dq._packed_weight_hh,
                                            cell_dq.bias_ih, cell_dq.bias_hh)

                # 调用模块的 __call__ 方法处理输入和状态，得到结果
                result_module = cell_dq(x, state[rnn_type])

                # 检查结果的一致性，确保模块的 API 调用正确
                self.assertEqual(result[0], result_module[0], msg="RNNCell module API failed")
                self.assertEqual(result[1], result_module[1], msg="RNNCell module API failed")

                # 检查模型的权重和偏置的 API
                weight_keys = ['weight_ih', 'weight_hh']
                bias_keys = ['bias_ih', 'bias_hh']
                self.check_eager_serialization(cell_dq, cell_dict[rnn_type](**kwargs), [x])
                self.check_weight_bias_api(cell_dq, weight_keys, bias_keys)
# 定义一个测试类 TestReferenceQuantizedModule，继承自 QuantizationTestCase，用于测试量化模块的功能
class TestReferenceQuantizedModule(QuantizationTestCase):

    # 定义一个私有方法 _quant_dequant_weight，用于量化和反量化权重
    def _quant_dequant_weight(self, weight, weight_qparams):
        # 从 weight_qparams 中获取量化方案
        qscheme = weight_qparams["qscheme"]
        # 从 weight_qparams 中获取量化的比例因子
        scale = weight_qparams["scale"]
        # 从 weight_qparams 中获取量化的零点
        zero_point = weight_qparams["zero_point"]
        # 从 weight_qparams 中获取量化后的数据类型
        dtype = weight_qparams["dtype"]
        
        # 根据量化方案选择不同的量化方法
        if qscheme == torch.per_tensor_affine:
            # 对权重进行张量级别的仿射量化
            weight = torch.quantize_per_tensor(weight, scale, zero_point, dtype)
        else:
            # 对权重进行通道级别的仿射量化
            # 从 weight_qparams 中获取量化的轴（通道）
            axis = weight_qparams["axis"]
            weight = torch.quantize_per_channel(weight, scale, zero_point, axis, dtype)
        
        # 对量化后的权重进行反量化操作，恢复到浮点数表示
        weight = weight.dequantize()
        
        # 返回反量化后的权重
        return weight

    # TODO: add tests for conv and linear
    # TODO: 添加卷积和线性层的测试（未完成）
    def test_rnn_cell(self):
        """ 检查 RNN 单元的参考量化模块是否具有正确的数值
        这包括 LSTMCell、GRUCell、RNNCell
        """
        batch = 7  # 批处理大小
        input_size = 3  # 输入大小
        hidden_size = 7  # 隐藏层大小
        bias = True  # 是否包含偏置

        x = torch.rand(batch, input_size)  # 随机生成输入张量 x
        h = torch.rand(batch, hidden_size)  # 随机生成隐藏状态张量 h
        
        # 定义包含不同 RNN 单元的字典
        cell_dict = {'LSTMCell': torch.nn.LSTMCell,
                     'GRUCell': torch.nn.GRUCell,
                     'RNNTanh': torch.nn.RNNCell,
                     'RNNReLU': torch.nn.RNNCell
                     }
        
        # 初始化各种 RNN 单元的初始状态
        state = {'LSTMCell': (h, h),
                 'GRUCell': h,
                 'RNNTanh': h,
                 'RNNReLU': h}
        
        # 定义包含量化 RNN 单元的字典
        qfn_dict = {'LSTMCell': nnqr.LSTMCell,
                    'GRUCell': nnqr.GRUCell,
                    'RNNTanh': nnqr.RNNCell,
                    'RNNReLU': nnqr.RNNCell}

        # 遍历每种 RNN 单元类型
        for rnn_type in cell_dict.keys():
            kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'bias': bias}
            if rnn_type == 'RNNReLU':
                kwargs['nonlinearity'] = "relu"
            elif rnn_type == 'RNNTanh':
                kwargs['nonlinearity'] = "tanh"

            # 创建基于浮点数的 RNN 单元模块
            fp_cell = cell_dict[rnn_type](**kwargs)
            
            # 初始化参考量化 RNN 单元模块
            weight_qparams = {
                'qscheme': torch.per_tensor_affine,
                'dtype': torch.quint8,
                'scale': 2.0,
                'zero_point': 5
            }
            weight_qparams_dict = {
                "weight_ih": weight_qparams,
                "weight_hh": weight_qparams,
                "is_decomposed": False,
            }
            ref_kwargs = kwargs.copy()
            ref_kwargs["weight_qparams_dict"] = weight_qparams_dict
            ref_cell = qfn_dict[rnn_type](**ref_kwargs)
            
            # 从浮点数的 RNN 单元模块复制权重到量化 RNN 单元模块
            ref_cell.weight_ih = fp_cell.weight_ih
            ref_cell.weight_hh = fp_cell.weight_hh
            ref_cell.bias_ih = fp_cell.bias_ih
            ref_cell.bias_hh = fp_cell.bias_hh

            # 运行参考量化 RNN 单元模块
            ref_res = ref_cell(x, state[rnn_type])

            # 将浮点数的权重进行量化和反量化处理，然后重新分配到权重
            fp_cell.weight_ih = torch.nn.Parameter(self._quant_dequant_weight(fp_cell.weight_ih, weight_qparams_dict["weight_ih"]))
            fp_cell.weight_hh = torch.nn.Parameter(self._quant_dequant_weight(fp_cell.weight_hh, weight_qparams_dict["weight_hh"]))
            fp_res = fp_cell(x, state[rnn_type])
            
            # 断言参考结果和浮点数结果一致，用于验证 RNN 单元模块的 API 是否正确
            self.assertEqual(ref_res[0], fp_res[0], msg="RNNCell module API failed")
            self.assertEqual(ref_res[1], fp_res[1], msg="RNNCell module API failed")
    def test_rnn(self):
        """检查 rnn 参考量化模块的数值正确性
        这包括 LSTM
        """
        # 设置序列长度、批量大小、输入维度、隐藏层维度和层数等参数
        seq_len = 4
        batch = 2
        input_size = 3
        hidden_size = 7
        num_layers = 2
        bias = True
        
        # 循环测试双向和单向的情况
        for bidirectional in [True, False]:
            # 生成随机输入张量 x，形状为 (seq_len, batch, input_size)
            x = torch.randn(seq_len, batch, input_size)
            # 生成随机隐藏状态张量 h 和细胞状态张量 c
            h = torch.randn(num_layers * (bidirectional + 1), batch, hidden_size)
            c = torch.randn(num_layers * (bidirectional + 1), batch, hidden_size)
            
            # 创建一个原始的 FP32 LSTM 模块
            fp32_rnn = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=False,
                dropout=0.0,
                bidirectional=bidirectional)
            
            # 初始化参考的量化 LSTM 模块
            weight_qparams = {
                "qscheme": torch.per_tensor_affine,
                "dtype": torch.qint8,
                "scale": 2.0,
                "zero_point": 5
            }
            # 创建权重量化参数字典，用于量化模块的权重
            weight_qparams_dict = {key: weight_qparams for key in fp32_rnn._flat_weights_names if key.startswith("weight")}
            weight_qparams_dict["is_decomposed"] = False
            ref_rnn = nnqr.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=False,
                dropout=0.0,
                bidirectional=bidirectional,
                weight_qparams_dict=weight_qparams_dict)
            
            # 将原始 FP32 模块的权重复制到参考量化模块中
            for wn in fp32_rnn._flat_weights_names:
                setattr(ref_rnn, wn, copy.deepcopy(getattr(fp32_rnn, wn)))
            
            # 深拷贝原始 FP32 模块的权重到参考量化模块中
            ref_rnn._flat_weights = copy.deepcopy(fp32_rnn._flat_weights)
            
            # 对 FP32 模块的权重进行量化和反量化处理
            flat_weights = []
            for wn in fp32_rnn._flat_weights_names:
                if wn.startswith("weight"):
                    weight = self._quant_dequant_weight(getattr(fp32_rnn, wn), weight_qparams)
                else:
                    weight = getattr(fp32_rnn, wn)
                flat_weights.append(weight)
            fp32_rnn._flat_weights = flat_weights
            
            # 分别使用 FP32 和参考量化模块计算结果
            fp32_res = fp32_rnn(x, (h, c))
            ref_res = ref_rnn(x, (h, c))
            
            # 断言 FP32 和参考量化模块的结果应该相等
            self.assertEqual(fp32_res, ref_res)
    def test_sparse(self):
        """测试稀疏表示和嵌入袋（Embedding and EmbeddingBag）"""

        num_embeddings = 10
        embedding_dim = 3
        # 定义嵌入输入
        ex = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

        # 定义嵌入袋输入
        ebx = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
        offsets = torch.tensor([0, 4], dtype=torch.long)

        fp_to_ref = {
            nn.Embedding: (nnqr.Embedding, (ex,)),  # 嵌入类映射到参考实现及其参数
            nn.EmbeddingBag: (nnqr.EmbeddingBag, (ebx, offsets)),  # 嵌入袋类映射到参考实现及其参数
        }

        per_tensor_weight_qparams = {
            "qscheme": torch.per_tensor_affine,
            "dtype": torch.quint8,
            "scale": 2.0,
            "zero_point": 5,
            "is_decomposed": False,
        }

        per_channel_weight_qparams = {
            "qscheme": torch.per_channel_affine,
            "dtype": torch.quint8,
            "scale": torch.randn(10),
            "zero_point": torch.randint(0, 255, (10,)),
            "axis": 0,
            "is_decomposed": False,
        }

        per_channel_weight_qparams_quint4x2 = {
            "qscheme": torch.per_channel_affine_float_qparams,
            "dtype": torch.quint4x2,
            "scale": torch.randn(10),
            "zero_point": torch.randint(0, 255, (10,)),
            "axis": 0,
            "is_decomposed": False,
        }

        weight_qparams_options = [
            per_tensor_weight_qparams,
            per_channel_weight_qparams,
            per_channel_weight_qparams_quint4x2,
        ]
        # 遍历嵌入类和权重参数组合的笛卡尔积
        for fp_cls, weight_qparams in itertools.product([nn.Embedding, nn.EmbeddingBag], weight_qparams_options):
            # TODO: torch.quint4x2 在 quantize_per_channel 中不受支持，需要添加支持
            if weight_qparams["dtype"] == torch.quint4x2:
                continue
            ref_cls, args = fp_to_ref[fp_cls]

            # 创建浮点32位嵌入对象
            fp32_embedding = fp_cls(num_embeddings, embedding_dim)

            # 创建参考嵌入对象，应用给定的权重量化参数
            ref_embedding = ref_cls(num_embeddings, embedding_dim, weight_qparams=weight_qparams)
            ref_embedding.weight = fp32_embedding.weight

            # 对浮点32位模块的权重进行量化和反量化
            fp32_embedding.weight = torch.nn.Parameter(self._quant_dequant_weight(fp32_embedding.weight, weight_qparams))

            # 计算浮点32位和参考嵌入的输出结果
            fp32_res = fp32_embedding(*args)
            ref_res = ref_embedding(*args)
            self.assertEqual(fp32_res, ref_res)
    # 定义一个测试方法，验证参考线性层对权重的自定义 qmin/qmax 的响应
    def test_linear_decomposed_weight_custom_qmin_qmax(self):
        """Verify that reference Linear respects custom qmin/qmax for weight
        """
        # 创建一个包含2个输入和2个输出的全连接层，初始为32位浮点数
        linear_fp32 = torch.nn.Linear(2, 2)
        # 获取默认的对称量化配置
        qconfig = torch.ao.quantization.default_symmetric_qnnpack_qconfig
        # 调用量化配置中的权重观察器，返回观察结果
        w_obs = qconfig.weight()
        # 断言权重的量化下限为-127
        self.assertTrue(w_obs.quant_min == -127)
        # 断言权重的量化上限为127
        self.assertTrue(w_obs.quant_max == 127)
        # 将观察到的量化配置应用于全连接层的权重
        w_obs(linear_fp32.weight)
        # 获取权重的量化参数字典
        weight_qparams = torch.ao.quantization.utils.get_qparam_dict(w_obs)
        # 将权重的分解状态标记为 True
        weight_qparams["is_decomposed"] = True
        # 使用权重量化参数初始化参考量化后的线性层
        linear_ref = nnqr.Linear.from_float(linear_fp32, weight_qparams)
        # 对参考量化后的线性层进行符号追踪
        linear_ref_traced = torch.fx.symbolic_trace(linear_ref)

        # 验证权重量化和反量化操作的 qmin/qmax 参数是否正确从观察器中取得
        found = 0
        for n in linear_ref_traced.graph.nodes:
            # 如果节点不是函数调用，则继续下一个节点
            if n.op != 'call_function':
                continue
            # 如果目标函数是量化或反量化函数之一
            if n.target in (
                torch.ops.quantized_decomposed.quantize_per_tensor,
                torch.ops.quantized_decomposed.dequantize_per_tensor,
            ):
                # 获取函数调用的参数，其中 qmin 和 qmax 是第四和第五个参数
                _0, _1, _2, qmin, qmax, _5 = n.args
                # 断言 qmin 和 qmax 是否等于 -127 和 127
                self.assertTrue(qmin == -127)
                self.assertTrue(qmax == 127)
                found += 1
        # 断言找到了两次符合条件的函数调用
        self.assertTrue(found == 2)
```
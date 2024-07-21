# `.\pytorch\test\quantization\eager\test_quantize_eager_ptq.py`

```py
# Owner(s): ["oncall: quantization"]

# 引入 PyTorch 相关模块
import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
from torch.nn.utils.rnn import PackedSequence

# 引入量化相关模块和函数
from torch.ao.quantization import (
    quantize,
    prepare,
    convert,
    prepare_qat,
    quantize_dynamic,
    QuantWrapper,
    QuantStub,
    DeQuantStub,
    default_qconfig,
    default_dynamic_qconfig,
    per_channel_dynamic_qconfig,
    float16_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    float_qparams_weight_only_qconfig_4bit,
    FixedQParamsObserver,
    PerChannelMinMaxObserver,
    default_dynamic_quant_observer,
    default_weight_observer,
    QConfig,
)

# 引入用于量化测试的常规模型和工具函数
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    AnnotatedSingleLayerLinearModel,
    QuantStubModel,
    ModelWithFunctionals,
    SingleLayerLinearDynamicModel,
    TwoLayerLinearModel,
    NestedModel,
    ResNetBase,
    RNNDynamicModel,
    RNNCellDynamicModel,
    ActivationsTestModel,
    NormalizationTestModel,
    test_only_eval_fn,
    prepare_dynamic,
    convert_dynamic,
    skipIfNoFBGEMM,
    EmbeddingBagModule,
    EmbeddingModule,
    EmbeddingWithStaticLinear,
    LinearReluLinearModel,
)

# 引入用于量化测试的带注释的模型
from torch.testing._internal.common_quantization import (
    AnnotatedTwoLayerLinearModel,
    AnnotatedNestedModel,
    AnnotatedSubNestedModel,
    AnnotatedCustomConfigNestedModel,
    AnnotatedSkipQuantModel,
)

# 引入用于量化测试的量化引擎相关函数
from torch.testing._internal.common_quantized import (
    override_quantized_engine,
    supported_qengines,
    override_qengines,
)

# 引入 Hypothesis 相关模块和函数
from hypothesis import given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu

# 禁用测试超时
hu.assert_deadline_disabled()

# 引入标准库
from typing import Tuple
import numpy as np

# 定义测试类 TestQuantizeEagerOps，继承自 QuantizationTestCase
class TestQuantizeEagerOps(QuantizationTestCase):
    @override_qengines
    def _test_reference_module_impl(self,
                                    float_module_class,
                                    quantized_module_class,
                                    extra_module_kwargs,
                                    input_size):
        # 定义一个内部测试函数，用于测试参考模块实现的量化效果
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化浮点数模块
                self.conv = float_module_class(**extra_module_kwargs)
                # 插入量化伪量化的占位符
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                # 量化输入
                x = self.quant(x)
                # 应用卷积操作
                x = self.conv(x)
                # 反量化输出
                x = self.dequant(x)
                return x

        class RefM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化浮点数模块
                self.conv = float_module_class(**extra_module_kwargs)
                # 插入多个量化伪量化的占位符
                self.quant1 = QuantStub()
                self.dequant1 = DeQuantStub()
                self.quant2 = QuantStub()
                self.dequant2 = DeQuantStub()

            def forward(self, x):
                # 分别量化和反量化输入
                x = self.quant1(x)
                x = self.dequant1(x)
                # 应用卷积操作
                x = self.conv(x)
                # 再次量化和反量化输出
                x = self.quant2(x)
                x = self.dequant2(x)
                return x

        # 获取当前的量化引擎并检查支持性
        qengine = torch.backends.quantized.engine
        if qengine not in supported_qengines or qengine == 'qnnpack':
            return   # 如果量化引擎不受支持或为qnnpack，则直接返回

        # 生成随机数据
        data = torch.randn(*input_size, dtype=torch.float)
        original_m = M()
        original_ref_m = RefM()

        # 复制浮点模型的权重和偏置到参考模型
        original_ref_m.conv.weight = torch.nn.Parameter(original_m.conv.weight.detach())
        original_ref_m.conv.bias = torch.nn.Parameter(original_m.conv.bias.detach())

        # 设置浮点模型的量化配置
        original_m.qconfig = torch.ao.quantization.default_qconfig

        # 准备模型以便量化
        m = prepare(original_m)
        # 进行校准
        m(data)
        # 转换为量化模型
        m = convert(m)
        # 检查模型是否正确量化
        self.assertEqual(type(m.quant), nnq.Quantize)
        self.assertEqual(type(m.conv), quantized_module_class)
        self.assertEqual(type(m.dequant), nnq.DeQuantize)
        # 获取量化模型的输出结果
        res = m(data)

        # 量化参考模型
        original_ref_m.eval()
        original_ref_m.qconfig = torch.ao.quantization.default_qconfig

        # 准备参考模型以便量化
        ref_m = prepare(original_ref_m)
        ref_m(data)
        # 将参考模型转换为量化模型，标记为参考
        ref_m = convert(ref_m, is_reference=True)
        # 获取参考模型的输出结果
        ref_res = ref_m(data)
        # 检查量化模型和参考模型输出是否一致
        self.assertEqual(res, ref_res)

    def test_conv_1d(self):
        # 测试一维卷积的量化效果
        self._test_reference_module_impl(
            nn.Conv1d,
            nnq.Conv1d,
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},
            (16, 1, 1)
        )

    def test_conv_2d(self):
        # 测试二维卷积的量化效果
        self._test_reference_module_impl(
            nn.Conv2d,
            nnq.Conv2d,
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},
            (16, 1, 10, 10)
        )
    # 定义测试方法，用于测试 3D 卷积操作
    def test_conv_3d(self):
        # 调用内部方法 _test_reference_module_impl 进行测试，比较 nn.Conv3d 和 nnq.Conv3d 的输出结果
        self._test_reference_module_impl(
            nn.Conv3d,  # 标准的 3D 卷积模块
            nnq.Conv3d,  # 量化后的 3D 卷积模块
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},  # 模块参数：输入通道数、输出通道数、核大小
            (16, 1, 10, 10, 10)  # 输入数据的形状：批大小 16，通道数 1，尺寸为 10x10x10
        )

    # 定义测试方法，用于测试 1D 转置卷积操作
    def test_conv_transpose_1d(self):
        # 调用内部方法 _test_reference_module_impl 进行测试，比较 nn.ConvTranspose1d 和 nnq.ConvTranspose1d 的输出结果
        self._test_reference_module_impl(
            nn.ConvTranspose1d,  # 标准的 1D 转置卷积模块
            nnq.ConvTranspose1d,  # 量化后的 1D 转置卷积模块
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},  # 模块参数：输入通道数、输出通道数、核大小
            (16, 1, 1)  # 输入数据的形状：批大小 16，通道数 1，尺寸为 1
        )

    # 定义测试方法，用于测试 2D 转置卷积操作
    def test_conv_transpose_2d(self):
        # 调用内部方法 _test_reference_module_impl 进行测试，比较 nn.ConvTranspose2d 和 nnq.ConvTranspose2d 的输出结果
        self._test_reference_module_impl(
            nn.ConvTranspose2d,  # 标准的 2D 转置卷积模块
            nnq.ConvTranspose2d,  # 量化后的 2D 转置卷积模块
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},  # 模块参数：输入通道数、输出通道数、核大小
            (16, 1, 10, 10)  # 输入数据的形状：批大小 16，通道数 1，尺寸为 10x10
        )

    # 定义测试方法，用于测试 3D 转置卷积操作
    def test_conv_transpose_3d(self):
        # 调用内部方法 _test_reference_module_impl 进行测试，比较 nn.ConvTranspose3d 和 nnq.ConvTranspose3d 的输出结果
        self._test_reference_module_impl(
            nn.ConvTranspose3d,  # 标准的 3D 转置卷积模块
            nnq.ConvTranspose3d,  # 量化后的 3D 转置卷积模块
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},  # 模块参数：输入通道数、输出通道数、核大小
            (16, 1, 10, 10, 10)  # 输入数据的形状：批大小 16，通道数 1，尺寸为 10x10x10
        )

    # 定义测试方法，用于测试线性层操作
    def test_linear(self):
        # 调用内部方法 _test_reference_module_impl 进行测试，比较 nn.Linear 和 nnq.Linear 的输出结果
        self._test_reference_module_impl(
            nn.Linear,  # 标准的线性层模块
            nnq.Linear,  # 量化后的线性层模块
            {'in_features': 5, 'out_features': 10},  # 模块参数：输入特征数、输出特征数
            (16, 5)  # 输入数据的形状：批大小 16，特征数 5
        )

    # 装饰器，覆盖量化引擎设置
    @override_qengines
    def test_int16_reference_module(self):
        # 定义一个名为 RefM 的内部类，继承自 torch.nn.Module
        class RefM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个 1x1 的反卷积层
                self.conv = nn.ConvTranspose2d(1, 1, 1)
                # 定义量化和反量化的占位符
                self.quant1 = QuantStub()
                self.dequant1 = DeQuantStub()
                self.quant2 = QuantStub()
                self.dequant2 = DeQuantStub()

            # 前向传播函数，依次应用量化、反量化、卷积操作及再次量化和反量化
            def forward(self, x):
                x = self.quant1(x)
                x = self.dequant1(x)
                x = self.conv(x)
                x = self.quant2(x)
                x = self.dequant2(x)
                return x

        # 定义输入数据大小
        input_size = (16, 1, 10, 10)
        # 生成符合标准正态分布的输入数据
        data = torch.randn(*input_size, dtype=torch.float)

        # 创建 RefM 类的一个实例 original_ref_m
        original_ref_m = RefM()
        # 随机生成与反卷积层权重相同形状的权重张量 rand_w 和偏置张量 rand_b
        rand_w = torch.randn_like(original_ref_m.conv.weight)
        rand_b = torch.randn_like(original_ref_m.conv.bias)
        # 将 rand_w 和 rand_b 设置为不可训练参数，并分别赋给 original_ref_m.conv 的权重和偏置属性
        original_ref_m.conv.weight = torch.nn.Parameter(rand_w, requires_grad=False)
        original_ref_m.conv.bias = torch.nn.Parameter(rand_b, requires_grad=False)

        # 获取当前量化引擎
        qengine = torch.backends.quantized.engine
        # 如果当前量化引擎不在支持的列表中，则返回
        if qengine not in supported_qengines:
            return

        # 导入 MovingAverageMinMaxObserver 类
        from torch.ao.quantization.observer import MovingAverageMinMaxObserver

        # 创建权重观察器 weight_obs 和激活观察器 act_obs，使用 qint32 数据类型
        weight_obs = MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint32,
            # 设置量化范围，以表示 qint16
            quant_min=-1 * (2 ** 15),
            quant_max=(2 ** 15) - 1,
            qscheme=torch.per_tensor_symmetric,
        )
        act_obs = MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint32,
            quant_min=-1 * (2 ** 15),
            quant_max=(2 ** 15) - 1,
        )
        # 创建自定义的量化配置 custom_qconfig，指定权重和激活的观察器
        custom_qconfig = QConfig(activation=act_obs, weight=weight_obs)

        # 将 original_ref_m 设置为评估模式
        original_ref_m.eval()
        # 将自定义的量化配置 custom_qconfig 分配给 original_ref_m 的量化配置
        original_ref_m.qconfig = custom_qconfig

        # 对 original_ref_m 进行量化准备
        ref_m = prepare(original_ref_m)
        # 进行校准
        ref_m(torch.randn(*input_size, dtype=torch.float))

        # 将 ref_m 转换为量化后的模型，设置为参考模式
        ref_m = convert(ref_m, is_reference=True)

        # 创建名为 myobs 的 MovingAverageMinMaxObserver 实例，设置一些参数
        myobs = MovingAverageMinMaxObserver(averaging_constant=0.5,
                                            dtype=torch.qint32,
                                            # 设置量化范围，以表示 qint16
                                            quant_min=-1 * (2 ** 15),
                                            quant_max=(2 ** 15) - 1,
                                            qscheme=torch.per_tensor_symmetric,
                                            )
        # 对 rand_w 应用 myobs 观察器，得到结果
        result = myobs(rand_w)
        # 计算 myobs 的量化参数
        qparams = myobs.calculate_qparams()
        # 断言 ref_m.conv 的权重缩放因子与 qparams 中的第一个元素相等
        self.assertEqual(ref_m.conv.weight_scale, qparams[0])
    def _test_activation_op_impl(
            self, float_module_class, quantized_module_class, extra_module_kwargs):
        """ Implementation for testing common activation ops like leaky relu
        Args:
            extra_module_kwargs: keyword args to instantiate the float module
        """
        # 定义内部测试函数 `_test_activation_op_impl`，用于测试常见的激活函数操作，如 leaky relu
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 实例化一个浮点数版本的激活函数模块
                self.activation_op = float_module_class(**extra_module_kwargs)
                # 添加量化和反量化占位符
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                # 对输入进行量化
                x = self.quant(x)
                # 应用激活函数操作
                x = self.activation_op(x)
                # 对输出进行反量化
                x = self.dequant(x)
                return x

        # 创建并评估模型实例
        m = M().eval()
        # 设置量化配置为默认配置
        m.qconfig = default_qconfig
        # 准备模型以便进行量化
        m = prepare(m)
        # 检查模型的观察器
        self.checkObservers(m)
        # 将模型转换为量化模型
        m = convert(m)
        # 断言激活函数模块的类型为量化模块的类型
        self.assertEqual(type(m.activation_op), quantized_module_class)

    # 测试 LeakyReLU 激活函数
    def test_leaky_relu(self):
        self._test_activation_op_impl(nn.LeakyReLU, nnq.LeakyReLU, {'negative_slope': 0.1, 'inplace': False})

    # 测试 ReLU 激活函数
    def test_relu(self):
        self._test_activation_op_impl(nn.ReLU, nn.ReLU, {'inplace': False})

    # 历史观察器比较慢，所以没有截止日期以确保测试不会超时
    @given(train_mode=st.booleans())
    # 测试功能性模块
    def test_functional_module(self, train_mode):
        # 创建一个 ModelWithFunctionals 模型实例
        model = ModelWithFunctionals()
        # 创建随机输入张量 x
        x = torch.rand(10, 1, dtype=torch.float)
        # 对 x 进行量化
        xq = torch.quantize_per_tensor(x, 0.01, 30, torch.quint8)
        # 检查模型是否可脚本化，并进行保存和加载检查
        self.checkScriptable(model, [[x]], check_save_load=True)
        if train_mode:
            # 如果处于训练模式，使用默认的量化训练配置 'fbgemm' 来配置模型的量化训练
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
            model = prepare_qat(model)
        else:
            # 否则，使用默认的量化配置 'qnnpack' 配置模型的量化
            model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
            model = prepare(model)
        # 检查模型中是否插入了观察器和量化/反量化节点
        self.checkNoPrepModules(model)
        self.checkObservers(model)
        # 进行模型校准
        model(xq.dequantize())
        # 将模型转换为量化模型
        model = convert(model)

        # 定义检查量化后模型的函数
        def checkQuantized(model):
            self.checkNoPrepModules(model)
            self.assertEqual(type(model.myadd), torch.ao.nn.quantized.QFunctional)
            self.assertEqual(type(model.mycat), torch.ao.nn.quantized.QFunctional)
            self.assertEqual(type(model.myadd_relu), torch.ao.nn.quantized.QFunctional)
            self.assertEqual(type(model.mymatmul), torch.ao.nn.quantized.QFunctional)
            self.checkNoQconfig(model)

        # 调用检查量化后模型的函数
        checkQuantized(model)
        # 再次检查模型是否可脚本化，并进行保存和加载检查
        self.checkScriptable(model, [[xq]], check_save_load=True)
class TestQuantizeEagerPTQStatic(QuantizationTestCase):

    def test_single_layer(self):
        r"""Quantize SingleLayerLinearModel which has one Linear module, make sure it is swapped
        to nnq.Linear which is the quantized version of the module
        """
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 使用指定的量化引擎覆盖当前的量化引擎环境
            with override_quantized_engine(qengine):
                # 获取默认的量化配置
                qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                # 创建一个带注释的单层线性模型，使用指定的量化引擎
                model = AnnotatedSingleLayerLinearModel(qengine)
                model.qconfig = qconfig
                # 准备模型以进行量化
                model = prepare(model)
                # 检查是否已插入观察器和量化/反量化节点
                self.checkNoPrepModules(model)
                self.checkHasPrepModules(model.fc1)
                self.checkObservers(model)

                # 使用校准数据测试仅评估函数
                test_only_eval_fn(model, self.calib_data)
                # 将模型转换为量化模型
                model = convert(model)

                def checkQuantized(model):
                    # 检查模型是否未准备好模块
                    self.checkNoPrepModules(model)
                    # 检查模型是否具有准备好的模块 fc1
                    self.checkHasPrepModules(model.fc1)
                    # 检查包装的量化线性模型 fc1
                    self.checkWrappedQuantizedLinear(model.fc1)
                    # 使用校准数据测试仅评估函数
                    test_only_eval_fn(model, self.calib_data)
                    # 检查模型是否可脚本化
                    self.checkScriptable(model, self.calib_data)
                    # 检查模型是否没有量化配置
                    self.checkNoQconfig(model)

                # 调用检查量化后的模型函数
                checkQuantized(model)

                # 测试单行 API - 非原位版本
                base = AnnotatedSingleLayerLinearModel(qengine)
                base.qconfig = qconfig
                keys_before = set(base.state_dict().keys())
                # 对基础模型进行量化，使用仅测试评估函数和校准数据
                model = quantize(base, test_only_eval_fn, [self.calib_data])
                # 再次调用检查量化后的模型函数
                checkQuantized(model)
                keys_after = set(base.state_dict().keys())
                # 简单检查，确保没有任何变化
                self.assertEqual(keys_before, keys_after)

                # 原位版本
                model = AnnotatedSingleLayerLinearModel(qengine)
                model.qconfig = qconfig
                # 对模型进行原位量化，使用测试仅评估函数和校准数据
                quantize(model, test_only_eval_fn, [self.calib_data], inplace=True)
                # 再次调用检查量化后的模型函数
                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_two_layers(self):
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        # 使用 'fbgemm' 引擎进行量化
        with override_quantized_engine('fbgemm'):
            # 创建一个带注释的 TwoLayerLinearModel 实例
            model = AnnotatedTwoLayerLinearModel()
            # 准备模型以进行量化
            model = prepare(model)

            # 检查模型整体没有准备过的模块
            self.checkNoPrepModules(model)
            # 检查模型整体的观察器是否添加了
            self.checkObservers(model)
            # 检查模型的第一个线性层没有准备过的模块
            self.checkNoPrepModules(model.fc1)
            # 检查模型的第二个线性层已经准备过的模块
            self.checkHasPrepModules(model.fc2)

            # 使用评估函数测试模型
            test_only_eval_fn(model, self.calib_data)
            # 将模型转换为量化模型
            model = convert(model)

            def checkQuantized(model):
                # 检查模型整体没有准备过的模块
                self.checkNoPrepModules(model)
                # 检查模型的第一个线性层没有准备过的模块
                self.checkNoPrepModules(model.fc1)
                # 检查模型的第二个线性层已经准备过的模块
                self.checkHasPrepModules(model.fc2)
                # 断言第一个线性层的类型为 torch.nn.Linear
                self.assertEqual(type(model.fc1), torch.nn.Linear)
                # 检查第二个线性层是否被包装成量化的线性层
                self.checkWrappedQuantizedLinear(model.fc2)
                # 使用评估函数测试模型
                test_only_eval_fn(model, self.calib_data)
                # 检查模型及其数据是否可以被脚本化
                self.checkScriptable(model, self.calib_data)
                # 检查模型是否没有量化配置
                self.checkNoQconfig(model)

            # 调用检查量化的函数
            checkQuantized(model)

            # 测试单行 API 的量化功能
            model = quantize(AnnotatedTwoLayerLinearModel(), test_only_eval_fn,
                             [self.calib_data])
            # 再次调用检查量化的函数
            checkQuantized(model)
    def test_nested1(self):
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', 'sub2.fc2' is not quantized
        """
        # 对于每个量化引擎，执行以下测试
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖默认量化引擎设置
            with override_quantized_engine(qengine):
                # 创建一个带注释的嵌套模型对象
                model = AnnotatedNestedModel(qengine)

                # 定义用于检查准备模块的函数
                def checkPrepModules(model, before_calib=False):
                    # 如果在校准之前，检查观察者是否已设置
                    if before_calib:
                        self.checkObservers(model)
                    # 检查模型及其子模块中不存在预处理模块
                    self.checkNoPrepModules(model)
                    self.checkNoPrepModules(model.sub1)
                    self.checkNoPrepModules(model.sub1.fc)
                    self.checkNoPrepModules(model.sub1.relu)
                    self.checkNoPrepModules(model.sub2)
                    # 检查模型的子模块 'sub2.fc1' 中存在预处理模块
                    self.checkHasPrepModules(model.sub2.fc1)
                    # 检查模型的子模块 'sub2.fc2' 中不存在预处理模块
                    self.checkNoPrepModules(model.sub2.fc2)
                    # 检查模型顶级模块 'fc3' 中存在预处理模块
                    self.checkHasPrepModules(model.fc3)

                # 对模型进行准备操作
                model = prepare(model)
                # 检查准备后的模块状态，包括观察者设置
                checkPrepModules(model, True)
                # 使用校准数据测试仅评估函数
                test_only_eval_fn(model, self.calib_data)
                # 将模型转换为量化版本
                model = convert(model)

                # 定义用于检查量化后模型的函数
                def checkQuantized(model):
                    # 检查模型的预处理模块状态
                    checkPrepModules(model)
                    # 检查模型的线性层 'sub1.fc' 是否已量化
                    self.checkLinear(model.sub1.fc)
                    # 检查模型的包装量化线性层 'fc3' 是否已量化
                    self.checkWrappedQuantizedLinear(model.fc3)
                    # 检查模型的包装量化线性层 'sub2.fc1' 是否已量化
                    self.checkWrappedQuantizedLinear(model.sub2.fc1)
                    # 检查模型的线性层 'sub2.fc2' 是否已量化
                    self.checkLinear(model.sub2.fc2)
                    # 使用校准数据测试仅评估函数
                    test_only_eval_fn(model, self.calib_data)
                    # 检查模型是否可脚本化
                    self.checkScriptable(model, self.calib_data)
                    # 检查模型中是否不存在量化配置
                    self.checkNoQconfig(model)

                # 检查量化后的模型状态
                checkQuantized(model)

                # 测试一行 API
                # 使用指定的量化引擎和校准数据，量化模型并检查状态
                model = quantize(AnnotatedNestedModel(qengine), test_only_eval_fn,
                                 [self.calib_data])
                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_nested2(self):
        # 创建一个 AnnotatedSubNestedModel 实例
        model = AnnotatedSubNestedModel()
        # 准备模型，可能进行一些预处理
        model = prepare(model)

        # 定义一个函数 checkPrepModules，用于检查模型的预处理模块
        def checkPrepModules(model, before_calib=False):
            # 如果在校准之前，检查模型的观察器
            if before_calib:
                self.checkObservers(model)
            # 检查模型及其子模块没有预处理模块
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            # 检查模型的子模块 sub2 有预处理模块
            self.checkHasPrepModules(model.sub2)
            self.checkNoPrepModules(model.sub2.module.fc1)
            self.checkNoPrepModules(model.sub2.module.fc2)
            # 检查模型的 fc3 层有预处理模块
            self.checkHasPrepModules(model.fc3)

        # 使用 checkPrepModules 函数检查模型在校准之前的情况
        checkPrepModules(model, True)

        # 使用 test_only_eval_fn 函数测试模型的评估功能
        test_only_eval_fn(model, self.calib_data)
        # 将模型转换为量化模型
        model = convert(model)

        # 定义一个函数 checkQuantized，用于检查量化后的模型
        def checkQuantized(model):
            # 使用 checkPrepModules 函数检查量化后的模型
            checkPrepModules(model)
            # 检查模型子模块 sub1.fc 是线性层
            self.checkLinear(model.sub1.fc)
            # 断言模型的 sub1.relu 是 ReLU 激活函数
            self.assertEqual(type(model.sub1.relu), torch.nn.ReLU)
            # 检查模型子模块 sub2.module.fc1 和 sub2.module.fc2 是量化后的线性层
            self.checkQuantizedLinear(model.sub2.module.fc1)
            self.checkQuantizedLinear(model.sub2.module.fc2)
            # 检查模型的 fc3 层是包装后的量化线性层
            self.checkWrappedQuantizedLinear(model.fc3)
            # 使用 test_only_eval_fn 函数测试量化后模型的评估功能
            test_only_eval_fn(model, self.calib_data)
            # 检查模型是否可以脚本化
            self.checkScriptable(model, self.calib_data)
            # 检查模型没有量化配置
            self.checkNoQconfig(model)

        # 使用 checkQuantized 函数检查模型的量化情况
        checkQuantized(model)

        # 测试单行 API
        # 对 AnnotatedSubNestedModel 进行量化，并使用 checkQuantized 函数检查
        model = quantize(AnnotatedSubNestedModel(), test_only_eval_fn,
                         [self.calib_data])
        checkQuantized(model)
    def test_nested3(self):
        r"""More complicated nested test case with child qconfig overrides
        parent qconfig
        """
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖量化配置
            with override_quantized_engine(qengine):
                # 创建一个注解的自定义配置嵌套模型
                model = AnnotatedCustomConfigNestedModel()
                # 准备模型以进行量化
                model = prepare(model)

                # 定义检查准备模块的函数
                def checkPrepModules(model, before_calib=False):
                    # 如果在校准之前，检查观察器是否正确设置
                    if before_calib:
                        self.checkObservers(model)
                    # 检查模型及其子模块没有准备模块
                    self.checkNoPrepModules(model)
                    self.checkNoPrepModules(model.sub1)
                    self.checkNoPrepModules(model.sub1.fc)
                    self.checkNoPrepModules(model.sub1.relu)
                    self.checkNoPrepModules(model.sub2)
                    # 检查模型的某些子模块具有准备模块
                    self.checkHasPrepModules(model.sub2.fc1)
                    self.checkHasPrepModules(model.sub2.fc2)
                    self.checkHasPrepModules(model.fc3)

                # 调用检查准备模块的函数，校准前的情况
                checkPrepModules(model, True)

                # 使用校准数据对模型进行仅评估测试
                test_only_eval_fn(model, self.calib_data)
                # 将模型转换为量化版本
                model = convert(model)

                # 检查量化后的模型
                def checkQuantized(model):
                    # 检查准备模块
                    checkPrepModules(model)
                    # 检查某些线性层是否被包装为量化版本
                    self.checkWrappedQuantizedLinear(model.sub2.fc1)
                    self.checkWrappedQuantizedLinear(model.sub2.fc2)
                    self.checkWrappedQuantizedLinear(model.fc3)
                    # 再次进行仅评估测试
                    test_only_eval_fn(model, self.calib_data)
                    # 检查模型的脚本化
                    self.checkScriptable(model, self.calib_data)
                    # 检查模型没有配置信息
                    self.checkNoQconfig(model)

                # 调用检查量化后模型的函数
                checkQuantized(model)

                # 测试一行 API 的功能
                # 将注解的自定义配置嵌套模型进行量化，并使用仅评估测试函数
                model = quantize(AnnotatedCustomConfigNestedModel(), test_only_eval_fn,
                                 [self.calib_data])
                # 再次调用检查量化后模型的函数
                checkQuantized(model)
    def test_skip_quant(self):
        r"""The case when we want to skip quantizing some layers
        """
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖量化引擎上下文
            with override_quantized_engine(qengine):
                # 创建带有指定量化引擎的 AnnotatedSkipQuantModel 模型
                model = AnnotatedSkipQuantModel(qengine)
                # 准备模型以进行量化
                model = prepare(model)
                # 检查模型中的观察器
                self.checkObservers(model)

                # 使用 test_only_eval_fn 函数对模型进行仅评估测试
                test_only_eval_fn(model, self.calib_data)
                # 将模型转换为量化表示
                model = convert(model)

                # 定义检查量化后模型的函数
                def checkQuantized(model):
                    # 检查线性层是否被量化
                    self.checkLinear(model.fc)
                    # 检查 sub 模块是否包含量化和反量化操作
                    self.checkQuantDequant(model.sub)
                    # 检查 sub 模块中的 fc1 和 fc2 线性层是否被量化
                    self.checkQuantizedLinear(model.sub.module.fc1)
                    self.checkQuantizedLinear(model.sub.module.fc2)
                    # 断言 sub 模块中的 relu1 和 relu2 层为 nn.ReLU 类型
                    self.assertEqual(type(model.sub.module.relu1), nn.ReLU)
                    self.assertEqual(type(model.sub.module.relu2), nn.ReLU)
                    # 检查模型在给定数据上是否可脚本化
                    self.checkScriptable(model, self.calib_data)
                    # 检查模型是否不含量化配置信息
                    self.checkNoQconfig(model)

                # 对当前模型进行量化后检查
                checkQuantized(model)

                # 测试单行 API 调用情况
                model = quantize(AnnotatedSkipQuantModel(qengine), test_only_eval_fn, [self.calib_data])
                # 再次检查量化后模型
                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_manual(self):
        r"""User inserts QuantStub and DeQuantStub in model code
        and call the quantization utility functions.
        """
        # 创建 QuantStubModel 模型实例
        model = QuantStubModel()
        # 将父模型的量化配置传播到子模型中，原地修改模型
        model = prepare(model)
        # 检查模型中的观察器
        self.checkObservers(model)

        # 使用 test_only_eval_fn 函数对模型进行仅评估测试
        test_only_eval_fn(model, self.calib_data)
        # 将模型转换为量化表示
        model = convert(model)

        # 定义检查量化后模型的函数
        def checkQuantized(model):
            # 断言模型中的 fc 层为 nnq.Linear 类型
            self.assertEqual(type(model.fc), nnq.Linear)
            # 再次使用 test_only_eval_fn 函数对模型进行仅评估测试
            test_only_eval_fn(model, self.calib_data)
            # 检查模型在给定数据上是否可脚本化
            self.checkScriptable(model, self.calib_data)
            # 检查模型是否不含量化配置信息
            self.checkNoQconfig(model)

        # 对当前模型进行量化后检查
        checkQuantized(model)

        # 测试单行 API 调用情况
        model = quantize(QuantStubModel(), test_only_eval_fn, [self.calib_data])
        # 再次检查量化后模型
        checkQuantized(model)
    def test_resnet_base(self):
        r"""Test quantization for bottleneck topology used in resnet/resnext
        and add coverage for conversion of average pool and float functional
        """
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖量化引擎设置
            with override_quantized_engine(qengine):
                # 获取当前量化引擎的默认量化配置
                qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                # 创建 ResNetBase 模型实例，并转换为浮点数类型并设为评估模式
                model = ResNetBase().float().eval()
                # 合并模型中可融合的操作
                model.fuse_model()
                # 将模型包装为量化器包装器
                model = QuantWrapper(model)
                # 设置模型的量化配置
                model.qconfig = qconfig
                # 对模型进行量化准备操作
                model = prepare(model)
                # 检查模型中的观察器
                self.checkObservers(model)
                # 使用测试函数评估模型，传入二维图像数据
                test_only_eval_fn(model, self.img_data_2d)
                # 将模型转换为量化后的版本
                model = convert(model)

                def checkQuantized(model):
                    # 断言模型的第一卷积层为量化的 ConvReLU2d 类型
                    self.assertEqual(type(model.module.conv1), nn.intrinsic.quantized.ConvReLU2d)
                    # 断言模型中自定义操作 'myop' 为 nn.quantized.QFunctional 类型
                    self.assertEqual(type(model.module.myop), nn.quantized.QFunctional)
                    # 断言模型中的平均池化层为 nn.AdaptiveAvgPool2d 类型
                    self.assertEqual(type(model.module.avgpool), nn.AdaptiveAvgPool2d)
                    # 断言模型中全连接层为量化后的 nnq.Linear 类型
                    self.assertEqual(type(model.module.fc), nnq.Linear)

                    # 再次使用测试函数评估模型，传入二维图像数据
                    test_only_eval_fn(model, self.img_data_2d)
                    # 检查模型中是否没有量化配置
                    self.checkNoQconfig(model)

                # 调用检查量化后模型的函数
                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_normalization(self):
        r"""
        Test quantization of normalization layers
        """
        # 创建 NormalizationTestModel 模型实例
        model = NormalizationTestModel()
        # 获取 'fbgemm' 量化引擎的默认量化配置
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        # 在原地准备模型进行量化
        prepare(model, inplace=True)
        # 检查模型中的观察器
        self.checkObservers(model)
        # 使用测试函数评估模型，传入校准数据
        test_only_eval_fn(model, self.calib_data)
        # 将模型转换为量化后的版本
        model = convert(model)

        def checkQuantized(model):
            # 检查模型的 layer_norm 层没有准备模块
            self.checkNoPrepModules(model.layer_norm)
            # 检查模型的 group_norm 层没有准备模块
            self.checkNoPrepModules(model.group_norm)
            # 检查模型的 instance_norm1d 层没有准备模块
            self.checkNoPrepModules(model.instance_norm1d)
            # 检查模型的 instance_norm2d 层没有准备模块
            self.checkNoPrepModules(model.instance_norm2d)
            # 检查模型的 instance_norm3d 层没有准备模块
            self.checkNoPrepModules(model.instance_norm3d)
            # 断言模型的 layer_norm 层为量化后的 nnq.LayerNorm 类型
            self.assertEqual(type(model.layer_norm), nnq.LayerNorm)
            # 断言模型的 group_norm 层为量化后的 nnq.GroupNorm 类型
            self.assertEqual(type(model.group_norm), nnq.GroupNorm)
            # 断言模型的 instance_norm1d 层为量化后的 nnq.InstanceNorm1d 类型
            self.assertEqual(type(model.instance_norm1d), nnq.InstanceNorm1d)
            # 断言模型的 instance_norm2d 层为量化后的 nnq.InstanceNorm2d 类型
            self.assertEqual(type(model.instance_norm2d), nnq.InstanceNorm2d)
            # 断言模型的 instance_norm3d 层为量化后的 nnq.InstanceNorm3d 类型
            self.assertEqual(type(model.instance_norm3d), nnq.InstanceNorm3d)
            # 再次使用测试函数评估模型，传入校准数据
            test_only_eval_fn(model, self.calib_data)
            # 检查模型的可脚本化性，传入校准数据
            self.checkScriptable(model, self.calib_data)
            # 检查模型中是否没有量化配置
            self.checkNoQconfig(model)

        # 调用检查量化后模型的函数
        checkQuantized(model)

        # 对模型使用 quantize 函数，传入 NormalizationTestModel 实例、测试函数和校准数据
        model_oneline = quantize(
            NormalizationTestModel(), test_only_eval_fn, [self.calib_data])
        # 再次调用检查量化后模型的函数
        checkQuantized(model)
    def test_save_load_state_dict(self):
        r"""Test PTQ flow of creating a model and quantizing it and saving the quantized state_dict
        Load the quantized state_dict for eval and compare results against original model
        """
        
        # 针对支持的量化引擎进行测试
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                # 创建一个两层线性模型
                model = TwoLayerLinearModel()
                # 将模型包装为量化模型
                model = torch.ao.quantization.QuantWrapper(model)
                # 设置模型的量化配置
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)

                # 准备模型以进行量化
                model = prepare(model)
                # 校准模型
                test_only_eval_fn(model, self.calib_data)
                # 转换模型为量化后的版本
                model = convert(model)
                # 创建输入张量 x
                x = torch.rand(2, 5, dtype=torch.float)
                # 获取参考输出
                ref = model(x)

                # 获取量化后的模型状态字典
                quant_state_dict = model.state_dict()

                # 再次创建模型以进行评估
                model = TwoLayerLinearModel()
                model = torch.ao.quantization.QuantWrapper(model)
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                model = prepare(model)
                model = convert(model)
                new_state_dict = model.state_dict()

                # 检查转换后的模型状态字典的键是否与原始模型匹配
                self.assertEqual(set(new_state_dict.keys()), set(quant_state_dict.keys()))

                # 加载量化后的模型状态字典
                model.load_state_dict(quant_state_dict)

                # 使用相同的输入 x 运行模型，并比较输出结果
                out = model(x)
                self.assertEqual(ref, out)

    @skipIfNoFBGEMM
    def test_activations(self):
        r"""
        Test quantization of activations
        """
        
        # 创建激活函数测试模型
        model = ActivationsTestModel()
        # 设置模型的量化配置为 'fbgemm'
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        # 准备模型以进行量化，直接在原地修改模型
        prepare(model, inplace=True)
        # 检查模型的观察器
        self.checkObservers(model)
        # 校准模型
        test_only_eval_fn(model, self.calib_data)
        # 转换模型为量化后的版本
        model = convert(model)

        def checkQuantized(model):
            # 检查模型的 hardswish 属性，确保没有预处理模块
            self.checkNoPrepModules(model.hardswish)
            # 确保 model.hardswish 属性是量化的 Hardswish 类型
            self.assertEqual(type(model.hardswish), nnq.Hardswish)
            # 确保 model.elu 属性是量化的 ELU 类型
            self.assertEqual(type(model.elu), nnq.ELU)
            # 再次校准模型
            test_only_eval_fn(model, self.calib_data)
            # 检查模型是否可脚本化
            self.checkScriptable(model, self.calib_data)
            # 检查模型是否没有量化配置
            self.checkNoQconfig(model)

        # 执行检查量化后的模型
        checkQuantized(model)

        # 测试单行 API
        model_oneline = quantize(ActivationsTestModel(), test_only_eval_fn,
                                 [self.calib_data])
        # 再次执行检查量化后的模型
        checkQuantized(model_oneline)

    @override_qengines
    def test_forward_hooks_preserved(self):
        r"""Test post-training static quantization on preserving
        pre forward and post forward hooks of original model
        """
        # 获取当前的量化引擎
        qengine = torch.backends.quantized.engine
        # 创建一个量化后的模型实例
        model = QuantStubModel()
        # 计数器，用于统计前向预处理和前向处理的调用次数
        counter = {
            'pre_forwards': 0,
            'forwards': 0,
        }

        # 前向预处理钩子函数，每次调用计数器增加
        def fw_pre_hook(h_module, input):
            counter['pre_forwards'] += 1

        # 前向处理钩子函数，每次调用计数器增加
        def fw_hook(h_module, input, output):
            counter['forwards'] += 1

        # 将前向预处理钩子函数注册到模型的全连接层
        model.fc.register_forward_pre_hook(fw_pre_hook)
        # 将前向处理钩子函数注册到模型的全连接层
        model.fc.register_forward_hook(fw_hook)

        # 设置模型的量化配置为默认配置
        model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        # 准备模型进行量化
        model = prepare(model)

        # 检查钩子函数是否存在的辅助函数
        def checkHooksIsPresent(model, before_convert=True):
            num_fwd_hooks = 1
            if before_convert:
                # 断言量化过程中观察者钩子是否存在
                self.assertEqual(len(model.quant._forward_hooks.values()), 1,
                                 "Quantization observer hook has disappeared")
                num_fwd_hooks = 2

            # 断言前向预处理钩子是否在模型的全连接层中
            self.assertObjectIn(fw_pre_hook, model.fc._forward_pre_hooks.values())
            # 断言前向处理钩子是否在模型的全连接层中
            self.assertObjectIn(fw_hook, model.fc._forward_hooks.values())
            # 断言模型的全连接层中只有一个前向预处理钩子
            self.assertEqual(len(model.fc._forward_pre_hooks.values()), 1,
                             "Extra pre forward hooks have appeared on a layer")
            # 静态量化期间，非存根层也会提供量化观察者钩子
            # 断言模型的全连接层中的前向处理钩子数量
            self.assertEqual(len(model.fc._forward_hooks.values()), num_fwd_hooks,
                             "Extra post forward hooks have appeared on a layer")
            # 隐式检查 fw_hook 是否在 _observer_forward_hook 之后
            self.assertEqual(list(model.fc._forward_hooks.values())[-1], fw_hook,
                             "_observer_forward_hook is not a first entry of the hooks list")

        # 检查钩子函数是否存在（模型未量化前）
        checkHooksIsPresent(model, True)
        # 使用测试数据对模型进行评估
        test_only_eval_fn(model, self.calib_data)
        # 将模型转换为量化模型（原地转换）
        torch.ao.quantization.convert(model, inplace=True)
        # 检查钩子函数是否存在（模型已量化后）
        checkHooksIsPresent(model, False)

    @skipIfNoFBGEMM
    def test_quantized_embedding(self):
        r""" Test the post-training quantization flow, serialization and scripting
        of embedding modules
        """

        # 遍历两种量化配置
        for qconfig in [float_qparams_weight_only_qconfig, float_qparams_weight_only_qconfig_4bit]:
            # 创建并评估嵌入模块
            model = EmbeddingModule().eval()
            # 定义索引张量
            indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
            # 随机初始化权重张量
            weights = torch.randn(10, 12, dtype=torch.float32)
            # 应用当前量化配置到模型
            model.qconfig = qconfig
            # 在原地准备模型以适应量化
            prepare(model, inplace=True)
            # 在原地转换模型以适应量化
            convert(model, inplace=True)
            # 断言模型中包含量化后的嵌入层
            self.assertTrue('QuantizedEmbedding' in str(model))
            # 断言模型中的嵌入层类型为 torch.ao.nn.quantized.Embedding
            self.assertEqual(type(model.emb), torch.ao.nn.quantized.Embedding)
            # 检查模型是否能被脚本化，同时检查保存和加载的有效性
            self.checkScriptable(model, [[indices]], check_save_load=True)

            # 定义索引和偏移张量
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            # 创建输入张量 x
            x = torch.randn(2, 4)
            # 创建并评估带静态线性层的嵌入模块
            model = EmbeddingWithStaticLinear().eval()
            # 在原地准备模型以适应量化
            prepare(model, inplace=True)
            # 在原地转换模型以适应量化
            convert(model, inplace=True)
            # 断言模型中包含量化后的嵌入层
            self.assertTrue('QuantizedEmbedding' in str(model))
            # 断言模型中包含量化后的线性层
            self.assertTrue('QuantizedLinear' in str(model))
            # 检查量化线性层的正确性
            self.checkQuantizedLinear(model.fc)
            # 执行模型推断
            model(idx, offsets, x)

    @skipIfNoFBGEMM
    def test_dequant_stub(self):
        # 创建并评估量化占位模型
        m = QuantStubModel().eval()
        # 在原地准备模型以适应量化
        prepare(m, inplace=True)
        # 检查模型的观察器
        self.checkObservers(m)
        # 在原地转换模型以适应量化
        convert(m, inplace=True)
        # 断言模型中量化对象为 Quantize 类型
        self.assertEqual(type(m.quant), nnq.Quantize)
        # 断言模型中的线性层为量化线性层
        self.assertEqual(type(m.fc), nnq.Linear)
        # 断言模型中的去量化对象为 DeQuantize 类型
        self.assertEqual(type(m.dequant), nnq.DeQuantize)

        # 检查当没有量化配置时，DeQuantStub 没有被替换
        m2 = QuantStubModel().eval()
        m2.dequant.qconfig = None
        # 在原地准备模型以适应量化
        prepare(m2, inplace=True)
        # 检查模型的观察器
        self.checkObservers(m2)
        # 在原地转换模型以适应量化
        convert(m2, inplace=True)
        # 断言模型中量化对象为 Quantize 类型
        self.assertEqual(type(m2.quant), nnq.Quantize)
        # 断言模型中的线性层为量化线性层
        self.assertEqual(type(m2.fc), nnq.Linear)
        # 断言模型中的去量化对象为 DeQuantStub 类型
        self.assertEqual(type(m2.dequant), DeQuantStub)
    def test_quantized_embedding_bag(self):
        r""" Test the post-training quantization flow, serialization and scripting
        of embedding_bag modules
        """
        # 创建包含指定索引的张量
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        # 创建包含偏移值的张量
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        # 创建具有指定形状的随机张量
        weights = torch.randn(10, 12, dtype=torch.float32)

        # 针对不同的量化数据类型进行测试
        for dtype in [torch.quint8, torch.quint4x2]:
            # 创建并将模型设置为评估模式
            model = EmbeddingBagModule().eval()
            # 设置观察器为逐通道最小-最大观察器，使用指定的量化方案
            float_qparams_observer = PerChannelMinMaxObserver.with_args(dtype=dtype,
                                                                        qscheme=torch.per_channel_affine_float_qparams,
                                                                        ch_axis=0)
            # 创建量化配置对象
            float_qparams_qconfig = QConfig(activation=default_dynamic_quant_observer,
                                            weight=float_qparams_observer)
            # 将模型的量化配置设置为上述配置
            model.qconfig = float_qparams_qconfig

            # 在原地准备模型以进行量化
            prepare(model, inplace=True)
            # 执行量化转换
            quantized_model = convert(model)

            # 创建随机权重张量
            per_sample_weights = torch.from_numpy(np.random.uniform(
                low=0.01, high=0.5, size=[len(indices)]).astype(np.float32))

            # 测试模型是否正确量化
            self.assertTrue('QuantizedEmbeddingBag' in str(quantized_model))
            # 检查量化后的EmbeddingBag模块
            self.checkDynamicQuantizedModule(quantized_model.emb, torch.ao.nn.quantized.EmbeddingBag, torch.quint8)
            # 检查模型是否可脚本化，并测试保存加载功能
            self.checkScriptable(quantized_model, [[indices, offsets, per_sample_weights]], check_save_load=True)

            # 定义包含Linear层的EmbeddingBag模块
            class EmbeddingBagWithLinear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建EmbeddingBag模块和Linear层
                    self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12,
                                                     include_last_offset=True, scale_grad_by_freq=False, mode='sum')
                    self.fc = torch.nn.Linear(5, 5)

                def forward(self, indices, offsets, per_sample_weights, linear_in):
                    # 前向传播函数，返回EmbeddingBag和Linear层的输出
                    return self.emb(indices, offsets, per_sample_weights), self.fc(linear_in)

            # 测试仅对EmbeddingBag层进行量化
            model2 = EmbeddingBagWithLinear().eval()
            # 设置EmbeddingBag层的量化配置
            model2.emb.qconfig = float_qparams_qconfig
            # 在原地准备模型以进行量化
            prepare(model2, inplace=True)
            # 执行量化转换
            quantized_model = convert(model2)

            # 断言量化后的模型中包含 'QuantizedEmbeddingBag' 字符串
            self.assertTrue('QuantizedEmbeddingBag' in str(quantized_model))
            # 检查Linear层是否保持不变
            self.checkLinear(model2.fc)
            # 检查量化后的EmbeddingBag模块
            self.checkDynamicQuantizedModule(quantized_model.emb, torch.ao.nn.quantized.EmbeddingBag, torch.quint8)

    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    def test_convtranspose_per_channel_fails_early(self):
        r"""
        Verifies that attempting to quantize a ConvTranspose module with per-Channel
        weight observers fails in the prepare step, as opposed to the convert step.
        """
        # 创建一个包含单个 ConvTranspose2d 模块的序列模型
        m = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 1, 1))
        # 设置量化配置为默认的 fbgemm 配置
        m.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        # 使用断言验证在 prepare 步骤中尝试使用每通道权重观察器会导致断言错误
        with self.assertRaises(AssertionError) as context:
            mp = torch.ao.quantization.prepare(m)
        # 使用断言验证异常消息是否与预期相符
        self.assertTrue(
            str(context.exception) ==
            'Per channel weight observer is not supported yet for ConvTranspose{n}d.')

    @skipIfNoFBGEMM
    def test_convtranspose_per_channel_qconfig_none(self):
        r"""
        Verifies that having qconfig==None for conv transpose does not crash
        """
        # 创建一个包含单个 ConvTranspose2d 模块的序列模型
        m = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 1, 1))
        # 设置量化配置为默认的 fbgemm 配置
        m.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        # 将第一个 ConvTranspose2d 模块的量化配置设为 None
        m[0].qconfig = None
        # 对模型进行量化准备
        mp = torch.ao.quantization.prepare(m)

    @skipIfNoFBGEMM
    def test_quantwrapper_attaches_qconfig_to_dequant(self):
        # 获取默认的量化配置
        qconfig = torch.ao.quantization.default_qconfig

        # 创建一个包含单个 Conv2d 模块的序列模型，并设置为评估模式
        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        # 为模型中的每个模块设置量化配置并使用 QuantWrapper 进行包装
        for i in range(len(m)):
            m[i].qconfig = qconfig
            m[i] = torch.ao.quantization.QuantWrapper(m[i])

        # 对模型进行量化准备
        mp = torch.ao.quantization.prepare(m)
        # 将准备后的模型转换为量化模型
        mq = torch.ao.quantization.convert(mp)
        # 使用断言验证第一个模块是否附加了 DeQuantize 模块
        self.assertTrue(isinstance(mq[0].dequant, nnq.DeQuantize))
    def test_activations_in_non_leaf_module_list(self):
        """
        Ensure activations like `nn.Sigmoid` and `nn.Tanh` are properly handled in
        `non_leaf_module_list`.
        """
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化量化模块
                self.quant = QuantStub()
                # sigmoid 激活函数
                self.sigmoid = torch.nn.Sigmoid()
                # hardsigmoid 激活函数
                self.hardsigmoid = torch.nn.Hardsigmoid()
                # softmax 激活函数
                self.softmax = torch.nn.Softmax()
                # tanh 激活函数
                self.tanh = torch.nn.Tanh()
                # 反量化模块
                self.dequant = DeQuantStub()

            def forward(self, x):
                # 量化输入数据
                x = self.quant(x)
                # 使用 sigmoid 激活函数
                x = self.sigmoid(x)
                # 使用 hardsigmoid 激活函数
                x = self.hardsigmoid(x)
                # 使用 softmax 激活函数
                x = self.softmax(x)
                # 使用 tanh 激活函数
                x = self.tanh(x)
                # 反量化输出数据
                x = self.dequant(x)
                return x

        # 定义量化配置 QConfig
        qconfig = QConfig(
            activation=FixedQParamsObserver.with_args(scale=123.0, zero_point=0),
            weight=default_weight_observer
        )
        # 创建模型实例
        m = MyModel()
        # 将 QConfig 应用到模型中
        m.qconfig = qconfig
        # 准备模型，设置非叶子模块列表为需要量化的激活函数类
        m = prepare(m, observer_non_leaf_module_list=[
            torch.nn.Sigmoid,
            torch.nn.Hardsigmoid,
            torch.nn.Softmax,
            torch.nn.Tanh,
        ])

        # 断言每个激活函数的后处理函数是 FixedQParamsObserver 类型，而不是默认的 FixedQParamsFakeQuantize
        self.assertTrue(isinstance(m.sigmoid.activation_post_process, FixedQParamsObserver))
        self.assertTrue(isinstance(m.hardsigmoid.activation_post_process, FixedQParamsObserver))
        self.assertTrue(isinstance(m.softmax.activation_post_process, FixedQParamsObserver))
        self.assertTrue(isinstance(m.tanh.activation_post_process, FixedQParamsObserver))

    @skipIfNoFBGEMM
    def test_mha_batch_first_attr_is_copied_in_prepare(self):
        class TransformerDecoderLayer(nn.Module):
            def __init__(self, d_model, nhead, batch_first):
                super().__init__()
                # 初始化 self-attention 层，根据 batch_first 参数选择处理方式
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=batch_first)

        # 获取当前量化引擎
        qengine = torch.backends.quantized.engine
        # 针对每个 batch_first 值进行测试
        for batch_first in [True, False]:
            # 创建 TransformerDecoderLayer 实例
            model = TransformerDecoderLayer(512, 8, batch_first)
            # 获取默认的量化配置
            quantization_config = torch.ao.quantization.get_default_qconfig(qengine)
            # 将量化配置应用到模型中
            model.qconfig = quantization_config
            # 准备模型，确保 batch_first 属性在准备过程中被正确复制
            prepared_model = torch.ao.quantization.prepare(model, inplace=False)
            self.assertTrue(prepared_model.self_attn.batch_first == model.self_attn.batch_first)
@skipIfNoFBGEMM
class TestQuantizeEagerPTQDynamic(QuantizationTestCase):
    def test_single_layer(self):
        r"""Dynamic Quantize SingleLayerLinearDynamicModel which has one Linear module,
        make sure it is swapped to nnqd.Linear which is the quantized version of
        the module
        """
        # 针对不同数据类型进行测试：torch.qint8 和 torch.float16
        for dtype in [torch.qint8, torch.float16]:
            # 创建评估模式下的 SingleLayerLinearDynamicModel 实例
            model = SingleLayerLinearDynamicModel().eval()
            # 根据数据类型选择量化配置
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            # 指定量化配置字典，仅对 'fc1' 模块进行量化
            qconfig_dict = {
                'fc1': qconfig
            }
            # 准备模型以使用动态量化
            prepare_dynamic(model, qconfig_dict)
            # 执行动态量化转换
            convert_dynamic(model)

            def checkQuantized(model):
                # 检查量化后的 Linear 模块是否正确
                self.checkDynamicQuantizedLinear(model.fc1, dtype)
                # 检查模型是否可脚本化，并检查保存和加载的功能
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                # 检查模型中是否没有遗留的量化配置
                self.checkNoQconfig(model)

            # 调用检查函数，验证量化后的模型
            checkQuantized(model)

            # 测试单行 API - 无地方性版本
            base = SingleLayerLinearDynamicModel()
            keys_before = set(base.state_dict().keys())
            model = quantize_dynamic(base, qconfig_dict)
            checkQuantized(model)
            keys_after = set(base.state_dict().keys())
            self.assertEqual(keys_before, keys_after)  # 简单检查，确保没有变化

            # 就地版本
            model = SingleLayerLinearDynamicModel()
            quantize_dynamic(model, qconfig_dict, inplace=True)
            checkQuantized(model)

            # 测试设置量化配置的 API
            model = SingleLayerLinearDynamicModel()
            quantize_dynamic(model, {nn.Linear}, inplace=True, dtype=dtype)
            checkQuantized(model)

    def test_two_layers(self):
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        # 针对不同数据类型进行测试：torch.qint8 和 torch.float16
        for dtype in [torch.qint8, torch.float16]:
            # 创建评估模式下的 TwoLayerLinearModel 实例
            model = TwoLayerLinearModel().eval()
            # 根据数据类型选择量化配置
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            # 指定量化配置字典，仅对 'fc2' 模块进行量化
            qconfig_dict = {
                'fc2': qconfig
            }
            # 准备模型以使用动态量化
            prepare_dynamic(model, qconfig_dict)
            # 执行动态量化转换
            convert_dynamic(model)

            def checkQuantized(model):
                # 确保 fc1 模块没有量化
                self.assertEqual(type(model.fc1), torch.nn.Linear)
                # 检查量化后的 fc2 模块是否正确
                self.checkDynamicQuantizedLinear(model.fc2, dtype=dtype)
                # 检查模型是否可脚本化，并检查保存和加载的功能
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                # 检查模型中是否没有遗留的量化配置
                self.checkNoQconfig(model)

            # 调用检查函数，验证量化后的模型
            checkQuantized(model)

            # 测试单行 API
            model = quantize_dynamic(TwoLayerLinearModel().eval(), qconfig_dict)
            checkQuantized(model)

            # 测试设置 API
            model = quantize_dynamic(TwoLayerLinearModel().eval(), {'fc2'}, dtype=dtype)
            checkQuantized(model)
    def test_nested1(self):
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', 'sub2.fc2' is not quantized
        """
        # 循环遍历 torch.qint8 和 torch.float16 两种数据类型
        for dtype in [torch.qint8, torch.float16]:
            # 创建 NestedModel 实例并设置为评估模式
            model = NestedModel().eval()
            # 根据数据类型选择量化配置
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            # 构建量化配置字典，指定需要量化的模块
            qconfig_dict = {
                'fc3': qconfig,
                'sub2.fc1': qconfig
            }

            # 准备动态量化
            prepare_dynamic(model, qconfig_dict)
            # 执行动态量化
            convert_dynamic(model)

            # 定义检查量化结果的函数
            def checkQuantized(model):
                # 检查线性模块是否被量化
                self.checkLinear(model.sub1.fc)
                # 检查动态量化线性模块是否正确
                self.checkDynamicQuantizedLinear(model.fc3, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=dtype)
                # 检查线性模块是否被量化
                self.checkLinear(model.sub2.fc2)
                # 检查模型是否可脚本化
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                # 检查模型是否没有量化配置
                self.checkNoQconfig(model)

            # 调用检查函数
            checkQuantized(model)

            # 测试一行 API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
            checkQuantized(model)

            # 使用集合 API 进行测试
            model = quantize_dynamic(NestedModel().eval(), {'fc3', 'sub2.fc1'}, dtype=dtype)
            checkQuantized(model)

    def test_nested2(self):
        r"""Another test case for quantized, we will quantize all submodules
        of submodule sub2
        """
        # 循环遍历 torch.qint8 和 torch.float16 两种数据类型
        for dtype in [torch.qint8, torch.float16]:
            # 创建 NestedModel 实例并设置为评估模式
            model = NestedModel().eval()
            # 根据数据类型选择量化配置
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            # 构建量化配置字典，指定需要量化的模块
            qconfig_dict = {
                'fc3': qconfig,
                'sub2': qconfig
            }
            # 准备动态量化
            prepare_dynamic(model, qconfig_dict)

            # 执行动态量化
            convert_dynamic(model)

            # 定义检查量化结果的函数
            def checkQuantized(model):
                # 检查线性模块是否被量化
                self.checkLinear(model.sub1.fc)
                # 断言 sub1.relu 的类型为 torch.nn.ReLU
                self.assertEqual(type(model.sub1.relu), torch.nn.ReLU)
                # 检查动态量化线性模块是否正确
                self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.fc3, dtype=dtype)
                # 检查模型是否可脚本化
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                # 检查模型是否没有量化配置
                self.checkNoQconfig(model)

            # 调用检查函数
            checkQuantized(model)

            # 测试一行 API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dict, dtype=dtype)
            checkQuantized(model)

            # 使用集合 API 进行测试
            model = quantize_dynamic(NestedModel().eval(), {'fc3', 'sub2'}, dtype=dtype)
            checkQuantized(model)
    def test_nested3(self):
        r"""More complicated nested test case with child qconfig overrides
        parent qconfig
        """
        # 遍历数据类型列表：torch.qint8 和 torch.float16
        for dtype in [torch.qint8, torch.float16]:
            # 创建并评估 NestedModel 模型
            model = NestedModel().eval()
            # 根据数据类型选择动态量化配置
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            # 定义动态量化配置字典，指定不同模块的量化配置
            qconfig_dynamic_dict = {
                'fc3': qconfig,
                'sub2': qconfig,
                'sub2.fc1': qconfig
            }
            # 准备动态量化，应用配置到模型
            prepare_dynamic(model, qconfig_dynamic_dict)

            # 执行动态量化转换
            convert_dynamic(model)

            # 定义函数：检查模型的量化结果
            def checkQuantized(model):
                # 检查模型的动态量化线性层 sub2.fc1
                self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=dtype)
                # 检查模型的动态量化线性层 sub2.fc2
                self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=dtype)
                # 检查模型的动态量化线性层 fc3
                self.checkDynamicQuantizedLinear(model.fc3, dtype=dtype)
                # 检查模型的脚本化结果及校准数据
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                # 检查模型是否有未配置的量化操作
                self.checkNoQconfig(model)

            # 调用检查函数，验证模型量化结果
            checkQuantized(model)

            # 测试单行 API 调用
            # 使用动态量化配置字典量化 NestedModel 模型
            model = quantize_dynamic(NestedModel().eval(), qconfig_dynamic_dict)
            # 再次调用检查函数，验证量化结果
            checkQuantized(model)

            # 测试集合 API 调用
            # 使用特定量化配置量化 NestedModel 模型
            model = quantize_dynamic(NestedModel().eval(), {'fc3', 'sub2', 'sub2.fc1'}, dtype=dtype)
            # 再次调用检查函数，验证量化结果
            checkQuantized(model)

    def test_type_match_rule(self):
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', All 'torch.nn.Linear' modules are quantized
        """
        # 遍历数据类型列表：torch.qint8 和 torch.float16
        for dtype in [torch.qint8, torch.float16]:
            # 创建并评估 NestedModel 模型
            model = NestedModel().eval()
            # 根据数据类型选择动态量化配置
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            # 定义量化配置字典，指定各模块的量化策略
            qconfig_dict = {
                'fc3': None,        # fc3 不量化
                'sub2.fc1': None,   # sub2.fc1 不量化
                torch.nn.Linear: qconfig  # torch.nn.Linear 模块使用指定的量化配置
            }

            # 准备动态量化，应用配置到模型
            prepare_dynamic(model, qconfig_dict)
            # 使用评估数据只测试模型
            test_only_eval_fn(model, self.calib_data)
            # 执行动态量化转换
            convert_dynamic(model)

            # 定义函数：检查模型的量化结果
            def checkQuantized(model):
                # 检查模型的动态量化线性层 sub1.fc
                self.checkDynamicQuantizedLinear(model.sub1.fc, dtype=dtype)
                # 检查模型的线性层 fc3
                self.checkLinear(model.fc3)
                # 检查模型的线性层 sub2.fc1
                self.checkLinear(model.sub2.fc1)
                # 检查模型的动态量化线性层 sub2.fc2
                self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=dtype)
                # 使用评估数据只测试模型
                test_only_eval_fn(model, self.calib_data)
                # 检查模型的脚本化结果及校准数据
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                # 检查模型是否有未配置的量化操作
                self.checkNoQconfig(model)

            # 调用检查函数，验证模型量化结果
            checkQuantized(model)

            # 测试单行 API 调用
            # 使用动态量化配置字典量化 NestedModel 模型
            model = quantize_dynamic(NestedModel().eval(), qconfig_dict, dtype=dtype)
            # 再次调用检查函数，验证量化结果
            checkQuantized(model)
    def test_per_channel_linear_quantize(self):
        r"""Test quantization for per_channel dynamic quantization
        """
        # 创建一个 NestedModel 实例并设置为评估模式
        model = NestedModel().eval()
        # 定义一个包含线性层的量化配置字典
        qconfig_dict = {
            torch.nn.Linear: per_channel_dynamic_qconfig
        }

        # 对模型应用动态量化准备阶段
        prepare_dynamic(model, qconfig_dict)
        # 使用评估模式下的模型进行测试
        test_only_eval_fn(model, self.calib_data)
        # 将模型转换为动态量化模型
        convert_dynamic(model)

        def checkQuantized(model):
            # 检查指定的线性层是否动态量化为 qint8 类型
            self.checkDynamicQuantizedLinear(model.sub1.fc, dtype=torch.qint8)
            self.checkDynamicQuantizedLinear(model.fc3, dtype=torch.qint8)
            self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=torch.qint8)
            self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=torch.qint8)
            # 使用评估模式下的模型进行测试
            test_only_eval_fn(model, self.calib_data)
            # 检查模型是否可脚本化，并验证保存和加载功能
            self.checkScriptable(model, self.calib_data, check_save_load=True)
            # 检查模型是否不存在任何量化配置
            self.checkNoQconfig(model)

        # 执行定义的检查函数以验证模型
        checkQuantized(model)
        # 使用一行 API 进行动态量化，并检查量化后的模型
        model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
        checkQuantized(model)

    def test_linear_relu_fusion(self):
        # 指定量化的数据类型为 qint8
        dtype = torch.qint8
        # 创建一个 LinearReluLinearModel 实例并设置为评估模式
        model = LinearReluLinearModel().eval()
        # 使用默认的动态量化配置
        qconfig = default_dynamic_qconfig
        qconfig_dict = {'' : qconfig}
        # 将模型中的 'fc1' 层和 'relu' 模块进行融合，操作是原地执行
        torch.ao.quantization.fuse_modules(model, [['fc1', 'relu']], inplace=True)
        # 对模型应用动态量化准备阶段
        prepare_dynamic(model, qconfig_dict)
        # 将模型转换为动态量化模型
        convert_dynamic(model)

        def checkQuantized(model):
            # 检查融合后的 'fc1' 层是否动态量化为指定类型的数据类型
            self.checkDynamicQuantizedLinearRelu(model.fc1, dtype)
            # 检查线性层 'fc2' 是否动态量化为指定类型的数据类型
            self.checkDynamicQuantizedLinear(model.fc2, dtype)
            # 检查模型是否可脚本化，并验证保存和加载功能
            self.checkScriptable(model, self.calib_data, check_save_load=True)
            # 检查模型是否不存在任何量化配置
            self.checkNoQconfig(model)

        # 执行定义的检查函数以验证模型
        checkQuantized(model)
    # 定义测试函数，用于测试动态量化的 LSTM 和 GRU 模块在 int8 和 fp16 数据类型上的脚本化和序列化功能
    def test_quantized_rnn(self, qconfig, dtype):
        r"""Test dynamic quantization, scriptability and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        # 迭代次数设定为 10
        niter = 10
        # 创建输入张量 x，包含重复的数据以模拟多次迭代
        x = torch.tensor([[100, -155],
                          [-155, 100],
                          [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1)
        # 定义用于不同模型的量化配置字典
        qconfig_dict = {
            torch.nn.LSTM : qconfig,
            torch.nn.GRU: qconfig
        }

        # 定义用于检查量化模型的函数
        def checkQuantized(model, module_type):
            # 映射不同模型类型到对应的量化动态模型
            mod_type_map = {'LSTM': torch.ao.nn.quantized.dynamic.LSTM,
                            'GRU': torch.ao.nn.quantized.dynamic.GRU}
            # 映射模型类型到相应的字符串表示
            mod_repr_map = {'LSTM': 'DynamicQuantizedLSTM',
                            'GRU': 'DynamicQuantizedGRU'}
            # 断言量化后的模型字符串表示中包含对应模型类型的字符串
            self.assertTrue(mod_repr_map[module_type] in str(model_quantized))
            # 检查动态量化模块的类型和数据类型
            self.checkDynamicQuantizedModule(model_quantized.mod, mod_type_map[module_type], dtype)

        # 遍历 LSTM 和 GRU 模型类型
        for module_type in ['LSTM', 'GRU']:
            # 创建指定类型的动态模型并设置为评估模式
            model = RNNDynamicModel(module_type).eval()

            # 根据数据类型选择进行动态量化
            if dtype == torch.float16:
                model_quantized = quantize_dynamic(model=model, dtype=dtype)
            else:
                model_quantized = quantize_dynamic(model=model, qconfig_spec=qconfig_dict, dtype=dtype)

            # 检查量化后的模型
            checkQuantized(model_quantized, module_type)
            # 检查模型是否可脚本化，并进行保存和加载的检查
            self.checkScriptable(model_quantized, [[x]], check_save_load=True)

            # 定义用于包装 PackedSequence 的脚本化包装类
            class ScriptWrapperPackedLSTM(torch.nn.Module):
                def __init__(self, cell):
                    super().__init__()
                    self.cell = cell

                def forward(self, x: PackedSequence) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
                    return self.cell(x)

            class ScriptWrapperPackedGRU(torch.nn.Module):
                def __init__(self, cell):
                    super().__init__()
                    self.cell = cell

                def forward(self, x: PackedSequence) -> Tuple[PackedSequence, torch.Tensor]:
                    return self.cell(x)

            # 映射模型类型到对应的脚本化包装类
            script_wrapper_map = {'LSTM': ScriptWrapperPackedLSTM,
                                  'GRU': ScriptWrapperPackedGRU}
            # 使用 pack_padded_sequence 封装输入数据
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, torch.tensor([10, 5, 2]))
            # 创建带有 PackedSequence 输入的脚本化模型
            model_with_packed_input = script_wrapper_map[module_type](model_quantized.mod)
            # 对脚本化模型执行前向传播
            model_with_packed_input(packed_input)
            # 对脚本化模型进行脚本化处理
            scripted = torch.jit.script(model_with_packed_input)
            # 使用脚本化模型执行前向传播
            scripted(packed_input)
            # 由于输入数据类型为 PackedSequence，无法使用追踪方法
            self._checkScriptable(model_with_packed_input, scripted, [[packed_input]], True)
    # 定义测试量化 RNN 单元的方法，接受量化配置和数据类型作为参数
    def test_quantized_rnn_cell(self, qconfig, dtype):
        r"""Test dynamic quantization, scriptability and serialization for dynamic quantized rnn cell modules on int8 and fp16
        """
        # 定义包含不同 RNN 单元类型和对应量化配置的字典
        qconfig_dict = {
            torch.nn.LSTMCell : qconfig,
            torch.nn.GRUCell : qconfig,
            torch.nn.RNNCell : qconfig
        }

        # 遍历每种 RNN 单元类型
        for module_type in ['LSTMCell', 'GRUCell', 'RNNTanh', 'RNNReLU']:
            # 创建指定类型的动态 RNNCell 模型并设置为评估模式
            model = RNNCellDynamicModel(module_type).eval()
            # 创建示例输入张量 x
            x = torch.tensor([[100, -155],
                             [-155, 100],
                             [100, -155]], dtype=torch.float)

            # 如果后端为 qnnpack 并且数据类型为 torch.float16，则跳过当前循环
            if torch.backends.quantized.engine == 'qnnpack' and dtype == torch.float16:
                continue
                # fp16 动态量化不支持 qnnpack 引擎

            # 根据数据类型选择是否对模型进行动态量化
            if dtype == torch.float16:
                model_quantized = quantize_dynamic(model=model, dtype=dtype)
            else:
                model_quantized = quantize_dynamic(model=model, qconfig_spec=qconfig_dict, dtype=dtype)

            # 定义用于检查量化模型的方法
            def checkQuantized(model, module_type):
                # 定义不同 RNN 单元类型到对应量化动态模型的映射
                mod_type_map = {'LSTMCell': torch.ao.nn.quantized.dynamic.LSTMCell,
                                'GRUCell': torch.ao.nn.quantized.dynamic.GRUCell,
                                'RNNTanh': torch.ao.nn.quantized.dynamic.RNNCell,
                                'RNNReLU': torch.ao.nn.quantized.dynamic.RNNCell}

                # 定义不同 RNN 单元类型到对应量化动态模型表示的映射
                mod_repr_map = {'LSTMCell': 'DynamicQuantizedLSTMCell',
                                'GRUCell': 'DynamicQuantizedGRUCell',
                                'RNNTanh': 'DynamicQuantizedRNNCell',
                                'RNNReLU': 'DynamicQuantizedRNNCell'}

                # 断言量化动态模型的表示中包含对应 RNN 单元类型的字符串
                self.assertTrue(mod_repr_map[module_type] in str(model_quantized))
                # 检查量化动态模型的模块是否为对应 RNN 单元类型的量化实现
                self.checkDynamicQuantizedModule(model_quantized.mod, mod_type_map[module_type], dtype)
                # 检查模型是否没有量化配置
                self.checkNoQconfig(model)

            # 对额外的表示进行烟雾测试（Smoke test）
            checkQuantized(model_quantized, module_type)
            # 检查模型是否可脚本化，并验证保存和加载过程
            self.checkScriptable(model_quantized, [[x]], check_save_load=True)
    def test_forward_hooks_preserved(self):
        r"""Test post-training dynamic quantization on preserving
        pre forward and post forward hooks of original model
        """
        # 循环测试不同数据类型的量化：torch.qint8 和 torch.float16
        for dtype in [torch.qint8, torch.float16]:
            # 创建一个单层线性动态模型，并设为评估模式
            model = SingleLayerLinearDynamicModel().eval()
            # 根据数据类型选择量化配置
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            qconfig_dict = {
                'fc1': qconfig
            }
            # 执行动态量化转换
            convert_dynamic(model)

            # 计数器，用于统计前向预处理和前向传播的调用次数
            counter = {
                'pre_forwards': 0,
                'forwards': 0,
            }

            # 定义前向预处理钩子函数，用于统计调用次数
            def fw_pre_hook(h_module, input):
                counter['pre_forwards'] += 1

            # 定义前向传播钩子函数，用于统计调用次数
            def fw_hook(h_module, input, output):
                counter['forwards'] += 1

            # 将钩子函数注册到模型的 fc1 层上
            model.fc1.register_forward_pre_hook(fw_pre_hook)
            model.fc1.register_forward_hook(fw_hook)
            # 准备模型进行动态量化
            prepare_dynamic(model, qconfig_dict)

            # 检查钩子函数是否成功注册到模型上
            def checkHooksIsPresent(model):
                self.assertObjectIn(fw_pre_hook, model.fc1._forward_pre_hooks.values())
                self.assertObjectIn(fw_hook, model.fc1._forward_hooks.values())
                self.assertEqual(len(model.fc1._forward_pre_hooks.values()), 1,
                                 "Extra pre forward hooks have appeared on a layer")
                self.assertEqual(len(model.fc1._forward_hooks.values()), 1,
                                 "Extra post forward hooks have appeared on a layer")

            # 调用检查函数，确认钩子函数已注册
            checkHooksIsPresent(model)
            # 使用评估数据测试模型
            test_only_eval_fn(model, self.calib_data)
            # 再次执行动态量化转换
            convert_dynamic(model)
            # 检查钩子函数是否仍然存在
            checkHooksIsPresent(model)

    @skipIfNoFBGEMM
    def test_embedding_bag_dynamic(self):
        class EmbeddingBagWithLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义包含线性层的嵌入袋模型
                self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12,
                                                 include_last_offset=True, scale_grad_by_freq=False, mode='sum')
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, indices, offsets, linear_in):
                # 前向传播函数
                return self.emb(indices, offsets), self.fc(linear_in)
        # 创建并评估嵌入袋模型
        model = EmbeddingBagWithLinear().eval()

        # 配置量化参数字典
        qconfig_dict = {
            torch.nn.EmbeddingBag : float_qparams_weight_only_qconfig,
            torch.nn.Linear: default_dynamic_qconfig
        }
        # 定义索引和偏移量张量
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        # 对模型进行动态量化
        q_model = quantize_dynamic(model, qconfig_dict)

        # 使用量化模型进行前向传播
        q_model(indices, offsets, torch.randn(5, 5))
        # 断言量化后的嵌入袋和线性层的类型
        self.assertTrue('QuantizedEmbeddingBag' in str(q_model.emb))
        self.assertTrue('DynamicQuantizedLinear' in str(q_model.fc))

    @skipIfNoFBGEMM
    def test_embedding_ops_dynamic(self):
        # 定义一个嵌套类 EmbeddingWithLinear，继承自 torch.nn.Module
        class EmbeddingWithLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个嵌入层，包含 10 个嵌入向量，每个向量的维度是 12
                self.emb = torch.nn.Embedding(
                    num_embeddings=10, embedding_dim=12, scale_grad_by_freq=False)
                # 初始化一个线性层，输入维度为 5，输出维度为 5
                self.fc = torch.nn.Linear(5, 5)

            # 定义模型的前向传播方法，接受 indices 和 linear_in 作为输入
            def forward(self, indices, linear_in):
                # 返回嵌入层的输出和线性层的输出
                return self.emb(indices), self.fc(linear_in)

        # 创建一个 EmbeddingWithLinear 类的实例，并设为评估模式
        model = EmbeddingWithLinear().eval()

        # 定义量化配置字典，指定嵌入层使用 float_qparams_weight_only_qconfig，线性层使用 default_dynamic_qconfig
        qconfig_dict = {
            torch.nn.Embedding : float_qparams_weight_only_qconfig,
            torch.nn.Linear: default_dynamic_qconfig
        }

        # 创建一个动态量化版本的模型，传入原始模型和量化配置字典
        q_model = quantize_dynamic(model, qconfig_dict)

        # 断言量化后的嵌入层包含 'QuantizedEmbedding' 字符串
        self.assertTrue('QuantizedEmbedding' in str(q_model.emb))
        # 断言量化后的线性层包含 'DynamicQuantizedLinear' 字符串
        self.assertTrue('DynamicQuantizedLinear' in str(q_model.fc))

        # 调用量化模型的前向传播方法，传入 indices 和随机生成的 5x5 的张量
        q_model(indices, torch.randn(5, 5))
# 如果当前脚本被直接运行，抛出运行时错误并显示提示信息
if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
```
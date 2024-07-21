# `.\pytorch\test\quantization\fx\test_model_report_fx.py`

```py
# Owner(s): ["oncall: quantization"]
from typing import Set

import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx  # 导入量化相关的函数和模块
import torch.nn.functional as F
from torch.ao.quantization import QConfig, QConfigMapping  # 导入量化配置相关模块
from torch.ao.quantization.fx._model_report.detector import (
    DynamicStaticDetector,  # 导入模型分析相关的检测器
    InputWeightEqualizationDetector,
    PerChannelDetector,
    OutlierDetector,
)
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver  # 导入模型分析报告观察器
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer  # 导入模型分析报告可视化器
from torch.ao.quantization.fx._model_report.model_report import ModelReport  # 导入模型分析报告
from torch.ao.quantization.observer import (
    HistogramObserver,  # 导入直方图观察器
    default_per_channel_weight_observer,  # 默认的按通道权重观察器
    default_observer  # 默认观察器
)
from torch.ao.nn.intrinsic.modules.fused import ConvReLU2d, LinearReLU  # 导入融合模块
from torch.testing._internal.common_quantization import (
    ConvModel,  # 导入量化测试相关的模型
    QuantizationTestCase,  # 量化测试用例
    SingleLayerLinearModel,  # 单层线性模型
    TwoLayerLinearModel,  # 双层线性模型
    skipIfNoFBGEMM,  # 如果没有FBGEMM则跳过测试
    skipIfNoQNNPACK,  # 如果没有QNNPACK则跳过测试
    override_quantized_engine,  # 覆盖量化引擎
)

"""
Partition of input domain:

Model contains: conv or linear, both conv and linear
    Model contains: ConvTransposeNd (not supported for per_channel)

Model is: post training quantization model, quantization aware training model
Model is: composed with nn.Sequential, composed in class structure

QConfig utilizes per_channel weight observer, backend uses non per_channel weight observer
QConfig_dict uses only one default qconfig, Qconfig dict uses > 1 unique qconfigs

Partition on output domain:

There are possible changes / suggestions, there are no changes / suggestions
"""

# Default output for string if no optimizations are possible
DEFAULT_NO_OPTIMS_ANSWER_STRING = (
    "Further Optimizations for backend {}: \nNo further per_channel optimizations possible."
)

# Example Sequential Model with multiple Conv and Linear with nesting involved
NESTED_CONV_LINEAR_EXAMPLE = torch.nn.Sequential(
    torch.nn.Conv2d(3, 3, 2, 1),  # 例子：包含多个卷积和线性层，存在嵌套
    torch.nn.Sequential(torch.nn.Linear(9, 27), torch.nn.ReLU()),
    torch.nn.Linear(27, 27),
    torch.nn.ReLU(),
    torch.nn.Conv2d(3, 3, 2, 1),
)

# Example Sequential Model with Conv sub-class example
LAZY_CONV_LINEAR_EXAMPLE = torch.nn.Sequential(
    torch.nn.LazyConv2d(3, 3, 2, 1),  # 例子：包含Conv子类的顺序模型
    torch.nn.Sequential(torch.nn.Linear(5, 27), torch.nn.ReLU()),
    torch.nn.ReLU(),
    torch.nn.Linear(27, 27),
    torch.nn.ReLU(),
    torch.nn.LazyConv2d(3, 3, 2, 1),
)

# Example Sequential Model with Fusion directly built into model
FUSION_CONV_LINEAR_EXAMPLE = torch.nn.Sequential(
    ConvReLU2d(torch.nn.Conv2d(3, 3, 2, 1), torch.nn.ReLU()),  # 例子：直接在模型中构建融合
    torch.nn.Sequential(LinearReLU(torch.nn.Linear(9, 27), torch.nn.ReLU())),
    LinearReLU(torch.nn.Linear(27, 27), torch.nn.ReLU()),
    torch.nn.Conv2d(3, 3, 2, 1),
)

# Test class
# example model to use for tests
class ThreeOps(nn.Module):
    # 初始化函数，继承父类的初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为3，输出维度为3
        self.linear = nn.Linear(3, 3)
        # 创建一个批量归一化层，输入通道数为3
        self.bn = nn.BatchNorm2d(3)
        # 创建一个ReLU激活函数层
        self.relu = nn.ReLU()

    # 前向传播函数，定义了数据流向
    def forward(self, x):
        # 输入x通过线性层计算得到输出
        x = self.linear(x)
        # 输出经过批量归一化层处理
        x = self.bn(x)
        # 处理后的输出经过ReLU激活函数
        x = self.relu(x)
        # 返回处理后的输出
        return x

    # 返回一个示例输入，这里返回一个形状为(1, 3, 3, 3)的张量元组
    def get_example_inputs(self):
        return (torch.randn(1, 3, 3, 3),)
class TwoThreeOps(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建两个 ThreeOps 模块作为子模块
        self.block1 = ThreeOps()
        self.block2 = ThreeOps()

    def forward(self, x):
        # 将输入 x 传递给 block1 模块
        x = self.block1(x)
        # 将 block1 的输出作为输入传递给 block2 模块
        y = self.block2(x)
        # 计算 x 和 y 的和
        z = x + y
        # 对 z 应用 ReLU 激活函数
        z = F.relu(z)
        return z

    def get_example_inputs(self):
        # 返回一个示例输入，这里是一个 shape 为 (1, 3, 3, 3) 的随机张量
        return (torch.randn(1, 3, 3, 3),)

class TestFxModelReportDetector(QuantizationTestCase):

    """Prepares and calibrate the model"""

    def _prepare_model_and_run_input(self, model, q_config_mapping, input):
        # 准备量化 FX 模型，并运行输入数据进行校准
        model_prep = torch.ao.quantization.quantize_fx.prepare_fx(model, q_config_mapping, input)  # prep model
        model_prep(input).sum()  # calibrate the model
        return model_prep

    """Case includes:
        one conv or linear
        post training quantization
        composed as module
        qconfig uses per_channel weight observer
        Only 1 qconfig in qconfig dict
        Output has no changes / suggestions
    """

    @skipIfNoFBGEMM
    def test_simple_conv(self):

        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = "fbgemm"

            # 创建 QConfigMapping 对象
            q_config_mapping = QConfigMapping()
            # 设置全局量化配置为当前引擎的默认配置
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

            # 创建一个随机输入张量
            input = torch.randn(1, 3, 10, 10)
            # 准备 ConvModel 模型并运行输入数据进行量化准备
            prepared_model = self._prepare_model_and_run_input(ConvModel(), q_config_mapping, input)

            # 运行检测器
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            # 生成检测器报告
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

            # 检查是否没有优化选项，并且 per_channel_status 中应该没有内容
            self.assertEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )

            # 检查模型中应只包含一个 conv 层
            self.assertEqual(per_channel_info["conv"]["backend"], torch.backends.quantized.engine)
            self.assertEqual(len(per_channel_info), 1)
            self.assertEqual(next(iter(per_channel_info)), "conv")
            # 检查是否支持逐通道量化
            self.assertEqual(
                per_channel_info["conv"]["per_channel_quantization_supported"],
                True,
            )
            # 检查是否使用了逐通道量化
            self.assertEqual(per_channel_info["conv"]["per_channel_quantization_used"], True)

    """Case includes:
        Multiple conv or linear
        post training quantization
        composed as module
        qconfig doesn't use per_channel weight observer
        Only 1 qconfig in qconfig dict
        Output has possible changes / suggestions
    """

    @skipIfNoQNNPACK
    def test_multi_linear_model_without_per_channel(self):
        # 使用 'qnnpack' 引擎覆盖当前的量化引擎设置
        with override_quantized_engine('qnnpack'):
            # 设置量化后端引擎为 "qnnpack"
            torch.backends.quantized.engine = "qnnpack"

            # 创建量化配置映射对象
            q_config_mapping = QConfigMapping()
            # 设置全局量化配置为当前引擎的默认量化配置
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

            # 准备模型并运行输入数据
            prepared_model = self._prepare_model_and_run_input(
                TwoLayerLinearModel(),  # 创建一个两层线性模型实例
                q_config_mapping,  # 使用上面创建的量化配置映射
                TwoLayerLinearModel().get_example_inputs()[0],  # 获取模型的示例输入
            )

            # 运行检测器
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            # 生成检测器报告
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

            # 检查是否存在优化建议
            self.assertNotEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )
            # 随机选择一个键来查看
            rand_key: str = next(iter(per_channel_info.keys()))
            # 断言所选键对应的通道量化信息的后端引擎与当前引擎一致
            self.assertEqual(per_channel_info[rand_key]["backend"], torch.backends.quantized.engine)
            # 断言通道量化信息字典的长度为2
            self.assertEqual(len(per_channel_info), 2)

            # 对于每个线性层，应该支持但未使用通道量化
            for linear_key in per_channel_info.keys():
                module_entry = per_channel_info[linear_key]

                # 断言当前模块支持通道量化
                self.assertEqual(module_entry["per_channel_quantization_supported"], True)
                # 断言当前模块未使用通道量化
                self.assertEqual(module_entry["per_channel_quantization_used"], False)
    # 定义一个测试方法，用于测试多个量化配置选项
    def test_multiple_q_config_options(self):

        # 使用上下文管理器修改量化引擎为'qnnpack'
        with override_quantized_engine('qnnpack'):
            # 设置全局量化引擎为'qnnpack'
            torch.backends.quantized.engine = "qnnpack"

            # 定义支持通道级量化的量化配置
            per_channel_qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=True),  # 激活函数的直方图观察器配置
                weight=default_per_channel_weight_observer,  # 默认的通道级权重观察器
            )

            # 设计一个包含卷积和线性层的模型类
            class ConvLinearModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(3, 3, 2, 1)  # 第一个卷积层
                    self.fc1 = torch.nn.Linear(9, 27)  # 第一个全连接层
                    self.relu = torch.nn.ReLU()  # ReLU激活函数
                    self.fc2 = torch.nn.Linear(27, 27)  # 第二个全连接层
                    self.conv2 = torch.nn.Conv2d(3, 3, 2, 1)  # 第二个卷积层

                # 前向传播函数
                def forward(self, x):
                    x = self.conv1(x)
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.fc2(x)
                    x = self.conv2(x)
                    return x

            # 创建量化配置映射对象
            q_config_mapping = QConfigMapping()
            # 设置全局量化配置为默认引擎对应的量化配置
            q_config_mapping.set_global(
                torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
            ).set_object_type(torch.nn.Conv2d, per_channel_qconfig)  # 设置卷积层的量化配置为通道级量化配置

            # 准备模型并使用输入数据运行
            prepared_model = self._prepare_model_and_run_input(
                ConvLinearModel(),
                q_config_mapping,
                torch.randn(1, 3, 10, 10),
            )

            # 运行检测器
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            # 生成检测器报告
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

            # 断言：应该有优化建议
            self.assertNotEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )

            # 断言：每个层都应该有通道信息
            self.assertEqual(len(per_channel_info), 4)

            # 对于每个层，检查其是否支持通道级量化但未使用
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]
                self.assertEqual(module_entry["per_channel_quantization_supported"], True)

                # 如果是全连接层，应该未使用通道级量化；如果是卷积层，应该已使用通道级量化
                if "fc" in key:
                    self.assertEqual(module_entry["per_channel_quantization_used"], False)
                elif "conv" in key:
                    self.assertEqual(module_entry["per_channel_quantization_used"], True)
                else:
                    raise ValueError("Should only contain conv and linear layers as key values")
    """
    Case includes:
        Multiple conv or linear
        post training quantization
        composed as sequential
        qconfig doesn't use per_channel weight observer
        Only 1 qconfig in qconfig dict
        Output has possible changes / suggestions
    """



    # 使用装饰器，如果不支持 QNNPACK，则跳过此测试
    @skipIfNoQNNPACK
    定义测试函数 test_sequential_model_format，用于测试顺序模型格式

        # 临时覆盖量化引擎为 qnnpack
        with override_quantized_engine('qnnpack'):
            设置 Torch 后端为 "qnnpack"
            torch.backends.quantized.engine = "qnnpack"

            # 创建 QConfigMapping 对象，设置全局量化配置为引擎默认配置
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

            # 准备模型并运行输入数据，获取准备后的模型
            prepared_model = self._prepare_model_and_run_input(
                NESTED_CONV_LINEAR_EXAMPLE,  # 使用 NESTED_CONV_LINEAR_EXAMPLE 模型
                q_config_mapping,  # 使用配置映射
                torch.randn(1, 3, 10, 10),  # 随机生成的输入张量
            )

            # 运行检测器生成检测报告
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

            # 检查优化信息字符串是否不等于默认无优化信息字符串
            self.assertNotEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )

            # 确保 per_channel_info 的长度为 4
            self.assertEqual(len(per_channel_info), 4)

            # 对于每个层，检查是否支持通道量化但未使用
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]

                self.assertEqual(module_entry["per_channel_quantization_supported"], True)
                self.assertEqual(module_entry["per_channel_quantization_used"], False)



    """
    Case includes:
        Multiple conv or linear
        post training quantization
        composed as sequential
        qconfig doesn't use per_channel weight observer
        Only 1 qconfig in qconfig dict
        Output has possible changes / suggestions
    """



    @skipIfNoQNNPACK
    # 定义一个测试方法，用于测试考虑子类的量化转换

    # 使用上下文管理器覆盖量化引擎为 'qnnpack'
    with override_quantized_engine('qnnpack'):
        # 设置 Torch 的量化引擎为 'qnnpack'
        torch.backends.quantized.engine = "qnnpack"

        # 创建 QConfigMapping 对象，用于管理量化配置
        q_config_mapping = QConfigMapping()
        # 设置全局量化配置为默认配置，使用当前量化引擎
        q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

        # 准备模型并运行输入数据，获取准备好的模型
        prepared_model = self._prepare_model_and_run_input(
            LAZY_CONV_LINEAR_EXAMPLE,
            q_config_mapping,
            torch.randn(1, 3, 10, 10),
        )

        # 创建 PerChannelDetector 对象，使用当前量化引擎
        per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
        # 生成检测器报告，返回优化建议和每通道信息
        optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

        # 断言优化字符串不等于默认无优化建议字符串格式化后的结果
        self.assertNotEqual(
            optims_str,
            DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
        )

        # 断言每通道信息列表长度为 4
        self.assertEqual(len(per_channel_info), 4)

        # 对每个层的每通道信息进行断言
        for key in per_channel_info.keys():
            module_entry = per_channel_info[key]

            # 断言每通道量化支持为 True
            self.assertEqual(module_entry["per_channel_quantization_supported"], True)
            # 断言每通道量化使用为 False
            self.assertEqual(module_entry["per_channel_quantization_used"], False)

"""Case includes:
    Multiple conv or linear
    post training quantization
    composed as sequential
    qconfig uses per_channel weight observer
    Only 1 qconfig in qconfig dict
    Output has no possible changes / suggestions
"""

# 跳过没有 FBGEMM 的测试
@skipIfNoFBGEMM
    # 定义一个测试方法，用于测试在顺序模型中的融合层

    # 使用 'fbgemm' 引擎覆盖量化引擎设置
    with override_quantized_engine('fbgemm'):
        # 设置当前的量化引擎为 "fbgemm"
        torch.backends.quantized.engine = "fbgemm"

        # 创建 QConfigMapping 对象
        q_config_mapping = QConfigMapping()
        # 设置全局量化配置为当前引擎的默认配置
        q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

        # 准备模型并运行输入数据
        prepared_model = self._prepare_model_and_run_input(
            FUSION_CONV_LINEAR_EXAMPLE,
            q_config_mapping,
            torch.randn(1, 3, 10, 10),
        )

        # 实例化 PerChannelDetector 对象，使用当前量化引擎
        per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
        # 生成检测器报告
        optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

        # 断言优化字符串与预期默认无优化答复字符串相等
        self.assertEqual(
            optims_str,
            DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
        )

        # 断言 per_channel_info 中条目数为 4
        self.assertEqual(len(per_channel_info), 4)

        # 遍历 per_channel_info 中的每个条目，检查量化支持和使用情况
        for key in per_channel_info.keys():
            module_entry = per_channel_info[key]
            # 断言每个条目的 per_channel_quantization_supported 属性为 True
            self.assertEqual(module_entry["per_channel_quantization_supported"], True)
            # 断言每个条目的 per_channel_quantization_used 属性为 True
            self.assertEqual(module_entry["per_channel_quantization_used"], True)
    def test_qat_aware_model_example(self):
        # 定义一个测试方法，测试量化感知训练（QAT）模型的示例

        # first we want a QAT model
        # 首先定义一个QAT模型类
        class QATConvLinearReluModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # QuantStub converts tensors from floating point to quantized
                # QuantStub将浮点张量转换为量化张量
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.bn = torch.nn.BatchNorm2d(1)
                self.relu = torch.nn.ReLU()
                # DeQuantStub converts tensors from quantized to floating point
                # DeQuantStub将量化张量转换为浮点张量
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        with override_quantized_engine('qnnpack'):
            # temporarily override the quantized engine to 'qnnpack'

            # create a model instance
            # 创建一个模型实例
            model_fp32 = QATConvLinearReluModel()

            # get QAT specific default QConfig for 'qnnpack'
            # 获取QAT特定的默认QConfig配置，这里是针对'qnnpack'引擎
            model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")

            # model must be in eval mode for fusion
            # 模型必须处于评估模式以进行模块融合
            model_fp32.eval()
            model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [["conv", "bn", "relu"]])

            # model must be set to train mode for QAT logic to work
            # 模型必须设置为训练模式以使QAT逻辑生效
            model_fp32_fused.train()

            # prepare the model for QAT, different than for post training quantization
            # 准备模型以进行QAT，与训练后量化不同的准备过程
            model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused)

            # run the detector
            # 运行检测器
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(model_fp32_prepared)

            # there should be optims possible
            # 应该存在优化的可能性
            self.assertNotEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )

            # make sure it was able to find the single conv in the fused model
            # 确保检测器能够找到融合模型中的单个卷积层
            self.assertEqual(len(per_channel_info), 1)

            # for the one conv, it should still give advice to use different qconfig
            # 对于单个卷积层，它仍然应该建议使用不同的QConfig配置
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]
                self.assertEqual(module_entry["per_channel_quantization_supported"], True)
                self.assertEqual(module_entry["per_channel_quantization_used"], False)
"""
Partition on Domain / Things to Test

- All zero tensor
- Multiple tensor dimensions
- All of the outward facing functions
- Epoch min max are correctly updating
- Batch range is correctly averaging as expected
- Reset for each epoch is correctly resetting the values

Partition on Output
- the calcuation of the ratio is occurring correctly

"""


class TestFxModelReportObserver(QuantizationTestCase):
    # 定义一个嵌套的修改后的单层线性模型类
    class NestedModifiedSingleLayerLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 初始化模型报告观察器1
            self.obs1 = ModelReportObserver()
            # 初始化单层线性模型
            self.mod1 = SingleLayerLinearModel()
            # 初始化模型报告观察器2
            self.obs2 = ModelReportObserver()
            # 定义一个输入维度为5，输出维度为5的线性层
            self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
            # 定义ReLU激活函数
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            # 在输入x上应用模型报告观察器1
            x = self.obs1(x)
            # 在输入x上应用单层线性模型
            x = self.mod1(x)
            # 在输入x上应用模型报告观察器2
            x = self.obs2(x)
            # 在输入x上应用线性层fc1
            x = self.fc1(x)
            # 在输入x上应用ReLU激活函数
            x = self.relu(x)
            # 返回处理后的输出x
            return x
    def run_model_and_common_checks(self, model, ex_input, num_epochs, batch_size):
        # 将输入数据按批次分割
        split_up_data = torch.split(ex_input, batch_size)
        
        # 遍历每个训练轮次
        for epoch in range(num_epochs):
            # 重置所有模型报告观察器的批次和轮次值
            model.apply(
                lambda module: module.reset_batch_and_epoch_values()
                if isinstance(module, ModelReportObserver)
                else None
            )

            # 快速检查重置是否发生
            self.assertEqual(
                model.obs1.average_batch_activation_range,
                torch.tensor(float(0)),
            )
            self.assertEqual(model.obs1.epoch_activation_min, torch.tensor(float("inf")))
            self.assertEqual(model.obs1.epoch_activation_max, torch.tensor(float("-inf")))

            # 遍历每个批次并执行模型训练
            for index, batch in enumerate(split_up_data):

                num_tracked_so_far = model.obs1.num_batches_tracked
                self.assertEqual(num_tracked_so_far, index)

                # 获取批次的最小值和最大值
                batch_min, batch_max = torch.aminmax(batch)
                current_average_range = model.obs1.average_batch_activation_range
                current_epoch_min = model.obs1.epoch_activation_min
                current_epoch_max = model.obs1.epoch_activation_max

                # 将输入数据通过模型
                model(ex_input)

                # 检查平均批次激活范围是否正确更新
                correct_updated_value = (current_average_range * num_tracked_so_far + (batch_max - batch_min)) / (
                    num_tracked_so_far + 1
                )
                self.assertEqual(
                    model.obs1.average_batch_activation_range,
                    correct_updated_value,
                )

                # 如果当前轮次的最大值和最小值之差大于0，则检查批次到轮次比率
                if current_epoch_max - current_epoch_min > 0:
                    self.assertEqual(
                        model.obs1.get_batch_to_epoch_ratio(),
                        correct_updated_value / (current_epoch_max - current_epoch_min),
                    )

    """
    案例包括：
        全零张量
        维度大小为2
        运行1个轮次
        运行10个批次
        测试输入数据观察器
    """
    def test_zero_tensor_errors(self):
        # initialize the model
        model = self.NestedModifiedSingleLayerLinear()

        # generate the desired input
        ex_input = torch.zeros((10, 1, 5))

        # run it through the model and do general tests
        self.run_model_and_common_checks(model, ex_input, 1, 1)

        # make sure final values are all 0
        self.assertEqual(model.obs1.epoch_activation_min, 0)  # Check minimum activation value after 1 epoch
        self.assertEqual(model.obs1.epoch_activation_max, 0)  # Check maximum activation value after 1 epoch
        self.assertEqual(model.obs1.average_batch_activation_range, 0)  # Check average batch activation range

        # we should get an error if we try to calculate the ratio
        with self.assertRaises(ValueError):
            ratio_val = model.obs1.get_batch_to_epoch_ratio()

    """Case includes:
    non-zero tensor
    dim size = 2
    run for 1 epoch
    run for 1 batch
    tests input data observer
    """

    def test_single_batch_of_ones(self):
        # initialize the model
        model = self.NestedModifiedSingleLayerLinear()

        # generate the desired input
        ex_input = torch.ones((1, 1, 5))

        # run it through the model and do general tests
        self.run_model_and_common_checks(model, ex_input, 1, 1)

        # make sure final values are all 0 except for range
        self.assertEqual(model.obs1.epoch_activation_min, 1)  # Check minimum activation value after 1 epoch
        self.assertEqual(model.obs1.epoch_activation_max, 1)  # Check maximum activation value after 1 epoch
        self.assertEqual(model.obs1.average_batch_activation_range, 0)  # Check average batch activation range

        # we should get an error if we try to calculate the ratio
        with self.assertRaises(ValueError):
            ratio_val = model.obs1.get_batch_to_epoch_ratio()

    """Case includes:
    non-zero tensor
    dim size = 2
    run for 10 epoch
    run for 15 batch
    tests non input data observer
    """

    def test_observer_after_relu(self):

        # model specific to this test
        class NestedModifiedObserverAfterRelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.mod1 = SingleLayerLinearModel()
                self.obs2 = ModelReportObserver()
                self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.obs1(x)
                x = self.mod1(x)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.obs2(x)
                return x

        # initialize the model
        model = NestedModifiedObserverAfterRelu()

        # generate the desired input
        ex_input = torch.randn((15, 1, 5))

        # run it through the model and do general tests
        self.run_model_and_common_checks(model, ex_input, 10, 15)

    """Case includes:
        non-zero tensor
        dim size = 2
        run for multiple epoch
        run for multiple batch
        tests input data observer
    """
"""
Partition on domain / things to test

There is only a single test case for now.

This will be more thoroughly tested with the implementation of the full end to end tool coming soon.
"""

# 定义一个测试类 TestFxModelReportDetectDynamicStatic，继承自 QuantizationTestCase
class TestFxModelReportDetectDynamicStatic(QuantizationTestCase):

    # 使用装饰器 skipIfNoFBGEMM，条件为没有 FBGEMM 引擎时跳过测试
    @skipIfNoFBGEMM
    # 定义测试类 TestFxModelReportClass，继承自 QuantizationTestCase
class TestFxModelReportClass(QuantizationTestCase):

    # 使用装饰器 skipIfNoFBGEMM，条件为没有 FBGEMM 引擎时跳过测试
    def test_constructor(self):
        """
        Tests the constructor of the ModelReport class.
        Specifically looks at:
        - The desired reports
        - Ensures that the observers of interest are properly initialized
        """

        # 使用 override_quantized_engine('fbgemm') 来设置测试的后端引擎为 'fbgemm'
        with override_quantized_engine('fbgemm'):
            # 设置 torch 后端量化引擎为 "fbgemm"
            torch.backends.quantized.engine = "fbgemm"
            backend = torch.backends.quantized.engine

            # 创建一个 ThreeOps 模型实例
            model = ThreeOps()
            # 创建 QConfigMapping 实例
            q_config_mapping = QConfigMapping()
            # 设置全局量化配置为默认配置
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))
            # 准备模型的量化过程
            model_prep = quantize_fx.prepare_fx(model, q_config_mapping, model.get_example_inputs()[0])

            # 创建一个包含 DynamicStaticDetector 和 PerChannelDetector(backend) 的测试检测器集合
            test_detector_set = {DynamicStaticDetector(), PerChannelDetector(backend)}
            # 使用空的检测器初始化 ModelReport 实例
            model_report = ModelReport(model_prep, test_detector_set)

            # 确保内部有效报告名称匹配
            detector_name_set = {detector.get_detector_name() for detector in test_detector_set}
            self.assertEqual(model_report.get_desired_reports_names(), detector_name_set)

            # 现在尝试没有有效报告的情况，应该会引发 ValueError 错误
            with self.assertRaises(ValueError):
                model_report = ModelReport(model, set())

            # 预期的兴趣观察者条目数量
            num_expected_entries = len(test_detector_set)
            self.assertEqual(len(model_report.get_observers_of_interest()), num_expected_entries)

            # 确保每个兴趣观察者列表长度为 0
            for value in model_report.get_observers_of_interest().values():
                self.assertEqual(len(value), 0)

    # 使用装饰器 skipIfNoFBGEMM
    @skipIfNoFBGEMM
    # 定义一个方法，用于获取模型中的 ModelReportObserver 模块数量以及图结构中的 model_report 节点数量
    def get_module_and_graph_cnts(self, callibrated_fx_module):
        """
        计算模型中 ModelReportObserver 模块的数量以及图结构的节点数量。
        返回一个包含两个元素的元组：
        int: 模型中找到的 ModelReportObservers 的数量
        int: 图中找到的 model_report 节点的数量
        """
        # 获取存储为模块的观察者数量
        modules_observer_cnt = 0
        # 遍历模型的命名模块
        for fqn, module in callibrated_fx_module.named_modules():
            # 如果模块是 ModelReportObserver 的实例
            if isinstance(module, ModelReportObserver):
                # 增加模块观察者计数
                modules_observer_cnt += 1

        # 获取图中观察者的数量
        model_report_str_check = "model_report"
        graph_observer_cnt = 0
        # 确保图中观察者的参数正确
        for node in callibrated_fx_module.graph.nodes:
            # 不是所有节点目标都是字符串，因此需要检查
            if isinstance(node.target, str) and model_report_str_check in node.target:
                # 如果找到了一个图中的观察者，则增加计数
                graph_observer_cnt += 1

        # 返回包含两个计数的元组
        return (modules_observer_cnt, graph_observer_cnt)

    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    def test_generate_visualizer(self):
        """
        Tests that the ModelReport class can properly create the ModelReportVisualizer instance
        Checks that:
            - Correct number of modules are represented
            - Modules are sorted
            - Correct number of features for each module
        """
        # 使用 'fbgemm' 引擎覆盖量化引擎设置
        with override_quantized_engine('fbgemm'):
            # 设置此测试的后端引擎为 "fbgemm"
            torch.backends.quantized.engine = "fbgemm"

            # 测试多个检测器的情况
            detector_set = set()
            # 添加异常检测器，参考百分位数为0.95
            detector_set.add(OutlierDetector(reference_percentile=0.95))
            # 添加输入权重均衡检测器，阈值为0.5
            detector_set.add(InputWeightEqualizationDetector(0.5))

            # 创建 TwoThreeOps 模型实例
            model = TwoThreeOps()

            # 获取测试模型并进行校准
            prepared_for_callibrate_model, mod_report = _get_prepped_for_calibration_model_helper(
                model, detector_set, model.get_example_inputs()[0]
            )

            # 现在实际校准模型
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)
            prepared_for_callibrate_model(example_input)

            # 尝试在不生成报告的情况下进行可视化，应抛出异常
            with self.assertRaises(Exception):
                mod_rep_visualizaiton = mod_report.generate_visualizer()

            # 通过运行 ModelReport 实例来获取报告
            generated_report = mod_report.generate_model_report(remove_inserted_observers=False)

            # 现在获取可视化器，不应该出错
            mod_rep_visualizer: ModelReportVisualizer = mod_report.generate_visualizer()

            # 由于我们使用了异常检测器，它检查每个基本级别模块，应该有六个条目在有序字典中
            mod_fqns_to_features = mod_rep_visualizer.generated_reports

            # 断言生成的模块数量为6
            self.assertEqual(len(mod_fqns_to_features), 6)

            # 线性模块具有的特征数为20，因为有一个共同的数据点
            for module_fqn in mod_fqns_to_features:
                if ".linear" in module_fqn:
                    linear_info = mod_fqns_to_features[module_fqn]
                    self.assertEqual(len(linear_info), 20)

    @skipIfNoFBGEMM
    def test_qconfig_mapping_generation(self):
        """
        Tests for generation of qconfigs by ModelReport API
        - Tests that qconfigmapping is generated
        - Tests that mappings include information for for relavent modules
        """
        # 使用 'fbgemm' 作为量化引擎进行测试
        with override_quantized_engine('fbgemm'):
            # 设置此测试的后端引擎为 'fbgemm'
            torch.backends.quantized.engine = "fbgemm"
            
            # 测试多个检测器的情况
            detector_set = set()
            detector_set.add(PerChannelDetector())
            detector_set.add(DynamicStaticDetector())

            # 创建一个包含两到三个操作的模型实例
            model = TwoThreeOps()

            # 获取测试模型并进行校准
            prepared_for_callibrate_model, mod_report = _get_prepped_for_calibration_model_helper(
                model, detector_set, model.get_example_inputs()[0]
            )

            # 现在实际进行模型校准
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)
            prepared_for_callibrate_model(example_input)

            # 获取不带错误的映射
            qconfig_mapping = mod_report.generate_qconfig_mapping()

            # 通过运行 ModelReport 实例来获取报告
            generated_report = mod_report.generate_model_report(remove_inserted_observers=False)

            # 获取可视化器以按模块的完全限定名访问重新格式化的报告
            mod_reports_by_fqn = mod_report.generate_visualizer().generated_reports

            # 比较映射的条目和报告的条目
            # 应该有相同数量的条目
            self.assertEqual(len(qconfig_mapping.module_name_qconfigs), len(mod_reports_by_fqn))

            # 对于非空条目，应该有2个，因为我们只有适用的线性模块
            # 所以应该对每个模块名称都有建议
            self.assertEqual(len(qconfig_mapping.module_name_qconfigs), 2)

            # 只有两个线性模块，确保权重使用每通道最小最大观察器，因为是 fbgemm
            # 还有静态分布，因为只有简单的单次校准
            for key in qconfig_mapping.module_name_qconfigs:
                config = qconfig_mapping.module_name_qconfigs[key]
                self.assertEqual(config.weight, default_per_channel_weight_observer)
                self.assertEqual(config.activation, default_observer)

            # 确保这些配置可以用于准备模型
            prepared = quantize_fx.prepare_fx(TwoThreeOps(), qconfig_mapping, example_input)

            # 现在转换模型以确保转换无误
            converted = quantize_fx.convert_fx(prepared)

    @skipIfNoFBGEMM
    def test_equalization_mapping_generation(self):
        """
        Tests for generation of qconfigs by ModelReport API
        - Tests that equalization config generated when input-weight equalization detector used
        - Tests that mappings include information for for relavent modules
        """
        with override_quantized_engine('fbgemm'):
            # 设置测试的后端引擎为 'fbgemm'
            torch.backends.quantized.engine = "fbgemm"
            
            # 使用一个输入权重均衡检测器的集合进行测试
            detector_set = set()
            detector_set.add(InputWeightEqualizationDetector(0.6))
            
            # 创建一个包含两到三个操作的模型
            model = TwoThreeOps()

            # 获取测试模型并进行校准
            prepared_for_callibrate_model, mod_report = _get_prepped_for_calibration_model_helper(
                model, detector_set, model.get_example_inputs()[0]
            )

            # 现在我们实际进行模型校准
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)

            prepared_for_callibrate_model(example_input)

            # 获取不包含错误的映射
            qconfig_mapping = mod_report.generate_qconfig_mapping()
            equalization_mapping = mod_report.generate_equalization_mapping()

            # 测试相等化映射的一些更简单的情况

            # 对于这种情况不应该有任何相等化建议
            self.assertEqual(len(qconfig_mapping.module_name_qconfigs), 2)

            # 确保这些可以用来准备模型
            prepared = quantize_fx.prepare_fx(
                TwoThreeOps(),
                qconfig_mapping,
                example_input,
                _equalization_config=equalization_mapping
            )

            # 现在转换模型以确保转换没有错误
            converted = quantize_fx.convert_fx(prepared)
class TestFxDetectInputWeightEqualization(QuantizationTestCase):
    
    class SimpleConv(torch.nn.Module):
        def __init__(self, con_dims):
            super().__init__()
            self.relu = torch.nn.ReLU()  # 初始化ReLU激活函数
            self.conv = torch.nn.Conv2d(con_dims[0], con_dims[1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # 初始化二维卷积层，设定参数包括输入和输出通道数、卷积核大小、步幅、填充，无偏置

        def forward(self, x):
            x = self.conv(x)  # 进行卷积操作
            x = self.relu(x)  # 应用ReLU激活函数
            return x

    class TwoBlockComplexNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = TestFxDetectInputWeightEqualization.SimpleConv((3, 32))
            # 初始化第一个简单卷积块
            self.block2 = TestFxDetectInputWeightEqualization.SimpleConv((3, 3))
            # 初始化第二个简单卷积块
            self.conv = torch.nn.Conv2d(32, 3, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
            # 初始化一维卷积层，用于融合过程，设定参数包括输入输出通道数、卷积核大小、步幅、填充，无偏置
            self.linear = torch.nn.Linear(768, 10)
            # 初始化线性层，设定输入和输出特征数
            self.relu = torch.nn.ReLU()
            # 初始化ReLU激活函数

        def forward(self, x):
            x = self.block1(x)  # 对输入数据应用第一个卷积块
            x = self.conv(x)  # 对第一个卷积块的输出进行卷积操作
            y = self.block2(x)  # 对卷积结果再应用第二个卷积块
            y = y.repeat(1, 1, 2, 2)  # 将第二个卷积块的输出在空间维度上重复两次
            z = x + y  # 将第一个卷积块的输出与重复后的第二个卷积块的输出相加
            z = z.flatten(start_dim=1)  # 将张量展平，从第一个维度开始
            z = self.linear(z)  # 应用线性层
            z = self.relu(z)  # 应用ReLU激活函数
            return z

        def get_fusion_modules(self):
            return [['conv', 'relu']]  # 返回需要融合的模块列表

        def get_example_inputs(self):
            return (torch.randn((1, 3, 28, 28)),)  # 返回一个示例输入数据元组

    class ReluOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()  # 初始化ReLU激活函数

        def forward(self, x):
            x = self.relu(x)  # 应用ReLU激活函数
            return x

        def get_example_inputs(self):
            return (torch.arange(27).reshape((1, 3, 3, 3)),)  # 返回一个示例输入数据元组

    def _get_prepped_for_calibration_model(self, model, detector_set, fused=False):
        r"""Returns a model that has been prepared for callibration and corresponding model_report"""

        # pass in necessary inputs to helper
        example_input = model.get_example_inputs()[0]  # 获取模型的示例输入数据
        return _get_prepped_for_calibration_model_helper(model, detector_set, example_input, fused)
        # 调用辅助函数，准备用于校准的模型和相应的模型报告

    @skipIfNoFBGEMM
    def test_input_weight_equalization_determine_points(self):
        # 使用 fbgemm 引擎覆盖当前的量化引擎环境
        with override_quantized_engine('fbgemm'):
            
            # 创建一个包含 InputWeightEqualizationDetector 的集合
            detector_set = {InputWeightEqualizationDetector(0.5)}

            # 获取经过预处理以进行校准的非融合模型和融合模型
            non_fused = self._get_prepped_for_calibration_model(self.TwoBlockComplexNet(), detector_set)
            fused = self._get_prepped_for_calibration_model(self.TwoBlockComplexNet(), detector_set, fused=True)

            # 断言报告应对融合模型和非融合模型都给出相同的节点数
            for prepared_for_callibrate_model, mod_report in [non_fused, fused]:

                # 要检查的支持模块集合
                mods_to_check = {nn.Linear, nn.Conv2d}

                # 获取图中所有节点的目标（全限定名）
                node_fqns = {node.target for node in prepared_for_callibrate_model.graph.nodes}

                # 应该有 4 个节点包含插入的观察器
                correct_number_of_obs_inserted = 4
                number_of_obs_found = 0
                obs_name_to_find = InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME

                for node in prepared_for_callibrate_model.graph.nodes:
                    # 如果目标中包含观察器名称，则找到一个观察器
                    if obs_name_to_find in str(node.target):
                        number_of_obs_found += 1

                self.assertEqual(number_of_obs_found, correct_number_of_obs_inserted)

                # 断言每个期望的模块都已插入观察器
                for fqn, module in prepared_for_callibrate_model.named_modules():
                    # 检查模块是否属于支持的模块之一
                    is_in_include_list = sum(isinstance(module, x) for x in mods_to_check) > 0

                    if is_in_include_list:
                        # 确保模块具有观察器属性
                        self.assertTrue(hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))
                    else:
                        # 如果不是支持的类型，则不应具有观察器属性
                        self.assertTrue(not hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))

    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    def test_input_weight_equalization_report_gen_empty(self):
        # 测试对没有任何层的模型生成报告
        # 使用 fbgemm 并创建我们的模型实例
        # 然后使用检测器创建模型报告实例
        with override_quantized_engine('fbgemm'):
            # 创建输入权重均衡检测器实例
            test_input_weight_detector = InputWeightEqualizationDetector(0.4)
            detector_set = {test_input_weight_detector}
            # 创建只包含 Relu 层的模型实例
            model = self.ReluOnly()
            # 准备模型以进行校准
            prepared_for_callibrate_model, model_report = self._get_prepped_for_calibration_model(model, detector_set)

            # 现在实际进行模型校准
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)

            prepared_for_callibrate_model(example_input)

            # 通过 ModelReport 实例生成模型报告
            generated_report = model_report.generate_model_report(True)

            # 检查生成的报告大小，应该只有一个检测器
            self.assertEqual(len(generated_report), 1)

            # 获取输入权重均衡的具体报告
            input_weight_str, input_weight_dict = generated_report[test_input_weight_detector.get_detector_name()]

            # 应该没有层，因为只有一个 Relu
            self.assertEqual(len(input_weight_dict), 0)

            # 确保字符串只有两行，如果没有建议的话应该是这样
            self.assertEqual(input_weight_str.count("\n"), 2)
class TestFxDetectOutliers(QuantizationTestCase):

    class LargeBatchModel(torch.nn.Module):
        def __init__(self, param_size):
            super().__init__()
            self.param_size = param_size
            self.linear = torch.nn.Linear(param_size, param_size)
            self.relu_1 = torch.nn.ReLU()
            self.conv = torch.nn.Conv2d(param_size, param_size, 1)
            self.relu_2 = torch.nn.ReLU()

        def forward(self, x):
            x = self.linear(x)        # 使用线性层对输入进行线性变换
            x = self.relu_1(x)        # 对线性层的输出进行 ReLU 激活
            x = self.conv(x)          # 使用卷积层对数据进行卷积操作
            x = self.relu_2(x)        # 对卷积层的输出进行 ReLU 激活
            return x

        def get_example_inputs(self):
            param_size = self.param_size
            return (torch.randn((1, param_size, param_size, param_size)),)  # 返回一个符合模型输入要求的示例输入

        def get_outlier_inputs(self):
            param_size = self.param_size
            random_vals = torch.randn((1, param_size, param_size, param_size))
            # 将一些输入中的值修改为异常大的值
            random_vals[:, 0:param_size:2, 0, 3] = torch.tensor([3.28e8])
            return (random_vals,)

    def _get_prepped_for_calibration_model(self, model, detector_set, use_outlier_data=False):
        r"""Returns a model that has been prepared for callibration and corresponding model_report"""
        # 调用通用的辅助函数进行模型校准
        example_input = model.get_example_inputs()[0]

        # 如果需要使用异常数据进行测试，替换示例输入
        if use_outlier_data:
            example_input = model.get_outlier_inputs()[0]

        return _get_prepped_for_calibration_model_helper(model, detector_set, example_input)

    @skipIfNoFBGEMM
    def test_outlier_detection_determine_points(self):
        # 使用 fbgemm 引擎进行测试，并创建我们的模型实例
        # 然后创建带有异常检测器的模型报告实例
        # 类似于 InputWeightEqualization 的测试，但由于重构不可行的关键差异
        # 不显式测试融合，因为 fx 工作流会自动处理

        with override_quantized_engine('fbgemm'):
            # 设置异常检测器集合，此处仅包含一个 Percentile 异常检测器
            detector_set = {OutlierDetector(reference_percentile=0.95)}

            # 获取测试模型并进行校准
            prepared_for_callibrate_model, mod_report = self._get_prepped_for_calibration_model(
                self.LargeBatchModel(param_size=128), detector_set
            )

            # 要检查的支持模块集合
            mods_to_check = {nn.Linear, nn.Conv2d, nn.ReLU}

            # 应该有 4 个节点具有插入的观察器
            correct_number_of_obs_inserted = 4
            number_of_obs_found = 0
            obs_name_to_find = InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME

            # 统计发现的观察器数量
            number_of_obs_found = sum(
                1 if obs_name_to_find in str(node.target) else 0 for node in prepared_for_callibrate_model.graph.nodes
            )
            self.assertEqual(number_of_obs_found, correct_number_of_obs_inserted)

            # 断言每个期望的模块是否已插入观察器
            for fqn, module in prepared_for_callibrate_model.named_modules():
                # 检查模块是否在支持模块列表中
                is_in_include_list = isinstance(module, tuple(mods_to_check))

                if is_in_include_list:
                    # 确保模块具有指定的观察器属性
                    self.assertTrue(hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))
                else:
                    # 如果不是支持的类型，则不应该有观察器附加
                    self.assertTrue(not hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))

    @skipIfNoFBGEMM
    `
        def test_no_outlier_report_gen(self):
            # 使用 fbgemm 引擎创建模型实例，并使用检测器创建模型报告实例
            with override_quantized_engine('fbgemm'):
    
                # 创建多个检测器的实例
                outlier_detector = OutlierDetector(reference_percentile=0.95)
                dynamic_static_detector = DynamicStaticDetector(tolerance=0.5)
    
                # 设置参数大小
                param_size: int = 4
                # 创建检测器集合
                detector_set = {outlier_detector, dynamic_static_detector}
                # 创建大型批处理模型实例
                model = self.LargeBatchModel(param_size=param_size)
    
                # 准备用于校准的模型，并获取模型报告实例
                prepared_for_callibrate_model, mod_report = self._get_prepped_for_calibration_model(
                    model, detector_set
                )
    
                # 获取示例输入并进行数据类型转换
                example_input = model.get_example_inputs()[0]
                example_input = example_input.to(torch.float)
    
                # 执行模型校准
                prepared_for_callibrate_model(example_input)
    
                # 通过 ModelReport 实例生成报告
                generated_report = mod_report.generate_model_report(True)
    
                # 检查报告中的检测器数量是否为 2
                self.assertEqual(len(generated_report), 2)
    
                # 获取输入权重平衡的特定报告
                outlier_str, outlier_dict = generated_report[outlier_detector.get_detector_name()]
    
                # 检查层数数量，应该是 4 个卷积层 + 1 个线性层 + 1 个 ReLU 层
                self.assertEqual(len(outlier_dict), 4)
    
                # 验证所有模块的断言条件
                for module_fqn in outlier_dict:
                    # 获取特定模块的信息
                    module_dict = outlier_dict[module_fqn]
    
                    # 由于使用了正态分布进行计算，实际应该没有异常值
                    outlier_info = module_dict[OutlierDetector.OUTLIER_KEY]
                    self.assertEqual(sum(outlier_info), 0)
    
                    # 确保比率和批次数量与参数数量相同
                    self.assertEqual(len(module_dict[OutlierDetector.COMP_METRIC_KEY]), param_size)
                    self.assertEqual(len(module_dict[OutlierDetector.NUM_BATCHES_KEY]), param_size)
    
        @skipIfNoFBGEMM
    def test_all_outlier_report_gen(self):
        # 使得百分位为0，比率为1，然后根据它判断所有内容是否为异常值
        # 使用fbgemm并创建我们的模型实例
        # 然后使用检测器创建模型报告实例
        with override_quantized_engine('fbgemm'):
            # 创建感兴趣的异常检测器
            outlier_detector = OutlierDetector(ratio_threshold=1, reference_percentile=0)

            param_size: int = 16
            detector_set = {outlier_detector}
            model = self.LargeBatchModel(param_size=param_size)

            # 获取测试模型并进行校准
            prepared_for_callibrate_model, mod_report = self._get_prepped_for_calibration_model(
                model, detector_set
            )

            # 现在实际上对模型进行校准
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)

            prepared_for_callibrate_model(example_input)

            # 通过运行它来获取模型报告实例
            generated_report = mod_report.generate_model_report(True)

            # 检查生成的报告大小是否合适，只有一个检测器
            self.assertEqual(len(generated_report), 1)

            # 获取输入权重均衡的特定报告
            outlier_str, outlier_dict = generated_report[outlier_detector.get_detector_name()]

            # 我们应该查看5层，因为有4个卷积层 + 1线性层 + 1ReLU
            self.assertEqual(len(outlier_dict), 4)

            # 断言所有模块的以下内容是正确的
            for module_fqn in outlier_dict:
                # 获取特定模块的信息
                module_dict = outlier_dict[module_fqn]

                # 所有的异常值应该是异常的，因为我们说所有的最大值应该等于它们的最小值
                # 然而，我们将只测试并说大多数应该是这样，以防有几个0通道值
                outlier_info = module_dict[OutlierDetector.OUTLIER_KEY]
                assert sum(outlier_info) >= len(outlier_info) / 2

                # 确保比率和批次计数的数量与参数的数量相同
                self.assertEqual(len(module_dict[OutlierDetector.COMP_METRIC_KEY]), param_size)
                self.assertEqual(len(module_dict[OutlierDetector.NUM_BATCHES_KEY]), param_size)
   `
class TestFxModelReportVisualizer(QuantizationTestCase):
    # 定义测试类，继承自 QuantizationTestCase

    def _callibrate_and_generate_visualizer(self, model, prepared_for_callibrate_model, mod_report):
        r"""
        Callibrates the passed in model, generates report, and returns the visualizer
        """
        # 定义一个函数，接收模型、用于校准的准备模型函数和模型报告实例，返回可视化器

        # 获取模型的示例输入
        example_input = model.get_example_inputs()[0]
        # 将示例输入的数据类型转换为浮点型
        example_input = example_input.to(torch.float)

        # 使用准备好的模型函数校准模型
        prepared_for_callibrate_model(example_input)

        # 使用 ModelReport 实例生成模型报告
        generated_report = mod_report.generate_model_report(remove_inserted_observers=False)

        # 生成模型报告的可视化器，确保不会报错
        mod_rep_visualizer: ModelReportVisualizer = mod_report.generate_visualizer()

        # 返回生成的可视化器
        return mod_rep_visualizer

    @skipIfNoFBGEMM
    # 使用装饰器跳过在没有 FBGEMM 的环境下的测试
    def test_get_modules_and_features(self):
        """
        Tests the get_all_unique_module_fqns and get_all_unique_feature_names methods of
        ModelReportVisualizer

        Checks whether returned sets are of proper size and filtered properly
        """
        with override_quantized_engine('fbgemm'):
            # 设置测试的后端引擎为 'fbgemm'
            torch.backends.quantized.engine = "fbgemm"

            # 创建包含两个检测器的集合
            detector_set = set()
            detector_set.add(OutlierDetector(reference_percentile=0.95))
            detector_set.add(InputWeightEqualizationDetector(0.5))

            # 创建 TwoThreeOps 模型实例
            model = TwoThreeOps()

            # 获取用于校准的模型和校准报告
            prepared_for_callibrate_model, mod_report = _get_prepped_for_calibration_model_helper(
                model, detector_set, model.get_example_inputs()[0]
            )

            # 调用方法进行校准并生成可视化器
            mod_rep_visualizer: ModelReportVisualizer = self._callibrate_and_generate_visualizer(
                model, prepared_for_callibrate_model, mod_report
            )

            # 确保模块全限定名与 get_all_unique_feature_names 方法返回的一致
            actual_model_fqns = set(mod_rep_visualizer.generated_reports.keys())
            returned_model_fqns = mod_rep_visualizer.get_all_unique_module_fqns()
            self.assertEqual(returned_model_fqns, actual_model_fqns)

            # 确保所有特征是否正确返回
            # 所有线性层均具有两个检测器的所有特征，用作方法可靠性检查
            b_1_linear_features = mod_rep_visualizer.generated_reports["block1.linear"]

            # 首先测试所有特征
            returned_all_feats = mod_rep_visualizer.get_all_unique_feature_names(False)
            self.assertEqual(returned_all_feats, set(b_1_linear_features.keys()))

            # 现在测试可绘制的特征
            plottable_set = set()

            for feature_name in b_1_linear_features:
                if type(b_1_linear_features[feature_name]) == torch.Tensor:
                    plottable_set.add(feature_name)

            returned_plottable_feats = mod_rep_visualizer.get_all_unique_feature_names()
            self.assertEqual(returned_plottable_feats, plottable_set)
    def _prep_visualizer_helper(self):
        r"""
        Returns a mod rep visualizer that we test in various ways
        """
        # 设置量化引擎为 "fbgemm"，用于测试
        torch.backends.quantized.engine = "fbgemm"

        # 使用多个异常检测器进行测试
        detector_set = set()
        detector_set.add(OutlierDetector(reference_percentile=0.95))  # 添加异常检测器：参考分位数为0.95
        detector_set.add(InputWeightEqualizationDetector(0.5))  # 添加输入权重均衡化检测器，阈值为0.5

        model = TwoThreeOps()  # 创建一个 TwoThreeOps 模型实例

        # 获取经过准备校准的模型和模型报告
        prepared_for_callibrate_model, mod_report = _get_prepped_for_calibration_model_helper(
            model, detector_set, model.get_example_inputs()[0]
        )

        # 调用私有方法，执行校准并生成可视化器
        mod_rep_visualizer: ModelReportVisualizer = self._callibrate_and_generate_visualizer(
            model, prepared_for_callibrate_model, mod_report
        )

        return mod_rep_visualizer  # 返回生成的模型报告可视化器实例

    @skipIfNoFBGEMM
    def test_generate_tables_match_with_report(self):
        """
        Tests the generate_table_view()
        ModelReportVisualizer

        Checks whether the generated dict has proper information
            Visual check that the tables look correct performed during testing
        """
        with override_quantized_engine('fbgemm'):  # 使用 'fbgemm' 作为量化引擎覆盖环境

            # 获取模型报告可视化器
            mod_rep_visualizer = self._prep_visualizer_helper()

            # 生成过滤后的表格字典
            table_dict = mod_rep_visualizer.generate_filtered_tables()

            # 主要测试字典，因为它与字符串信息相同
            tensor_headers, tensor_table = table_dict[ModelReportVisualizer.TABLE_TENSOR_KEY]
            channel_headers, channel_table = table_dict[ModelReportVisualizer.TABLE_CHANNEL_KEY]

            # 这两者组合应与生成的报告信息中的键相同
            tensor_info_modules = {row[1] for row in tensor_table}
            channel_info_modules = {row[1] for row in channel_table}
            combined_modules: Set = tensor_info_modules.union(channel_info_modules)

            # 检查生成的模块键是否与模型报告可视化器的生成报告键相等
            generated_report_keys: Set = set(mod_rep_visualizer.generated_reports.keys())
            self.assertEqual(combined_modules, generated_report_keys)
    def test_generate_tables_no_match(self):
        """
        Tests the generate_table_view() method in ModelReportVisualizer.

        Checks whether the generated dict has proper information.
        Visual check that the tables look correct performed during testing.
        """
        with override_quantized_engine('fbgemm'):
            # 获取可视化器实例
            mod_rep_visualizer = self._prep_visualizer_helper()

            # 尝试使用一个随机过滤器，并确保两个表格中都没有行
            empty_tables_dict = mod_rep_visualizer.generate_filtered_tables(module_fqn_filter="random not there module")

            # 主要测试字典，因为其信息与字符串相同
            tensor_headers, tensor_table = empty_tables_dict[ModelReportVisualizer.TABLE_TENSOR_KEY]
            channel_headers, channel_table = empty_tables_dict[ModelReportVisualizer.TABLE_CHANNEL_KEY]

            # 提取张量表格中的模块信息集合
            tensor_info_modules = {row[1] for row in tensor_table}
            # 提取通道表格中的模块信息集合
            channel_info_modules = {row[1] for row in channel_table}
            # 合并张量和通道表格中的模块信息集合
            combined_modules: Set = tensor_info_modules.union(channel_info_modules)
            self.assertEqual(len(combined_modules), 0)  # 应该没有匹配的模块

    @skipIfNoFBGEMM
    def test_generate_tables_single_feat_match(self):
        """
        Tests the generate_table_view() method in ModelReportVisualizer.

        Checks whether the generated dict has proper information.
        Visual check that the tables look correct performed during testing.
        """
        with override_quantized_engine('fbgemm'):
            # 获取可视化器实例
            mod_rep_visualizer = self._prep_visualizer_helper()

            # 尝试匹配特征过滤器，并确保只有这些特征显示
            # 如果我们过滤到非常具体的特征名称，每个表格行应该只有一个额外列
            single_feat_dict = mod_rep_visualizer.generate_filtered_tables(feature_filter=OutlierDetector.MAX_VALS_KEY)

            # 主要测试字典，因为其信息与字符串相同
            tensor_headers, tensor_table = single_feat_dict[ModelReportVisualizer.TABLE_TENSOR_KEY]
            channel_headers, channel_table = single_feat_dict[ModelReportVisualizer.TABLE_CHANNEL_KEY]

            # 获取每个表格中特征的数量
            tensor_info_features = len(tensor_headers)
            channel_info_features = len(channel_headers) - ModelReportVisualizer.NUM_NON_FEATURE_CHANNEL_HEADERS

            # 确保张量表格中没有特征，并且通道级别表格中有一个特征
            self.assertEqual(tensor_info_features, 0)
            self.assertEqual(channel_info_features, 1)
# 定义一个辅助函数，准备用于校准的模型和相应的模型报告
def _get_prepped_for_calibration_model_helper(model, detector_set, example_input, fused: bool = False):
    r"""Returns a model that has been prepared for callibration and corresponding model_report"""
    
    # 设置此测试的后端引擎为 "fbgemm"
    torch.backends.quantized.engine = "fbgemm"

    # 将示例输入转换为浮点类型
    example_input = example_input.to(torch.float)
    
    # 获取默认的量化配置映射
    q_config_mapping = torch.ao.quantization.get_default_qconfig_mapping()

    # 如果传入了融合参数，确保测试融合模块
    if fused:
        model = torch.ao.quantization.fuse_modules(model, model.get_fusion_modules())

    # 准备模型以进行量化
    model_prep = quantize_fx.prepare_fx(model, q_config_mapping, example_input)

    # 创建一个模型报告，用于存储量化后的模型和检测器集合
    model_report = ModelReport(model_prep, detector_set)

    # 准备模型以进行详细的校准
    prepared_for_calibrate_model = model_report.prepare_detailed_calibration()

    # 返回包含准备好的校准模型和模型报告的元组
    return (prepared_for_calibrate_model, model_report)
```
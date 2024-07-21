# `.\pytorch\test\quantization\eager\test_quantize_eager_qat.py`

```
# 导入所需的模块和库
import copy  # 导入 copy 模块，用于深复制对象
import math  # 导入 math 模块，提供数学函数实现

import torch  # 导入 PyTorch 深度学习库
import torch.ao.nn.intrinsic.qat as nniqat  # 导入量化自动微分推理算子模块
import torch.ao.nn.qat as nnqat  # 导入量化自动微分模块
import torch.ao.nn.qat.dynamic as nnqatd  # 导入动态量化自动微分模块
import torch.ao.nn.quantized as nnq  # 导入量化模块
import torch.ao.nn.quantized.dynamic as nnqd  # 导入动态量化模块
import torch.backends.mkldnn  # 导入 Torch 的 MKLDNN 后端支持
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.testing._internal.hypothesis_utils as hu  # 导入测试工具模块

from hypothesis import given, strategies as st  # 导入假设测试框架相关函数和策略模块
from torch.ao.nn.intrinsic.qat import ConvBn2d, ConvBnReLU2d  # 导入量化自动微分卷积-BN融合模块
from torch.ao.quantization import (  # 导入 Torch 自动量化模块
    convert,  # 导入转换函数
    default_embedding_qat_qconfig,  # 默认嵌入量化自动微分量化配置
    default_qat_qconfig,  # 默认量化自动微分量化配置
    default_qconfig,  # 默认量化配置
    default_symmetric_qnnpack_qat_qconfig,  # 默认对称 QNNPACK 量化自动微分量化配置
    DeQuantStub,  # 反量化存根类
    FixedQParamsFakeQuantize,  # 固定量化参数的伪量化类
    FusedMovingAvgObsFakeQuantize,  # 融合移动平均观察者伪量化类
    get_default_qat_qconfig,  # 获取默认量化自动微分量化配置
    get_embedding_qat_module_mappings,  # 获取嵌入量化自动微分模块映射
    get_embedding_static_quant_module_mappings,  # 获取嵌入静态量化模块映射
    NoopObserver,  # 空操作观察者类
    prepare,  # 准备函数
    prepare_qat,  # 准备量化自动微分
    quantize_qat,  # 量化自动微分函数
    QuantStub,  # 量化存根类
)
from torch.ao.quantization.qconfig import qconfig_equals  # 量化配置相等函数
from torch.nn import BatchNorm2d, Conv2d, init, ReLU  # 导入批标准化、卷积、初始化、ReLU 模块
from torch.nn.modules.utils import _pair  # 导入 _pair 工具函数
from torch.testing._internal.common_quantization import (  # 导入 Torch 通用量化测试模块
    DeFusedEmbeddingBagLinear,  # 解融合的嵌入包线性模型类
    ManualConvLinearQATModel,  # 手动卷积线性量化自动微分模型类
    ManualConvLinearSymmQATModel,  # 手动卷积线性对称量化自动微分模型类
    ManualDropoutQATModel,  # 手动丢弃量化自动微分模型类
    ManualEmbeddingBagLinear,  # 手动嵌入包线性模型类
    ManualLinearDynamicQATModel,  # 手动线性动态量化自动微分模型类
    ManualLinearQATModel,  # 手动线性量化自动微分模型类
    QuantizationTestCase,  # 量化测试用例类
    QuantStubModel,  # 量化存根模型类
    test_only_eval_fn,  # 仅测试评估函数
    test_only_train_fn,  # 仅测试训练函数
    TwoLayerLinearModel,  # 双层线性模型类
)

from torch.testing._internal.common_quantized import (  # 导入 Torch 通用量化模块
    override_qengines,  # 覆盖量化引擎函数
    override_quantized_engine,  # 覆盖量化引擎函数
    supported_qengines,  # 支持的量化引擎函数
)

from torch.testing._internal.common_utils import skipIfNoXNNPACK  # 如果没有 XNNPACK 则跳过

hu.assert_deadline_disabled()  # 禁用测试截止时间断言
from functools import reduce  # 导入 reduce 函数，用于迭代计算

class _ReferenceConvBnNd(torch.nn.Conv2d, torch.nn.modules.conv._ConvNd):
    """
    Conv-BN fusion implemented with explicit folding. Useful
    to verify numerical equivalency with non-folded version.
    """
    # 参考 Conv-BN 融合的实现，通过显式折叠实现。用于验证与非融合版本的数值等价性。
    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups,
                 bias,
                 padding_mode,
                 # BatchNormNd args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        # 继承父类 nn.modules.conv._ConvNd 的初始化方法，传入卷积层的参数
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, transposed,
                                         output_padding, groups, False, padding_mode)
        # 断言，确保 qconfig 不为空，用于量化训练模块
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig  # 设置 qconfig 属性
        self.eps = eps  # 设置 eps 属性，默认为 1e-05
        self.momentum = momentum  # 设置 momentum 属性，默认为 0.1
        self.freeze_bn = freeze_bn if self.training else True  # 冻结 Batch Normalization 统计信息的标志
        self.num_features = out_channels  # 设置 num_features 属性为输出通道数
        self.gamma = nn.Parameter(torch.empty(out_channels))  # 初始化 gamma 参数
        self.beta = nn.Parameter(torch.empty(out_channels))  # 初始化 beta 参数
        self.affine = True  # 设置 affine 属性为 True，表示使用可学习的拉伸和偏移
        self.track_running_stats = True  # 设置 track_running_stats 属性为 True，表示追踪运行时统计信息
        self.register_buffer('running_mean', torch.zeros(out_channels))  # 注册缓冲区 running_mean
        self.register_buffer('running_var', torch.ones(out_channels))  # 注册缓冲区 running_var
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))  # 注册缓冲区 num_batches_tracked
        self.activation_post_process = self.qconfig.activation()  # 获取激活后处理的方法
        self.weight_fake_quant = self.qconfig.weight()  # 获取权重伪量化的方法
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))  # 如果 bias 为 True，则初始化 bias 参数
        else:
            self.register_parameter('bias', None)  # 否则注册 bias 参数为 None
        self.reset_bn_parameters()  # 调用函数重置 Batch Normalization 参数

    def reset_running_stats(self):
        self.running_mean.zero_()  # 将 running_mean 缓冲区置零
        self.running_var.fill_(1)  # 将 running_var 缓冲区填充为 1
        self.num_batches_tracked.zero_()  # 将 num_batches_tracked 缓冲区置零

    def reset_bn_parameters(self):
        self.reset_running_stats()  # 调用函数重置运行时统计信息
        init.uniform_(self.gamma)  # 使用均匀分布初始化 gamma 参数
        init.zeros_(self.beta)  # 使用零初始化 beta 参数
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)  # 计算输入和输出的扇入
            bound = 1 / math.sqrt(fan_in)  # 计算初始化的界限
            init.uniform_(self.bias, -bound, bound)  # 使用均匀分布初始化 bias 参数

    def reset_parameters(self):
        super().reset_parameters()  # 调用父类的重置参数方法
        # A hack to avoid resetting on undefined parameters
        if hasattr(self, 'gamma'):  # 如果存在 gamma 属性
            self.reset_bn_parameters()  # 调用函数重置 Batch Normalization 参数

    def update_bn_stats(self):
        self.freeze_bn = False  # 设置 freeze_bn 属性为 False，用于更新 Batch Normalization 统计信息
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True  # 设置 freeze_bn 属性为 True，用于冻结 Batch Normalization 统计信息
        return self

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super().extra_repr()  # 调用父类的额外表示方法

    def forward(self, input):
        return self.activation_post_process(self._forward(input))  # 前向传播方法，应用激活后处理

    @classmethod
    # 从浮点数模块或 qparams_dict 创建量化训练模块
    def from_float(cls, mod, qconfig=None):
        # 断言输入的 mod 类型必须为 cls._FLOAT_MODULE
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        
        # 如果未提供 qconfig 参数，则从 mod 中获取 qconfig
        if not qconfig:
            # 断言输入的浮点数模块必须有定义 qconfig
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            # 断言 mod 的 qconfig 必须有效
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            # 将 mod 的 qconfig 赋给 qconfig
            qconfig = mod.qconfig
        
        # 从 mod 中分离出卷积层和批归一化层
        conv, bn = mod[0], mod[1]
        
        # 使用 cls 创建量化训练版本的卷积批归一化模块
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, conv.bias is not None,
                         conv.padding_mode,
                         bn.eps, bn.momentum,
                         False,
                         qconfig)
        
        # 将浮点数模块的权重和偏置复制给量化训练模块
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.gamma = bn.weight
        qat_convbn.beta = bn.bias
        qat_convbn.running_mean = bn.running_mean
        qat_convbn.running_var = bn.running_var
        qat_convbn.num_batches_tracked = bn.num_batches_tracked
        
        # 返回创建的量化训练模块
        return qat_convbn
class _ReferenceConvBn2d(_ReferenceConvBnNd, nn.Conv2d):
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvBn2d

    def __init__(self,
                 # ConvNd args：继承自 ConvNd 的参数
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm2d args：继承自 BatchNorm2d 的参数
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module：本模块的参数
                 freeze_bn=False,
                 qconfig=None):
        kernel_size = _pair(kernel_size)  # 将 kernel_size 转换成二元组
        stride = _pair(stride)  # 将 stride 转换成二元组
        padding = _pair(padding)  # 将 padding 转换成二元组
        dilation = _pair(dilation)  # 将 dilation 转换成二元组
        _ReferenceConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, False, _pair(0), groups, bias, padding_mode,
                                    eps, momentum, freeze_bn, qconfig)

class TestQuantizeEagerQAT(QuantizationTestCase):
    def setUp(self):
        super().setUp()

        self.embed_linear_data_train = [[torch.randint(0, 10, (12, 12), dtype=torch.long),
                                         torch.randn((12, 1), dtype=torch.float)]
                                        for _ in range(2)]
        self.embed_data = [[torch.randint(0, 10, (12, 1))]]

    def test_manual(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):  # 使用指定的量化引擎进行覆盖
                model = ManualLinearQATModel(qengine)  # 创建基于指定引擎的模型
                model = prepare_qat(model)  # 准备模型以进行量化训练
                self.checkObservers(model)  # 检查模型的观察器
                test_only_train_fn(model, self.train_data)  # 对训练数据执行仅用于测试的训练函数
                model = convert(model)  # 将模型转换为量化模型

                def checkQuantized(model):
                    self.assertEqual(type(model.fc1), nnq.Linear)  # 断言模型的第一个全连接层已经量化
                    self.assertEqual(type(model.fc2), nnq.Linear)  # 断言模型的第二个全连接层已经量化
                    test_only_eval_fn(model, self.calib_data)  # 对校准数据执行仅用于测试的评估函数
                    self.checkScriptable(model, self.calib_data)  # 检查模型的可脚本化性
                    self.checkNoQconfig(model)  # 检查模型是否没有配置信息

                checkQuantized(model)  # 对量化后的模型进行检查

                model = quantize_qat(ManualLinearQATModel(qengine), test_only_train_fn,
                                     [self.train_data])  # 使用量化训练函数对模型进行量化
                checkQuantized(model)  # 对量化后的模型再次进行检查
    def test_dropout(self):
        # 遍历所有支持的量化引擎
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖默认设置
            with override_quantized_engine(qengine):
                # 创建一个使用指定量化引擎的手动 Dropout 量化训练模型
                model = ManualDropoutQATModel(qengine)
                # 准备模型以进行量化训练
                model = prepare_qat(model)
                # 检查模型中的观察者（observers）
                self.checkObservers(model)
                # 使用仅训练函数测试模型
                test_only_train_fn(model, self.train_data)
                # 将模型转换为量化模型
                model = convert(model)

                def checkQuantized(model):
                    # 断言模型的第一层是量化的线性层
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    # 断言模型的 dropout 层是量化的 dropout
                    self.assertEqual(type(model.dropout), nnq.Dropout)
                    # 使用仅评估函数测试模型
                    test_only_eval_fn(model, self.calib_data)
                    # 检查模型是否可脚本化
                    self.checkScriptable(model, self.calib_data)
                    # 检查模型是否没有量化配置
                    self.checkNoQconfig(model)

                # 调用检查量化函数
                checkQuantized(model)

                # 使用仅训练函数和训练数据量化手动 Dropout 量化训练模型
                model = quantize_qat(ManualDropoutQATModel(qengine), test_only_train_fn,
                                     [self.train_data])
                # 再次调用检查量化函数
                checkQuantized(model)

    def test_eval_only_fake_quant(self):
        r"""Using FakeQuant in evaluation only mode,
        this is useful for estimating accuracy loss when we quantize the
        network
        """
        # 遍历所有支持的量化引擎
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖默认设置
            with override_quantized_engine(qengine):
                # 创建一个使用指定量化引擎的手动线性量化训练模型
                model = ManualLinearQATModel(qengine)

                # 准备模型以进行量化训练
                model = prepare_qat(model)
                # 检查模型中的观察者（observers）
                self.checkObservers(model)

                # 将模型设置为评估模式
                model.eval()
                # 使用仅评估函数测试模型
                test_only_eval_fn(model, self.calib_data)

    def test_conv_linear(self):
        # 遍历所有支持的量化引擎
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖默认设置
            with override_quantized_engine(qengine):
                # 创建一个手动卷积线性量化训练模型
                model = ManualConvLinearQATModel()

                # 准备模型以进行量化训练
                model = prepare_qat(model)
                # 检查模型中的观察者（observers）
                self.checkObservers(model)

                # 使用仅训练函数测试模型
                test_only_train_fn(model, self.img_data_2d_train)
                # 将模型转换为量化模型
                model = convert(model)

                def checkQuantized(model):
                    # 断言模型的卷积层是量化的卷积层
                    self.assertEqual(type(model.conv), nnq.Conv2d)
                    # 断言模型的第一线性层是量化的线性层
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    # 断言模型的第二线性层是量化的线性层
                    self.assertEqual(type(model.fc2), nnq.Linear)
                    # 使用仅评估函数测试模型
                    test_only_eval_fn(model, self.img_data_2d)
                    # 检查模型是否可脚本化
                    self.checkScriptable(model, self.img_data_2d)
                    # 检查模型是否没有量化配置
                    self.checkNoQconfig(model)

                # 调用检查量化函数
                checkQuantized(model)

                # 创建一个手动卷积线性量化训练模型
                model = ManualConvLinearQATModel()
                # 使用仅训练函数和训练数据量化模型
                model = quantize_qat(model, test_only_train_fn, [self.img_data_2d_train])
                # 再次调用检查量化函数
                checkQuantized(model)

    @skipIfNoXNNPACK
    def test_conv_linear_symm(self):
        r"""Same as test_conv_linear but with Symmetric quantization.
        Supported only with qengine=qnnpack, which uses symmetric
        kernels from xnnpack library."""
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 如果当前量化引擎不是 'qnnpack'，则跳过当前循环
            if qengine != 'qnnpack':
                continue
            # 使用 override_quantized_engine() 上下文管理器设置当前的量化引擎为 'qnnpack'
            with override_quantized_engine(qengine):
                # 创建 ManualConvLinearSymmQATModel 模型实例
                model = ManualConvLinearSymmQATModel()

                # 准备 QAT（量化感知训练）
                model = prepare_qat(model)
                # 检查模型中的观察者
                self.checkObservers(model)

                # 在训练数据上运行仅测试函数 test_only_train_fn
                test_only_train_fn(model, self.img_data_2d_train)
                # 将模型转换为量化后的模型
                model = convert(model)

                # 定义函数 checkQuantized，用于检查模型是否正确量化
                def checkQuantized(model):
                    # 断言模型中 conv 层被量化为 nnq.Conv2d 类型
                    self.assertEqual(type(model.conv), nnq.Conv2d)
                    # 断言模型中 fc1 层被量化为 nnq.Linear 类型
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    # 断言模型中 fc2 层被量化为 nnq.Linear 类型
                    self.assertEqual(type(model.fc2), nnq.Linear)
                    # 在评估数据上运行仅测试函数 test_only_eval_fn
                    test_only_eval_fn(model, self.img_data_2d)
                    # 检查模型是否可脚本化
                    self.checkScriptable(model, self.img_data_2d)
                    # 检查模型是否没有量化配置
                    self.checkNoQconfig(model)

                # 调用 checkQuantized 函数，检查当前的模型量化情况
                checkQuantized(model)

                # 创建 ManualConvLinearSymmQATModel 模型的新实例
                model = ManualConvLinearSymmQATModel()
                # 将模型进行 QAT 量化，并在 self.img_data_2d_train 数据上运行仅测试函数 test_only_train_fn
                model = quantize_qat(model, test_only_train_fn, [self.img_data_2d_train])
                # 调用 checkQuantized 函数，检查当前的模型量化情况
                checkQuantized(model)

    def test_dynamic_qat_linear(self):
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 使用 override_quantized_engine() 上下文管理器设置当前的量化引擎为 qengine
            with override_quantized_engine(qengine):
                # 动态 QAT（量化感知训练）需要具有无记忆观察者，否则会失败
                with self.assertRaisesRegex(ValueError,
                                            "Dynamic QAT requires a memoryless observer." +
                                            "This means a MovingAverage observer with averaging constant equal to 1"
                                            ):
                    # 创建 ManualLinearDynamicQATModel 模型实例，并使用默认的 QAT 配置
                    model = ManualLinearDynamicQATModel(default_qat_qconfig)
                    # 准备 QAT（量化感知训练），映射 torch.nn.Linear 层到 nnqatd.Linear
                    model = prepare_qat(model, mapping={torch.nn.Linear: nnqatd.Linear})

                # 创建 ManualLinearDynamicQATModel 模型实例
                model = ManualLinearDynamicQATModel()
                # 准备 QAT（量化感知训练），映射 torch.nn.Linear 层到 nnqatd.Linear
                model = prepare_qat(model, mapping={torch.nn.Linear: nnqatd.Linear})
                # 断言模型中 fc1 层被量化为 nnqatd.Linear 类型
                self.assertEqual(type(model.fc1), nnqatd.Linear)
                # 断言模型中 fc2 层被量化为 nnqatd.Linear 类型
                self.assertEqual(type(model.fc2), nnqatd.Linear)
                # 检查模型中的观察者
                self.checkObservers(model)
                # 在训练数据上运行仅测试函数 test_only_train_fn
                test_only_train_fn(model, self.train_data)
                # 将模型转换为非 QAT 的量化模型，映射 nnqatd.Linear 层到 nnqd.Linear
                model = convert(model, mapping={nnqatd.Linear: nnqd.Linear})
                # 断言模型中 fc1 层被量化为 nnqd.Linear 类型
                self.assertEqual(type(model.fc1), nnqd.Linear)
                # 断言模型中 fc2 层被量化为 nnqd.Linear 类型
                self.assertEqual(type(model.fc2), nnqd.Linear)
                # 在校准数据上运行仅测试函数 test_only_eval_fn
                test_only_eval_fn(model, self.calib_data)
                # 检查模型是否可脚本化
                self.checkScriptable(model, self.calib_data)
                # 检查模型是否没有量化配置
                self.checkNoQconfig(model)
    def test_defused_embedding_bag_linear(self):
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖上下文环境
            with override_quantized_engine(qengine):
                # 创建一个训练模式下的 DeFusedEmbeddingBagLinear 模型实例
                model = DeFusedEmbeddingBagLinear().train()
                # 准备量化训练模型，使用给定的映射配置
                model = prepare_qat(model, mapping=get_embedding_qat_module_mappings())
                # 检查模型的观察器
                self.checkObservers(model)

                # 使用训练函数测试模型，传入训练数据集 self.embed_linear_data_train
                test_only_train_fn(model, self.embed_linear_data_train)
                # 确保在 Linear 层之后插入了 activation_post_process
                self.assertEqual(type(model.linear.activation_post_process), FusedMovingAvgObsFakeQuantize)
                # 确保 Embedding 层的激活后处理是 NoopObserver
                self.assertEqual(type(model.emb.activation_post_process), NoopObserver)
                # 确保 FakeQuant 的 zero_points 是正确的数据类型 torch.float32
                self.assertEqual(model.emb.weight_fake_quant.zero_point.dtype, torch.float32)
                self.assertEqual(model.linear.weight_fake_quant.zero_point.dtype, torch.int32)

                # 转换模型为静态量化模型，使用给定的映射配置
                model = convert(model, mapping=get_embedding_static_quant_module_mappings())

                def checkQuantized(model):
                    # 确保现在 Embedding 是一个 QuantizedEmbedding 类型
                    self.assertEqual(type(model.emb), nn.quantized.Embedding)
                    # 确保现在 Linear 是一个 QuantizedLinear 类型
                    self.assertEqual(type(model.linear), nn.quantized.Linear)

                    # 使用评估函数测试模型，传入评估数据集 self.embed_data
                    test_only_eval_fn(model, self.embed_data)
                    # 检查模型是否可以被脚本化
                    self.checkScriptable(model, self.embed_data)
                    # 检查模型是否没有量化配置
                    self.checkNoQconfig(model)

                # 执行模型量化后的检查
                checkQuantized(model)
    # 定义测试函数，用于测试嵌入包（EmbeddingBag）和线性层的量化
    def test_embedding_bag_linear(self):
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖上下文
            with override_quantized_engine(qengine):
                # 创建并训练手动嵌入包线性模型
                model = ManualEmbeddingBagLinear().train()
                # 准备模型进行量化训练，并指定嵌入量化模块映射
                model = prepare_qat(model, mapping=get_embedding_qat_module_mappings())
                # 检查模型观察器的设置
                self.checkObservers(model)

                # 使用训练数据测试模型，仅验证训练时效果
                test_only_train_fn(model, self.embed_linear_data_train)
                # 确保嵌入包中未插入激活后处理过程
                self.assertFalse(hasattr(model, "activation_post_process"))
                # 确保伪量化零点的数据类型为 torch.float32
                self.assertEqual(model.emb.weight_fake_quant.zero_point.dtype, torch.float32)
                # 确保线性层的伪量化零点的数据类型为 torch.int32
                self.assertEqual(model.linear.weight_fake_quant.zero_point.dtype, torch.int32)

                # 将模型转换为静态量化模型，并指定嵌入静态量化模块映射
                model = convert(model, mapping=get_embedding_static_quant_module_mappings())

                # 定义检查量化后模型的函数
                def checkQuantized(model):
                    # 确保嵌入包现在是量化嵌入包
                    self.assertTrue(type(model.emb), nn.quantized.EmbeddingBag)
                    # 同时验证线性层已被量化
                    self.assertTrue(type(model.linear), nnq.Linear)

                    # 使用评估数据测试模型，仅验证评估时效果
                    test_only_eval_fn(model, self.embed_data)
                    # 检查模型是否可脚本化
                    self.checkScriptable(model, self.embed_data)
                    # 确保模型没有量化配置
                    self.checkNoQconfig(model)

                # 执行检查量化后模型的函数
                checkQuantized(model)

                # 重新创建未量化的手动嵌入包线性模型，用于下一轮量化引擎测试
                model = ManualEmbeddingBagLinear()
    # 定义一个测试方法，用于测试量化训练的流程，包括创建模型、执行量化训练、保存量化状态字典以及评估结果
    def test_train_save_load_eval(self):
        r"""Test QAT flow of creating a model, doing QAT and saving the quantized state_dict
        During eval, we first call prepare_qat and conver on the model and then load the state_dict
        and compare results against original model
        """
        # 遍历支持的量化引擎
        for qengine in supported_qengines:
            # 使用指定的量化引擎覆盖当前上下文
            with override_quantized_engine(qengine):
                # 创建一个两层线性模型
                model = TwoLayerLinearModel()
                # 将模型包装成量化包装器
                model = torch.ao.quantization.QuantWrapper(model)
                # 获取指定量化引擎的默认量化训练配置
                model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
                # 准备模型进行量化训练
                model = prepare_qat(model)

                # 获取量化训练后的模型状态字典
                fq_state_dict = model.state_dict()

                # 使用测试数据执行仅用于训练的函数
                test_only_train_fn(model, self.train_data)
                # 将模型转换为量化状态
                model = convert(model)

                # 获取转换后的模型状态字典
                quant_state_dict = model.state_dict()

                # 创建随机张量作为输入
                x = torch.rand(2, 5, dtype=torch.float)
                # 获取参考输出结果
                ref = model(x)

                # 重新创建模型进行评估，使用量化状态字典进行结果验证
                model = TwoLayerLinearModel()
                model = torch.ao.quantization.QuantWrapper(model)
                model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
                torch.ao.quantization.prepare_qat(model, inplace=True)
                new_state_dict = model.state_dict()

                # 检查 prepare_qat 后的模型是否具有与原始模型相同的状态字典
                self.assertEqual(set(fq_state_dict.keys()), set(new_state_dict.keys()))

                # 将模型转换为量化状态
                torch.ao.quantization.convert(model, inplace=True)
                # 将模型设为评估模式
                model.eval()
                # 加载量化状态字典
                model.load_state_dict(quant_state_dict)
                # 使用输入进行评估
                out = model(x)
                # 检查评估结果与参考输出是否一致
                self.assertEqual(ref, out)

                # 检查使用 prepare 方法创建的模型是否具有与量化状态字典相同的状态字典
                model = TwoLayerLinearModel()
                model.eval()
                model = torch.ao.quantization.QuantWrapper(model)
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                torch.ao.quantization.prepare(model, inplace=True)
                torch.ao.quantization.convert(model, inplace=True)
                self.assertEqual(set(model.state_dict().keys()), set(quant_state_dict.keys()))
                model.eval()
                model.load_state_dict(quant_state_dict)
                out = model(x)
                self.assertEqual(ref, out)

    @override_qengines


这段代码是一个测试方法，用于验证量化训练流程中创建、训练、保存和加载量化模型状态的正确性。
    def test_forward_hooks_preserved(self):
        r"""Test QAT on preserving pre forward and post forward hooks of original model
        """
        # 获取当前量化引擎
        qengine = torch.backends.quantized.engine
        # 创建量化前模型
        model = QuantStubModel()
        # 计数器，用于统计前向预处理和前向传播的次数
        counter = {
            'pre_forwards': 0,
            'forwards': 0,
        }

        # 前向预处理钩子函数
        def fw_pre_hook(h_module, input):
            counter['pre_forwards'] += 1

        # 前向传播钩子函数
        def fw_hook(h_module, input, output):
            counter['forwards'] += 1

        # 注册前向预处理钩子到全连接层
        model.fc.register_forward_pre_hook(fw_pre_hook)
        # 注册前向传播钩子到全连接层
        model.fc.register_forward_hook(fw_hook)

        # 获取默认的量化训练配置
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        # 准备量化感知训练模型
        model = prepare_qat(model)

        # 检查钩子函数是否存在的函数
        def checkHooksIsPresent(model, before_convert=True):
            forward_hooks = 1
            if before_convert:
                # 断言：检查量化观察器钩子是否存在
                self.assertEqual(len(model.quant._forward_hooks.values()), 1,
                                 "Quantization observer hook has disappeared")
                forward_hooks = 2
            # 断言：检查前向预处理钩子是否存在于全连接层
            self.assertObjectIn(fw_pre_hook, model.fc._forward_pre_hooks.values())
            # 断言：检查前向传播钩子是否存在于全连接层
            self.assertObjectIn(fw_hook, model.fc._forward_hooks.values())
            # 断言：检查全连接层前向预处理钩子的数量
            self.assertEqual(len(model.fc._forward_pre_hooks.values()), 1,
                             "Extra pre forward hooks have appeared on a layer")
            # 断言：检查全连接层前向传播钩子的数量
            self.assertEqual(len(model.fc._forward_hooks.values()), forward_hooks,
                             "Extra post forward hooks have appeared on a layer")

        # 在转换前检查钩子是否存在
        checkHooksIsPresent(model, True)
        # 创建随机张量作为输入
        x = torch.rand(2, 5, dtype=torch.float)
        # 进行模型前向传播
        model(x)
        # 执行量化转换
        torch.ao.quantization.convert(model, inplace=True)
        # 在转换后检查钩子是否存在
        checkHooksIsPresent(model, False)

    def test_add_scalar_uses_input_qparams(self):
        # 定义一个简单的模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.ff = torch.ao.nn.quantized.FloatFunctional()

            def forward(self, x):
                # 将输入张量进行量化
                x = self.quant(x)
                # 使用FloatFunctional类中的add_scalar方法，向输入张量添加标量1.0
                x = self.ff.add_scalar(x, 1.0)
                return x

        # 创建模型实例
        m = M()
        # 设置模型的量化配置为默认的量化配置
        m.qconfig = torch.ao.quantization.default_qconfig
        # 准备量化感知训练模型
        mp = torch.ao.quantization.prepare_qat(m)
        # 对准备好的量化感知训练模型进行前向传播
        mp(torch.randn(4, 4))
        # 将准备好的量化感知训练模型转换为量化模型
        mq = torch.ao.quantization.convert(mp)
        # 创建随机张量
        res = mq(torch.randn(4, 4))
        # 定义一个小的误差范围
        eps = 1e-5
        # 断言：检查量化模型的量化参数是否与结果的量化比例尺度接近
        self.assertTrue(torch.abs(mq.quant.scale - res.q_scale()) < eps)
    def test_mul_scalar_uses_input_qparams(self):
        # 定义一个继承自 torch.nn.Module 的模型类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加量化存根和量化功能
                self.quant = torch.ao.quantization.QuantStub()
                self.ff = torch.ao.nn.quantized.FloatFunctional()

            def forward(self, x):
                # 对输入 x 进行量化
                x = self.quant(x)
                # 使用量化功能对象 ff 对 x 进行标量乘法操作
                x = self.ff.mul_scalar(x, 2.0)
                return x

        # 创建模型实例 m
        m = M()
        # 设置量化配置为默认 QAT 配置
        m.qconfig = torch.ao.quantization.default_qconfig
        # 准备模型以支持量化训练
        mp = torch.ao.quantization.prepare_qat(m)
        # 对模型进行随机输入
        mp(torch.randn(4, 4))
        # 将准备好的模型转换为量化模型
        mq = torch.ao.quantization.convert(mp)
        # 对模型进行随机输入
        res = mq(torch.randn(4, 4))
        # 定义误差阈值
        eps = 1e-5
        # 断言量化后的量化比例乘以 2 与结果的量化比例之差小于误差阈值
        self.assertTrue(torch.abs(mq.quant.scale * 2 - res.q_scale()) < eps)

    @override_qengines
    def test_qat_embedding_bag_errors(self):
        # 获取默认的 QAT 配置
        default_qat_qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)

        # 测试构造函数参数检查
        with self.assertRaisesRegex(AssertionError,
                                    "qconfig must be provided for QAT module"):
            # 检查未提供 qconfig 时是否会抛出异常
            nnqat.EmbeddingBag(10, 5, qconfig=None)

        with self.assertRaisesRegex(AssertionError,
                                    "Embedding Bag weights requires a qscheme of " +
                                    "torch.per_channel_affine_float_qparams"):
            # 检查使用默认 QAT 配置时是否会抛出异常
            nnqat.EmbeddingBag(10, 5, qconfig=default_qat_qconfig)

        # 测试从 float 类型转换检查
        embed = nn.Embedding(10, 5)
        with self.assertRaisesRegex(AssertionError,
                                    "qat.EmbeddingBag.from_float only works for EmbeddingBag"):
            # 检查非 EmbeddingBag 类型转换时是否会抛出异常
            nnqat.EmbeddingBag.from_float(embed)
        embed_bag = nn.EmbeddingBag(10, 5)
        with self.assertRaisesRegex(AssertionError,
                                    "Input float module must have qconfig defined"):
            # 检查未定义 qconfig 时是否会抛出异常
            nnqat.EmbeddingBag.from_float(embed_bag)
        embed_bag.qconfig = None
        with self.assertRaisesRegex(AssertionError,
                                    "Input float module must have a valid qconfig"):
            # 检查无效 qconfig 时是否会抛出异常
            nnqat.EmbeddingBag.from_float(embed_bag)
        embed_bag.qconfig = default_qat_qconfig
        with self.assertRaisesRegex(AssertionError,
                                    "Embedding Bag weights requires a qscheme of " +
                                    "torch.per_channel_affine_float_qparams"):
            # 检查使用默认 QAT 配置时是否会抛出异常
            nnqat.EmbeddingBag.from_float(embed_bag)

    def test_embedding_qat_qconfig_equal(self):
        # 对比 Embedding QAT 使用的观察器类 NoopObserver 和权重的 FakeQuant
        # 确保 qconfig 在函数和类的混合情况下能够正确比较
        model = ManualEmbeddingBagLinear().train()
        # 准备模型以支持量化训练
        model = prepare_qat(model)

        # 断言模型的嵌入层的 qconfig 是否等于默认的嵌入 QAT 配置
        self.assertTrue(qconfig_equals(model.emb.qconfig,
                                       default_embedding_qat_qconfig))
class TestQuantizeEagerQATNumerics(QuantizationTestCase):
    # QuantizationTestCase 类的子类，用于测试量化相关功能

    def _test_activation_convert_numerics_impl(self, Act, data):
        # 实现测试激活函数转换数值计算

        class M(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的内部类 M

            def __init__(self):
                # 初始化方法
                super().__init__()
                self.act = Act()  # 初始化激活函数实例
                self.quant = QuantStub()  # 初始化量化存根
                self.dequant = DeQuantStub()  # 初始化去量化存根

            def forward(self, x):
                # 前向传播方法
                x = self.quant(x)  # 对输入 x 进行量化
                x = self.act(x)  # 使用设定的激活函数进行激活
                x = self.dequant(x)  # 对输出 x 进行去量化
                return x

        m = M().train()  # 创建 M 类实例并设置为训练模式
        m.qconfig = default_qat_qconfig  # 设置量化配置为默认 QAT 配置
        m = prepare_qat(m)  # 准备模型以进行 QAT（量化感知训练）
        before_convert = m(data)  # 在转换前使用模型处理输入数据并记录结果
        m = convert(m)  # 对模型进行转换（量化转换）
        after_convert = m(data)  # 在转换后使用模型处理输入数据并记录结果
        self.assertEqual(before_convert, after_convert)  # 断言转换前后处理结果是否一致

    def test_fixed_qparam_ops(self):
        # 测试固定量化参数操作

        class M(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的内部类 M

            def __init__(self):
                # 初始化方法
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()  # 初始化 Sigmoid 激活函数
                self.hardsigmoid = torch.nn.Hardsigmoid()  # 初始化 Hardsigmoid 激活函数
                self.tanh = torch.nn.Tanh()  # 初始化 Tanh 激活函数
                self.quant = QuantStub()  # 初始化量化存根
                self.dequant = DeQuantStub()  # 初始化去量化存根

            def forward(self, x):
                # 前向传播方法
                x = self.quant(x)  # 对输入 x 进行量化
                x = self.sigmoid(x)  # 使用 Sigmoid 激活函数
                x = self.hardsigmoid(x)  # 使用 Hardsigmoid 激活函数
                x = self.tanh(x)  # 使用 Tanh 激活函数
                x = self.dequant(x)  # 对输出 x 进行去量化
                return x

        m = M().train()  # 创建 M 类实例并设置为训练模式
        m.qconfig = default_qat_qconfig  # 设置量化配置为默认 QAT 配置
        m = prepare_qat(m)  # 准备模型以进行 QAT（量化感知训练）

        # 遍历激活函数属性，并断言其后处理过程的类型为 FixedQParamsFakeQuantize
        for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
            self.assertEqual(type(getattr(m, attr).activation_post_process), FixedQParamsFakeQuantize)

        data = torch.randn(1, 3, 2, 4)  # 创建一个随机数据张量
        before_convert = m(data)  # 在转换前使用模型处理输入数据并记录结果
        m = convert(m)  # 对模型进行转换（量化转换）
        after_convert = m(data)  # 在转换后使用模型处理输入数据并记录结果
        self.assertEqual(before_convert, after_convert)  # 断言转换前后处理结果是否一致

        # 确保去量化后处理过程已移除
        for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
            self.assertFalse(hasattr(getattr(m, attr), 'activation_post_process'))
            self.assertTrue(len(getattr(m, attr)._forward_hooks.items()) == 0)

        # 确保在评估模式下不会插入虚假量化模块
        def checkNoFQModule(m):
            for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
                self.assertFalse(hasattr(getattr(m, attr), "activation_post_process"))
                self.assertTrue(len(getattr(m, attr)._forward_hooks.items()) == 0)

        m = M().eval()  # 创建 M 类实例并设置为评估模式
        m.qconfig = default_qconfig  # 设置量化配置为默认配置
        m = prepare(m)  # 准备模型
        checkNoFQModule(m)  # 检查是否没有虚假量化模块
        m = convert(m)  # 对模型进行转换（量化转换）
        checkNoFQModule(m)  # 再次检查是否没有虚假量化模块

    def test_leaky_relu(self):
        # 测试 LeakyReLU 激活函数

        data = torch.randn(1, 3, 2, 4)  # 创建一个随机数据张量
        self._test_activation_convert_numerics_impl(nn.LeakyReLU, data)  # 调用内部方法进行激活函数转换数值计算测试
    # 定义测试函数 test_relu，用于测试 ReLU 激活函数的行为
    def test_relu(self):
        # 定义一个继承自 nn.Module 的内嵌类 M
        class M(torch.nn.Module):
            # 初始化函数，构建神经网络模型
            def __init__(self):
                super().__init__()
                # 创建 ReLU 激活函数对象
                self.relu = nn.ReLU()

            # 前向传播函数
            def forward(self, x):
                # 应用 ReLU 激活函数到输入 x
                x = self.relu(x)
                return x

        # 创建 M 类的实例 m，并设置为训练模式
        m = M().train()
        # 设置量化配置为默认配置
        m.qconfig = default_qconfig
        # 对模型 m 进行量化感知训练准备
        m = prepare_qat(m)
        
        # 断言确保在 ReLU 模块中没有插入 activation_post_process 属性
        self.assertFalse(hasattr(m, "activation_post_process"))
        
        # 将模型 m 转换为量化模型
        m = convert(m)
        
        # 断言确保 ReLU 模块未被改变
        self.assertTrue(type(m.relu), nn.ReLU)

    # 使用 @given 装饰器定义测试函数 test_conv_bn_relu，并指定多个参数生成策略
    @given(batch_size=st.integers(2, 4),
           input_channels_per_group=st.sampled_from([2, 3, 4]),
           height=st.integers(5, 10),
           width=st.integers(5, 10),
           output_channels_per_group=st.sampled_from([2, 3]),
           groups=st.integers(1, 3),
           kernel_h=st.integers(1, 3),
           kernel_w=st.integers(1, 3),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 1),
           padding_mode=st.sampled_from(['zeros', 'circular']),
           use_relu=st.booleans(),
           eps=st.sampled_from([1e-5, 1e-4, 1e-3]),
           momentum=st.sampled_from([0.1, 0.2, 0.3]),
           freeze_bn=st.booleans(),
           zero_gamma=st.booleans(),
           has_bias=st.booleans(),
           use_slow_fusion=st.booleans())
    # 定义测试卷积-批归一化-ReLU函数
    def test_conv_bn_relu(
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
            padding_mode,
            use_relu,
            eps,
            momentum,
            freeze_bn,
            zero_gamma,
            has_bias,
            use_slow_fusion,
    # 使用 @given 装饰器定义测试函数 test_conv_bn_relu，并指定多个参数生成策略
    @given(batch_size=st.integers(2, 4),
           input_channels_per_group=st.sampled_from([2, 3, 4]),
           height=st.integers(5, 10),
           width=st.integers(5, 10),
           output_channels_per_group=st.sampled_from([2, 3]),
           groups=st.integers(1, 3),
           kernel_h=st.integers(1, 3),
           kernel_w=st.integers(1, 3),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 1),
           padding_mode=st.sampled_from(['zeros', 'circular']),
           eps=st.sampled_from([1e-5, 1e-4, 1e-3]),
           momentum=st.sampled_from([0.1, 0.2, 0.3]),
           freeze_bn=st.booleans(),
           bias=st.booleans())
    @override_qengines
    def test_conv_bn_folded_vs_unfolded(
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
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            bias,
    ):
        # 测试卷积和批归一化（BN）折叠与展开的性能对比
        qengine = torch.backends.quantized.engine
        
        # 创建一个参考的模型，包含线性层和BN层
        m_ref = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
        )
        
        # 深度复制参考模型
        m_ref_copy = copy.deepcopy(m_ref)
        
        # 对参考模型进行量化感知训练（QAT）模块的融合
        m_ref_copy = torch.ao.quantization.fuse_modules_qat(m_ref_copy, [['0', '1']])
        
        # 获取默认的QAT量化配置
        qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        
        # 将量化配置应用于融合后的线性层
        m_ref_copy[0].qconfig = qconfig
        
        # 将QAT融合后的模型转换为量化感知训练（QAT）模型
        m = nniqat.LinearBn1d.from_float(m_ref_copy[0])

        # 禁用虚假量化，验证融合后的QAT模块与fp32模块的匹配性
        m.apply(torch.ao.quantization.disable_fake_quant)
        
        # 创建随机输入数据
        data = torch.randn(4, 4)
        
        # 计算参考模型和QAT模型的输出
        r1 = m_ref(data)
        r2 = m(data)
        
        # 断言两者输出近似相等
        self.assertTrue(torch.allclose(r1, r2))

    @skipIfNoXNNPACK
    @override_qengines
    def test_linear_bn_symm_numerics(self):
        # 测试线性层和对称量化的数值计算
        qengine = torch.backends.quantized.engine
        
        # 只有qnnpack支持对称量化
        if qengine != "qnnpack":
            return
        
        # 创建一个参考的模型，包含线性层和BN层
        m_ref = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
        )
        
        # 深度复制参考模型
        m_ref_copy = copy.deepcopy(m_ref)
        
        # 对参考模型进行量化感知训练（QAT）模块的融合
        m_ref_copy = torch.ao.quantization.fuse_modules_qat(m_ref_copy, [['0', '1']])
        
        # 获取默认的qnnpack对称量化的QAT量化配置
        qconfig = default_symmetric_qnnpack_qat_qconfig
        
        # 将量化配置应用于融合后的线性层
        m_ref_copy[0].qconfig = qconfig
        
        # 将QAT融合后的模型转换为量化感知训练（QAT）模型
        m = nniqat.LinearBn1d.from_float(m_ref_copy[0])

        # 禁用虚假量化，验证融合后的QAT模块与fp32模块的匹配性
        m.apply(torch.ao.quantization.disable_fake_quant)
        
        # 创建随机输入数据
        data = torch.randn(4, 4)
        
        # 计算参考模型和QAT模型的输出
        r1 = m_ref(data)
        r2 = m(data)
        
        # 断言两者输出近似相等
        self.assertTrue(torch.allclose(r1, r2))

    @override_qengines
    def test_linear_bn_workflow(self):
        # 测试线性层和BN层的量化感知训练（QAT）工作流程
        qengine = torch.backends.quantized.engine
        
        # 创建一个包含量化存根、线性层和BN层的序列模型
        m = nn.Sequential(
            QuantStub(),
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
        )
        
        # 创建随机输入数据
        data = torch.randn(4, 4)
        
        # 获取默认的QAT量化配置
        m.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        
        # 将线性层和BN层融合为单个QAT模块
        m = torch.ao.quantization.fuse_modules_qat(m, [['1', '2']])
        
        # 准备模型以进行量化感知训练
        mp = prepare_qat(m)
        
        # 对输入数据进行模型预测
        mp(data)
        
        # 转换量化感知训练模型为量化模型
        mq = convert(mp)
        
        # 断言转换后的模型的第一层为量化线性层，第二层为恒等映射
        self.assertTrue(type(mq[1]) == nnq.Linear)
        self.assertTrue(type(mq[2]) == nn.Identity)
    # 定义一个测试函数，用于测试预计算的伪量化
    def test_linear_precomputed_fake_quant(self):
        # 获取当前量化引擎
        qengine = torch.backends.quantized.engine
        # 如果量化引擎不是 "qnnpack"，则退出测试（只有 qnnpack 支持对称量化）
        if qengine != "qnnpack":
            return
        # 创建一个普通的线性层模型，输入和输出维度均为 4
        m_ref = nn.Linear(4, 4)

        # 深度复制模型 m_ref
        m_ref_copy = copy.deepcopy(m_ref)
        # 设置默认的量化配置
        qconfig = default_qconfig
        # 将量化配置应用到深度复制的模型上
        m_ref_copy.qconfig = qconfig
        # 深度复制量化配置中的权重后处理器
        weight_post_process = copy.deepcopy(qconfig.weight())
        # 深度复制量化配置中的激活后处理器
        activation = copy.deepcopy(qconfig.activation())
        # 对一个 4x4 的随机张量应用激活后处理器
        activation(torch.randn(4, 4))
        # 将激活后处理器应用到深度复制的模型的激活后处理器属性上
        m_ref_copy.activation_post_process = activation
        # 将深度复制的模型转换为量化后的模型 nnq.Linear
        m_ref_copy = nnq.Linear.from_float(m_ref_copy)
        # 获取量化配置中的权重后处理器
        weight_post_process = qconfig.weight()
        # 设置权重后处理器的最小值为 -1
        weight_post_process.min_val = torch.tensor(-1)
        # 设置权重后处理器的最大值为 1
        weight_post_process.max_val = torch.tensor(1)
        # 将权重后处理器应用到原始模型的权重后处理器属性上
        m_ref.weight_post_process = weight_post_process
        # 将激活后处理器应用到原始模型的激活后处理器属性上
        m_ref.activation_post_process = activation
        # 将量化配置应用到原始模型上
        m_ref.qconfig = qconfig
        # 将原始模型转换为量化后的模型 nnq.Linear，使用预计算的伪量化
        m_ref = nnq.Linear.from_float(m_ref, use_precomputed_fake_quant=True)
        # 断言量化后的模型和深度复制后量化的模型的量化尺度不相等
        self.assertTrue(m_ref._weight_bias()[0].q_scale != m_ref_copy._weight_bias()[0].q_scale)
# 如果当前脚本被直接运行（而不是被作为模块导入），则抛出运行时错误并显示以下消息：
# "This test file is not meant to be run directly, use:"
# 接着显示如何正确使用该脚本的命令，即：
# "\tpython test/test_quantization.py TESTNAME\n\n"
# 最后指示用户应该如何正确地使用该脚本。
if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
```
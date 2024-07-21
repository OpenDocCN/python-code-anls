# `.\pytorch\test\quantization\fx\test_quantize_fx.py`

```py
# Owner(s): ["oncall: quantization"]

# 导入所需的模块和类
from collections import OrderedDict  # 导入有序字典模块
import contextlib  # 导入上下文管理模块
import torch  # 导入PyTorch主模块
import torch.nn.functional as F  # 导入PyTorch的函数式接口模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.ao.nn.quantized as nnq  # 导入PyTorch的量化神经网络模块
import torch.ao.nn.quantized.reference as nnqr  # 导入PyTorch的参考量化神经网络模块
import torch.ao.nn.quantized.dynamic as nnqd  # 导入PyTorch的动态量化神经网络模块
import torch.ao.nn.intrinsic as nni  # 导入PyTorch的内在操作模块
import torch.ao.nn.intrinsic.quantized as nniq  # 导入PyTorch的量化内在操作模块
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd  # 导入PyTorch的动态量化内在操作模块
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY  # 导入PyTorch的图模块，用于用户保留属性键

# graph mode quantization based on fx
from torch.ao.quantization.quantize_fx import (  # 导入基于fx的图模式量化函数
    prepare_fx,  # 导入准备函数
    convert_fx,  # 导入转换函数
    convert_to_reference_fx,  # 导入转换为参考函数
    _convert_to_reference_decomposed_fx,  # 导入分解转换为参考函数
    prepare_qat_fx,  # 导入准备量化感知训练函数
    fuse_fx,  # 导入融合函数
)

from torch.ao.quantization.fx.quantize_handler import DefaultNodeQuantizeHandler  # 导入默认节点量化处理器类

from torch.ao.quantization.fx.match_utils import (  # 导入匹配工具函数
    _is_match,  # 导入判断匹配函数
    MatchAllNode,  # 导入匹配所有节点类
)

from torch.ao.quantization import (  # 导入量化模块
    QuantType,  # 导入量化类型枚举
)

from torch.ao.quantization.quant_type import _get_quant_type_to_str  # 导入获取量化类型字符串函数

from torch.ao.quantization import (  # 导入量化模块
    QuantStub,  # 导入量化存根类
    DeQuantStub,  # 导入去量化存根类
    QuantWrapper,  # 导入量化包装器类
    default_qconfig,  # 导入默认量化配置
    default_dynamic_qconfig,  # 导入默认动态量化配置
    default_per_channel_qconfig,  # 导入默认逐通道量化配置
    default_qat_qconfig,  # 导入默认量化感知训练配置
    default_reuse_input_qconfig,  # 导入默认重用输入量化配置
    default_symmetric_qnnpack_qconfig,  # 导入默认对称QNNPACK量化配置
    default_symmetric_qnnpack_qat_qconfig,  # 导入默认对称QNNPACK量化感知训练配置
    per_channel_dynamic_qconfig,  # 导入逐通道动态量化配置
    float16_dynamic_qconfig,  # 导入动态float16量化配置
    float16_static_qconfig,  # 导入静态float16量化配置
    float_qparams_weight_only_qconfig,  # 导入仅权重float量化配置
    float_qparams_weight_only_qconfig_4bit,  # 导入4位仅权重float量化配置
    get_default_qconfig,  # 导入获取默认量化配置函数
    get_default_qat_qconfig,  # 导入获取默认量化感知训练配置函数
    get_default_qconfig_mapping,  # 导入获取默认量化配置映射函数
    get_default_qat_qconfig_mapping,  # 导入获取默认量化感知训练配置映射函数
    fuse_modules,  # 导入模块融合函数
    fuse_modules_qat,  # 导入模块融合（量化感知训练）函数
    prepare,  # 导入准备函数
    prepare_qat,  # 导入准备量化感知训练函数
    convert,  # 导入转换函数
    quantize_dynamic,  # 导入动态量化函数
    default_placeholder_observer,  # 导入默认占位符观察器
    default_weight_observer,  # 导入默认权重观察器
    PerChannelMinMaxObserver,  # 导入逐通道最小最大观察器类
    FixedQParamsFakeQuantize,  # 导入固定量化参数假量化类
    FixedQParamsObserver,  # 导入固定量化参数观察器类
    FusedMovingAvgObsFakeQuantize,  # 导入融合移动平均假量化类
    FakeQuantize,  # 导入假量化类
    MovingAverageMinMaxObserver,  # 导入移动平均最小最大观察器类
    HistogramObserver,  # 导入直方图观察器类
    ReuseInputObserver,  # 导入重用输入观察器类
    QConfig,  # 导入量化配置类
    default_embedding_qat_qconfig,  # 导入默认嵌入量化感知训练配置
)

from torch.ao.quantization.backend_config import (  # 导入后端配置模块
    get_fbgemm_backend_config,  # 导入获取FBGEMM后端配置函数
    get_qnnpack_backend_config,  # 导入获取QNNPACK后端配置函数
    BackendConfig,  # 导入后端配置类
    BackendPatternConfig,  # 导入后端模式配置类
    DTypeConfig,  # 导入数据类型配置类
    DTypeWithConstraints,  # 导入带约束的数据类型类
    ObservationType  # 导入观察类型枚举
)

from torch.ao.quantization.backend_config.native import (  # 导入本地后端配置模块
    get_test_only_legacy_native_backend_config,  # 导入获取仅用于测试的遗留本地后端配置函数
)

from torch.ao.quantization.qconfig_mapping import (  # 导入量化配置映射模块
    _get_symmetric_qnnpack_qconfig_mapping,  # 导入获取对称QNNPACK量化配置映射函数
    _get_symmetric_qnnpack_qat_qconfig_mapping,  # 导入获取对称QNNPACK量化感知训练配置映射函数
    _GLOBAL_DICT_KEY,  # 导入全局字典键
    _MODULE_NAME_DICT_KEY,  # 导入模块名字典键
    _MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY,  # 导入模块名称对象类型顺序字典键
    _MODULE_NAME_REGEX_DICT_KEY,  # 导入模块名字正则表达式字典键
    _OBJECT_TYPE_DICT_KEY,  # 导入对象类型字典键
    QConfigMapping,  # 导入量化配置映射类
)

from torch.ao.quantization.fx.qconfig_mapping_utils import (  # 导入量化配置映射工具函数模块
    _get_object_type_qconfig,  # 导入获取对象类型量化配置函数
    _get_module_name_qconfig,  # 导入获取模块名量化配置函数
    _get_module_name_regex_qconfig,  # 导入获取模块名正则表达式量化配置函数
    _maybe_adjust_qconfig_for_module_name_object_type_order,  # 导入
# 导入 torch.ao.quantization.fx.pattern_utils 模块中的相关函数和变量
from torch.ao.quantization.fx.pattern_utils import (
    _DEFAULT_FUSION_PATTERNS,
    _DEFAULT_QUANTIZATION_PATTERNS,
    _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP,
    _DEFAULT_OUTPUT_OBSERVER_MAP,
    _register_fusion_pattern,
    _register_quant_pattern,
    get_default_output_activation_post_process_map
)

# 导入 torch.ao.quantization.fx.custom_config 模块中的相关函数和变量
from torch.ao.quantization.fx.custom_config import (
    STANDALONE_MODULE_NAME_DICT_KEY,
    STANDALONE_MODULE_CLASS_DICT_KEY,
    FLOAT_TO_OBSERVED_DICT_KEY,
    OBSERVED_TO_QUANTIZED_DICT_KEY,
    NON_TRACEABLE_MODULE_NAME_DICT_KEY,
    NON_TRACEABLE_MODULE_CLASS_DICT_KEY,
    INPUT_QUANTIZED_INDEXES_DICT_KEY,
    OUTPUT_QUANTIZED_INDEXES_DICT_KEY,
    PRESERVED_ATTRIBUTES_DICT_KEY,
    FuseCustomConfig,
    ConvertCustomConfig,
    PrepareCustomConfig,
    StandaloneModuleConfigEntry,
)

# 导入 torch.ao.quantization.fx.lstm_utils 模块
import torch.ao.quantization.fx.lstm_utils

# 导入 torch.ao.quantization.fx.utils 模块中的相关函数和变量
from torch.ao.quantization.fx.utils import (
    _reroute_tuple_getitem_pattern,
    NodeInfo,
)

# 导入 torch.ao.quantization.fake_quantize 模块中的相关函数
from torch.ao.quantization.fake_quantize import (
    default_fixed_qparams_range_0to1_fake_quant,
    default_fixed_qparams_range_neg1to1_fake_quant,
)

# 导入 torch.ao.quantization.observer 模块中的相关函数和类
from torch.ao.quantization.observer import (
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    MinMaxObserver,
    _is_activation_post_process,
)

# 导入 hypothesis 库中的 given 和 settings 函数
from hypothesis import given, settings
# 导入 hypothesis 库中的 strategies 模块并重命名为 st
from hypothesis import strategies as st
# 导入 torch.testing._internal.common_cuda 模块中的 TEST_MULTIGPU 和 TEST_CUDA
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
# 导入 torch.testing._internal.common_quantization 模块中的相关类和函数
from torch.testing._internal.common_quantization import (
    LinearReluLinearModel,
    LinearReluModel,
    LinearBnLeakyReluModel,
    LinearTanhModel,
    ConvBnAddReluModel,
    QuantizationTestCase,
    skipIfNoFBGEMM,
    skipIfNoQNNPACK,
    skip_if_no_torchvision,
    train_one_epoch,
    run_ddp,
    test_only_eval_fn,
    test_only_train_fn,
    ModelForConvTransposeBNFusion,
    get_supported_device_types,
    skipIfNoONEDNN,
)

# 导入 torch.testing._internal.common_quantization 模块中的相关类
from torch.testing._internal.common_quantization import (
    LinearModelWithSubmodule,
    ResNetBase,
    RNNDynamicModel,
    RNNCellDynamicModel,
)

# 导入 torch.testing._internal.common_quantized 模块中的相关函数
from torch.testing._internal.common_quantized import (
    supported_qengines,
    override_qengines,
    override_quantized_engine,
)

# 导入 torch.testing._internal.common_utils 模块中的相关函数
from torch.testing._internal.common_utils import (
    TemporaryFileName,
    IS_ARM64,
    skipIfTorchDynamo,
)

# 导入 torch.testing._internal.common_quantization 模块中的 NodeSpec 类并重命名为 ns
from torch.testing._internal.common_quantization import NodeSpec as ns

# 导入 torch.testing 模块中的 FileCheck 类
from torch.testing import FileCheck

# 导入 copy、itertools、operator、unittest 和 io 模块
import copy
import itertools
import operator
import unittest
import io

# 导入 typing 模块中的类型定义
from typing import Callable, Optional, List, Tuple

# 定义一个名为 BinaryOp 的 torch.nn.Module 子类
class BinaryOp(torch.nn.Module):
    def __init__(self, binary_op, ibinary_op, is_inplace, is_scalar):
        """ ibinary_op means inplace binary op
        """
        # 调用父类的构造函数
        super().__init__()
        # 创建一个输入通道数为 1，输出通道数为 1，卷积核大小为 1 的 Conv2d 层，并将其转换为 float 类型
        self.conv1 = torch.nn.Conv2d(1, 1, 1).float()
        # 创建另一个类似的 Conv2d 层
        self.conv2 = torch.nn.Conv2d(1, 1, 1).float()
        # 根据传入的参数设置是否为标量操作
        self.is_scalar = is_scalar
        # 根据 ibinary_op 和 is_inplace 参数选择相应的操作作为 self.op
        self.op = ibinary_op if ibinary_op and is_inplace else binary_op
    # 定义一个前向传播方法，接受输入参数 x 和 y
    def forward(self, x, y):
        # 使用 self.conv1 对输入 x 进行卷积操作
        x = self.conv1(x)
        # 如果 self.is_scalar 不为真，则对输入 y 进行 self.conv2 的卷积操作；否则将 y 设为 3
        y = 3 if self.is_scalar else self.conv2(y)
        # 对 x 和 y 进行某种操作，通常是某种元素级的操作
        x = self.op(x, y)
        # 再次对 y 和 x 进行某种操作，可能是另一种元素级的操作
        x = self.op(y, x)
        # 返回最终的处理结果 x
        return x
class BinaryOpNonQuantizedInput(torch.nn.Module):
    def __init__(self, binary_op, ibinary_op, is_inplace, is_scalar):
        """ ibinary_op means inplace binary op
        初始化函数，定义了一个非量化输入的二元操作模块。
        """
        super().__init__()
        # 设置是否为标量
        self.is_scalar = is_scalar
        # 根据是否原地操作选择合适的二元操作
        self.op = ibinary_op if ibinary_op and is_inplace else binary_op

    def forward(self, x, y):
        # 如果是标量，将y设为3，否则保持不变
        y = 3 if self.is_scalar else y
        # 执行二元操作
        x = self.op(x, y)
        return x

class BinaryOpRelu(torch.nn.Module):
    def __init__(self, binary_op, ibinary_op, is_inplace, relu_callable,
                 is_scalar):
        """ ibinary_op means inplace binary op
        初始化函数，定义了一个带有ReLU激活的二元操作模块。
        """
        super().__init__()
        # 创建两个1x1的卷积层
        self.conv1 = torch.nn.Conv2d(1, 1, 1).float()
        self.conv2 = torch.nn.Conv2d(1, 1, 1).float()
        # 根据是否原地操作选择合适的二元操作
        self.op = ibinary_op if ibinary_op and is_inplace else binary_op
        # 设置ReLU函数
        self.relu_callable = relu_callable
        self.is_scalar = is_scalar
        # 根据relu_callable选择ReLU激活函数
        if relu_callable is torch.nn.ReLU:
            self.relu = torch.nn.ReLU()
        else:
            self.relu = relu_callable

    def forward(self, x, y):
        # 对输入x进行卷积操作
        x = self.conv1(x)
        # 如果是标量，对y进行常数转换，否则对y进行卷积操作
        y = 3 if self.is_scalar else self.conv2(y)
        # 执行第一次二元操作
        x = self.op(x, y)
        # 对结果执行ReLU激活
        x = self.relu(x)
        # 执行第二次二元操作
        x = self.op(y, x)
        # 对结果执行ReLU激活
        x = self.relu(x)
        return x

@torch.fx.wrap
def _user_func_with_complex_return_type(x):
    # 使用torch.split在第一维度上将输入x分割成多个张量并组成列表
    return list(torch.split(x, 1, 1))

class TestFuseFx(QuantizationTestCase):
    # 这部分代码没有需要添加注释的内容，因此保持原样不变
    def test_fuse_conv_bn_relu(self):
        # 定义一个测试函数，用于验证融合 Conv-BN-ReLU 的功能

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义多个卷积层和对应的批归一化层，并初始化
                self.conv1d = nn.Conv1d(1, 1, 1)
                self.conv2d = nn.Conv2d(1, 1, 1)
                self.conv3d = nn.Conv3d(1, 1, 1)
                self.bn1d = nn.BatchNorm1d(1)
                self.bn2d = nn.BatchNorm2d(1)
                self.bn3d = nn.BatchNorm3d(1)
                self.conv1d2 = nn.Conv1d(1, 1, 1)
                self.conv2d2 = nn.Conv2d(1, 1, 1)
                self.conv3d2 = nn.Conv3d(1, 1, 1)
                self.bn1d2 = nn.BatchNorm1d(1)
                self.bn2d2 = nn.BatchNorm2d(1)
                self.bn3d2 = nn.BatchNorm3d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                # 网络的前向传播过程
                x = self.conv1d(x)
                x = self.bn1d(x)
                x = self.conv2d(x)
                x = self.bn2d(x)
                x = self.conv3d(x)
                x = self.bn3d(x)
                x = self.conv1d2(x)
                x = self.bn1d2(x)
                x = self.relu(x)
                x = self.conv2d2(x)
                x = self.bn2d2(x)
                x = self.relu(x)
                x = self.conv3d2(x)
                x = self.bn3d2(x)
                x = self.relu(x)
                return x

        # 测试模型处于训练模式
        m = M().train()
        # 当前不检查模块是否配置了 qconfig 就进行融合
        # TODO: 如果未来决定进行这样的检查，需要更新该测试
        # 在准备量化训练的过程中调用融合函数
        m = prepare_qat_fx(m, {}, example_inputs=(torch.randn(1, 1, 1, 1),))
        expected_nodes = [
            ns.call_module(nni.ConvBn1d),
            ns.call_module(nni.ConvBn2d),
            ns.call_module(nni.ConvBn3d),
            ns.call_module(nni.ConvBnReLU1d),
            ns.call_module(nni.ConvBnReLU2d),
            ns.call_module(nni.ConvBnReLU3d),
        ]
        expected_occurrence = {
            ns.call_module(nn.ReLU): 0
        }
        # 检查模型中的节点是否符合预期
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

        # 测试模型处于评估模式
        m = M().eval()
        # 顶层 API 只支持评估模式下的融合
        m = fuse_fx(m)
        expected_nodes = [
            ns.call_module(nn.Conv1d),
            ns.call_module(nn.Conv2d),
            ns.call_module(nn.Conv3d),
            ns.call_module(nni.ConvReLU1d),
            ns.call_module(nni.ConvReLU2d),
            ns.call_module(nni.ConvReLU3d),
        ]
        # ConvBnReLU1d 不被融合
        expected_occurrence = {
            ns.call_module(nn.ReLU): 0
        }
        # 检查模型中的节点是否符合预期
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)
    # 定义一个名为 test_fuse_linear_bn_eval 的测试方法
    def test_fuse_linear_bn_eval(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个线性层，输入维度为1，输出维度为1
                self.linear = nn.Linear(1, 1)
                # 添加一个批标准化层，输入特征维度为1
                self.bn1d = nn.BatchNorm1d(1)

            # 前向传播方法
            def forward(self, x):
                # 线性层的前向传播
                x = self.linear(x)
                # 批标准化层的前向传播
                x = self.bn1d(x)
                return x

        # 创建 M 类的实例，并设置为评估模式
        m = M().eval()
        # 对模型 m 进行融合优化处理
        m = fuse_fx(m)
        # 预期的节点列表，包含一个调用 nn.Linear 的节点
        expected_nodes = [
            ns.call_module(nn.Linear),
        ]
        # 预期的节点出现次数，对应一个调用 nn.BatchNorm1d 的节点，出现次数为0
        expected_occurrence = {
            ns.call_module(nn.BatchNorm1d): 0,
        }
        # 检查模型 m 的图模块节点
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

    # 如果没有 ONEDNN，跳过测试
    @skipIfNoONEDNN
    def test_fuse_linear_bn_leaky_relu_onednn(self):
        # 对于 ONEDNN 后端，linear - bn - leaky_relu 被融合
        from torch.ao.quantization.backend_config import get_onednn_backend_config
        # 预期的节点列表，包含一个调用 nni.LinearLeakyReLU 的节点
        expected_nodes = [
            ns.call_module(nni.LinearLeakyReLU),
        ]
        # 预期的节点出现次数，包含调用 nn.BatchNorm1d 和 nn.LeakyReLU 的节点，均出现次数为0
        expected_occurrence = {
            ns.call_module(nn.BatchNorm1d): 0,
            ns.call_module(nn.LeakyReLU): 0,
        }

        # 对于每个是否有 bn 的情况，进行测试
        for with_bn in [True, False]:
            # 创建 LinearBnLeakyReluModel 类的实例，并设置为评估模式
            m = LinearBnLeakyReluModel(with_bn).eval()
            # 对模型 m 进行融合优化处理，使用 ONEDNN 后端配置
            m = fuse_fx(m,
                        backend_config=get_onednn_backend_config())
            # 检查模型 m 的图模块节点
            self.checkGraphModuleNodes(
                m,
                expected_node_list=expected_nodes,
                expected_node_occurrence=expected_occurrence)

    # 测试线性层 - 批标准化层 - leaky_relu 默认不融合
    def test_linear_bn_leaky_relu_not_fused_by_default(self):
        # 确保线性层 - 批标准化层 - leaky_relu 默认不融合
        for with_bn in [True, False]:
            # 创建 LinearBnLeakyReluModel 类的实例，并设置为评估模式
            m = LinearBnLeakyReluModel(with_bn).eval()
            # 对模型 m 进行融合优化处理
            m = fuse_fx(m)
            # 预期的节点列表，包含调用 nn.Linear 和 nn.LeakyReLU 的节点
            expected_nodes = [
                ns.call_module(nn.Linear),
                ns.call_module(nn.LeakyReLU),
            ]
            # 预期的节点出现次数，包含一个调用 nni.LinearLeakyReLU 的节点，出现次数为0
            expected_occurrence = {
                ns.call_module(nni.LinearLeakyReLU): 0,
            }
            # 检查模型 m 的图模块节点
            self.checkGraphModuleNodes(
                m,
                expected_node_list=expected_nodes,
                expected_node_occurrence=expected_occurrence)

    # 如果没有 ONEDNN，跳过测试
    @skipIfNoONEDNN


这段代码是针对神经网络模型中的线性层、批标准化层和 leaky_relu 激活函数的融合优化进行测试。
    def test_fuse_linear_tanh_for_onednn_backend(self):
        # linear - tanh is fused for onednn backend only
        # 导入从torch.ao.quantization.backend_config模块获取onednn后端配置的函数
        from torch.ao.quantization.backend_config import get_onednn_backend_config
        # 期望的节点列表，包括一个nni.LinearTanh模块调用
        expected_nodes = [
            ns.call_module(nni.LinearTanh),
        ]
        # 期望的节点出现情况，包括nn.Linear和nn.Tanh模块调用次数都为0
        expected_occurrence = {
            ns.call_module(nn.Linear): 0,
            ns.call_module(nn.Tanh): 0,
        }

        # 测试评估模式
        m = LinearTanhModel().eval()
        # fuse_fx是一个顶级API，仅支持评估模式，并且使用onednn后端配置
        m = fuse_fx(m,
                    backend_config=get_onednn_backend_config())
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

    def test_linear_tanh_not_fused_by_default(self):
        # Make sure linear - tanh is not fused by default
        # 测试评估模式
        m = LinearTanhModel().eval()
        # fuse_fx是一个顶级API，仅支持评估模式，但不使用任何特定的后端配置
        m = fuse_fx(m)
        # 期望的节点列表，包括nn.Linear和nn.Tanh模块调用
        expected_nodes = [
            ns.call_module(nn.Linear),
            ns.call_module(nn.Tanh),
        ]
        # 期望的节点出现情况，包括nni.LinearTanh模块调用次数为0
        expected_occurrence = {
            ns.call_module(nni.LinearTanh): 0,
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

    def test_fuse_conv_bn_add_relu_onednn(self):
        # conv - bn - add - relu is fused for onednn backend only
        # 导入从torch.ao.quantization.backend_config模块获取onednn后端配置的函数
        from torch.ao.quantization.backend_config import get_onednn_backend_config
        # 使用itertools.product生成所有选项的组合
        options = itertools.product(
            [True, False],  # with_bn
            [True, False],  # with_relu
            [True, False],  # conv in the left
            [True, False],  # with_two_conv
            [True, False],  # use_torch_add
        )
        for with_bn, with_relu, left_conv, two_conv, use_torch_add in options:
            # 期望的节点列表，根据不同的选项决定是nni.ConvAddReLU2d还是nni.ConvAdd2d模块调用
            expected_nodes = [
                ns.call_module(nni.ConvAddReLU2d if with_relu else nni.ConvAdd2d),
            ]
            # 期望的节点出现情况，包括nn.BatchNorm2d模块调用次数为0
            expected_occurrence = {
                ns.call_module(nni.ConvAddReLU2d if with_relu else nni.ConvAdd2d): 1,
                ns.call_module(nn.BatchNorm2d): 0,
            }

            # 测试评估模式
            m = ConvBnAddReluModel(
                with_bn=with_bn,
                with_relu=with_relu,
                left_conv=left_conv,
                two_conv=two_conv,
                use_torch_add=use_torch_add).eval()

            # fuse_fx是一个顶级API，仅支持评估模式，并且使用onednn后端配置
            m = fuse_fx(m,
                        backend_config=get_onednn_backend_config())
            self.checkGraphModuleNodes(
                m,
                expected_node_list=expected_nodes,
                expected_node_occurrence=expected_occurrence)
    # 定义测试方法，用于测试融合 Conv-BN-Add-ReLU 的默认行为
    def test_fuse_conv_bn_add_relu_by_default(self):
        # 生成所有可能的测试选项组合
        options = itertools.product(
            [True, False],  # 是否包含 BN（Batch Normalization）
            [True, False],  # 是否包含 ReLU 激活函数
            [True, False],  # Conv 是否在左侧
            [True, False],  # 是否有两个 Conv
            [True, False],  # 是否使用 torch.add
        )
        # 对于每一种选项组合，执行以下操作
        for with_bn, with_relu, left_conv, two_conv, use_torch_add in options:
            # 预期的节点列表，初始化为包含一个 Conv2d 模块的列表
            expected_nodes = [
                ns.call_module(nn.Conv2d),
            ]
            # 预期的节点出现次数，初始化为 ConvAdd2d 模块不出现（即出现次数为 0）
            expected_occurrence = {
                ns.call_module(nni.ConvAdd2d): 0,
            }
            # 创建 ConvBnAddReluModel 实例，并设置为评估模式（eval）
            m = ConvBnAddReluModel(
                with_bn=with_bn,
                with_relu=with_relu,
                left_conv=left_conv,
                two_conv=two_conv,
                use_torch_add=use_torch_add).eval()
            # 对模型 m 应用融合操作
            m = fuse_fx(m)
            # 调用 self.checkGraphModuleNodes 方法检查模型 m
            # 检查预期的节点列表和预期的节点出现次数
            self.checkGraphModuleNodes(
                m,
                expected_node_list=expected_nodes,
                expected_node_occurrence=expected_occurrence)

    # 如果没有 ONEDNN 支持，则跳过此测试
    @skipIfNoONEDNN
    def test_fuse_conv_bn_add_relu_lowering(self):
        """ Test fusion and lowering of Conv2d - (bn -) ReLU
            by FX. For onednn backend only.
        """
        from torch.ao.quantization.backend_config import get_onednn_backend_config
        # 获取默认的 onednn 配置映射
        qconfig_mapping = get_default_qconfig_mapping('onednn')
        # 使用 onednn 引擎覆盖量化引擎
        with override_quantized_engine('onednn'):
            # 定义各种选项组合的迭代器
            options = itertools.product(
                [True, False],  # with_bn 是否包含 BatchNorm
                [True, False],  # with_relu 是否包含 ReLU
                [True, False],  # conv in the left 是否在左侧进行卷积
                [True, False],  # two_conv 是否有两个卷积操作
                [True, False],  # use_torch_add 是否使用 torch 的加法操作
            )
            for with_bn, with_relu, left_conv, two_conv, use_torch_add in options:
                # 定义节点出现次数的期望字典
                node_occurrence = {
                    ns.call_function(torch.quantize_per_tensor): 1 if two_conv else 2,
                    ns.call_method("dequantize"): 1,
                    ns.call_module(nniq.ConvAddReLU2d if with_relu else nniq.ConvAdd2d): 1,
                    ns.call_module(nn.Conv2d): 0,
                    ns.call_module(nn.ReLU): 0,
                }
                # 参考节点出现次数的期望字典
                node_occurrence_ref = {
                    ns.call_function(torch.quantize_per_tensor): 3,
                    ns.call_method("dequantize"): 3,
                }

                # 测试 eval 模式
                # 创建 ConvBnAddReluModel 模型并转为 eval 模式
                m = ConvBnAddReluModel(
                    with_bn=with_bn,
                    with_relu=with_relu,
                    left_conv=left_conv,
                    two_conv=two_conv,
                    use_torch_add=use_torch_add).eval()
                # 获取模型的示例输入
                example_x = m.get_example_inputs()
                # 准备 FX 格式的模型
                m = prepare_fx(m, qconfig_mapping,
                               example_inputs=example_x,
                               backend_config=get_onednn_backend_config())
                # 深拷贝模型备份
                m_copy = copy.deepcopy(m)
                # 转换 FX 模型至 onednn 后端
                m = convert_fx(m, backend_config=get_onednn_backend_config())
                # 转换为参考 FX 模型
                m_ref = convert_to_reference_fx(m_copy)
                # 检查模型节点出现次数是否符合预期
                self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
                self.checkGraphModuleNodes(m_ref, expected_node_occurrence=node_occurrence_ref)
                # 执行模型前向传播
                m(*example_x)

    def test_fuse_convtranspose_bn_eval(self):
        # 创建 ConvTransposeBNFusion 模型并转为 eval 模式
        m = ModelForConvTransposeBNFusion().eval()
        # 执行 FX 格式的融合
        m = fuse_fx(m)

        # 期望的节点列表
        expected_nodes = [
            ns.call_module(nn.ConvTranspose1d),
            ns.call_module(nn.ConvTranspose2d),
            ns.call_module(nn.ConvTranspose3d),
        ]
        # 期望的节点出现次数字典
        expected_occurrence = {
            ns.call_module(nn.BatchNorm1d): 0,
            ns.call_module(nn.BatchNorm2d): 0,
            ns.call_module(nn.BatchNorm3d): 0,
        }
        # 检查模型节点是否符合预期
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)
    # 定义测试函数，用于测试融合模块中的ReLU操作
    def test_fuse_module_relu(self):
        # 定义一个继承自torch.nn.Module的内部类M，用于构建模型
        class M(torch.nn.Module):
            # 模型初始化函数
            def __init__(self):
                super().__init__()
                # 定义1维卷积层、2维卷积层、3维卷积层各一层
                self.conv1d = nn.Conv1d(1, 1, 1)
                self.conv2d = nn.Conv2d(1, 1, 1)
                self.conv3d = nn.Conv3d(1, 1, 1)
                # 定义1维批归一化层、2维批归一化层、3维批归一化层各一层
                self.bn1d = nn.BatchNorm1d(1)
                self.bn2d = nn.BatchNorm2d(1)
                self.bn3d = nn.BatchNorm3d(1)
                # 定义ReLU激活函数
                self.relu = nn.ReLU()

            # 前向传播函数
            def forward(self, x):
                # 进行1维卷积操作，并使用ReLU激活函数
                x = self.conv1d(x)
                x = self.relu(x)
                # 进行2维卷积操作，并使用ReLU激活函数
                x = self.conv2d(x)
                x = self.relu(x)
                # 进行3维卷积操作，并使用ReLU激活函数
                x = self.conv3d(x)
                x = self.relu(x)
                # 执行1维批归一化操作，并使用ReLU激活函数
                x = self.bn1d(x)
                x = self.relu(x)
                # 执行2维批归一化操作，并使用ReLU激活函数
                x = self.bn2d(x)
                x = self.relu(x)
                # 执行3维批归一化操作，并使用ReLU激活函数
                x = self.bn3d(x)
                x = self.relu(x)
                return x

        # 创建M类的实例，并设置为评估模式
        m = M().eval()
        # 对模型m进行融合操作
        m = fuse_fx(m)
        # 预期的节点列表，包含了各种融合模块的调用
        expected_nodes = [
            ns.call_module(nni.ConvReLU1d),  # 期望调用ConvReLU1d融合模块
            ns.call_module(nni.ConvReLU2d),  # 期望调用ConvReLU2d融合模块
            ns.call_module(nni.ConvReLU3d),  # 期望调用ConvReLU3d融合模块
            ns.call_module(nni.BNReLU2d),    # 期望调用BNReLU2d融合模块
            ns.call_module(nni.BNReLU3d),    # 期望调用BNReLU3d融合模块
        ]
        # 检查图形化模块的节点，确认是否符合预期
        self.checkGraphModuleNodes(m, expected_node_list=expected_nodes)

    # 装饰器函数，用于在没有FBGEMM的情况下跳过测试
    @skipIfNoFBGEMM
    def test_qconfig_fused_module(self):
        """ TODO: add test for all fused modules
        """
        # 定义量化配置字典，包括线性层、ReLU激活函数、以及F.relu函数的量化配置
        qconfig_dict = {
            "": None,
            "object_type": [(nn.Linear, default_qconfig),
                            (nn.ReLU, default_qconfig),
                            (F.relu, default_qconfig)]
        }

        # 线性-ReLU节点列表，包含量化、线性-ReLU融合、反量化操作
        linearRelu_node_list = [
            ns.call_function(torch.quantize_per_tensor),  # 调用torch.quantize_per_tensor函数
            ns.call_module(nniq.LinearReLU),              # 期望调用LinearReLU融合模块
            ns.call_method('dequantize')                  # 调用dequantize方法进行反量化
        ]

        # 线性-ReLU-线性节点列表，包含量化、线性-ReLU融合、线性、反量化操作
        linearReluLinear_node_list = [
            ns.call_function(torch.quantize_per_tensor),  # 调用torch.quantize_per_tensor函数
            ns.call_module(nniq.LinearReLU),              # 期望调用LinearReLU融合模块
            ns.call_module(nnq.Linear),                   # 期望调用线性层模块
            ns.call_method('dequantize')                  # 调用dequantize方法进行反量化
        ]

        # 测试用例列表，每个元素包含一个模型和其对应的节点列表
        tests = [(LinearReluModel, linearRelu_node_list),               # 线性-ReLU模型及其节点列表
                 (LinearReluLinearModel, linearReluLinear_node_list)]  # 线性-ReLU-线性模型及其节点列表

        # 遍历每个测试用例
        for M, node_list in tests:
            # 创建M类的实例，并设置为评估模式
            m = M().eval()
            # 创建一个例子的输入
            example_inputs = (torch.rand(5, 5),)
            # 使用prepare_fx函数，对模型进行准备，包括量化配置字典和例子的输入
            prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)

            # 对准备好的模型执行例子的输入
            prepared(*example_inputs)
            # 将准备好的模型转换为量化表示
            quantized = convert_fx(prepared)

            # 检查图形化模块的节点，确认是否符合预期节点列表
            self.checkGraphModuleNodes(quantized, expected_node_list=node_list)
    # 定义一个测试用例函数，用于测试存在问题的融合示例
    def test_problematic_fuse_example(self):
        # 定义一个继承自 nn.Sequential 的自定义类 LinearRelu
        class LinearRelu(nn.Sequential):
            def __init__(self):
                super().__init__(
                    nn.Linear(5, 5),  # 添加线性层，输入和输出维度均为 5
                    nn.ReLU(),        # 添加 ReLU 激活函数层
                )

        # 定义一个继承自 torch.nn.Module 的自定义类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin_relu = LinearRelu()  # 创建 LinearRelu 实例
                self.linear = nn.Linear(5, 5)  # 添加另一个线性层，输入和输出维度均为 5

            def forward(self, x):
                x = self.lin_relu(x)   # 使用 LinearRelu 实例处理输入 x
                x = self.linear(x)     # 使用线性层处理上一步的输出 x
                return x

        model = M().eval()  # 创建 M 类的实例并设置为评估模式

        # 定义一个量化配置字典 qconfig_dict
        # 注意：这里的 qconfig_dict 变量名可能有误，实际上应为 qconfig_dict
        # 这些 qconfigs 在某种情况下与默认的 qconfig 不相等
        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.Linear, get_default_qconfig('fbgemm')),  # 设置 Linear 层的量化配置
                (torch.nn.ReLU, get_default_qconfig('fbgemm')),     # 设置 ReLU 层的量化配置
            ],
        }

        # 准备模型以进行量化，并提供示例输入
        m = prepare_fx(model, qconfig_dict, example_inputs=(torch.randn(1, 5),))

        # 检查图模块中的节点，预期会调用 LinearReLU 的融合模块
        self.checkGraphModuleNodes(m, expected_node=ns.call_module(torch.ao.nn.intrinsic.modules.fused.LinearReLU))

    # 标记为跳过的测试用例，临时跳过该测试案例，稍后在支持简单模式格式后将启用
    def test_fuse_addtional_fuser_method(self):
        # 定义一个继承自 torch.nn.Module 的自定义类 MyConvReLU
        class MyConvReLU(torch.nn.Module):
            pass

        # 自定义的卷积ReLU融合函数，将卷积和ReLU层融合成 MyConvReLU 实例
        def my_conv_relu_fuser(conv, relu):
            return MyConvReLU()

        # 定义一个继承自 torch.nn.Module 的自定义类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)  # 添加一个2D卷积层，输入和输出通道数为3，卷积核大小为3
                self.relu = torch.nn.ReLU()          # 添加 ReLU 激活函数层

            def forward(self, x):
                return self.relu(self.conv(x))  # 在输入 x 上应用卷积和 ReLU 激活函数

        m = M().eval()  # 创建 M 类的实例并设置为评估模式

        # 将模型中的卷积和ReLU层融合为 MyConvReLU 类的实例
        m = fuse_fx(m, fuse_custom_config={
            "additional_fuser_method_mapping": {
                (torch.nn.Conv2d, torch.nn.ReLU): my_conv_relu_fuser
            }
        })

        # 检查图模块中的节点，预期会调用 MyConvReLU
        self.checkGraphModuleNodes(m, expected_node=ns.call_module(MyConvReLU))
    def test_fuse_custom_pattern(self):
        # 定义一个测试函数，用于测试自定义模式融合

        class M(torch.nn.Module):
            # 定义一个简单的神经网络模型类
            def __init__(self, use_torch_add=True):
                super().__init__()
                # 初始化网络层
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(3)
                # 根据参数选择不同的加法函数
                if use_torch_add:
                    self.add = torch.add
                else:
                    self.add = operator.add

            def forward(self, x):
                # 前向传播函数
                y = x
                y = self.maxpool(x)
                x = self.conv(x)
                x = self.bn(x)
                x = self.add(y, x)  # 使用选择的加法函数
                x = self.relu(x)
                return x

        for use_torch_add in [True, False]:
            # 循环测试两种不同的加法函数设置
            m = M(use_torch_add).eval()

            def fuse_conv_bn_relu(is_qat, relu, add_pattern):
                # 定义一个融合函数，用于模式匹配和替换
                _, _, bn_pattern = add_pattern
                bn, conv = bn_pattern
                return conv

            # 设置两种不同的后端模式配置
            conv_bn_res_relu_config1 = BackendPatternConfig() \
                ._set_pattern_complex_format((nn.ReLU, (torch.add, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d)))) \
                .set_fuser_method(fuse_conv_bn_relu)
            conv_bn_res_relu_config2 = BackendPatternConfig() \
                ._set_pattern_complex_format((nn.ReLU, (operator.add, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d)))) \
                .set_fuser_method(fuse_conv_bn_relu)
            # 设置后端配置对象
            backend_config = BackendConfig() \
                .set_backend_pattern_config(conv_bn_res_relu_config1) \
                .set_backend_pattern_config(conv_bn_res_relu_config2)
            # 融合模型
            m = fuse_fx(m, backend_config=backend_config)
            # 断言确保融合后 conv 层的类型仍为 Conv2d
            self.assertEqual(type(m.conv), torch.nn.Conv2d)
            # 检查 bn 和 relu 层已被移除，因为整个模式已替换为 conv
            self.assertFalse(hasattr(m, "bn"))
            self.assertFalse(hasattr(m, "relu"))
    def test_fusion_pattern_with_multiple_inputs(self):
        """ This test tests two keys in backend_config: root_node_getter and
        extra_inputs_getter,
        root_node_getter is used to identify a "root" module in the node pattern,
        the node that we'll keep after fusion.
        extra_inputs_getter will return a list of node that needs to be added to the
        fused node as extra inputs.
        """
        # 定义一个测试函数，用于验证多输入的融合模式
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)  # 添加一个卷积层
                self.bn = torch.nn.BatchNorm2d(3)  # 添加一个批量归一化层
                self.relu = torch.nn.ReLU()  # 添加一个ReLU激活层
                self.maxpool = torch.nn.MaxPool2d(3)  # 添加一个最大池化层

            def forward(self, x):
                y = x
                y = self.maxpool(x)  # 对输入进行最大池化操作
                x = self.conv(x)  # 对输入进行卷积操作
                x = self.bn(x)  # 对卷积结果进行批量归一化
                x = torch.add(x, y)  # 将归一化结果与最大池化结果相加
                x = self.relu(x)  # 对相加后的结果进行ReLU激活
                return x

        m = M().eval()  # 创建并评估模型实例

        def fuse_conv_bn_relu(is_qat, relu, add_pattern):
            _, bn_pattern, _ = add_pattern
            bn, conv = bn_pattern
            return conv
        # 定义一个函数，用于融合卷积、批量归一化和ReLU激活的模式，返回融合后的卷积层

        def conv_bn_res_relu_root_node_getter(pattern):
            relu, add_pattern = pattern
            _, bn_pattern, _ = add_pattern
            bn, conv = bn_pattern
            return conv
        # 定义一个函数，用于从模式中获取卷积层作为根节点的函数

        def conv_bn_res_relu_extra_inputs_getter(pattern):
            """ get inputs pattern for extra inputs, inputs for root node
            are assumed to be copied over from root node to the fused node
            """
            relu, add_pattern = pattern
            _, bn_pattern, extra_input = add_pattern
            bn, conv = bn_pattern
            return [extra_input]
        # 定义一个函数，用于从模式中获取额外输入的函数，这些输入将被添加到融合节点作为额外的输入

        conv_bn_res_relu_config = BackendPatternConfig() \
            ._set_pattern_complex_format((nn.ReLU, (torch.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))) \
            .set_fuser_method(fuse_conv_bn_relu) \
            ._set_root_node_getter(conv_bn_res_relu_root_node_getter) \
            ._set_extra_inputs_getter(conv_bn_res_relu_extra_inputs_getter)
        # 配置融合模式的后端配置，包括模式格式、融合方法、根节点获取器和额外输入获取器

        backend_config = BackendConfig().set_backend_pattern_config(conv_bn_res_relu_config)
        # 创建后端配置，并设置融合模式的配置

        m = fuse_fx(m, backend_config=backend_config)
        # 使用配置对模型进行融合操作

        self.assertEqual(type(m.conv), torch.nn.Conv2d)
        # 断言融合后的模型的卷积层类型为torch.nn.Conv2d

        # check bn and relu are gone since we replaced the whole pattern to conv
        self.assertFalse(hasattr(m, "bn"))
        self.assertFalse(hasattr(m, "relu"))
        # 断言融合后的模型中不再包含批量归一化层和ReLU激活层，因为整个模式已被替换为卷积层

        # check conv module has two inputs
        named_modules = dict(m.named_modules())
        for node in m.graph.nodes:
            if node.op == "call_module" and type(named_modules[node.target]) == torch.nn.Conv2d:
                self.assertTrue(len(node.args) == 2), "Expecting the fused op to have two arguments"
        # 验证融合后的卷积模块是否有两个输入
    def test_fusion_pattern_with_matchallnode(self):
        """
        This test tests that the node matched by MatchAllNode will be regarded as an input
        instead of a module to be fused. For instance, we have two patterns:
            (nn.ReLU, (torch.add, MatchAllNode, nn.Conv2d))
            (nn.ReLU, nn.Conv2d)
        And we want to fuse the following model:
            Conv2d -> ReLU +
            Conv2d ------ Add -> ReLU
        ReLU in the first row is matched as MatchAllNode in the residual pattern. But it won't be
        fused as part of that pattern. It needs to be properly fused with the upstream Conv2d.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                y = self.conv1(x)  # Apply first convolution layer
                y = self.relu1(y)  # Apply ReLU activation

                x = self.conv2(x)  # Apply second convolution layer
                x = torch.add(x, y)  # Element-wise addition with y
                x = self.relu2(x)  # Apply ReLU activation
                return x

        m = M().eval()  # Instantiate and set model to evaluation mode

        def fuse_conv_relu(is_qat, conv, relu):
            return conv  # Return the convolution layer unchanged

        def fuse_conv_res_relu(is_qat, relu, add_pattern):
            _, conv, _ = add_pattern
            return conv  # Return the convolution layer from the add pattern

        def conv_res_relu_root_node_getter(pattern):
            relu, (_, conv, _) = pattern
            return conv  # Return the convolution layer as the root node

        def conv_res_relu_extra_inputs_getter(pattern):
            relu, (_, _, extra_input) = pattern
            return [extra_input]  # Return the extra input from the pattern

        # Configure pattern for (Conv2d, ReLU) fusion
        conv_relu_config = BackendPatternConfig((nn.Conv2d, nn.ReLU)) \
            .set_fuser_method(fuse_conv_relu)

        # Configure pattern for (ReLU, (Add, Conv2d, MatchAllNode)) fusion
        conv_res_relu_config = BackendPatternConfig() \
            ._set_pattern_complex_format((nn.ReLU, (torch.add, nn.Conv2d, MatchAllNode))) \
            .set_fuser_method(fuse_conv_res_relu) \
            ._set_root_node_getter(conv_res_relu_root_node_getter) \
            ._set_extra_inputs_getter(conv_res_relu_extra_inputs_getter)

        # Configure backend with both fusion patterns
        backend_config = BackendConfig() \
            .set_backend_pattern_config(conv_relu_config) \
            .set_backend_pattern_config(conv_res_relu_config)

        # Fuse the model using the configured backend patterns
        m = fuse_fx(m, backend_config=backend_config)

        # Assertions to check if convolution layers are still Conv2d
        self.assertEqual(type(m.conv1), torch.nn.Conv2d)
        self.assertEqual(type(m.conv2), torch.nn.Conv2d)

        # Check that ReLU activations are removed since they are replaced by convolutions
        self.assertFalse(hasattr(m, "relu1"))
        self.assertFalse(hasattr(m, "relu2"))
@skipIfNoFBGEMM
# 使用装饰器 @skipIfNoFBGEMM 来标记这个测试类，如果没有支持 FBGEMM 的条件，则跳过执行该类的所有测试用例
class TestQuantizeFx(QuantizationTestCase):
    # 定义一个测试类 TestQuantizeFx，继承自 QuantizationTestCase

    def test_pattern_match(self):
        """ test MatchAllNode with
            conv - bn - add - relu pattern
        """
        # 定义一个测试方法 test_pattern_match，测试 MatchAllNode 模式，该模式为 conv - bn - add - relu

        class M(torch.nn.Module):
            # 定义一个内部模型类 M，继承自 torch.nn.Module
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.bn = nn.BatchNorm2d(1)
                self.relu = nn.ReLU()

            def forward(self, x, y):
                # M 类的前向传播方法，接收两个输入参数 x 和 y
                x = self.conv(x)
                x = self.bn(x)
                x = x + y
                x = self.relu(x)
                return x

        # 定义一个模式 pattern，包含了 (nn.ReLU, (operator.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))
        pattern = (nn.ReLU, (operator.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))
        # 对模型 M 进行符号跟踪
        m = torch.fx.symbolic_trace(M())
        # 构建模块名称到模块对象的字典
        modules = dict(m.named_modules())
        # 遍历图中的节点
        for n in m.graph.nodes:
            # 如果节点的操作为 'call_module'，且模块类型为 nn.ReLU
            if n.op == 'call_module' and type(modules[n.target]) == nn.ReLU:
                # 断言节点 n 是否匹配给定的模式 pattern
                self.assertTrue(_is_match(modules, n, pattern))

    def test_pattern_match_constant(self):
        # 定义一个测试方法 test_pattern_match_constant

        class M(torch.nn.Module):
            # 定义一个内部模型类 M，继承自 torch.nn.Module
            def forward(self, x):
                # M 类的前向传播方法，接收输入参数 x
                x, _ = torch.ops.aten.max_pool2d_with_indices.default(x)
                return x

        # 定义一个模式 pattern，包含了 (operator.getitem, torch.ops.aten.max_pool2d_with_indices.default, 0)
        pattern = (operator.getitem, torch.ops.aten.max_pool2d_with_indices.default, 0)
        # 对模型 M 进行符号跟踪
        m = torch.fx.symbolic_trace(M())
        # 清除图中死代码，以便匹配模式
        m.graph.eliminate_dead_code()
        # 构建模块名称到模块对象的字典
        modules = dict(m.named_modules())
        # 遍历图中的节点
        for n in m.graph.nodes:
            # 如果节点的操作为 'call_function'，且目标函数为 operator.getitem
            if n.op == "call_function" and n.target == operator.getitem:
                # 断言节点 n 是否匹配给定的模式 pattern
                self.assertTrue(_is_match(modules, n, pattern))

    def test_fused_module_qat_swap(self):
        # 定义一个测试方法 test_fused_module_qat_swap

        class Tmp(torch.nn.Module):
            # 定义一个内部模型类 Tmp，继承自 torch.nn.Module
            def __init__(self):
                super().__init__()
                self.tmp = torch.nn.Linear(5, 5)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # Tmp 类的前向传播方法，接收输入参数 x
                x = self.tmp(x)
                return self.relu(x)

        class M(torch.nn.Module):
            # 定义一个内部模型类 M，继承自 torch.nn.Module
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(Tmp(), torch.nn.Linear(5, 5))
                self.mods2 = torch.nn.Linear(5, 5)

            def forward(self, x):
                # M 类的前向传播方法，接收输入参数 x
                a = self.mods1(x)
                x = torch.add(x, 5)
                x = self.mods2(x)
                x = torch.add(x, 5)
                return a, x

        # 创建一个 M 类的实例 model，并设置为训练模式
        model = M().train()
        # 定义量化配置字典 qconfig_dict
        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.Linear, default_qat_qconfig),
                (torch.nn.ReLU, default_qat_qconfig),
            ],
        }
        # 对模型进行量化准备，使用 prepare_qat_fx 函数，传入模型、量化配置字典和示例输入
        prepared = prepare_qat_fx(model, qconfig_dict, example_inputs=(torch.randn(1, 5),))
        # 断言 mods1 中的第一个模块的 tmp 属性是否为 torch.ao.nn.intrinsic.qat.LinearReLU 类型
        self.assertTrue(isinstance(getattr(prepared.mods1, "0").tmp, torch.ao.nn.intrinsic.qat.LinearReLU))

    @skipIfNoFBGEMM
    def test_conv_linear_not_reference(self):
        """ Test quantizing conv and linear
        """
        # 获取所有非参考测试用例（即不是参考模式的测试用例）
        tests = self._get_conv_linear_test_cases(is_reference=False)
        # 遍历每个测试用例
        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            # 确定量化类型为动态或静态
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            # 初始化节点出现次数的字典
            node_occurrence = {}
            # 如果存在预打包权重节点，则设置其初始出现次数为0
            if weight_prepack_node:
                node_occurrence[weight_prepack_node] = 0
            # 运行图模式下的量化操作检查，验证预期的量化节点和节点出现次数
            self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node=quantized_node,
                expected_node_occurrence=node_occurrence,
                is_reference=False)

    @skipIfNoFBGEMM
    # 定义测试函数，用于测试带有引用选项的量化函数 conv 和 linear
    def test_conv_linear_reference(self):
        """ Test quantizing functional conv and linear with reference option
        """
        # 获取测试用例，这些用例具有引用选项
        tests = self._get_conv_linear_test_cases(is_reference=True)

        # 定义内部函数，用于生成键值列表
        def _get_keys(prefix, is_dynamic):
            # 初始化键列表，包括权重量化方案和数据类型
            all_keys = [prefix + "." + k for k in ["weight_qscheme", "weight_dtype"]]
            # 如果不是动态模型，添加权重比例因子和零点的键
            if not is_dynamic:
                all_keys.extend([prefix + "." + k for k in ["weight_scale", "weight_zero_point"]])
            return all_keys

        # 遍历测试用例
        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            # 确定量化类型，动态或静态
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            # 初始化节点出现次数字典
            node_occurrence = {}
            # 如果存在权重预打包节点，将其加入节点出现次数字典
            if weight_prepack_node:
                node_occurrence[weight_prepack_node] = 0

            # 运行图模式下的操作检查函数，获取结果字典
            result_dict = self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node=quantized_node,
                expected_node_occurrence=node_occurrence,
                is_reference=True)
            # 获取量化引用对象
            qr = result_dict["quantized_reference"]

            # 内部函数，用于检查权重量化参数
            def checkWeightQParams(model):
                # 遍历模型的线性和卷积模块
                for module_name in ("linear", "conv"):
                    if hasattr(model, module_name):
                        # 断言模块具有权重量化方案、比例因子和零点属性，并且名称中包含"Reference"
                        self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_qscheme"))
                        self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_scale"))
                        self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_zero_point"))
                        self.assertTrue("Reference" in qr.get_submodule(module_name)._get_name())

            # 内部函数，用于检查序列化和反序列化操作
            def checkSerDeser(model, is_dynamic):
                # 遍历模型的线性和卷积模块
                for module_name in ("linear", "conv"):
                    if hasattr(model, module_name):
                        # 确保序列化操作正常工作
                        state_dict = copy.deepcopy(model.state_dict())
                        all_keys = _get_keys(module_name, is_dynamic)
                        for key in all_keys:
                            self.assertTrue(key in state_dict)
                        # 检查load_state_dict是否能恢复状态
                        module = getattr(model, module_name)
                        prev_scale = module.weight_scale
                        module.weight_scale = None
                        model.load_state_dict(state_dict)
                        module = getattr(model, module_name)
                        self.assertTrue(torch.equal(prev_scale, module.weight_scale))

            # 检查权重量化参数
            checkWeightQParams(qr)
            # 深拷贝量化引用对象，确保拷贝后仍然保留量化参数
            qr = copy.deepcopy(qr)
            checkWeightQParams(qr)

            # 检查序列化和反序列化操作
            checkSerDeser(qr, is_dynamic)

    # 如果没有FBGEMM，则跳过测试
    @skipIfNoFBGEMM
    # 定义一个测试方法，用于验证转置卷积的量化结果，此方法不使用参考实现
    def test_conv_transpose_not_reference(self):
        """ Test quantizing transposed conv
        """
        # 获取转置卷积的测试用例，不使用ReLU，且不是参考实现
        tests = self._get_conv_transpose_test_cases(use_relu=False, is_reference=False)
        
        # 遍历测试用例
        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            # 根据是否动态量化来选择量化类型
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            
            # 初始化节点出现次数的字典
            node_occurrence = {}
            if weight_prepack_node:
                # 若有预打包权重节点，则初始化其出现次数为0
                node_occurrence[weight_prepack_node] = 0
            
            # 调用方法验证图模式下的量化操作
            self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),  # 创建模块对象
                inputs, quant_type,  # 输入数据和量化类型
                expected_node=quantized_node,  # 期望的量化节点
                expected_node_occurrence=node_occurrence,  # 预期节点出现次数
                is_reference=False  # 标志不是参考实现
            )

    # 如果没有FBGEMM，跳过此测试
    @skipIfNoFBGEMM
    def test_conv_transpose_reference(self):
        """ Test quantizing transposed conv with reference option
        """
        # 获取使用参考选项的反向卷积量化测试用例
        tests = self._get_conv_transpose_test_cases(use_relu=False, is_reference=True)

        def _get_keys(prefix, is_dynamic):
            # 返回给定前缀和动态/静态属性的键列表
            all_keys = [prefix + "." + k for k in ["weight_qscheme", "weight_dtype"]]
            if not is_dynamic:
                # 如果不是动态模式，则添加权重缩放和零点的键
                all_keys.extend([prefix + "." + k for k in ["weight_scale", "weight_zero_point"]])
            return all_keys

        # 遍历每个测试用例
        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            # 确定量化类型为动态或静态
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            # 初始化节点出现次数字典
            node_occurrence = {}
            if weight_prepack_node:
                # 如果有权重预打包节点，则将其出现次数设置为0
                node_occurrence[weight_prepack_node] = 0
            # 运行图模式下的操作检查，返回结果字典
            result_dict = self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node=quantized_node,
                expected_node_occurrence=node_occurrence,
                is_reference=True)
            qr = result_dict["quantized_reference"]

            def checkWeightQParams(model):
                # 检查权重量化参数的存在和正确性
                module_name = "deconv"
                if hasattr(model, module_name):
                    self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_qscheme"))
                    self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_scale"))
                    self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_zero_point"))
                    self.assertTrue("Reference" in qr.get_submodule(module_name)._get_name())

            def checkSerDeser(model, is_dynamic):
                # 检查模型的序列化和反序列化操作
                module_name = "deconv"
                if hasattr(model, module_name):
                    # 确保序列化操作正常
                    state_dict = copy.deepcopy(model.state_dict())
                    all_keys = _get_keys(module_name, is_dynamic)
                    for key in all_keys:
                        self.assertTrue(key in state_dict)
                    # 检查加载状态字典后能够恢复状态
                    module = getattr(model, module_name)
                    prev_scale = module.weight_scale
                    module.weight_scale = None
                    model.load_state_dict(state_dict)
                    module = getattr(model, module_name)
                    self.assertTrue(torch.equal(prev_scale, module.weight_scale))

            # 检查权重量化参数
            checkWeightQParams(qr)
            qr = copy.deepcopy(qr)
            # 确保在复制后仍保留量化参数
            checkWeightQParams(qr)

            # 检查模型的序列化和反序列化
            checkSerDeser(qr, is_dynamic)
    # 定义一个测试函数，用于验证转置卷积加 relu 的量化
    # Fusion with relu is not supported.（不支持与 relu 合并。）
    def test_conv_transpose_relu_not_reference(self):
        """ Test quantizing transposed conv + relu
            Fusion with relu is not supported.
        """
        # 获取使用 relu 和非参考模式的转置卷积测试用例
        tests = self._get_conv_transpose_test_cases(use_relu=True, is_reference=False)
        # 遍历测试用例
        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            # 根据 is_dynamic 确定量化类型
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            # 创建一个空的节点出现次数字典
            node_occurrence = {}
            # 如果存在 weight_prepack_node，设置其出现次数为 0
            if weight_prepack_node:
                node_occurrence[weight_prepack_node] = 0
            # 如果 quantized_node 的操作是 'call_module'
            if quantized_node.op == 'call_module':
                # 设置 nn.ReLU 的调用次数为 1
                node_occurrence[ns.call_module(nn.ReLU)] = 1
            else:
                # 否则设置 F.relu 的调用次数为 1
                node_occurrence[ns.call_function(F.relu)] = 1
            # 调用父类的方法检查图模式下的操作
            self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node=quantized_node,
                expected_node_occurrence=node_occurrence,
                is_reference=False)

    # 如果没有 FBGEMM，跳过该测试
    @skipIfNoFBGEMM
    def test_conv_transpose_relu_reference(self):
        """ Test quantizing transposed conv with reference option
            Fusion with relu is not supported.
        """
        # 获取使用 relu 和 reference 选项的转置卷积的测试用例
        tests = self._get_conv_transpose_test_cases(use_relu=True, is_reference=True)

        def _get_keys(prefix, is_dynamic):
            # 构建参数键列表，包括量化方案和数据类型
            all_keys = [prefix + "." + k for k in ["weight_qscheme", "weight_dtype"]]
            if not is_dynamic:
                # 如果不是动态量化，则还需添加量化的比例因子和零点
                all_keys.extend([prefix + "." + k for k in ["weight_scale", "weight_zero_point"]])
            return all_keys

        # 遍历测试用例
        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            node_occurrence = {}
            if weight_prepack_node:
                # 设置权重预打包节点出现次数为零
                node_occurrence[weight_prepack_node] = 0
            if quantized_node.op == 'call_module':
                # 如果量化节点为调用模块，则期望 ReLU 被调用一次
                node_occurrence[ns.call_module(nn.ReLU)] = 1
            else:
                # 否则期望调用 F.relu 一次
                node_occurrence[ns.call_function(F.relu)] = 1
            # 进行图模式 FX 操作的检查
            result_dict = self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node=quantized_node,
                expected_node_occurrence=node_occurrence,
                is_reference=True)
            qr = result_dict["quantized_reference"]

            def checkWeightQParams(model):
                module_name = "deconv"
                if hasattr(model, module_name):
                    # 断言模型的子模块具有量化参数
                    self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_qscheme"))
                    self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_scale"))
                    self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_zero_point"))
                    # 断言模型的名称中包含 "Reference"
                    self.assertTrue("Reference" in qr.get_submodule(module_name)._get_name())

            def checkSerDeser(model, is_dynamic):
                module_name = "deconv"
                if hasattr(model, module_name):
                    # 确保序列化工作正常
                    state_dict = copy.deepcopy(model.state_dict())
                    all_keys = _get_keys(module_name, is_dynamic)
                    for key in all_keys:
                        # 断言键存在于状态字典中
                        self.assertTrue(key in state_dict)
                    # 检查 load_state_dict 是否恢复状态
                    module = getattr(model, module_name)
                    prev_scale = module.weight_scale
                    module.weight_scale = None
                    model.load_state_dict(state_dict)
                    module = getattr(model, module_name)
                    # 断言量化参数在加载状态后得到恢复
                    self.assertTrue(torch.equal(prev_scale, module.weight_scale))

            # 检查量化参数
            checkWeightQParams(qr)
            qr = copy.deepcopy(qr)
            # 确保复制后量化参数保持不变
            checkWeightQParams(qr)

            # 检查序列化和反序列化
            checkSerDeser(qr, is_dynamic)

    @skipIfNoFBGEMM
    # 定义一个测试函数，用于测试动态量化权重观察器是否在转换步骤中运行
    def test_dynamic_quant_weight_observer(self):
        ''' Test that weight observer is run in convert step
        '''
        
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                # 将传入的权重参数转换为模型的可训练参数
                self.weight = torch.nn.Parameter(weight)

            # 定义模型的前向传播方法
            def forward(self, x):
                return F.linear(x, self.weight)

        # 创建一个模型实例，并设置为评估模式
        m = M(torch.rand(1, 1)).eval()
        
        # 定义动态量化的配置
        qconfig = default_dynamic_qconfig
        qconfig_dict = {'': qconfig}
        
        # 准备模型以备转换，传入示例输入
        example_inputs = (torch.rand(1, 1),)
        prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        
        # 将准备好的模型转换为参考实现
        quantized = convert_to_reference_fx(prepared)
        
        # 获取量化后的权重参数的量化参数（量化比例和零点）
        qparams = (quantized._scale_0, quantized._zero_point_0)
        
        # 创建权重观察器实例
        weight_obs = qconfig.weight()
        
        # 对量化后的权重进行观察
        weight_obs(quantized.weight)
        
        # 获取实际值以避免张量大小不匹配错误，torch.Size([]) vs torch.Size([1])
        ref_qparams = (weight_obs.calculate_qparams()[0].item(), weight_obs.calculate_qparams()[1].item())
        
        # 使用断言检查量化参数是否与参考量化参数相等
        self.assertEqual(qparams, ref_qparams)
    def test_conv_bn_relu(self):
        """ Tests fusion and quantization for "Conv - Bn" and "Conv - Bn - ReLU"
        """
        # 定义不同维度的卷积层类别
        convs = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }
        # 定义不同维度的批标准化层类别
        bns = {
            1: nn.BatchNorm1d,
            2: nn.BatchNorm2d,
            3: nn.BatchNorm3d,
        }
        # 定义量化后的不同维度的卷积层类别
        quantized_convs = {
            1: nnq.Conv1d,
            2: nnq.Conv2d,
            3: nnq.Conv3d,
        }
        # 定义量化后的不同维度的卷积-ReLU层类别
        quantized_conv_relus = {
            1: nniq.ConvReLU1d,
            2: nniq.ConvReLU2d,
            3: nniq.ConvReLU3d,
        }

        # 定义模型类，包含卷积层、批标准化层、ReLU激活函数等
        class M(torch.nn.Module):
            def __init__(self, dim, has_relu):
                super().__init__()
                # 初始化卷积层
                self.conv = convs[dim](3, 3, 3)
                # 初始化批标准化层
                self.bn = bns[dim](3                # 如果有ReLU激活函数则使用，否则使用恒等映射
                self.relu = nn.ReLU() if has_relu else nn.Identity()
                self.has_relu = has_relu
                # 量化前的占位符
                self.quant = QuantStub()
                # 反量化的占位符
                self.dequant = DeQuantStub()

            def forward(self, x):
                # 对输入进行量化
                x = self.quant(x)
                # 进行卷积操作
                x = self.conv(x)
                # 进行批标准化操作
                x = self.bn(x)
                # 如果有ReLU激活函数则使用，否则不作处理
                if self.has_relu:
                    x = self.relu(x)
                # 对输出进行反量化
                x = self.dequant(x)
                return x

        # 遍历所有可能的参数组合，测试不同配置下的量化效果
        options = itertools.product([1, 2, 3], [True, False], self.static_quant_types)
        for dim, has_relu, quant_type in options:
            # 根据是否有ReLU激活函数选择期望的量化节点
            expected_node = ns.call_module(
                quantized_conv_relus[dim] if has_relu
                else quantized_convs[dim])
            # 创建模型实例
            m = M(dim, has_relu)
            # 深度复制模型，用于立即模式下的图模式操作检查
            m_eager = copy.deepcopy(m)
            # 检查图模式下的操作
            result_dict = self.checkGraphModeFxOp(
                m,
                self.img_data_dict[dim],
                quant_type,
                expected_node=expected_node,
            )
            # 获取量化后的输出结果
            result = result_dict["quantized_output"]

            # 检查数值结果
            qengine = torch.backends.quantized.engine
            if quant_type == QuantType.STATIC:
                # 静态量化模式下的配置
                m_eager.eval()
                qconfig = get_default_qconfig(qengine)
                prepare_fn = prepare
                is_qat = False
            else:
                # 动态量化模式下的配置
                m_eager.train()
                qconfig = get_default_qat_qconfig(qengine)
                prepare_fn = prepare_qat
                is_qat = True

            # 合并列表中的模块（卷积、批标准化、ReLU）
            fuse_list = ["conv", "bn"]
            if has_relu:
                fuse_list.append("relu")
            if is_qat:
                # 如果是动态量化模式，则原地合并模块
                fuse_modules_qat(m_eager, fuse_list, inplace=True)
            else:
                # 否则，在原地合并模块
                fuse_modules(m_eager, fuse_list, inplace=True)
            # 设置模型的量化配置
            m_eager.qconfig = qconfig
            # 对模型进行准备
            m_eager = prepare_fn(m_eager)
            # 获取准备后的结果
            prepared_fx = result_dict["prepared"]

            # 在立即模式下运行模型
            m_eager(*self.img_data_dict[dim][0])
            # 执行模型转换
            m_eager = convert(m_eager)
            # 获取立即模式下的结果
            result_eager = m_eager(*self.img_data_dict[dim][0])
            # 断言两种模式下的输出结果一致
            self.assertEqual(result, result_eager)
    # 定义一个测试方法，用于测试带有批标准化的线性层量化
    def test_linear_bn(self):
        # 定义一个内部模型类M，继承自torch.nn.Module
        class M(torch.nn.Module):
            # 构造方法，初始化模型的各个组件
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入维度为4，输出维度为4
                self.linear = nn.Linear(4, 4)
                # 创建一个批标准化层，输入维度为4
                self.bn = nn.BatchNorm1d(4)
                # 创建一个量化存根
                self.quant = QuantStub()
                # 创建一个反量化存根
                self.dequant = DeQuantStub()

            # 前向传播方法
            def forward(self, x):
                # 对输入进行量化
                x = self.quant(x)
                # 线性层处理量化后的输入
                x = self.linear(x)
                # 批标准化层处理线性层的输出
                x = self.bn(x)
                # 反量化处理批标准化层的输出
                x = self.dequant(x)
                return x

        # 准备测试数据，包含一个4x4的随机张量
        data = (torch.randn(4, 4),)
        # 遍历静态量化类型列表
        for quant_type in self.static_quant_types:
            # 期望的节点类型为带量化线性层
            expected_node = ns.call_module(nnq.Linear)
            # 创建模型实例m
            m = M()
            # 使用深拷贝创建eager模式下的模型实例m_eager
            m_eager = copy.deepcopy(m)
            # 调用checkGraphModeFxOp方法，进行图模式下的操作检查
            result_dict = self.checkGraphModeFxOp(m, data, quant_type, expected_node=expected_node)
            # 获取量化后的输出结果
            result = result_dict["quantized_output"]

            # 按照量化引擎选择融合模块的顺序为线性层和批标准化层
            fuse_list = ["linear", "bn"]
            qengine = torch.backends.quantized.engine
            # 如果是静态量化类型
            if quant_type == QuantType.STATIC:
                # 将模型设置为评估模式
                m_eager.eval()
                # 获取默认的量化配置
                qconfig = get_default_qconfig(qengine)
                # 使用prepare函数准备模型
                prepare_fn = prepare
                # 在原地融合指定的模块列表
                fuse_modules(m_eager, fuse_list, inplace=True)
            else:
                # 否则，设置模型为训练模式
                m_eager.train()
                # 获取默认的量化训练配置
                qconfig = get_default_qat_qconfig(qengine)
                # 使用prepare_qat函数准备模型
                prepare_fn = prepare_qat
                # 在原地融合指定的模块列表
                fuse_modules_qat(m_eager, fuse_list, inplace=True)
            # 设置模型的量化配置
            m_eager.qconfig = qconfig
            # 对模型进行配置准备
            m_eager = prepare_fn(m_eager)
            # 在数据上执行模型的前向传播
            m_eager(*data)
            # 将模型转换为量化表示
            m_eager = convert(m_eager)
            # 计算量化后的输出结果
            result_eager = m_eager(*data)
            # 断言量化后的结果与eager模式下的结果一致
            self.assertEqual(result, result_eager)

    # 如果没有FBGEMM引擎，跳过测试
    @skipIfNoFBGEMM
    # 定义一个测试方法，用于测试动态量化和 FP16 量化
    def test_dynamic_quant_fp16(self):
        # 使用 'fbgemm' 引擎覆盖默认量化引擎
        with override_quantized_engine('fbgemm'):
            # 定义一个简单的线性层模型
            class Linear(torch.nn.Module):
                def __init__(self, weight):
                    super().__init__()
                    self.weight = torch.nn.Parameter(weight)

                def forward(self, x):
                    return F.linear(x, self.weight)

            # 创建一个随机输入张量和随机权重张量
            linear_input = torch.rand(8, 5)
            linear_weight = torch.rand(10, 5)

            # 定义一个包含线性层的模块
            class LinearModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(5, 10)

                def forward(self, x):
                    return self.linear(x)

            # 创建一个随机输入张量用于线性层模块
            linear_module_input = torch.rand(8, 5)

            # 定义测试用例列表，每个元素为一个元组，包含模块类、模块构造输入、输入数据、量化节点和权重预打包节点
            tests = [
                (Linear, (linear_weight,), (linear_input,),
                 ns.call_function(torch.ops.quantized.linear_dynamic_fp16),
                 ns.call_function(torch.ops.quantized.linear_prepack_fp16)),
                (LinearModule, (), (linear_module_input,),
                 ns.call_module(nnqd.Linear),
                 None),
            ]
            # 迭代执行每个测试用例
            for (ModuleClass, module_constructor_inputs,
                 inputs, quantized_node, weight_prepack_node) in tests:
                # 对于每种量化是否参考的情况（True 或 False）
                for is_reference in [True, False]:
                    # 初始化节点出现次数的字典
                    node_occurrence = {}
                    # 如果有权重预打包节点，则将其加入节点出现次数字典
                    if weight_prepack_node:
                        node_occurrence[weight_prepack_node] = 0
                    # 创建模块实例并设置为评估模式
                    m = ModuleClass(*module_constructor_inputs).eval()
                    # 定义量化配置字典
                    qconfig_dict = {"": float16_dynamic_qconfig}
                    # 使用输入数据准备函数将模块准备为量化模块
                    m = prepare_fx(m, qconfig_dict, example_inputs=inputs)
                    # 根据量化是否参考进行转换函数选择，并转换模块
                    convert_fn = convert_to_reference_fx if is_reference else convert_fx
                    m = convert_fn(m)
                    # 检查图模块节点，验证期望的节点出现次数
                    self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    # 如果不支持多 GPU，则跳过该测试
    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    # 如果 CUDA 不可用，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 使用覆盖量化引擎装饰器
    @override_qengines
    def test_qat_prepare_device_affinity(self):
        """
        Tests that FX QAT prepare pass respects device affinity
        """
        # 定义一个简单的神经网络模型
        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                # 添加一个1x1的卷积层，输入和输出通道数均为1
                self.conv = nn.Conv2d(1, 1, 1)
                # 添加一个批归一化层，输入通道数为1
                self.bn = nn.BatchNorm2d(1)
                # 添加一个ReLU激活层
                self.relu = nn.ReLU()

            def forward(self, x):
                # 前向传播过程
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        # 创建一个模型实例
        model = Model()
        # 获取量化推理引擎
        qengine = torch.backends.quantized.engine
        # 构建量化配置字典，使用默认的QAT量化配置
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig(qengine)}
        # 指定设备为cuda:0，将模型移动到该设备上
        device = torch.device('cuda:0')
        model.to(device)

        # 准备示例输入数据
        example_inputs = (torch.randn(4, 1, 4, 4, device=device),)
        # 进行QAT准备
        model = prepare_qat_fx(model, qconfig_dict, example_inputs=example_inputs)

        # 确保在CUDA上运行输入数据无需任何更改
        model(*example_inputs)

        # 确保所有参数和缓冲区都在预期的设备上
        model_devices = {p.device for p in model.parameters()} | \
            {p.device for p in model.buffers()}
        self.assertEqual(len(model_devices), 1)
        model_device = next(iter(model_devices))
        self.assertEqual(model_device, device)

    @skipIfNoFBGEMM
    def test_dict_output(self):
        """ Make sure quantization runs for models with dictionary output
        """
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个1x1的卷积层，输入和输出通道数均为1
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 前向传播过程，返回一个字典类型的输出
                return {"output": self.conv(x["input"])}

        # 准备示例输入数据
        example_inputs = ({"input": torch.randn(1, 1, 1, 1)},)
        # 创建模型实例，并设置为评估模式
        m = M().eval()
        # 构建量化配置字典，使用默认的量化配置
        qconfig_dict = {"": default_qconfig}
        # 对模型进行量化准备
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 运行模型，验证量化后的行为
        m(*example_inputs)
        # 将模型转换为量化版本
        m = convert_fx(m)
        # 再次运行模型，验证转换后的行为
        m(*example_inputs)

    @override_qengines
    def test_attention(self):
        """ Make sure quantization runs for a corner case in attention module
        """
        # 定义一个继承自torch.nn.Module的子类M，用于测试注意力模块的量化情况
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个1x1的卷积层
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 执行前向传播
                x = self.conv(x)
                # 将输出x切片成q、k、v三部分，沿着dim=0维度
                q, k, v = x.chunk(3, dim=0)
                # 将q、k、v分别进行连续性处理和维度变换
                q = q.contiguous().view(-1, 1).transpose(0, 1)
                k = k.contiguous().view(-1, 1).transpose(0, 1)
                v = v.contiguous().view(-1, 1).transpose(0, 1)
                # 断言k的大小为1，即key的大小应该等于1
                torch._assert(
                    k.size(1) == 1, "key size should be equal to 1"
                )
                # 计算r为k与v的矩阵乘积
                r = torch.mm(k, v)
                # 返回量化后的结果，q * k + r
                return q * k + r

        # 设置一个示例输入
        example_inputs = (torch.randn(3, 1, 1, 1),)
        # 实例化M类，并设置为评估模式
        m = M().eval()
        # 设置量化配置字典
        qconfig_dict = {
            "": None,
            "object_type": [
                (nn.Conv2d, default_qconfig),
            ]
        }
        # 准备模型以进行量化仿真，传入量化配置和示例输入
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 运行量化仿真后的模型
        m(*example_inputs)
        # 将模型转换为量化模型
        m = convert_fx(m)
        # 再次运行转换后的量化模型
        m(*example_inputs)

    def test_standalone_module_float_interface(self):
        # 设置浮点接口配置字典
        float_interface_config = {
            "input_quantized_idxs": [],  # 浮点输入
            "output_quantized_idxs": [],  # 浮点输出
        }
        interface_config = float_interface_config
        # 设置准备检查计数字典，用于检查调用最小最大观察器的次数
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 2
        }
        # 设置独立模块准备检查计数字典，用于检查调用最小最大观察器的次数
        standalone_prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 2
        }
        # 设置转换检查计数字典，用于检查调用量化和反量化函数以及Conv2d模块的次数
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_module(nnq.Conv2d): 1,
            ns.call_method("dequantize"): 1,
        }
        # 设置独立模块转换检查计数字典，用于检查调用量化、反量化函数以及Conv2d模块的次数
        standalone_convert_count_check = {
            # 独立模块将以浮点作为输入和输出，因此我们会在模块中看到量化和反量化
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_module(nnq.Conv2d): 1,
            ns.call_method("dequantize"): 1,
        }
        # 调用测试函数，测试独立模块的浮点接口
        self._test_standalone_module(
            interface_config,
            prepare_count_check,
            standalone_prepare_count_check,
            convert_count_check,
            standalone_convert_count_check)
    # 定义一个测试函数，用于测试独立模块的量化接口
    def test_standalone_module_quantized_interface(self):
        # 设置量化接口的配置，指定输入和输出量化的索引
        quantized_interface_config = {
            "input_quantized_idxs": [0],  # 量化输入
            "output_quantized_idxs": [0],  # 量化输出
        }
        # 将量化接口配置赋值给接口配置变量
        interface_config = quantized_interface_config
        
        # 准备检查预处理计数的字典，用于第一个卷积层的输入和输出观察器
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 2
        }
        
        # 准备检查预处理计数的字典，用于独立模块中卷积层的输出观察器
        standalone_prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 1
        }
        
        # 转换计数检查字典，用于量化输入和独立模块中卷积层的输出反量化
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 1,  # 为卷积层量化输入
            ns.call_module(nnq.Conv2d): 1,  # 调用量化的卷积层模块
            ns.call_method("dequantize"): 1,  # 反量化独立模块的卷积层输出
        }
        
        # 独立模块转换计数检查字典，指定在父模块中进行输入量化，在量化卷积模块中进行输出量化，父模块中进行输出反量化
        standalone_convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 0,  # 输入量化在父模块中完成
            ns.call_module(nnq.Conv2d): 1,  # 调用量化的卷积层模块
            ns.call_method("dequantize"): 0,  # 输出反量化在父模块中完成
        }
        
        # 调用测试函数，传入以上各项配置进行测试
        self._test_standalone_module(
            interface_config,
            prepare_count_check,
            standalone_prepare_count_check,
            convert_count_check,
            standalone_convert_count_check)

    # 如果没有 FBGEMM，跳过这个测试
    @skipIfNoFBGEMM
    def test_qconfig_none(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        # 创建模型实例并设为评估模式
        m = M().eval()
        
        # 定义量化配置字典，指定对"conv2"模块不进行量化
        qconfig_dict = {"": default_qconfig,
                        "module_name": [("conv2", None)]}
        
        # 准备示例输入
        example_inputs = (torch.randn(1, 1, 1, 1),)
        
        # 准备模型，应用量化配置并进行一次前向传播
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        
        # 将模型转换为量化模型
        m = convert_fx(m)
        
        # 再次进行一次前向传播
        m(*example_inputs)
        
        # 定义预期的节点列表，用于验证模型的图模块节点
        node_list = [
            ns.call_function(torch.quantize_per_tensor),  # 对张量进行量化
            ns.call_module(nnq.Conv2d),  # 调用量化的卷积层模块
            ns.call_method("dequantize"),  # 对张量进行反量化
            ns.call_module(nn.Conv2d),  # 普通卷积层模块
        ]
        
        # 调用检查函数，验证模型的图模块节点是否与预期一致
        self.checkGraphModuleNodes(m, expected_node_list=node_list)
    def test_qconfig_module_type(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 添加一个 2D 卷积层
                self.conv = nn.Conv2d(1, 1, 1)
                # 添加一个全连接层
                self.linear = nn.Linear(9, 3)

            # 前向传播函数
            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 将输出张量展平为一维
                x = x.reshape((1, -1))
                # 执行线性层操作
                x = self.linear(x)
                return x

        # 创建 M 类的实例并设置为评估模式
        m = M().eval()
        # 定义量化配置字典，仅对 Conv2d 类型对象进行量化
        qconfig_dict = {"object_type": [(torch.nn.Conv2d, default_qconfig)]}
        # 准备模型以便量化，并使用示例输入
        example_inputs = (torch.randn(1, 1, 3, 3),)
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 对准备好的量化模型执行一次前向传播
        m(*example_inputs)
        # 将准备好的量化模型转换为量化模型
        m = convert_fx(m)
        # 再次执行一次前向传播
        m(*example_inputs)
        # 定义预期的节点列表，描述模型图中的操作顺序
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_method("dequantize"),
            ns.call_module(nn.Linear),
        ]
        # 检查模型图的节点是否符合预期
        self.checkGraphModuleNodes(m, expected_node_list=node_list)

    def test_qconfig_qat_module_type(self):
        # 定义一个继承自 nn.Sequential 的类 LinearRelu
        class LinearRelu(nn.Sequential):
            # 初始化函数
            def __init__(self):
                super().__init__(
                    # 添加一个线性层
                    nn.Linear(5, 5),
                    # 添加一个 ReLU 激活函数层
                    nn.ReLU(),
                )

        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 添加 LinearRelu 类的实例作为模型的一部分
                self.lin_relu = LinearRelu()
                # 添加一个线性层
                self.linear = nn.Linear(5, 5)

            # 前向传播函数
            def forward(self, x):
                # 执行 LinearRelu 类的前向传播
                x = self.lin_relu(x)
                # 执行线性层操作
                x = self.linear(x)
                return x

        # 创建 M 类的实例并设置为训练模式
        model = M().train()

        # 定义量化训练配置字典，对 Linear 和 ReLU 类型对象进行量化训练
        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.Linear, default_qat_qconfig),
                (torch.nn.ReLU, default_qat_qconfig),
            ],
        }
        # 准备模型以便量化训练，并使用示例输入
        example_inputs = (torch.rand(5, 5),)
        m = prepare_qat_fx(model, qconfig_dict, example_inputs=example_inputs)
        # 对准备好的量化训练模型执行一次前向传播
        m(*example_inputs)
        # 将准备好的量化训练模型转换为量化模型
        m = convert_fx(m)
        # 再次执行一次前向传播
        m(*example_inputs)
        # 定义预期的节点列表，描述模型图中的操作顺序
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nniq.LinearReLU),
            ns.call_module(nnq.Linear),
            ns.call_method("dequantize"),
        ]
        # 检查模型图的节点是否符合预期
        self.checkGraphModuleNodes(m, expected_node_list=node_list)
    # 定义一个测试函数，用于测试量化配置功能
    def test_qconfig_function(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            # 实现神经网络前向传播方法
            def forward(self, x, y):
                return x + y

        # 创建一个 M 类的实例并设置为评估模式
        m = M().eval()
        # 定义量化配置字典，指定对加法操作进行量化
        qconfig_dict = {"object_type": [(operator.add, default_qconfig)]}
        # 生成一个随机张量作为输入数据
        data = torch.randn(1, 1, 1, 1)
        example_inputs = (data, data)
        # 准备量化的函数，根据给定的量化配置对模型 m 进行准备
        m = prepare_fx(m, qconfig_dict, example_inputs)
        # 对准备好的模型进行前向传播
        m(*example_inputs)
        # 将准备好的模型转换为量化模型
        m = convert_fx(m)
        # 对转换后的量化模型进行前向传播
        m(*example_inputs)
        # 定义期望的节点操作序列，表示预期的量化节点操作顺序
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.add),
            ns.call_method("dequantize"),
        ]
        # 调用检查方法，验证模型的节点操作序列是否与期望一致
        self.checkGraphModuleNodes(m, expected_node_list=node_list)

    # 定义一个测试函数，用于测试模块名称正则表达式的量化配置功能
    def test_qconfig_module_name_regex(self):
        # 定义一个包含两个卷积层的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            # 实现神经网络前向传播方法
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        # 创建一个 M 类的实例并设置为评估模式
        m = M().eval()
        # 定义量化配置字典，使用模块名称正则表达式对模型中的卷积层进行量化
        qconfig_dict = {"module_name_regex": [("conv*", default_qconfig)]}
        # 生成一个随机张量作为输入数据
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 准备量化的函数，根据给定的量化配置对模型 m 进行准备
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 对准备好的模型进行前向传播
        m(*example_inputs)
        # 将准备好的模型转换为量化模型
        m = convert_fx(m)
        # 对转换后的量化模型进行前向传播
        m(*example_inputs)
        # 定义期望的节点操作序列，表示预期的量化节点操作顺序
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),  # 第一个卷积层被量化
            ns.call_module(nnq.Conv2d),  # 第二个卷积层未被量化
            ns.call_method("dequantize"),
        ]
        # 调用检查方法，验证模型的节点操作序列是否与期望一致
        self.checkGraphModuleNodes(m, expected_node_list=node_list)
    # 定义测试方法，用于验证量化配置的优先级
    def test_qconfig_precedence(self):
        # 遍历所有支持的设备类型
        for device in get_supported_device_types():
            # 定义一个继承自 torch.nn.Module 的子类 M
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 定义神经网络结构，包括线性层和卷积层
                    self.linear = nn.Linear(1, 1)
                    self.conv = nn.Conv2d(1, 1, 1)
                    self.module_conv1 = nn.Conv2d(1, 1, 1)
                    self.module_conv2 = nn.Conv2d(1, 1, 1)

                # 前向传播函数
                def forward(self, x):
                    # 应用全局量化配置
                    x = self.linear(x)
                    # 应用对象类型的动态量化配置
                    x = self.conv(x)
                    # 应用模块名正则表达式匹配的动态量化配置
                    x = self.module_conv1(x)
                    # 应用指定模块名的量化训练配置
                    x = self.module_conv2(x)
                    return x

            # 创建 M 类的实例，并将其移动到指定设备并设置为评估模式
            m = M().to(device).eval()

            # 定义全局量化配置、对象类型配置、模块名正则表达式配置和指定模块名配置
            global_qconfig = default_qconfig
            object_type_qconfig = default_dynamic_qconfig
            module_name_regex_qconfig = float16_dynamic_qconfig
            module_name_qconfig = default_qat_qconfig

            # 定义量化配置字典，指定了各种条件下的量化配置
            qconfig_dict = {
                "": global_qconfig,
                "object_type": [(nn.Conv2d, object_type_qconfig)],
                "module_name_regex": [("module_conv*", module_name_regex_qconfig)],
                "module_name": [("module_conv2", module_name_qconfig)]
            }

            # 对模型进行量化准备，使用定义好的量化配置字典和示例输入
            m_prep = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1),))

            # 断言各层的量化配置是否符合预期
            self.assertEqual(m_prep.linear.qconfig.activation.p.func, global_qconfig.activation.p.func)
            self.assertEqual(m_prep.linear.qconfig.weight.p.func, global_qconfig.weight.p.func)
            self.assertEqual(m_prep.conv.qconfig.activation.p.func, object_type_qconfig.activation.p.func)
            self.assertEqual(m_prep.conv.qconfig.weight.p.func, object_type_qconfig.weight.p.func)
            self.assertEqual(m_prep.module_conv1.qconfig.activation.p.func, module_name_regex_qconfig.activation.p.func)
            self.assertEqual(m_prep.module_conv1.qconfig.weight.p.func, module_name_regex_qconfig.weight.p.func)
            self.assertEqual(m_prep.module_conv2.qconfig.activation.p.func, module_name_qconfig.activation.p.func)
            self.assertEqual(m_prep.module_conv2.qconfig.weight.p.func, module_name_qconfig.weight.p.func)
    def test_qconfig_dict_with_fused_modules(self):
        # 定义一个包含线性层和ReLU激活的模型类
        class LinearReLUModel(torch.nn.Module):
            def __init__(self, relu):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)  # 创建一个线性层，输入输出维度均为3
                self.relu = relu  # 设置激活函数为传入的relu参数

            def forward(self, x):
                x = self.linear(x)  # 执行线性层操作
                x = self.relu(x)  # 执行ReLU激活操作
                return x

        # 定义一个包含卷积层和ReLU激活的模型类
        class ConvReLUModel(torch.nn.Module):
            def __init__(self, relu):
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 3, 3)  # 创建一个卷积层，输入输出维度均为3，卷积核大小为3
                self.relu = relu  # 设置激活函数为传入的relu参数

            def forward(self, x):
                x = self.conv(x)  # 执行卷积层操作
                x = self.relu(x)  # 执行ReLU激活操作
                return x

        # 定义一个包含卷积层、批标准化和ReLU激活的模型类
        class ConvBnReLUModel(torch.nn.Module):
            def __init__(self, relu):
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 3, 3)  # 创建一个卷积层，输入输出维度均为3，卷积核大小为3
                self.bn = torch.nn.BatchNorm1d(3)  # 创建一个批标准化层，输入维度为3
                self.relu = relu  # 设置激活函数为传入的relu参数

            def forward(self, x):
                x = self.conv(x)  # 执行卷积层操作
                x = self.bn(x)  # 执行批标准化操作
                x = self.relu(x)  # 执行ReLU激活操作
                return x

        # 遍历不同的模型和ReLU激活函数的组合进行测试
        for model in [LinearReLUModel, ConvReLUModel, ConvBnReLUModel]:
            for relu in [torch.nn.ReLU(), torch.nn.functional.relu, torch.relu]:
                m = model(relu).eval()  # 创建模型实例并设为评估模式
                qengine = torch.backends.quantized.engine  # 获取量化引擎
                qconfig_dict = torch.ao.quantization.get_default_qconfig_mapping(qengine)
                # 调用prepare_fx函数以准备模型，传入量化配置字典和示例输入数据
                prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 3, 3, 3),))

    # TODO: move QConfigMapping tests to test/quantization/core
    def test_qconfig_mapping_set_global(self):
        # 获取默认的量化配置
        qconfig = get_default_qconfig()
        qconfig_mapping = QConfigMapping()
        self.assertEqual(qconfig_mapping.global_qconfig, None)  # 断言全局量化配置为None
        qconfig_mapping.set_global(qconfig)  # 设置全局量化配置
        self.assertEqual(qconfig_mapping.global_qconfig, qconfig)  # 断言全局量化配置与设定的配置一致
    # 测试方法：测试设置对象类型的配置映射
    def test_qconfig_mapping_set_object_type(self):
        # 获取默认的配置对象
        qconfig1 = get_default_qconfig()
        qconfig2 = get_default_qconfig()
        qconfig3 = get_default_qconfig()
        # 断言确保三个配置对象不相等
        self.assertNotEqual(qconfig1, qconfig2)
        self.assertNotEqual(qconfig1, qconfig3)
        # 创建一个空的配置映射对象
        qconfig_mapping = QConfigMapping()
        # 断言配置映射对象中的对象类型-配置字典长度为0
        self.assertEqual(len(qconfig_mapping.object_type_qconfigs), 0)
        
        # 插入一些条目到配置映射中
        qconfig_mapping.set_object_type(torch.nn.Linear, qconfig1)
        qconfig_mapping.set_object_type(torch.nn.ReLU, qconfig2)
        # 断言配置映射对象中的对象类型-配置字典长度为2
        self.assertEqual(len(qconfig_mapping.object_type_qconfigs), 2)
        # 断言配置映射对象中 torch.nn.Linear 对应的配置是 qconfig1
        self.assertEqual(qconfig_mapping.object_type_qconfigs[torch.nn.Linear], qconfig1)
        # 断言配置映射对象中 torch.nn.ReLU 对应的配置是 qconfig2
        self.assertEqual(qconfig_mapping.object_type_qconfigs[torch.nn.ReLU], qconfig2)
        
        # 覆盖已有的键
        qconfig_mapping.set_object_type(torch.nn.Linear, qconfig3)
        # 断言配置映射对象中 torch.nn.Linear 对应的配置变为 qconfig3
        self.assertEqual(qconfig_mapping.object_type_qconfigs[torch.nn.Linear], qconfig3)
        # 断言配置映射对象中 torch.nn.ReLU 对应的配置仍然是 qconfig2
        self.assertEqual(qconfig_mapping.object_type_qconfigs[torch.nn.ReLU], qconfig2)
        # 使用辅助函数 _get_object_type_qconfig 验证映射的正确性
        self.assertEqual(_get_object_type_qconfig(qconfig_mapping, torch.nn.Linear, None), qconfig3)
        self.assertEqual(_get_object_type_qconfig(qconfig_mapping, torch.nn.ReLU, None), qconfig2)
        # 验证没有匹配的情况下返回 None
        self.assertEqual(_get_object_type_qconfig(qconfig_mapping, "nomatch", None), None)

    # 测试方法：测试设置模块名正则表达式的配置映射
    def test_qconfig_mapping_set_module_name_regex(self):
        # 获取默认的配置对象
        qconfig1 = get_default_qconfig()
        qconfig2 = get_default_qconfig()
        qconfig3 = get_default_qconfig()
        # 断言确保三个配置对象不相等
        self.assertNotEqual(qconfig1, qconfig2)
        self.assertNotEqual(qconfig1, qconfig3)
        # 创建一个空的配置映射对象
        qconfig_mapping = QConfigMapping()
        # 断言配置映射对象中的模块名正则表达式-配置字典长度为0
        self.assertEqual(len(qconfig_mapping.module_name_regex_qconfigs), 0)
        
        # 插入一些条目到配置映射中
        qconfig_mapping.set_module_name_regex("foo.*bar", qconfig1)
        qconfig_mapping.set_module_name_regex("foo.*", qconfig2)
        # 断言配置映射对象中的模块名正则表达式-配置字典长度为2
        self.assertEqual(len(qconfig_mapping.module_name_regex_qconfigs), 2)
        # 断言配置映射对象中 "foo.*bar" 对应的配置是 qconfig1
        self.assertEqual(qconfig_mapping.module_name_regex_qconfigs["foo.*bar"], qconfig1)
        # 断言配置映射对象中 "foo.*" 对应的配置是 qconfig2
        self.assertEqual(qconfig_mapping.module_name_regex_qconfigs["foo.*"], qconfig2)
        
        # 覆盖已有的键
        qconfig_mapping.set_module_name_regex("foo.*bar", qconfig3)
        # 断言配置映射对象中 "foo.*bar" 对应的配置变为 qconfig3
        self.assertEqual(qconfig_mapping.module_name_regex_qconfigs["foo.*bar"], qconfig3)
        # 断言配置映射对象中 "foo.*" 对应的配置仍然是 qconfig2
        self.assertEqual(qconfig_mapping.module_name_regex_qconfigs["foo.*"], qconfig2)
        # 使用辅助函数 _get_module_name_regex_qconfig 验证映射的正确性
        self.assertEqual(_get_module_name_regex_qconfig(qconfig_mapping, "foo123bar", None), qconfig3)
        self.assertEqual(_get_module_name_regex_qconfig(qconfig_mapping, "foobar", None), qconfig3)
        self.assertEqual(_get_module_name_regex_qconfig(qconfig_mapping, "foobaz", None), qconfig2)
        self.assertEqual(_get_module_name_regex_qconfig(qconfig_mapping, "foo", None), qconfig2)
        # 验证没有匹配的情况下返回 None
        self.assertEqual(_get_module_name_regex_qconfig(qconfig_mapping, "nomatch", None), None)
    # 定义测试方法，验证设置模块名称的 QConfig 映射
    def test_qconfig_mapping_set_module_name(self):
        # 获取默认的 QConfig 实例
        qconfig1 = get_default_qconfig()
        qconfig2 = get_default_qconfig()
        qconfig3 = get_default_qconfig()
        
        # 确保每个 QConfig 实例不同
        self.assertNotEqual(qconfig1, qconfig2)
        self.assertNotEqual(qconfig1, qconfig3)
        
        # 创建 QConfigMapping 实例，验证起始时模块名到 QConfig 的映射为空
        qconfig_mapping = QConfigMapping()
        self.assertEqual(len(qconfig_mapping.module_name_qconfigs), 0)
        
        # 插入一些条目并验证插入后的映射正确
        qconfig_mapping.set_module_name("mod1", qconfig1)
        qconfig_mapping.set_module_name("mod2", qconfig2)
        self.assertEqual(len(qconfig_mapping.module_name_qconfigs), 2)
        self.assertEqual(qconfig_mapping.module_name_qconfigs["mod1"], qconfig1)
        self.assertEqual(qconfig_mapping.module_name_qconfigs["mod2"], qconfig2)
        
        # 覆盖现有键并验证覆盖后的映射正确
        qconfig_mapping.set_module_name("mod1", qconfig3)
        self.assertEqual(qconfig_mapping.module_name_qconfigs["mod1"], qconfig3)
        self.assertEqual(qconfig_mapping.module_name_qconfigs["mod2"], qconfig2)
        
        # 验证内部函数调用获取正确的 QConfig
        self.assertEqual(_get_module_name_qconfig(qconfig_mapping, "mod1", None), qconfig3)
        self.assertEqual(_get_module_name_qconfig(qconfig_mapping, "mod2", None), qconfig2)
        
        # 验证对于未匹配的模块名，获取的 QConfig 为 None
        self.assertEqual(_get_module_name_qconfig(qconfig_mapping, "nomatch", None), None)

    # 用于 QConfigMapping 测试的内部方法，返回一个测试用的 qconfig_dict
    def _get_qconfig_dict_for_qconfig_mapping_test(self, global_qconfig, qconfig1, qconfig2):
        """
        Return a dummy qconfig_dict to test QConfigMapping's to_dict and from_dict methods.
        """
        return {
            _GLOBAL_DICT_KEY: global_qconfig,
            _OBJECT_TYPE_DICT_KEY: [
                (torch.nn.Linear, qconfig1),
                (torch.nn.ReLU, qconfig2),
            ],
            _MODULE_NAME_REGEX_DICT_KEY: [
                ("foo.*bar", qconfig1),
                ("foo.*", qconfig2),
            ],
            _MODULE_NAME_DICT_KEY: [
                ("bazbaz", qconfig1),
                ("borbor", qconfig2),
            ],
            _MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY: [
                ("bazbaz", torch.nn.Linear, 0, qconfig1),
                ("foofoo", torch.nn.ReLU, 1, qconfig2),
            ],
        }

        # 验证在使用未知键时抛出 ValueError 异常，并检查异常信息
        with self.assertRaises(ValueError) as context:
            m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 3, 3, 3),))  # noqa: F821
        self.assertTrue(
            'Expected qconfig_dict to have the following keys:' in str(context.exception)
        )
        self.assertTrue('But found \'object_typo\' instead.' in str(context.exception))
    # 测试函数，用于测试从字典创建 QConfigMapping 对象的功能
    def test_qconfig_mapping_from_dict(self):
        # 创建全局 QConfig 对象
        global_qconfig = QConfig(123, "global")
        # 创建两个 QConfig 对象
        qconfig1 = QConfig(1, "one")
        qconfig2 = QConfig(2, "two")
        # 调用辅助函数获取 QConfig 字典
        qconfig_dict = self._get_qconfig_dict_for_qconfig_mapping_test(global_qconfig, qconfig1, qconfig2)
        # 向字典中添加一个未定义的键值对，值为列表，包含元组
        qconfig_dict["undefined_dict_key"] = [(123, qconfig1), (234, qconfig2)]
        # 使用 QConfigMapping 类的静态方法 from_dict 创建 QConfigMapping 对象
        qconfig_mapping = QConfigMapping.from_dict(qconfig_dict)
        # 断言全局 QConfig 属性与 global_qconfig 相等
        self.assertEqual(qconfig_mapping.global_qconfig, global_qconfig)
        # 断言 object_type_qconfigs 属性为按顺序的 OrderedDict
        self.assertEqual(qconfig_mapping.object_type_qconfigs, OrderedDict({
            torch.nn.Linear: qconfig1,
            torch.nn.ReLU: qconfig2,
        }))
        # 断言 module_name_regex_qconfigs 属性为按顺序的 OrderedDict
        self.assertEqual(qconfig_mapping.module_name_regex_qconfigs, OrderedDict({
            "foo.*bar": qconfig1,
            "foo.*": qconfig2,
        }))
        # 断言 module_name_qconfigs 属性为按顺序的 OrderedDict
        self.assertEqual(qconfig_mapping.module_name_qconfigs, OrderedDict({
            "bazbaz": qconfig1,
            "borbor": qconfig2,
        }))
        # 断言 module_name_object_type_order_qconfigs 属性为按顺序的 OrderedDict
        self.assertEqual(qconfig_mapping.module_name_object_type_order_qconfigs, OrderedDict({
            ("bazbaz", torch.nn.Linear, 0): qconfig1,
            ("foofoo", torch.nn.ReLU, 1): qconfig2,
        }))

    # 测试函数，用于测试将 QConfigMapping 对象转换为字典的功能
    def test_qconfig_mapping_to_dict(self):
        # 创建全局 QConfig 对象
        global_qconfig = QConfig(123, "global")
        # 创建两个 QConfig 对象
        qconfig1 = QConfig(1, "one")
        qconfig2 = QConfig(2, "two")
        # 创建 QConfigMapping 对象，并设置其属性
        qconfig_mapping = QConfigMapping().set_global(global_qconfig) \
            .set_object_type(torch.nn.Linear, qconfig1) \
            .set_object_type(torch.nn.ReLU, qconfig2) \
            .set_module_name_regex("foo.*bar", qconfig1) \
            .set_module_name_regex("foo.*", qconfig2) \
            .set_module_name("bazbaz", qconfig1) \
            .set_module_name("borbor", qconfig2) \
            .set_module_name_object_type_order("bazbaz", torch.nn.Linear, 0, qconfig1) \
            .set_module_name_object_type_order("foofoo", torch.nn.ReLU, 1, qconfig2)
        # 调用辅助函数获取 QConfig 字典
        qconfig_dict = self._get_qconfig_dict_for_qconfig_mapping_test(global_qconfig, qconfig1, qconfig2)
        # 断言 QConfigMapping 对象转换为的字典与预期的 qconfig_dict 相等
        self.assertEqual(qconfig_mapping.to_dict(), qconfig_dict)

    # 测试函数，用于测试 QConfigMapping 对象的字符串表示形式是否为字符串类型
    def test_qconfig_mapping_repr(self):
        # 断言默认 QConfigMapping 对象的字符串表示形式为字符串类型
        self.assertTrue(isinstance(get_default_qconfig_mapping().__repr__(), str))
    # 定义一个测试方法，用于测试默认的量化配置映射覆盖全局设置
    def test_default_qconfig_mapping_override_global(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            # 定义模型的前向传播方法
            def forward(self, x):
                return self.conv(x)

        # 创建类 M 的实例并设置为评估模式
        m = M().eval()
        # 定义一个自定义的量化配置对象 my_qconfig
        my_qconfig = QConfig(activation=MinMaxObserver, weight=default_weight_observer)
        # 调用函数获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping()
        # 记录原全局量化配置
        old_global_qconfig = qconfig_mapping.global_qconfig
        # 设置全局量化配置为自定义的 my_qconfig
        qconfig_mapping.set_global(my_qconfig)
        # 准备一个示例输入
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 使用 prepare_fx 函数准备模型 m，传入量化配置映射和示例输入
        m = prepare_fx(m, qconfig_mapping, example_inputs)
        # 断言旧的全局激活函数量化观察器为 HistogramObserver 类型
        self.assertTrue(isinstance(old_global_qconfig.activation(), HistogramObserver))
        # 断言新设置的 my_qconfig 的激活函数量化观察器为 MinMaxObserver 类型
        self.assertTrue(isinstance(my_qconfig.activation(), MinMaxObserver))
        # 断言模型 m 具有 "activation_post_process_0" 属性
        self.assertTrue(hasattr(m, "activation_post_process_0"))
        # 断言模型 m 具有 "activation_post_process_1" 属性
        self.assertTrue(hasattr(m, "activation_post_process_1"))
        # 断言模型 m 的 "activation_post_process_0" 是 MinMaxObserver 类型
        self.assertTrue(isinstance(m.activation_post_process_0, MinMaxObserver))
        # 断言模型 m 的 "activation_post_process_1" 是 MinMaxObserver 类型
        self.assertTrue(isinstance(m.activation_post_process_1, MinMaxObserver))

    # 用于 PrepareCustomConfig 测试的虚拟类定义

    class _DummyStandaloneModule:
        pass

    class _DummyFloatModule:
        pass

    class _DummyObservedModule:
        pass

    class _DummyQuantizedModule:
        pass

    class _DummyNonTraceableModule1:
        pass

    class _DummyNonTraceableModule2:
        pass

    # 测试设置独立模块名称的自定义配置准备方法
    def test_prepare_custom_config_set_standalone_module_name(self):
        # 创建一个空的量化配置映射对象
        qconfig_mapping = QConfigMapping()
        # 准备一个示例输入
        example_inputs = (torch.randn(3),)
        # 创建一个 PrepareCustomConfig 对象的实例 child_prepare_custom_config
        child_prepare_custom_config = PrepareCustomConfig()
        # 创建一个名为 "my_backend" 的后端配置对象 backend_config
        backend_config = BackendConfig("my_backend")
        # 创建一个独立模块配置条目对象 config_entry
        config_entry = StandaloneModuleConfigEntry(
            qconfig_mapping, example_inputs, child_prepare_custom_config, backend_config)
        # 创建一个 PrepareCustomConfig 对象的实例 prepare_custom_config
        prepare_custom_config = PrepareCustomConfig()
        # 断言 prepare_custom_config 的独立模块名称列表长度为 0
        self.assertEqual(len(prepare_custom_config.standalone_module_names), 0)
        # 设置独立模块名称为 "module1"，并添加对应的配置条目
        prepare_custom_config.set_standalone_module_name(
            "module1", qconfig_mapping, example_inputs, child_prepare_custom_config, backend_config)
        # 断言 prepare_custom_config 的独立模块名称列表包含 "module1"
        self.assertEqual(list(prepare_custom_config.standalone_module_names.keys()), ["module1"])
        # 断言 prepare_custom_config 中 "module1" 对应的配置条目与 config_entry 相同
        self.assertEqual(prepare_custom_config.standalone_module_names["module1"], config_entry)
    # 测试准备自定义配置的独立模块类
    def test_prepare_custom_config_set_standalone_module_class(self):
        # 创建 QConfigMapping 实例
        qconfig_mapping = QConfigMapping()
        # 创建示例输入
        example_inputs = (torch.randn(3),)
        # 创建子准备自定义配置对象
        child_prepare_custom_config = PrepareCustomConfig()
        # 创建后端配置对象
        backend_config = BackendConfig("my_backend")
        # 创建独立模块配置条目对象
        config_entry = StandaloneModuleConfigEntry(
            qconfig_mapping, example_inputs, child_prepare_custom_config, backend_config)
        # 创建准备自定义配置对象
        prepare_custom_config = PrepareCustomConfig()
        # 断言独立模块类列表长度为 0
        self.assertEqual(len(prepare_custom_config.standalone_module_classes), 0)
        # 设置独立模块类为 DummyStandaloneModule，并关联相关配置信息
        prepare_custom_config.set_standalone_module_class(
            self._DummyStandaloneModule, qconfig_mapping, example_inputs, child_prepare_custom_config, backend_config)
        # 断言独立模块类列表长度为 1
        self.assertEqual(len(prepare_custom_config.standalone_module_classes), 1)
        # 断言 DummyStandaloneModule 存在于独立模块类列表中
        self.assertTrue(self._DummyStandaloneModule in prepare_custom_config.standalone_module_classes)
        # 断言独立模块类的配置条目与预期相等
        self.assertEqual(prepare_custom_config.standalone_module_classes[self._DummyStandaloneModule], config_entry)

    # 测试准备自定义配置的浮点数到观察映射
    def test_prepare_custom_config_set_float_to_observed_mapping(self):
        # 创建准备自定义配置对象
        prepare_custom_config = PrepareCustomConfig()
        # 断言浮点数到观察映射的长度为 0
        self.assertEqual(len(prepare_custom_config.float_to_observed_mapping), 0)
        # 设置浮点数到观察映射，关联 DummyFloatModule 和 DummyObservedModule，使用静态量化类型
        prepare_custom_config.set_float_to_observed_mapping(self._DummyFloatModule, self._DummyObservedModule, QuantType.STATIC)
        # 断言浮点数到观察映射的长度为 1
        self.assertEqual(len(prepare_custom_config.float_to_observed_mapping), 1)
        # 断言浮点数到观察映射的键为 [QuantType.STATIC]
        self.assertEqual(list(prepare_custom_config.float_to_observed_mapping.keys()), [QuantType.STATIC])
        # 断言静态量化类型映射的长度为 1
        self.assertEqual(len(prepare_custom_config.float_to_observed_mapping[QuantType.STATIC]), 1)
        # 断言 DummyFloatModule 存在于静态量化类型映射中
        self.assertTrue(self._DummyFloatModule in prepare_custom_config.float_to_observed_mapping[QuantType.STATIC])
        # 断言 DummyFloatModule 对应的观察模块为 DummyObservedModule
        self.assertEqual(prepare_custom_config.float_to_observed_mapping[QuantType.STATIC][self._DummyFloatModule],
                         self._DummyObservedModule)

    # 测试准备自定义配置设置不可追踪模块名列表
    def test_prepare_custom_config_set_non_traceable_module_names(self):
        # 创建准备自定义配置对象
        prepare_custom_config = PrepareCustomConfig()
        # 断言不可追踪模块名列表长度为 0
        self.assertEqual(len(prepare_custom_config.non_traceable_module_names), 0)
        # 设置不可追踪模块名列表为 ["module1", "module2"]
        prepare_custom_config.set_non_traceable_module_names(["module1", "module2"])
        # 断言不可追踪模块名列表与预期相等
        self.assertEqual(prepare_custom_config.non_traceable_module_names, ["module1", "module2"])

    # 测试准备自定义配置设置不可追踪模块类列表
    def test_prepare_custom_config_set_non_traceable_module_classes(self):
        # 创建准备自定义配置对象
        prepare_custom_config = PrepareCustomConfig()
        # 断言不可追踪模块类列表长度为 0
        self.assertEqual(len(prepare_custom_config.non_traceable_module_classes), 0)
        # 设置不可追踪模块类列表为 [DummyNonTraceableModule1, DummyNonTraceableModule2]
        prepare_custom_config.set_non_traceable_module_classes([self._DummyNonTraceableModule1, self._DummyNonTraceableModule2])
        # 断言不可追踪模块类列表与预期相等
        self.assertEqual(prepare_custom_config.non_traceable_module_classes,
                         [self._DummyNonTraceableModule1, self._DummyNonTraceableModule2])
    # 定义测试方法，验证设置输入量化索引功能
    def test_prepare_custom_config_set_input_quantized_indexes(self):
        # 创建 PrepareCustomConfig 实例
        prepare_custom_config = PrepareCustomConfig()
        # 断言输入量化索引初始长度为0
        self.assertEqual(len(prepare_custom_config.input_quantized_indexes), 0)
        # 调用设置输入量化索引方法
        prepare_custom_config.set_input_quantized_indexes([0, 1])
        # 断言设置后的输入量化索引是否符合预期
        self.assertEqual(prepare_custom_config.input_quantized_indexes, [0, 1])

    # 定义测试方法，验证设置输出量化索引功能
    def test_prepare_custom_config_set_output_quantized_indexes(self):
        # 创建 PrepareCustomConfig 实例
        prepare_custom_config = PrepareCustomConfig()
        # 断言输出量化索引初始长度为0
        self.assertEqual(len(prepare_custom_config.output_quantized_indexes), 0)
        # 调用设置输出量化索引方法
        prepare_custom_config.set_output_quantized_indexes([0, 1])
        # 断言设置后的输出量化索引是否符合预期
        self.assertEqual(prepare_custom_config.output_quantized_indexes, [0, 1])

    # 定义测试方法，验证设置保留属性功能
    def test_prepare_custom_config_set_preserved_attributes(self):
        # 创建 PrepareCustomConfig 实例
        prepare_custom_config = PrepareCustomConfig()
        # 断言保留属性初始长度为0
        self.assertEqual(len(prepare_custom_config.preserved_attributes), 0)
        # 调用设置保留属性方法
        prepare_custom_config.set_preserved_attributes(["attr1", "attr2"])
        # 断言设置后的保留属性是否符合预期
        self.assertEqual(prepare_custom_config.preserved_attributes, ["attr1", "attr2"])

    # 定义私有方法，返回一个用于测试的虚拟 prepare_custom_config_dict
    def _get_dummy_prepare_custom_config_dict(self):
        """
        Return a dummy prepare_custom_config_dict to test PrepareCustomConfig's to_dict and from_dict methods.
        """
        return {
            # 单独模块名称字典键，包含元组列表
            STANDALONE_MODULE_NAME_DICT_KEY: [(
                "module1",                      # 模块名称
                QConfigMapping(),               # QConfig 映射
                (torch.randn(3),),              # 模拟的输入数据
                PrepareCustomConfig(),          # 自定义配置准备对象
                BackendConfig("my_backend"),    # 后端配置对象
            )],
            # 单独模块类别字典键，包含元组列表
            STANDALONE_MODULE_CLASS_DICT_KEY: [(
                self._DummyStandaloneModule,    # 虚拟的独立模块类
                QConfigMapping(),               # QConfig 映射
                (torch.randn(10),),             # 模拟的输入数据
                PrepareCustomConfig(),          # 自定义配置准备对象
                BackendConfig("my_backend"),    # 后端配置对象
            )],
            # 浮点到观察字典键，包含静态映射
            FLOAT_TO_OBSERVED_DICT_KEY: {
                "static": {
                    self._DummyFloatModule: self._DummyObservedModule  # 虚拟的浮点模块和观察模块映射
                },
            },
            # 不可追踪模块名称字典键，包含不可追踪模块列表
            NON_TRACEABLE_MODULE_NAME_DICT_KEY: ["module2", "module3"],
            # 不可追踪模块类别字典键，包含不可追踪模块类别列表
            NON_TRACEABLE_MODULE_CLASS_DICT_KEY: [self._DummyNonTraceableModule1, self._DummyNonTraceableModule2],
            # 输入量化索引字典键，包含索引列表
            INPUT_QUANTIZED_INDEXES_DICT_KEY: [0, 1],
            # 输出量化索引字典键，包含索引列表
            OUTPUT_QUANTIZED_INDEXES_DICT_KEY: [0, 1],
            # 保留属性字典键，包含属性列表
            PRESERVED_ATTRIBUTES_DICT_KEY: ["attr1", "attr2"]
        }
    def test_prepare_custom_config_from_dict(self):
        # 从测试辅助方法获取准备自定义配置的虚拟字典
        prepare_custom_config_dict = self._get_dummy_prepare_custom_config_dict()
        # 解构从准备自定义配置字典中获取的第一个独立模块名称和相关配置
        (sm_name, qm1, ei1, pcc1, bcd1) = prepare_custom_config_dict[STANDALONE_MODULE_NAME_DICT_KEY][0]
        # 解构从准备自定义配置字典中获取的第一个独立模块类和相关配置
        (sm_class, qm2, ei2, pcc2, bcd2) = prepare_custom_config_dict[STANDALONE_MODULE_CLASS_DICT_KEY][0]
        # 创建独立模块配置条目1，使用提取的相关配置
        sm_config_entry1 = StandaloneModuleConfigEntry(qm1, ei1, pcc1, bcd1)
        # 创建独立模块配置条目2，使用提取的相关配置
        sm_config_entry2 = StandaloneModuleConfigEntry(qm2, ei2, pcc2, bcd2)
        # 从字典形式的准备自定义配置创建准备自定义配置对象
        prepare_custom_config = PrepareCustomConfig.from_dict(prepare_custom_config_dict)

        # 验证独立模块名称集合的长度为1
        self.assertEqual(len(prepare_custom_config.standalone_module_names), 1)
        # 验证指定的独立模块名称在准备自定义配置对象的独立模块名称集合中
        self.assertTrue(sm_name in prepare_custom_config.standalone_module_names)
        # 验证指定的独立模块名称对应的配置条目与之前创建的sm_config_entry1相等
        self.assertEqual(prepare_custom_config.standalone_module_names[sm_name], sm_config_entry1)
        # 验证独立模块类集合的长度为1
        self.assertEqual(len(prepare_custom_config.standalone_module_classes), 1)
        # 验证指定的独立模块类在准备自定义配置对象的独立模块类集合中
        self.assertTrue(sm_class in prepare_custom_config.standalone_module_classes)
        # 验证指定的独立模块类对应的配置条目与之前创建的sm_config_entry2相等
        self.assertEqual(prepare_custom_config.standalone_module_classes[sm_class], sm_config_entry2)

        # 验证浮点数到观察映射的长度为1
        self.assertEqual(len(prepare_custom_config.float_to_observed_mapping), 1)
        # 验证浮点数到观察映射的键列表包含静态量化类型
        self.assertEqual(list(prepare_custom_config.float_to_observed_mapping.keys()), [QuantType.STATIC])
        # 验证静态量化类型下的映射列表长度为1
        self.assertEqual(len(prepare_custom_config.float_to_observed_mapping[QuantType.STATIC]), 1)
        # 验证_DummyFloatModule存在于静态量化类型的观察映射中
        self.assertTrue(self._DummyFloatModule in prepare_custom_config.float_to_observed_mapping[QuantType.STATIC])
        # 验证_DummyFloatModule对应的观察模块为_DummyObservedModule
        self.assertEqual(prepare_custom_config.float_to_observed_mapping[QuantType.STATIC][self._DummyFloatModule],
                         self._DummyObservedModule)

        # 验证非可追踪模块名称列表与预期的["module2", "module3"]相等
        self.assertEqual(prepare_custom_config.non_traceable_module_names, ["module2", "module3"])
        # 验证非可追踪模块类列表与预期的[self._DummyNonTraceableModule1, self._DummyNonTraceableModule2]相等
        self.assertEqual(prepare_custom_config.non_traceable_module_classes,
                         [self._DummyNonTraceableModule1, self._DummyNonTraceableModule2])
        # 验证输入量化索引列表与预期的[0, 1]相等
        self.assertEqual(prepare_custom_config.input_quantized_indexes, [0, 1])
        # 验证输出量化索引列表与预期的[0, 1]相等
        self.assertEqual(prepare_custom_config.output_quantized_indexes, [0, 1])
        # 验证保留属性列表与预期的["attr1", "attr2"]相等
        self.assertEqual(prepare_custom_config.preserved_attributes, ["attr1", "attr2"])
    # 测试准备自定义配置转换为字典
    def test_prepare_custom_config_to_dict(self):
        # 获取虚拟的准备自定义配置字典
        prepare_custom_config_dict = self._get_dummy_prepare_custom_config_dict()
        # 解构赋值获取元组中的值
        (sm_name, qm1, ei1, pcc1, bcd1) = prepare_custom_config_dict[STANDALONE_MODULE_NAME_DICT_KEY][0]
        (sm_class, qm2, ei2, pcc2, bcd2) = prepare_custom_config_dict[STANDALONE_MODULE_CLASS_DICT_KEY][0]
        # 创建准备自定义配置对象，并设置属性
        prepare_custom_config = PrepareCustomConfig() \
            .set_standalone_module_name(sm_name, qm1, ei1, pcc1, bcd1) \
            .set_standalone_module_class(sm_class, qm2, ei2, pcc2, bcd2) \
            .set_float_to_observed_mapping(self._DummyFloatModule, self._DummyObservedModule) \
            .set_non_traceable_module_names(["module2", "module3"]) \
            .set_non_traceable_module_classes([self._DummyNonTraceableModule1, self._DummyNonTraceableModule2]) \
            .set_input_quantized_indexes([0, 1]) \
            .set_output_quantized_indexes([0, 1]) \
            .set_preserved_attributes(["attr1", "attr2"])
        # 准备自定义配置对象转换为字典，同时将内部 QConfigMappings 和 PrepareCustomConfigs 转换为字典
        prepare_custom_config_dict[STANDALONE_MODULE_NAME_DICT_KEY][0] = (sm_name, qm1.to_dict(), ei1, pcc1.to_dict(), bcd1)
        prepare_custom_config_dict[STANDALONE_MODULE_CLASS_DICT_KEY][0] = (sm_class, qm2.to_dict(), ei2, pcc2.to_dict(), bcd2)
        # 断言准备自定义配置对象转换为的字典与预期字典相等
        self.assertEqual(prepare_custom_config.to_dict(), prepare_custom_config_dict)

    # 测试转换自定义配置设置观察到的到量化的映射
    def test_convert_custom_config_set_observed_to_quantized_mapping(self):
        convert_custom_config = ConvertCustomConfig()
        # 断言观察到的到量化的映射长度为 0
        self.assertEqual(len(convert_custom_config.observed_to_quantized_mapping), 0)
        # 设置观察到的到量化的映射
        convert_custom_config.set_observed_to_quantized_mapping(
            self._DummyObservedModule, self._DummyQuantizedModule, QuantType.STATIC)
        # 断言观察到的到量化的映射长度为 1
        self.assertEqual(len(convert_custom_config.observed_to_quantized_mapping), 1)
        # 断言观察到的到量化的映射的键列表为 [QuantType.STATIC]
        self.assertEqual(list(convert_custom_config.observed_to_quantized_mapping.keys()), [QuantType.STATIC])
        # 断言观察到的到量化的映射中包含 DummyObservedModule
        self.assertTrue(self._DummyObservedModule in convert_custom_config.observed_to_quantized_mapping[QuantType.STATIC])
        # 断言观察到的到量化的映射中 DummyObservedModule 对应 DummyQuantizedModule
        self.assertEqual(convert_custom_config.observed_to_quantized_mapping[QuantType.STATIC][self._DummyObservedModule],
                         self._DummyQuantizedModule)

    # 测试转换自定义配置设置保留的属性
    def test_convert_custom_config_set_preserved_attributes(self):
        convert_custom_config = ConvertCustomConfig()
        # 断言保留的属性列表长度为 0
        self.assertEqual(len(convert_custom_config.preserved_attributes), 0)
        # 设置保留的属性列表
        convert_custom_config.set_preserved_attributes(["attr1", "attr2"])
        # 断言保留的属性列表与预期列表相等
        self.assertEqual(convert_custom_config.preserved_attributes, ["attr1", "attr2"])
    # 返回一个用于测试ConvertCustomConfig的to_dict和from_dict方法的虚拟配置字典
    def _get_dummy_convert_custom_config_dict(self):
        """
        Return a dummy convert_custom_config_dict to test ConvertCustomConfig's to_dict and from_dict methods.
        """
        return {
            # 观察到量化映射的字典键
            OBSERVED_TO_QUANTIZED_DICT_KEY: {
                "static": {
                    # 将_DummyObservedModule映射到_DummyQuantizedModule
                    self._DummyObservedModule: self._DummyQuantizedModule
                },
            },
            # 保留属性的列表键
            PRESERVED_ATTRIBUTES_DICT_KEY: ["attr1", "attr2"]
        }

    # 测试从字典创建ConvertCustomConfig对象的方法
    def test_convert_custom_config_from_dict(self):
        convert_custom_config_dict = self._get_dummy_convert_custom_config_dict()
        convert_custom_config = ConvertCustomConfig.from_dict(convert_custom_config_dict)
        # 断言观察到量化映射的长度为1
        self.assertEqual(len(convert_custom_config.observed_to_quantized_mapping), 1)
        # 断言观察到量化映射的键列表为[QuantType.STATIC]
        self.assertEqual(list(convert_custom_config.observed_to_quantized_mapping.keys()), [QuantType.STATIC])
        # 断言静态量化映射的长度为1
        self.assertEqual(len(convert_custom_config.observed_to_quantized_mapping[QuantType.STATIC]), 1)
        # 断言_DummyObservedModule在静态量化映射中
        self.assertTrue(self._DummyObservedModule in convert_custom_config.observed_to_quantized_mapping[QuantType.STATIC])
        # 断言_DummyObservedModule映射到_DummyQuantizedModule
        self.assertEqual(convert_custom_config.observed_to_quantized_mapping[QuantType.STATIC][self._DummyObservedModule],
                         self._DummyQuantizedModule)
        # 断言保留属性与预期列表相同
        self.assertEqual(convert_custom_config.preserved_attributes, ["attr1", "attr2"])

    # 测试将ConvertCustomConfig对象转换为字典的方法
    def test_convert_custom_config_to_dict(self):
        convert_custom_config = ConvertCustomConfig() \
            .set_observed_to_quantized_mapping(self._DummyObservedModule, self._DummyQuantizedModule) \
            .set_preserved_attributes(["attr1", "attr2"])
        # 断言ConvertCustomConfig对象转换为字典与预期字典相同
        self.assertEqual(convert_custom_config.to_dict(), self._get_dummy_convert_custom_config_dict())

    # 测试设置保留属性的方法
    def test_fuse_custom_config_set_preserved_attributes(self):
        fuse_custom_config = FuseCustomConfig()
        # 断言初始时保留属性列表为空
        self.assertEqual(len(fuse_custom_config.preserved_attributes), 0)
        # 设置保留属性为["attr1", "attr2"]
        fuse_custom_config.set_preserved_attributes(["attr1", "attr2"])
        # 断言设置后保留属性与预期列表相同
        self.assertEqual(fuse_custom_config.preserved_attributes, ["attr1", "attr2"])

    # 测试从字典创建FuseCustomConfig对象的方法
    def test_fuse_custom_config_from_dict(self):
        fuse_custom_config_dict = {PRESERVED_ATTRIBUTES_DICT_KEY: ["attr1", "attr2"]}
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config_dict)
        # 断言从字典创建的FuseCustomConfig对象的保留属性与预期列表相同
        self.assertEqual(fuse_custom_config.preserved_attributes, ["attr1", "attr2"])

    # 测试将FuseCustomConfig对象转换为字典的方法
    def test_fuse_custom_config_to_dict(self):
        fuse_custom_config_dict = {PRESERVED_ATTRIBUTES_DICT_KEY: ["attr1", "attr2"]}
        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(["attr1", "attr2"])
        # 断言FuseCustomConfig对象转换为字典与预期字典相同
        self.assertEqual(fuse_custom_config.to_dict(), fuse_custom_config_dict)
    def test_remove_qconfig(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool = torch.nn.AvgPool2d(1)

            def forward(self, x):
                return self.avg_pool(x)

        # 创建一个 M 类的实例并设置为评估模式
        m = M().eval()
        # 定义量化配置字典，包含一个默认的量化配置
        qconfig_dict = {'': default_qconfig}
        # 准备模型 m 进行量化，提供示例输入 example_inputs
        m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1, 1, 1),))
        # 使用示例输入 example_inputs 对模型 m 进行前向传播
        m(*example_inputs)
        # 将准备好的量化模型 m 转换为量化后的模型
        m = convert_fx(m)
        # 再次使用示例输入 example_inputs 对量化后的模型 m 进行前向传播
        m(*example_inputs)
        # 遍历模型 m 中的所有模块和它们的名称
        for name, module in m.named_modules():
            # 断言每个模块 module 没有属性 'qconfig'，如果有则抛出异常，显示相应模块名称
            self.assertFalse(hasattr(module, 'qconfig'),
                             'qconfig is not removed for ' + name)

    def test_return_none(self):
        class M(torch.nn.Module):
            def forward(self, x):
                pass

        # 创建一个 M 类的实例并设置为评估模式
        m = M().eval()
        # 定义量化配置字典，包含一个默认的量化配置
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        # 准备模型 m 进行量化，提供示例输入 (torch.randn(1),)
        m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1),))
        # 将准备好的模型 m 转换为量化后的模型
        m = convert_fx(m)

    def test_default_quant_after_none_qconfig(self):
        """ Make sure default quant is inserted properly"""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = x.transpose(1, 2)
                x = self.conv2(x)

        # 创建一个 M 类的实例并设置为评估模式
        m = M().eval()
        # 定义量化配置字典，包含一个默认的量化配置和一个针对 'conv1' 的量化配置
        qconfig_dict = {
            "": default_qconfig,
            "module_name": [
                ("conv1", None)
            ]
        }
        # 准备模型 m 进行量化，提供示例输入 (torch.randn(1, 1, 1, 1),)
        m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1, 1, 1),))
        # 将准备好的模型 m 转换为量化后的模型
        m = convert_fx(m)
    def test_qconfig_for_call_method(self):
        # 定义一个名为 test_qconfig_for_call_method 的测试方法
        class Sub(torch.nn.Module):
            # 定义一个名为 Sub 的子类，继承自 torch.nn.Module
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                # 初始化方法，包括一个 1x1 的 Conv2d 层

            def forward(self, x):
                # 前向传播方法，接受输入 x
                x = x.transpose(2, 3)
                # 对输入 x 进行维度转置
                x = self.conv(x)
                # 将转置后的 x 输入到 self.conv 中
                return x.transpose(2, 3)
                # 返回经过维度转置后的输出

        class M(torch.nn.Module):
            # 定义一个名为 M 的子类，继承自 torch.nn.Module
            def __init__(self):
                super().__init__()
                self.sub = Sub()
                # 初始化方法，包括一个 Sub 实例作为子模块 self.sub
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                # 初始化方法，包括一个 1x1 的 Conv2d 层 self.conv1
                self.conv2 = torch.nn.Conv2d(1, 1, 1)
                # 初始化方法，包括一个 1x1 的 Conv2d 层 self.conv2

            def forward(self, x):
                # 前向传播方法，接受输入 x
                x = self.conv1(x)
                # 将输入 x 输入到 self.conv1 中
                x = self.sub(x)
                # 将经过 self.conv1 处理后的 x 输入到 self.sub 中
                x = self.conv2(x)
                # 将经过 self.sub 处理后的 x 输入到 self.conv2 中
                return x.transpose(2, 3)
                # 返回经过维度转置后的输出

        qconfig_dict1 = {"": default_qconfig, "module_name": [("sub", None)]}
        # 定义一个名为 qconfig_dict1 的字典，设定了 quantization 配置
        # sub 模块的 quantization 配置为 None，因此在 self.conv1 的输出需要反量化，
        # self.conv2 的输入需要量化
        # transpose 后的 dequantize 应该在 transpose 之后，因为它使用了 default_qconfig
        # Sub 模块实例中的节点不被量化
        node_list1 = [
            ns.call_function(torch.quantize_per_tensor),
            # 使用 torch.quantize_per_tensor 函数调用
            ns.call_module(nnq.Conv2d),
            # 使用 nnq.Conv2d 模块调用
            ns.call_method("dequantize"),
            # 调用 "dequantize" 方法
            ns.call_method("transpose"),
            # 调用 "transpose" 方法
            ns.call_module(nn.Conv2d),
            # 使用 nn.Conv2d 模块调用
            ns.call_method("transpose"),
            # 调用 "transpose" 方法
            ns.call_function(torch.quantize_per_tensor),
            # 使用 torch.quantize_per_tensor 函数调用
            ns.call_module(nnq.Conv2d),
            # 使用 nnq.Conv2d 模块调用
            ns.call_method("transpose"),
            # 调用 "transpose" 方法
            ns.call_method("dequantize")
            # 调用 "dequantize" 方法
        ]

        qconfig_dict2 = {"": None, "module_name": [("sub", default_qconfig)]}
        # 定义一个名为 qconfig_dict2 的字典，设定了 quantization 配置
        # 只有 Sub 模块实例中的节点被量化
        # 第一个 transpose 不被量化，因为输入没有被量化
        node_list2 = [
            ns.call_module(nn.Conv2d),
            # 使用 nn.Conv2d 模块调用
            ns.call_function(torch.quantize_per_tensor),
            # 使用 torch.quantize_per_tensor 函数调用
            ns.call_method("transpose"),
            # 调用 "transpose" 方法
            ns.call_module(nnq.Conv2d),
            # 使用 nnq.Conv2d 模块调用
            ns.call_method("transpose"),
            # 调用 "transpose" 方法
            ns.call_method("dequantize"),
            # 调用 "dequantize" 方法
            ns.call_module(nn.Conv2d),
            # 使用 nn.Conv2d 模块调用
            ns.call_method("transpose"),
            # 调用 "transpose" 方法
        ]

        for qconfig_dict, node_list in [
                (qconfig_dict1, node_list1),
                (qconfig_dict2, node_list2)
        ]:
            example_inputs = (torch.randn(2, 1, 3, 3),)
            # 定义示例输入，形状为 (2, 1, 3, 3)
            m = M().eval()
            # 创建 M 类的实例 m，并设置为 evaluation 模式
            m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
            # 使用 prepare_fx 函数对模型 m 进行量化准备
            m(torch.randn(2, 1, 3, 3))
            # 对模型 m 进行前向传播
            m = convert_fx(m)
            # 将模型 m 转换为 FX 模式
            self.checkGraphModuleNodes(m, expected_node_list=node_list)
            # 调用 self.checkGraphModuleNodes 方法，检查模型 m 的图结构节点是否符合预期 node_list
            # 确保模型能够运行
            m(*example_inputs)
            # 对示例输入进行前向传播
    # 定义一个测试函数，用于测试量化配置对调用函数的影响
    def test_qconfig_for_call_func(self):
        # 定义一个简单的线性模块
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)  # 初始化权重矩阵为全1
                self.b = torch.zeros(5)     # 初始化偏置向量为全0

            def forward(self, x):
                # 使用 torch.nn.functional.linear 函数进行前向传播
                return torch.nn.functional.linear(x, self.w, self.b)

        # 定义一个复合模块 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # mods1 包含两个 Linear 实例
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )
                # mods2 包含一个 Linear 实例
                self.mods2 = Linear()

            def forward(self, x):
                x = self.mods1(x)  # 对输入 x 进行 mods1 的前向传播
                x = self.mods2(x)  # 对上一步结果进行 mods2 的前向传播
                return x

        # 创建模型 M 的实例并设为评估模式
        model = M().eval()
        # 准备一个示例输入
        example_inputs = (torch.rand(5, 5),)
        # 定义量化配置字典
        qconfig_dict = {"": default_qconfig, "module_name": [("mods2", None)]}
        # 对模型进行准备，使用函数 prepare_fx
        m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
        # 对准备后的模型执行前向传播
        m(*example_inputs)

        # 将准备后的模型转换为量化版本
        m = convert_fx(m)
        # 定义期望的节点列表
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.linear),
            ns.call_function(torch.ops.quantized.linear),
            ns.call_method('dequantize'),
            ns.call_function(torch.nn.functional.linear)
        ]
        # 检查模型的图模块节点是否符合预期
        self.checkGraphModuleNodes(m, expected_node_list=node_list)
        # 对转换后的模型执行前向传播
        m(torch.rand(5, 5))

    # 定义一个测试函数，用于测试保留属性功能
    def test_preserve_attributes(self):
        # 定义一个简单的模块 M，包含一个卷积层
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 执行卷积操作
                return self.conv(x)

        # 创建模块 M 的实例
        m = M()
        # 设为评估模式
        m.eval()
        # 添加一个自定义的保留属性
        m.preserved_attr = 3
        # 定义准备阶段的自定义配置字典，指定要保留的属性
        prepare_custom_config_dict = {
            "preserved_attributes": ["preserved_attr"]
        }
        # 准备示例输入
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 对模型进行准备，使用函数 prepare_fx，传入量化配置和自定义配置
        m = prepare_fx(
            m,
            {"": default_qconfig},
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict)

        # 定义断言函数，用于检查保留属性是否被正确设置
        def assertAttrPreserved(m):
            self.assertTrue(hasattr(m, "preserved_attr"))  # 断言模型包含属性 "preserved_attr"
            self.assertEqual(m.preserved_attr, 3)          # 断言属性值为 3

        # 执行断言函数，检查属性是否被正确保留
        assertAttrPreserved(m)

        # 定义转换阶段的自定义配置字典，指定要保留的属性
        convert_custom_config_dict = {
            "preserved_attributes": ["preserved_attr"]
        }
        # 对模型进行转换为量化版本，传入转换配置
        m = convert_fx(m, convert_custom_config=convert_custom_config_dict)
        # 再次执行断言函数，检查属性是否仍然被正确保留
        assertAttrPreserved(m)

    # 标记条件测试依赖于是否支持 FBGEMM 的装饰器
    @skipIfNoFBGEMM
    # 定义一个测试函数，用于测试量化训练和脚本化功能
    def test_qat_and_script(self):
        # 创建一个带子模块的线性模型，并将其设置为训练模式
        model = LinearModelWithSubmodule().train()
        # 获取当前量化后端引擎
        qengine = torch.backends.quantized.engine
        # 创建一个包含默认量化训练配置的字典
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig(qengine)}
        # 生成一个形状为(5, 5)的随机张量作为示例输入
        x = torch.randn(5, 5)
        example_inputs = (x,)
        # 准备模型以进行量化训练和脚本化
        model = prepare_qat_fx(model, qconfig_dict, example_inputs=example_inputs)

        # 确保脚本化功能正常运行
        scripted = torch.jit.script(model)
        # 运行一轮以确保模型能够执行
        scripted(x)
        # 检查脚本化后的模型图中"FakeQuantize"属性的数量为4
        FileCheck().check_count('FakeQuantize = prim::GetAttr[name="', 4, exactly=True) \
                   .run(scripted.graph)

        # 禁用 fake_quant 和 observer
        for epoch in range(3):
            if epoch == 1:
                scripted.apply(torch.ao.quantization.disable_observer)
            if epoch == 2:
                scripted.apply(torch.ao.quantization.disable_fake_quant)

        # 确保 fake_quant 和 observer 已被禁用
        matches = ['.fake_quant_enabled', '.observer_enabled']
        for key, v in scripted.state_dict().items():
            if any(x in key for x in matches):
                self.assertEqual(v, torch.tensor([0], dtype=torch.int64))

        # 再次启用 fake_quant 和 observer
        scripted.apply(torch.ao.quantization.enable_fake_quant)
        scripted.apply(torch.ao.quantization.enable_observer)
        for key, v in scripted.state_dict().items():
            if any(x in key for x in matches):
                self.assertEqual(v, torch.tensor([1], dtype=torch.int64))

    # 如果没有支持的FBGEMM，跳过该测试
    @skipIfNoFBGEMM
    # 定义一个测试函数，用于测试保存观察者状态字典功能
    def test_save_observer_state_dict(self):
        # 创建一个带子模块的线性模型，并将其设置为评估模式
        orig = LinearModelWithSubmodule().eval()
        model = orig
        # 创建一个包含默认FBGEMM配置的量化配置字典
        qconfig_dict = {'': torch.ao.quantization.get_default_qconfig('fbgemm')}
        # 生成一个形状为(5, 5)的随机张量作为示例输入
        x = torch.randn(5, 5)
        # 准备模型以进行量化
        model = prepare_fx(model, qconfig_dict, example_inputs=(x,))

        # 通过输入运行模型
        model(x)
        # 保存模型的状态字典
        obs_dict = torch.ao.quantization.get_observer_state_dict(model)

        # 将模型转换为量化表示
        quant = convert_fx(model)

        b = io.BytesIO()
        # 将状态字典保存到字节流中
        torch.save(obs_dict, b)

        # 加载状态字典到新模型中
        for weights_only in [True, False]:
            b.seek(0)
            model_2 = orig
            model_2 = prepare_fx(model_2, qconfig_dict, example_inputs=(x,))

            loaded_dict = torch.load(b, weights_only=weights_only)
            # 加载观察者状态字典到模型中
            torch.ao.quantization.load_observer_state_dict(model_2, loaded_dict)

            quant_2 = convert_fx(model_2)

            # 验证加载的状态字典产生相同的结果
            self.assertEqual(quant(x), quant_2(x))

    # 如果没有支持的FBGEMM，跳过该测试
    @skipIfNoFBGEMM
    # 如果没有支持的FBGEMM，跳过该测试
        def test_custom_module_class_input_has_multiple_users(self):
            """ Tests that the flow still works when the input of custom module
            has multiple users
            """
            # 定义一个自定义的 PyTorch 模块
            class CustomModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建一个线性层，输入和输出都是3维
                    self.linear = torch.nn.Linear(3, 3)

                def forward(self, x):
                    # 在前向传播中应用线性层
                    return self.linear(x)

            # 观察过的自定义模块，继承自 `torch.nn.Module`
            class ObservedCustomModule(torch.nn.Module):
                def __init__(self, linear):
                    super().__init__()
                    # 使用给定的线性层初始化
                    self.linear = linear

                def forward(self, x):
                    # 在前向传播中应用线性层
                    return self.linear(x)

                @classmethod
                def from_float(cls, float_module):
                    # 确保浮点模块具有 'qconfig' 属性
                    assert hasattr(float_module, 'qconfig')
                    # 创建观察过的模块实例，并复制 'qconfig' 属性
                    observed = cls(float_module.linear)
                    observed.qconfig = float_module.qconfig
                    return observed

            # 静态量化的自定义模块，继承自 `torch.nn.Module`
            class StaticQuantCustomModule(torch.nn.Module):
                def __init__(self, linear):
                    super().__init__()
                    # 使用给定的线性层初始化
                    self.linear = linear

                def forward(self, x):
                    # 在前向传播中应用线性层
                    return self.linear(x)

                @classmethod
                def from_observed(cls, observed_module):
                    # 确保观察过的模块具有 'qconfig' 和 'activation_post_process' 属性
                    assert hasattr(observed_module, 'qconfig')
                    assert hasattr(observed_module, 'activation_post_process')
                    # 复制激活后处理属性并创建静态量化的模块实例
                    observed_module.linear.activation_post_process = \
                        observed_module.activation_post_process
                    quantized = cls(nnq.Linear.from_float(observed_module.linear))
                    return quantized

            # 主模块 M，继承自 `torch.nn.Module`
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建一个线性层，输入和输出都是3维
                    self.linear = torch.nn.Linear(3, 3)
                    # 创建一个自定义模块实例
                    self.custom = CustomModule()

                def forward(self, x0):
                    # 在前向传播中应用自定义模块和线性层，并返回它们的和
                    x1 = self.custom(x0)
                    x2 = self.linear(x0)
                    return x1 + x2

            # 准备量化配置字典
            prepare_custom_config_dict = {
                "float_to_observed_custom_module_class": {
                    "static": {
                        CustomModule: ObservedCustomModule
                    }
                }
            }
            # 转换量化配置字典
            convert_custom_config_dict = {
                "observed_to_quantized_custom_module_class": {
                    "static": {
                        ObservedCustomModule: StaticQuantCustomModule
                    }
                }
            }

            # 创建并评估模块 M 的实例
            m = M().eval()
            # 创建示例输入
            example_inputs = (torch.randn(3, 3),)
            # 准备模型以便量化
            m = prepare_fx(
                m,
                {"": default_qconfig},
                example_inputs=example_inputs,
                prepare_custom_config=prepare_custom_config_dict)
            # 确保转换有效
            m = convert_fx(
                m,
                convert_custom_config=convert_custom_config_dict)
            # 确保模型能够运行
            m(*example_inputs)
        
        @skipIfNoFBGEMM
    def test_custom_module_class_input_has_duplicate_nodes(self):
        """ Tests that the flow still works when the graph has
        multiple nodes with the same custom module target.
        """
        # 定义一个自定义的PyTorch模块
        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        # 定义一个观察到的自定义模块，从浮点数模块转换而来
        class ObservedCustomModule(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self.linear(x)

            @classmethod
            def from_float(cls, float_module):
                assert hasattr(float_module, 'qconfig')
                observed = cls(float_module.linear)
                observed.qconfig = float_module.qconfig
                return observed

        # 定义一个静态量化的自定义模块，从观察到的模块转换而来
        class StaticQuantCustomModule(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self.linear(x)

            @classmethod
            def from_observed(cls, observed_module):
                assert hasattr(observed_module, 'qconfig')
                assert hasattr(observed_module, 'activation_post_process')
                # 将观察到的模块的激活后处理方法传递给静态量化模块的线性层
                observed_module.linear.activation_post_process = \
                    observed_module.activation_post_process
                quantized = cls(nnq.Linear.from_float(observed_module.linear))
                return quantized

        # 定义一个包含自定义模块的主模块
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.custom = CustomModule()

            def forward(self, x0):
                # 多次使用自定义模块，模拟重复节点
                x1 = self.custom(x0)
                x2 = self.custom(x0)
                return x1 + x2

        # 准备量化配置字典，将浮点数模块转换为观察到的自定义模块类
        prepare_custom_config_dict = {
            "float_to_observed_custom_module_class": {
                "static": {
                    CustomModule: ObservedCustomModule
                }
            }
        }
        
        # 转换配置字典，将观察到的自定义模块转换为静态量化自定义模块类
        convert_custom_config_dict = {
            "observed_to_quantized_custom_module_class": {
                "static": {
                    ObservedCustomModule: StaticQuantCustomModule
                }
            }
        }
        
        # 实例化主模块，并设为评估模式
        m = M().eval()
        example_inputs = (torch.randn(3, 3),)
        
        # 使用准备函数，将主模块进行配置转换，应用默认量化配置和自定义配置
        m = prepare_fx(
            m,
            {"": default_qconfig},
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict)
        
        # 使用转换函数，将准备好的模块进一步转换为量化模块
        m = convert_fx(
            m,
            convert_custom_config=convert_custom_config_dict)
        
        # 确保模块可以成功运行
        m(*example_inputs)

    @skipIfNoFBGEMM
    def test_non_traceable_module(self):
        # 定义一个不可追踪的模块 NonTraceable
        class NonTraceable(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, x):
                # 对输入的字典 x 的每个键进行迭代并打印其值
                for k in x.keys():
                    print(x[k])
                return x

        # 定义另一个不可追踪的模块 NonTraceable2
        class NonTraceable2(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, x):
                # 数据依赖的控制流不可追踪
                for i in x:
                    print(i)
                return x

        # 定义包含 NonTraceable 和 NonTraceable2 模块的主模块 M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 实例化 NonTraceable 模块并赋值给 self.m1
                self.m1 = NonTraceable()
                # 实例化 NonTraceable2 模块并赋值给 self.m2
                self.m2 = NonTraceable2()

            # 前向传播方法
            def forward(self, x):
                # 依次调用 self.m1 和 self.m2 的前向传播方法
                x = self.m1(x)
                x = self.m2(x)
                return x

        # 创建 M 类的实例并将其设置为评估模式
        m = M().eval()
        # 定义量化配置字典
        qconfig_dict = {"": default_qconfig}
        # 定义准备过程中的自定义配置字典
        prepare_custom_config_dict = {
            "non_traceable_module_name": [
                "m1"
            ],
            "non_traceable_module_class": [
                NonTraceable2
            ]
        }
        # 对模型 m 进行 FX 准备过程
        m = prepare_fx(
            m, qconfig_dict,
            example_inputs=({"key": torch.randn(1)},),
            prepare_custom_config=prepare_custom_config_dict)

        # 定义预期的节点出现次数字典
        node_occurrence = {
            ns.call_module(NonTraceable) : 1,
            ns.call_module(NonTraceable2) : 1,
        }
        # 确保这些模块没有被追踪
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_prepared_model_deepcopy(self):
        """Ensures that copy.deepcopy works correctly on a prepared model.
        """
        # 定义一个简单的神经网络模型 M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个卷积层
                self.conv = torch.nn.Conv2d(1, 1, 1)
                # 添加一个私有属性 _foobar
                self._foobar = 'foobar'
                # 添加一个公共属性 foobar2
                self.foobar2 = 'foobar2'

            # 前向传播方法
            def forward(self, x):
                # 将输入 x 传入卷积层并返回结果
                x = self.conv(x)
                return x

        # 创建 M 类的实例
        m = M()
        # 将模型设置为评估模式
        m.eval()
        # 定义量化配置字典
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        # 定义示例输入
        example_inputs = (torch.randn(4, 1, 4, 4),)
        # 对模型进行 FX 准备过程
        prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 执行校准
        prepared(*example_inputs)
        # 对准备好的模型进行深拷贝
        prepared_copy = copy.deepcopy(prepared)
        # 进行量化，确保无错误
        quantized = convert_fx(prepared_copy)
    def test_quantized_model_type(self):
        """ Test state_dict and deepcopy works properly in the quantized model
        """
        # 定义一个简单的神经网络模型，包含一个线性层
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        # 创建一个示例输入
        example_inputs = (torch.rand(8, 5),)
        # 实例化模型并设置为评估模式
        m = M().eval()
        # 准备模型以进行量化，使用默认的量化配置
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        # 将模型转换为量化模型
        m = convert_fx(m)
        # 测试深拷贝功能
        m_copy = copy.deepcopy(m)
        # 断言深拷贝后的模型与原模型在相同输入下的输出相等
        self.assertEqual(m_copy(*example_inputs), m(*example_inputs))

        # 测试模型状态字典的加载
        state_dict = m.state_dict()
        # 实例化一个新的模型
        m_new = M().eval()
        # 准备新模型以进行量化，使用默认的量化配置
        m_new = prepare_fx(m_new, {"": default_qconfig}, example_inputs=example_inputs)
        # 将新模型转换为量化模型
        m_new = convert_fx(m_new)
        # 加载先前保存的状态字典到新模型中
        m_new.load_state_dict(state_dict)
        # 断言加载状态字典后的模型与原模型在相同输入下的输出相等
        self.assertEqual(m_new(*example_inputs), m(*example_inputs))

    def test_dequantize(self):
        r""" Test to make sure dequantize node are placed before
        non-quantizable node
        """
        # 定义一个包含卷积和激活函数的神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.act = torch.nn.GELU()

            def forward(self, x):
                x = self.conv(x)
                return self.act(x)

        # 创建随机数据作为输入
        data = torch.rand(5, 1, 3, 3, dtype=torch.float)
        # 遍历静态量化类型列表
        for quant_type in self.static_quant_types:
            # 预期的节点列表，包含量化后的卷积层、去量化操作和激活函数
            node_list = [
                ns.call_module(nnq.Conv2d),
                ns.call_method("dequantize"),
                ns.call_module(nn.GELU),
            ]
            # 使用自定义函数检查图模式下操作的正确性
            self.checkGraphModeFxOp(
                M().eval(), (data,), quant_type, expected_node_list=node_list)

    def test_sequential(self):
        # 定义一个包含两个卷积层的顺序模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 1, 1),
                    torch.nn.Conv2d(1, 1, 1)
                )

            def forward(self, x):
                x = self.convs(x)
                return x

        # 创建随机数据作为输入
        data = torch.rand(5, 1, 3, 3, dtype=torch.float)
        # 遍历静态量化类型列表
        for quant_type in self.static_quant_types:
            # 预期的节点列表，包含两个量化后的卷积层
            node_list = [
                ns.call_module(nnq.Conv2d),
                ns.call_module(nnq.Conv2d),
            ]
            # 使用自定义函数检查图模式下操作的正确性
            self.checkGraphModeFxOp(
                M().eval(), (data,), quant_type, expected_node_list=node_list)
    # 测试带有图量化输入和输出选项的功能
    def _test_quantized_inputs_outputs(
            self, prepare_custom_config_dict, prepare_count_check,
            convert_count_check):
        """
        Test the option to have inputs and outputs of the graph quantized
        """
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 模型的前向传播
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        # 创建模型实例
        m = M()
        # 设置量化配置字典
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        # 创建示例输入
        example_inputs = (torch.randn(1, 1, 4, 4),)
        # 将模型设为评估模式
        m.eval()
        # 准备量化的函数表示
        mp = torch.ao.quantization.quantize_fx.prepare_fx(
            m, qconfig_dict,
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict)
        # 检查量化后的函数表示中节点的出现次数是否符合预期
        self.checkGraphModuleNodes(mp, expected_node_occurrence=prepare_count_check)
        # 对量化后的模型进行例子输入测试
        mp(*example_inputs)
        # 将量化后的函数表示转换为持久化表示
        mq = torch.ao.quantization.quantize_fx.convert_fx(mp)
        # 检查转换后的模型中节点的出现次数是否符合预期
        self.checkGraphModuleNodes(mq, expected_node_occurrence=convert_count_check)

    # 测试图量化输入和图量化输出的情况
    def test_quantized_input_quantized_output(self):
        # 准备自定义的配置字典，指定输入和输出量化的索引
        prepare_custom_config_dict = {
            'input_quantized_idxs': [0], 'output_quantized_idxs': [0]}
        # 检查预备过程中量化观察器模块的出现次数是否符合预期
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 2,
        }
        # 检查转换过程中函数调用和方法调用的出现次数是否符合预期
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 0,
            ns.call_method('dequantize'): 0,
        }
        # 调用测试函数进行测试
        self._test_quantized_inputs_outputs(
            prepare_custom_config_dict, prepare_count_check, convert_count_check)

    # 测试浮点输入和图量化输出的情况
    def test_fp32_input_quantized_output(self):
        # 准备自定义的配置字典，指定输出量化的索引
        prepare_custom_config_dict = {
            'output_quantized_idxs': [0]}
        # 检查预备过程中量化观察器模块的出现次数是否符合预期
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 3,
        }
        # 检查转换过程中函数调用和方法调用的出现次数是否符合预期
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method('dequantize'): 0,
        }
        # 调用测试函数进行测试
        self._test_quantized_inputs_outputs(
            prepare_custom_config_dict, prepare_count_check, convert_count_check)

    # 测试图量化输入和浮点输出的情况
    def test_quantized_input_fp32_output(self):
        # 准备自定义的配置字典，指定输入量化的索引
        prepare_custom_config_dict = {
            'input_quantized_idxs': [0]}
        # 检查预备过程中量化观察器模块的出现次数是否符合预期
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 2,
        }
        # 检查转换过程中函数调用和方法调用的出现次数是否符合预期
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 0,
            ns.call_method('dequantize'): 1,
        }
        # 调用测试函数进行测试
        self._test_quantized_inputs_outputs(
            prepare_custom_config_dict, prepare_count_check, convert_count_check)
    # 定义测试函数，测试输入为 fp32，输出也为 fp32 的情况
    def test_fp32_input_fp32_output(self):
        # 准备自定义配置字典，这里为空字典
        prepare_custom_config_dict = {}
        # 定义准备阶段检查计数字典，对 MinMaxObserver 的调用应为 3 次
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 3,
        }
        # 定义转换阶段检查计数字典，对 quantize_per_tensor 函数调用 1 次，对 'dequantize' 方法调用 1 次
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method('dequantize'): 1,
        }
        # 调用内部方法 _test_quantized_inputs_outputs 进行测试
        self._test_quantized_inputs_outputs(
            prepare_custom_config_dict, prepare_count_check, convert_count_check)

    # 标记如果没有 FBGEMM 库，则跳过此测试函数
    @skipIfNoFBGEMM
    def test_convtranspose_per_channel_fails_early(self):
        """
        验证尝试对具有每通道权重观察器的 ConvTranspose 模块进行量化是否在准备阶段失败，而不是转换阶段。
        """
        # 创建一个包含单个 ConvTranspose2d 模块的 Sequential 模型
        m = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 1, 1))
        # 设置模型为评估模式
        m.eval()
        # 定义量化配置字典，使用默认的 fbgemm qconfig
        qconfig_dict = {'': torch.ao.quantization.get_default_qconfig('fbgemm')}
        # 使用断言检查是否会引发 AssertionError 异常
        with self.assertRaises(AssertionError) as context:
            # 准备模型 mp，传入量化配置字典和示例输入
            mp = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1, 1, 1),))
        # 使用断言检查异常信息是否与预期相符
        self.assertTrue(
            str(context.exception) ==
            'Per channel weight observer is not supported yet for ConvTranspose{n}d.')

    # 标记如果没有 FBGEMM 库，则跳过此测试函数
    @skipIfNoFBGEMM
    def test_qparams_buffers(self):
        # 定义一个包含多个嵌套模块的测试类，用于测试量化参数缓冲区的状态
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化权重为全1，偏置为全0的线性层模块
                self.w = torch.ones(5, 5)
                self.b = torch.zeros(5)

            def forward(self, x):
                # 使用 torch.nn.functional.linear 进行前向传播计算
                return torch.nn.functional.linear(x, self.w, self.b)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建包含两个 Linear 实例的序列模块 mods1 和一个单独的 Linear 实例 mods2
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )
                self.mods2 = Linear()

            def forward(self, x):
                # 对输入 x 分别应用 mods1 和 mods2
                x = self.mods1(x)
                x = self.mods2(x)
                return x

        # 创建 M 类的实例并设为评估模式
        model = M().eval()
        # 定义量化配置字典，包含一个默认的量化配置
        qconfig_dict = {"": default_qconfig}
        # 创建一个示例输入，由一个大小为 5x5 的随机张量组成的元组
        example_inputs = (torch.rand(5, 5),)
        # 使用 prepare_fx 对模型进行准备，传入量化配置字典和示例输入
        m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
        # 执行模型的前向传播
        m(*example_inputs)
        # 将准备好的模型转换为量化版本
        m = convert_fx(m)
        # 获取模型的状态字典的键值列表
        keys = m.state_dict().keys()
        # 初始化计数器来统计不同类型的量化参数
        quant_scale_count = quant_zero_point = scale_count = zero_point_count = 0
        # 遍历状态字典的键
        for k in keys:
            # 根据键名中的特定字符串进行计数
            if 'input_scale' in k:
                quant_scale_count = quant_scale_count + 1
            elif 'input_zero_point' in k:
                quant_zero_point = quant_zero_point + 1
            elif 'scale' in k:
                scale_count = scale_count + 1
            elif 'zero_point' in k:
                zero_point_count = zero_point_count + 1

        # 断言：期望每个量化的线性操作都有一个 scale 和一个 zero point
        self.assertTrue(scale_count == 3, "Expect each quantized linear op to have a scale in state_dict")
        self.assertTrue(zero_point_count == 3, "Expect each quantized linear op to have a zero_point in state_dict")
        # 再次执行模型的前向传播
        m(*example_inputs)
        # 确保模型可以被转为脚本形式
        scripted = torch.jit.script(m)
        # 获取脚本模型的状态字典的键值列表
        scripted_keys = scripted.state_dict().keys()
        # 将脚本模型中的一个权重属性设为原模型的相应属性
        scripted.mods1_0_packed_weight_0 = m.state_dict()["mods1_0_packed_weight_0"]
        # 获取非打包权重的键列表
        non_packed_weight_keys = [key for key in keys if "_packed_weight" not in key]
        # 断言：期望脚本模型保留非打包权重属性的状态字典
        self.assertTrue(
            set(scripted_keys) == set(non_packed_weight_keys),
            "Expected the scripted model to preserve the state_dict for non-packed weight attributes")
        # TODO: 可能不希望硬编码属性名，因为它们是自动生成的
        # 遍历属性名列表，对每个属性名进行断言检查
        for attr_name in [
                "mods1_0_input_scale_0", "mods1_0_input_zero_point_0",
                "mods1_0_scale_1", "mods1_0_zero_point_1",
                "mods1_1_scale_1", "mods1_1_zero_point_1",
                "mods2_scale_1", "mods2_zero_point_1"]:
            self.assertTrue(hasattr(m, attr_name), attr_name + " not found.")

    @skipIfNoFBGEMM
    def test_packed_weight_fused_op(self):
        # 定义一个简单的线性层模型
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)  # 初始化权重矩阵为全1
                self.b = torch.zeros(5)    # 初始化偏置向量为全0

            def forward(self, x):
                return F.linear(x, self.w, self.b)  # 执行线性变换操作

        # 定义一个包含两个线性层模型的序列
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )
                self.mods2 = Linear()  # 定义单独的一个线性层模型
                self.relu = F.relu    # 定义ReLU激活函数的引用

            def forward(self, x):
                x = self.mods1(x)  # 使用mods1执行前向传播
                x = self.mods2(x)  # 使用mods2执行前向传播
                x = self.relu(x)   # 对结果应用ReLU激活函数
                return x

        model = M().eval()  # 创建模型实例并设置为评估模式
        example_inputs = (torch.rand(5, 5),)  # 创建一个示例输入
        qconfig_dict = {"": default_qconfig}  # 定义量化配置字典
        m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)  # 准备量化模型
        m(*example_inputs)  # 运行量化模型以确保其正常工作
        m = convert_fx(m)   # 将准备好的模型转换为量化表示
        assert hasattr(m, "mods1_0_packed_weight_0")  # 断言模型具有预期的属性
        assert hasattr(m, "mods1_1_packed_weight_0")  # 断言模型具有预期的属性
        assert hasattr(m, "mods2_packed_weight_0")    # 断言模型具有预期的属性

    @skipIfNoFBGEMM
    def test_mul_add_fp16_config(self):
        # 使用fbgemm引擎覆盖量化引擎
        with override_quantized_engine('fbgemm'):
            # 定义一个简单的线性层模型
            class Linear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = torch.ones(5, 5)   # 初始化权重矩阵为全1
                    self.b = torch.zeros(5)     # 初始化偏置向量为全0

                def forward(self, x):
                    return torch.nn.functional.linear(x, self.w, self.b)  # 执行线性变换操作

            # 定义一个包含两个线性层模型的序列
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mods1 = torch.nn.Sequential(
                        Linear(),
                        Linear()
                    )
                    self.mods2 = Linear()  # 定义单独的一个线性层模型

                def forward(self, x):
                    x = x * 5    # 执行乘法操作
                    x = x + 5    # 执行加法操作
                    x = self.mods1(x)  # 使用mods1执行前向传播
                    x = self.mods2(x)  # 使用mods2执行前向传播
                    return x

            model = M().eval()   # 创建模型实例并设置为评估模式
            qconfig_dict = {"": float16_dynamic_qconfig}  # 定义量化配置字典
            example_inputs = (torch.rand(5, 5),)   # 创建一个示例输入
            m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)  # 准备量化模型
            m = convert_fx(m)    # 将准备好的模型转换为量化表示
            # 确保模型可以成功运行
            m(*example_inputs)
    def test_getattr_with_nontensor_result(self):
        """
        Verifies that binary ops get quantized correctly if some
        of the args are nodes but not Tensors, such as an `x.ndim`
        pattern.
        """
        # 定义测试函数，验证当某些参数是节点但不是张量时（如 `x.ndim` 模式），二进制操作是否正确量化。

        class M1(torch.nn.Module):
            def forward(self, x):
                # 获取输入张量的维度数
                dims = x.ndim
                # 对维度数进行减一操作
                dims_sub = dims - 1
                # 对减一后的维度数再次减一
                dims_sub2 = dims_sub - 1
                # 使用 torch.add 函数对输入张量 x 和 dims_sub2 相加
                x = torch.add(x, dims_sub2)
                return x

        class M2(torch.nn.Module):
            def forward(self, x):
                # 获取输入张量的维度数
                dims = x.ndim
                # 对维度数进行减二操作
                dims_sub = dims - 2
                # 创建一个包含多个 1 的列表，长度为 dims_sub
                mul = [1] * dims_sub
                # 构建新的维度列表，将 -1 和输入张量第二个维度大小加入其中
                dims_list = [-1, x.size(1)] + mul
                # 使用 x.view 函数按照新的维度列表对输入张量进行变形
                x = x.view(dims_list)
                return x

        class M3(torch.nn.Module):
            def forward(self, x):
                # 获取输入张量的形状
                shape = x.shape
                # 使用 x.view 函数按照原始形状对输入张量进行变形
                x = x.view(shape)
                return x

        # 遍历所有的类，对每个类进行以下操作
        for cls in (M1, M2, M3):
            # 创建类的实例并设置为评估模式
            m = cls().eval()
            # 创建一个示例输入张量
            example_inputs = (torch.rand(4, 4, 4, 4),)
            # 调用类实例的 __call__ 方法，执行模型前向传播
            m(*example_inputs)
            # 定义量化配置字典
            qconfig_dict = {'': torch.ao.quantization.default_qconfig}
            # 使用 prepare_fx 函数准备模型
            mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
            # 调用准备好的模型，执行模型前向传播
            mp(torch.rand(4, 4, 4, 4))
            # 使用 convert_fx 函数将准备好的模型转换为量化模型
            mc = convert_fx(mp)

    class _NonReferenceTestModel(nn.Module):
        def __init__(self, func, lin_in, lin_out):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.lin = nn.Linear(lin_in, lin_out)
            self.func = func

        def forward(self, x, y, z):
            # 执行卷积、ReLU和池化操作
            x = self.pool(F.relu(self.conv1(x)))
            # 对张量 x 进行展平操作
            x = torch.flatten(x, 1)
            # 使用 func 函数处理张量 x、y、z
            x = self.func(x, y, z)
            # 使用线性层进行最终的线性变换
            x = self.lin(x)
            return x

    # This function looks at the node specified by the NodeInfo in the key of
    # node_info_to_non_tensor_args and checks that the args at specified indices
    # are not observed (since they are non tensors). If the args at those indices
    # are a tuple/list (which do not show up as nodes) the function checks the
    # individual elements of the tuple/list recursively.

    # 此函数查看由 node_info_to_non_tensor_args 键中的 NodeInfo 指定的节点，并检查指定索引处的参数是否未被观察（因为它们不是张量）。
    # 如果这些索引处的参数是元组/列表（它们不会显示为节点），则函数递归地检查元组/列表的各个元素。
    # 检查给定模型中未观察到的节点
    def _check_not_observed(self, model, node_info_to_non_tensor_args):

        # 这是一个辅助函数（用于更容易地递归），检查 arg_node 是否被观察到
        def _check_node_not_observed(model, arg_node, node):
            if isinstance(arg_node, (tuple, list)):
                for new_node in arg_node:
                    _check_node_not_observed(model, new_node, node)
            elif arg_node.op == "call_module":
                # 断言：检查模块是否为激活后处理，如果是则抛出 AssertionError
                self.assertTrue(
                    not _is_activation_post_process(getattr(model, arg_node.target)),
                    f"Arg: {arg_node} of node: {node} is observed but is not a float tensor",
                )

        # 遍历模型图中的每个节点
        for node in model.graph.nodes:
            # 获取当前节点相关的非张量参数的索引列表
            indices = node_info_to_non_tensor_args.get(
                NodeInfo(node.op, node.target), []
            )
            for index in indices:
                # 检查索引是否在参数列表长度范围内，防止索引超出界限
                if index < len(node.args):
                    arg_node = node.args[index]
                    # 调用辅助函数检查参数节点是否被观察到
                    _check_node_not_observed(model, arg_node, node)

    # 这个测试检查模型是否正确准备，不在特定操作上有观察者（参见 _check_not_observed），并且准备好的模型运行正常
    def _test_dtype_propagation(self, model, node_info_to_non_tensor_args, *args):
        # 将模型设置为评估模式
        model.eval()
        # 定义量化配置字典
        qconfig_dict = {"": torch.ao.quantization.get_default_qconfig("fbgemm")}
        # 准备模型，包括量化和准备输入示例
        prepared_model = prepare_fx(model, qconfig_dict, example_inputs=tuple(args))
        # 调用 _check_not_observed 方法检查模型是否有未观察到的节点
        self._check_not_observed(prepared_model, node_info_to_non_tensor_args)
        # 执行准备好的模型，传入参数 *args
        prepared_model(*args)

    # 测试非张量参数在 masked_fill 操作中是否未观察到
    def test_masked_fill_nontensor_args_not_observed(self):
        # 定义一个函数 func，对输入 x 进行 masked_fill 操作
        def func(x, y, z):
            return x.masked_fill(y, z)

        # 使用 _NonReferenceTestModel 初始化模型，设置相关参数和方法
        model = self._NonReferenceTestModel(func, 1176, 1)
        # 定义测试函数的输入参数 args
        args = [torch.randn(5, 3, 32, 32), torch.randn(1176) > 0, 0.1]
        # 定义 node_info_to_non_tensor_args 字典，指定特定操作（此处为 masked_fill）的非张量参数索引
        node_info_to_non_tensor_args = {NodeInfo("call_method", "masked_fill"): [1, 2]}
        # 调用 _test_dtype_propagation 方法执行测试
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 测试非张量参数在 permute 操作中是否未观察到
    def test_permute_nontensor_args_not_observed(self):
        # 定义一个函数 func，对输入 x 进行 permute 操作
        def func(x, y, z):
            return x.permute(y, z)

        # 使用 _NonReferenceTestModel 初始化模型，设置相关参数和方法
        model = self._NonReferenceTestModel(func, 1176, 1)
        # 定义测试函数的输入参数 args
        args = [torch.randn(5, 3, 32, 32), 0, 1]
        # 定义 node_info_to_non_tensor_args 字典，指定特定操作（此处为 permute）的非张量参数索引
        node_info_to_non_tensor_args = {NodeInfo("call_method", "permute"): [1, 2]}
        # 调用 _test_dtype_propagation 方法执行测试
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 测试非张量参数在 repeat 操作中是否未观察到
    def test_repeat_nontensor_args_not_observed(self):
        # 定义一个函数 func，对输入 x 进行 repeat 操作
        def func(x, y, z):
            return x.repeat(y, z)

        # 使用 _NonReferenceTestModel 初始化模型，设置相关参数和方法
        model = self._NonReferenceTestModel(func, 1176, 1)
        # 定义测试函数的输入参数 args
        args = [torch.randn(5, 3, 32, 32), 2, 1]
        # 定义 node_info_to_non_tensor_args 字典，指定特定操作（此处为 repeat）的非张量参数索引
        node_info_to_non_tensor_args = {NodeInfo("call_method", "repeat"): [1, 2]}
        # 调用 _test_dtype_propagation 方法执行测试
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)
    # 定义一个测试函数，测试 reshape 操作对于非张量参数的行为，不会被观察到
    def test_reshape_nontensor_args_not_observed(self):
        # 定义一个函数 func，接受 x, y, z 作为参数，返回 x 被 reshape 后的结果
        def func(x, y, z):
            return x.reshape(-1, y)

        # 创建一个 _NonReferenceTestModel 的实例 model，使用 func 作为处理函数，5 和 1 作为额外参数
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义参数 args，其中包括一个大小为 [5, 3, 32, 32] 的随机张量，5，和一个空值
        args = [torch.randn(5, 3, 32, 32), 5, None]
        # 定义 node_info_to_non_tensor_args 字典，指定 NodeInfo("call_method", "reshape") 作为键，值为 [2]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [2]}
        # 调用 self._test_dtype_propagation 方法，传递 model, node_info_to_non_tensor_args, *args 作为参数
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 定义一个测试函数，测试 size 操作对于非张量参数的行为，不会被观察到
    def test_size_nontensor_args_not_observed(self):
        # 定义一个函数 func，接受 x, y, z 作为参数，返回 x 被 reshape 后的结果
        def func(x, y, z):
            return x.reshape((-1, x.size(y)))

        # 创建一个 _NonReferenceTestModel 的实例 model，使用 func 作为处理函数，5 和 1 作为额外参数
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义参数 args，其中包括一个大小为 [5, 3, 32, 32] 的随机张量，0，和一个空值
        args = [torch.randn(5, 3, 32, 32), 0, None]
        # 定义 node_info_to_non_tensor_args 字典，指定 NodeInfo("call_method", "size") 作为键，值为 [1]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "size"): [1]}
        # 调用 self._test_dtype_propagation 方法，传递 model, node_info_to_non_tensor_args, *args 作为参数
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 定义一个测试函数，测试 transpose 操作对于非张量参数的行为，不会被观察到
    def test_transpose_nontensor_args_not_observed(self):
        # 定义一个函数 func，接受 x, y, z 作为参数，返回 x 被 transpose 后的结果
        def func(x, y, z):
            return x.transpose(y, z)

        # 创建一个 _NonReferenceTestModel 的实例 model，使用 func 作为处理函数，5 和 1 作为额外参数
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义参数 args，其中包括一个大小为 [5, 3, 32, 32] 的随机张量，0 和 1
        args = [torch.randn(5, 3, 32, 32), 0, 1]
        # 定义 node_info_to_non_tensor_args 字典，指定 NodeInfo("call_method", "transpose") 作为键，值为 [1, 2]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "transpose"): [1, 2]}
        # 调用 self._test_dtype_propagation 方法，传递 model, node_info_to_non_tensor_args, *args 作为参数
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 定义一个测试函数，测试 torch.transpose 操作对于非张量参数的行为，不会被观察到
    def test_torch_transpose_nontensor_args_not_observed(self):
        # 定义一个函数 func，接受 x, y, z 作为参数，返回 torch.transpose(x, 0, 1) 的结果
        def func(x, y, z):
            return torch.transpose(x, 0, 1)

        # 创建一个 _NonReferenceTestModel 的实例 model，使用 func 作为处理函数，5 和 1 作为额外参数
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义 node_info_to_non_tensor_args 字典，指定 NodeInfo("call_method", torch.transpose) 作为键，值为 [1, 2]
        node_info_to_non_tensor_args = {NodeInfo("call_method", torch.transpose): [1, 2]}
        # 定义参数 args，其中包括一个大小为 [5, 3, 32, 32] 的随机张量，0 和 1
        args = [torch.randn(5, 3, 32, 32), 0, 1]
        # 调用 self._test_dtype_propagation 方法，传递 model, node_info_to_non_tensor_args, *args 作为参数
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 定义一个测试函数，测试 unsqueeze 操作对于非张量参数的行为，不会被观察到
    def test_unsqueeze_nontensor_args_not_observed(self):
        # 定义一个函数 func，接受 x, y, z 作为参数，返回 x 被 unsqueeze(y) 后的结果
        def func(x, y, z):
            return x.unsqueeze(y)

        # 创建一个 _NonReferenceTestModel 的实例 model，使用 func 作为处理函数，1176 和 1 作为额外参数
        model = self._NonReferenceTestModel(func, 1176, 1)
        # 定义参数 args，其中包括一个大小为 [5, 3, 32, 32] 的随机张量，1 和一个空值
        args = [torch.randn(5, 3, 32, 32), 1, None]
        # 定义 node_info_to_non_tensor_args 字典，指定 NodeInfo("call_method", "unsqueeze") 作为键，值为 [1]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "unsqueeze"): [1]}
        # 调用 self._test_dtype_propagation 方法，传递 model, node_info_to_non_tensor_args, *args 作为参数
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 定义一个测试函数，测试 unsqueeze_ 操作对于非张量参数的行为，不会被观察到
    def test_unsqueeze__nontensor_args_not_observed(self):
        # 定义一个函数 func，接受 x, y, z 作为参数，返回 x 被 unsqueeze_(y) 后的结果
        def func(x, y, z):
            return x.unsqueeze_(y)

        # 创建一个 _NonReferenceTestModel 的实例 model，使用 func 作为处理函数，1176 和 1 作为额外参数
        model = self._NonReferenceTestModel(func, 1176, 1)
        # 定义参数 args，其中包括一个大小为 [5, 3, 32, 32] 的随机张量，1 和一个空值
        args = [torch.randn(5, 3, 32, 32), 1, None]
        # 定义 node_info_to_non_tensor_args 字典，指定 NodeInfo("call_method", "unsqueeze_") 作为键，值为 [1]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "unsqueeze_"): [1]}
        # 调用 self._test_dtype_propagation 方法，传递 model, node_info_to_non_tensor_args, *args 作为参数
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)
    # 测试 torch.unsqueeze 函数在不观察非张量参数时的行为
    def test_torch_unsqueeze_nontensor_args_not_observed(self):
        # TODO: make torch.unsqueeze scriptable by fx when using
        # variable nontensor arguments
        # 定义一个函数 func，它接受参数 x, y, z，并调用 torch.unsqueeze 函数
        def func(x, y, z):
            return torch.unsqueeze(x, 1)

        # 创建一个 _NonReferenceTestModel 对象，使用 func 作为函数
        model = self._NonReferenceTestModel(func, 1176, 1)
        # 定义函数调用的参数列表 args
        args = [torch.randn(5, 3, 32, 32), 1, None]
        # 定义一个映射，指定哪些节点信息对应于非张量参数
        node_info_to_non_tensor_args = {NodeInfo("call_method", torch.unsqueeze): [1]}
        # 调用测试函数，验证数据类型的传播
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 测试 view 方法在不观察非张量参数时的行为
    def test_view_nontensor_args_not_observed(self):
        # 定义一个函数 func，它接受参数 x, y, z，并调用 x 的 view 方法
        def func(x, y, z):
            return x.view(-1, y)

        # 创建一个 _NonReferenceTestModel 对象，使用 func 作为函数
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义函数调用的参数列表 args
        args = [torch.randn(5, 3, 32, 32), 5, None]
        # 定义一个映射，指定哪些节点信息对应于非张量参数
        node_info_to_non_tensor_args = {NodeInfo("call_method", "view"): [2]}
        # 调用测试函数，验证数据类型的传播
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 测试对已知节点的列表参数的数据类型传播行为
    def test_propagate_dtypes_for_known_nodes_list_args(self):
        # 定义一个函数 func，它接受参数 x, y, z，并调用 x 的 reshape 方法
        def func(x, y, z):
            return x.reshape(y)

        # 创建一个 _NonReferenceTestModel 对象，使用 func 作为函数
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义函数调用的参数列表 args
        args = [torch.randn(5, 3, 32, 32), [-1, 5], None]
        # 定义一个映射，指定哪些节点信息对应于非张量参数
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [1]}
        # 调用测试函数，验证数据类型的传播
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 测试对已知节点的分离列表参数的数据类型传播行为
    def test_propagate_dtypes_for_known_nodes_split_list_args(self):
        # 定义一个函数 func，它接受参数 x, y, z，并调用 x 的 reshape 方法
        def func(x, y, z):
            return x.reshape([y, z])

        # 创建一个 _NonReferenceTestModel 对象，使用 func 作为函数
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义函数调用的参数列表 args
        args = [torch.randn(5, 3, 32, 32), -1, 5]
        # 定义一个映射，指定哪些节点信息对应于非张量参数
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [1]}
        # 调用测试函数，验证数据类型的传播
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 测试对已知节点的元组参数的数据类型传播行为
    def test_propagate_dtypes_for_known_nodes_tuple_args(self):
        # 定义一个函数 func，它接受参数 x, y, z，并调用 x 的 reshape 方法
        def func(x, y, z):
            return x.reshape(y)

        # 创建一个 _NonReferenceTestModel 对象，使用 func 作为函数
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义函数调用的参数列表 args
        args = [torch.randn(5, 3, 32, 32), (-1, 5), None]
        # 定义一个映射，指定哪些节点信息对应于非张量参数
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [1]}
        # 调用测试函数，验证数据类型的传播
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 测试对已知节点的分离元组参数的数据类型传播行为
    def test_propagate_dtypes_for_known_nodes_split_tuple_args(self):
        # 定义一个函数 func，它接受参数 x, y, z，并调用 x 的 reshape 方法
        def func(x, y, z):
            return x.reshape((y, z))

        # 创建一个 _NonReferenceTestModel 对象，使用 func 作为函数
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义函数调用的参数列表 args
        args = [torch.randn(5, 3, 32, 32), -1, 5]
        # 定义一个映射，指定哪些节点信息对应于非张量参数
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [1]}
        # 调用测试函数，验证数据类型的传播
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)
    # 测试函数：对已知节点的字典参数进行数据类型传播测试
    def test_propagate_dtypes_for_known_nodes_dict_args(self):
        # 定义一个简单的函数，使用字典 y 的 "first" 和 "second" 键来转置张量 x
        def func(x, y, z):
            return x.transpose(y["first"], y["second"])

        # 使用 _NonReferenceTestModel 类初始化模型，传入 func 函数作为模型的核心
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义测试参数 args：包括一个张量，一个字典（包含 "first" 和 "second" 键），以及一个空值
        args = [torch.randn(5, 3, 32, 32), {"first": 0, "second": 1}, None]
        # 定义节点信息到非张量参数的映射，这里将 "transpose" 方法映射到字典 y 的 "first" 和 "second" 键
        node_info_to_non_tensor_args = {NodeInfo("call_method", "transpose"): [1, 2]}
        # 调用 _test_dtype_propagation 方法，测试数据类型传播
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 测试函数：对已知节点的字典和元组参数进行数据类型传播测试
    def test_propagate_dtypes_for_known_nodes_dict_tuple_args(self):
        # 定义一个简单的 nn.Module 类 reshape_module，重写 forward 方法，使用字典 y 的 "shape" 键来重塑张量 x
        class reshape_module(nn.Module):
            def forward(self, x, y, z):
                return x.reshape(y["shape"])

        # 使用 _NonReferenceTestModel 类初始化模型，传入 reshape_module 实例作为模型的核心
        model = self._NonReferenceTestModel(reshape_module(), 5, 1)
        # 定义测试参数 args：包括一个张量，一个字典（包含 "shape" 键），以及一个空值
        args = [torch.randn(5, 3, 32, 32), {"shape": (-1, 5)}, None]
        # 定义节点信息到非张量参数的映射，这里将 "reshape" 方法映射到字典 y 的 "shape" 键
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [1]}
        # 调用 _test_dtype_propagation 方法，测试数据类型传播
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 测试函数：对已知节点的字典和元组参数进行数据类型传播测试（错误的节点信息）
    def test_propagate_dtypes_for_known_nodes_dict_split_tuple_args(self):
        # 定义一个简单的函数，使用字典 y 的 "first" 和 "second" 键来重塑张量 x
        def func(x, y, z):
            return x.reshape((y["first"], y["second"]))

        # 使用 _NonReferenceTestModel 类初始化模型，传入 func 函数作为模型的核心
        model = self._NonReferenceTestModel(func, 5, 1)
        # 定义测试参数 args：包括一个张量，一个字典（包含 "first" 和 "second" 键），以及一个空值
        args = [torch.randn(5, 3, 32, 32), {"first": -1, "second": 5}, None]
        # 定义节点信息到非张量参数的映射，这里错误地将 "transpose" 方法映射到字典 y 的 "first" 键
        node_info_to_non_tensor_args = {NodeInfo("call_method", "transpose"): [1]}
        # 调用 _test_dtype_propagation 方法，测试数据类型传播
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    # 测试函数：在量化层后验证计算量化张量大小的正确性
    """
    Verifies that calculating a size of a quantized tensor works
    correctly in quantization passes.
    """
    def test_assert_on_size_after_quant_layer(self):
        # 定义一个简单的 nn.Module 类 M，包含一个卷积层 conv1
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 在 forward 方法中，对输入 x 进行卷积操作
                x = self.conv1(x)
                # 断言卷积后张量的通道数为 1
                torch._assert(x.size(1) == 1, 'foobar')
                return x

        # 创建 M 类的实例 m，并设置为评估模式
        m = M().eval()
        # 定义示例输入 example_inputs：一个 4 维随机张量
        example_inputs = (torch.rand(4, 1, 4, 4),)
        # 在 m 上执行示例输入，得到量化配置字典 qconfig_dict
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 在量化模型 mp 上执行示例输入
        mp(*example_inputs)
        # 将量化模型 mp 转换为量化后的模型 mc
        mc = convert_fx(mp)
        # 在量化后的模型 mc 上执行示例输入
        mc(*example_inputs)
    def test_fp32_sum(self):
        """
        Verifies that fp32 sum works correctly if it's before or after
        quantized layers.
        """
        # 定义一个继承自 torch.nn.Module 的子类 M1
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个卷积层
                self.conv1 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 将输入 x 经过 conv1 卷积层处理
                x = self.conv1(x)
                # 将处理后的结果以列表形式堆叠
                x = torch.stack([x])
                # 对堆叠后的结果进行求和操作
                x = torch.sum(x)
                return x

        # 定义另一个继承自 torch.nn.Module 的子类 M2
        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加两个卷积层
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 将输入 x 经过 conv1 卷积层处理
                x = self.conv1(x)
                # 将处理后的结果以列表形式堆叠
                x1 = torch.stack([x])
                # 沿着第 0 维度对堆叠后的结果进行求和操作
                x1 = torch.sum(x1, dim=0)
                # 将 x1 输入到 conv2 卷积层进行处理
                x2 = self.conv2(x1)
                return x2

        # 遍历子类 M1 和 M2
        for cls in (M1, M2):
            # 创建子类实例，并设置为评估模式
            m = cls().eval()
            # 创建一个示例输入
            example_inputs = (torch.rand(4, 1, 4, 4),)
            # 使用默认的量化配置 qconfig_dict 准备模型 mp
            mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
            # 对准备好的模型 mp 执行示例输入
            mp(*example_inputs)
            # 将准备好的模型 mp 转换成量化模型 mc
            mc = convert_fx(mp)
            # 对转换后的量化模型 mc 执行示例输入
            mc(*example_inputs)

    def test_fusion_pattern_unquantized(self):
        """
        Ensure that leaving a possible fusion pattern of multiple nodes
        unquantized runs through the APIs without errors.
        """
        # 定义一个继承自 torch.nn.Module 的子类 Child
        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个 ReLU 激活函数层
                self.relu = nn.ReLU()

            def forward(self, x):
                # 对输入 x 加上常数 1.0
                x = torch.add(x, 1.0)
                # 对加上常数后的结果应用 ReLU 激活函数
                x = torch.nn.functional.relu(x)
                return x

        # 定义一个继承自 torch.nn.Module 的子类 Parent
        class Parent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个 Child 类型的子模块
                self.child = Child()
                # 添加一个卷积层
                self.conv = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 将输入 x 经过 Child 模块处理
                x = self.child(x)
                # 将处理后的结果经过卷积层处理
                x = self.conv(x)
                return x

        # 创建 Parent 类型的实例，并设置为评估模式
        m = Parent().eval()
        # 设置量化配置 qconfig_dict
        qconfig_dict = {
            '': torch.ao.quantization.default_qconfig,
            'module_name': [
                ('child', None),
            ],
        }
        # 创建一个示例输入
        example_inputs = (torch.rand(1, 1, 1, 1),)
        # 使用默认的量化配置 qconfig_dict 准备模型 mp
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 对准备好的模型 mp 执行示例输入
        mp(*example_inputs)
        # 将准备好的模型 mp 转换成量化模型 mc
        mc = convert_fx(mp)
    def test_state_dict(self):
        """ 确保打包的参数出现在 state_dict 中
        """

        # test linear packed weight
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.rand(4, 30)  # 初始化线性层权重
                self.b = torch.rand(4)      # 初始化线性层偏置

            def forward(self, x):
                return F.linear(x, self.w, self.b)  # 前向传播函数使用线性函数

        m = M1().eval()  # 实例化并设置为评估模式
        qconfig_dict = {"": default_qconfig}  # 设置量化配置字典
        m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 30),))  # 准备将模型转换为量化模型
        m = convert_fx(m)  # 执行模型转换
        state_dict = m.state_dict()  # 获取模型的状态字典
        self.assertTrue("_packed_weight_0" in state_dict)  # 断言确保状态字典中包含 "_packed_weight_0"

        # test conv packed weight
        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.rand(3, 3, 3, 3)  # 初始化卷积层权重
                self.b = torch.rand(3)          # 初始化卷积层偏置
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x):
                return F.conv2d(x, self.w, self.b, self.stride, self.padding, self.dilation, self.groups)  # 使用卷积函数进行前向传播

        m = M2().eval()  # 实例化并设置为评估模式
        qconfig_dict = {"": default_qconfig}  # 设置量化配置字典
        m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 3, 3, 3),))  # 准备将模型转换为量化模型
        m = convert_fx(m)  # 执行模型转换
        state_dict = m.state_dict()  # 获取模型的状态字典
        self.assertTrue("_packed_weight_0" in state_dict)  # 断言确保状态字典中包含 "_packed_weight_0"

        # test load
        ref_weight, ref_bias = torch.ops.quantized.conv2d_unpack(state_dict["_packed_weight_0"])  # 解压缩量化卷积的权重和偏置
        data = torch.rand(1, 3, 5, 5)
        ref_res = m(data)  # 获取参考结果
        m = M2().eval()
        m = prepare_fx(m, qconfig_dict, (data,))
        m = convert_fx(m)
        res = m(data)
        weight, bias = m._packed_weight_0.unpack()  # 解包当前模型的权重和偏置
        # 检查随机模型的权重/偏置不匹配参考的权重/偏置
        self.assertNotEqual(weight, ref_weight)
        self.assertNotEqual(bias, ref_bias)
        self.assertNotEqual(res, ref_res)
        m.load_state_dict(state_dict)  # 加载状态字典到模型

        def checkModel(m, data, ref_weight, ref_bias, ref_res):
            res = m(data)
            weight, bias = m._packed_weight_0.unpack()
            # 检查加载状态字典后权重/偏置匹配
            self.assertEqual(weight, ref_weight)
            self.assertEqual(bias, ref_bias)
            self.assertEqual(res, ref_res)

        checkModel(m, data, ref_weight, ref_bias, ref_res)

        # Test save to disk and load back
        m = M2().eval()
        m = prepare_fx(m, qconfig_dict, example_inputs=(data,))
        m = convert_fx(m)
        m.load_state_dict(state_dict)
        with TemporaryFileName() as fname:
            torch.save(m.state_dict(), fname)  # 保存模型状态字典到文件
            # 在这里不测试 weights_only，因为这是加载一个 ScriptModule
            m.load_state_dict(torch.load(fname))  # 加载保存的状态字典

        checkModel(m, data, ref_weight, ref_bias, ref_res)

    @skipIfNoFBGEMM
    def test_preserve_qconfig(self):
        """
        Test to make sure the temporary config option to preserve qconfig attributes
        in the model works
        """
        # 使用上下文管理器修改量化引擎为 'fbgemm'
        with override_quantized_engine('fbgemm'):
            # 定义一个简单的线性层模块
            class Linear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = torch.ones(5, 5)  # 初始化权重为全1
                    self.b = torch.zeros(5)    # 初始化偏置为全0

                def forward(self, x):
                    # 返回输入 x 的线性变换结果
                    return torch.nn.functional.linear(x, self.w, self.b)

            # 定义一个包含多层模块的主模块
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mods1 = torch.nn.Sequential(
                        Linear(),  # 包含两个线性层
                        Linear()
                    )
                    self.mods2 = torch.nn.Sigmoid()  # sigmoid 激活层

                def forward(self, x):
                    x = self.mods1(x)  # 对输入 x 应用第一组模块
                    x = self.mods2(x)  # 对结果应用第二组模块
                    return x

            model = M().eval()  # 创建模型并设置为评估模式
            # 定义量化配置字典
            qconfig_dict = {
                "object_type": [
                    (torch.nn.functional.linear, float16_dynamic_qconfig),
                ],
            }
            example_inputs = (torch.rand(5, 5),)  # 生成输入示例
            m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)  # 准备模型进行量化
            m(*example_inputs)  # 对模型进行一次前向传播
            m = convert_fx(m, _remove_qconfig=False)  # 将准备好的模型转换为量化模型并保留 qconfig

            # 断言模块 mods2 是否有 qconfig 属性
            self.assertTrue(hasattr(m.mods2, 'qconfig'))

    def test_not_used(self):
        """ Test quantizing a not used value"""

        # 定义一个简单的模块
        class M(torch.nn.Module):
            def forward(self, x):
                x = x + x  # 对输入 x 进行加法操作
                x.sigmoid_()  # 对结果应用 sigmoid 激活函数
                return x

        m = M().eval()  # 创建模型并设置为评估模式
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping().set_global(float16_static_qconfig)
        # 确保量化过程运行
        m = prepare_fx(m, qconfig_mapping, example_inputs=(torch.randn(1),))
        m = convert_fx(m)  # 将准备好的模型转换为量化模型
    def test_qparams_fqn(self):
        """ Test that the FQN of input_scale/zero_point is set
        to that of first linear use. """
        # 定义一个简单的线性层模块
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)  # 初始化权重为全一张量
                self.b = torch.zeros(5)    # 初始化偏置为全零张量

            def forward(self, x):
                return torch.nn.functional.linear(x, self.w, self.b)  # 执行线性变换

        # 定义包含两个线性层的模块
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )

            def forward(self, x):
                x = torch.cat((x,), 1)  # 在第一个维度上对输入张量进行连接
                tmp = x.size()         # 计算连接后张量的尺寸
                x = self.mods1(x)      # 执行包含两个线性层的序列模块
                y = x * tmp[0]         # 乘以连接前张量的第一个维度大小
                return y

        model = M().eval()  # 创建并转换为评估模式
        qconfig_dict = {
            "": None,  # 根据空键来应用默认的量化配置
            "object_type": [
                (torch.nn.functional.linear, default_qconfig),  # 设置线性函数的量化配置
                (torch.nn.functional.relu, default_qconfig),    # 设置ReLU函数的量化配置
            ],
        }
        example_inputs = (torch.rand(5, 5),)  # 创建一个示例输入张量
        m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)  # 准备模型以便量化
        m(*example_inputs)  # 对示例输入进行模型调用
        m = convert_fx(m)   # 将模型转换为量化表示
        keys = m.state_dict().keys()  # 获取模型状态字典的键
        m(torch.randn(5, 5))  # 对一个随机输入执行模型调用
        # TODO: probably don't want to hardcode the attribute names, since they are generated
        # 遍历属性名列表，确保每个属性名存在于模型中
        for attr_name in [
                "mods1_0_input_scale_0", "mods1_0_input_zero_point_0",
                "mods1_0_scale_0", "mods1_0_zero_point_0",
                "mods1_1_scale_0", "mods1_1_zero_point_0"]:
            self.assertTrue(hasattr(m, attr_name), attr_name + " not found.")

    def test_no_obs_between_unmatched_node_and_copy_node(self):
        """
        Verifies that an observer is not inserted between an unmatched
        node and a node matched to CopyNodeQuantizeHandler.  This is done
        because observers require activations to be Tensors, and there is
        no guarantee that an output of an unmatched node is a Tensor.
        """
        # 定义一个简单的模块M，包含ReLU激活函数
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                x = _user_func_with_complex_return_type(x)  # 执行一个复杂返回类型的用户函数
                x1 = x[0] + 1  # 对返回的第一个元素执行加一操作
                return x1, x[1]  # 返回修改后的张量和原始第二个元素

        m = M().eval()  # 创建并转换为评估模式

        qconfig_dict = {'': torch.ao.quantization.default_qconfig}  # 使用默认的量化配置
        example_inputs = (torch.randn(4, 4, 4, 4),)  # 创建一个示例输入张量
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)  # 准备模型以便量化
        # 如果在_user_func_with_complex_return_type后插入了观察者，以下调用将失败
        mp(*example_inputs)  # 对示例输入执行模型调用
        mc = convert_fx(mp)   # 将模型转换为量化表示
        mc(*example_inputs)   # 对示例输入执行量化后的模型调用
    def test_fold_quant_dequant(self):
        """ Test that the sequence of quant-dequant nodes in the
            graph, get folded and we erase the extra dequant nodes.
        """
        # 定义一个测试方法，验证图中量化-反量化节点的顺序是否被折叠并且多余的反量化节点被删除

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)  # 初始化权重矩阵为 5x5 全为1的张量
                self.b = torch.zeros(5)     # 初始化偏置向量为长度为5的全零张量

            def forward(self, x):
                x = torch.cat((x,), 1)  # 在输入张量x的第1维度上连接x自身，构成一个新张量
                tmp = x.size()  # 获取x张量的大小（形状）
                x = torch.nn.functional.linear(x, self.w, self.b)  # 对输入张量x进行线性变换
                y = x * tmp[0]  # 计算结果张量y为x与tmp[0]的乘积
                return y

        model = M().eval()  # 创建M类的实例，并设置为评估模式
        qconfig_dict = {
            "": None,  # 默认配置为空
            "object_type": [
                (torch.nn.functional.linear, default_qconfig),  # 为torch.nn.functional.linear函数设置默认量化配置
            ],
        }
        example_inputs = (torch.rand(5, 5),)  # 创建一个示例输入，大小为5x5的随机张量
        m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)  # 准备模型m，应用量化配置和示例输入
        m(*example_inputs)  # 对模型m使用示例输入进行前向传播
        m = convert_fx(m)  # 将模型m转换为量化后的模型
        keys = m.state_dict().keys()  # 获取量化后模型m的状态字典的键集合
        m(*example_inputs)  # 再次对量化后模型m使用示例输入进行前向传播
        dequant = 0  # 初始化反量化节点计数为0
        quant = 0  # 初始化量化节点计数为0
        for n in m.graph.nodes:  # 遍历量化后模型m的计算图节点
            if n.op == "call_method" and n.target == "dequantize":  # 如果节点是调用方法且目标是"dequantize"
                dequant = dequant + 1  # 增加反量化节点计数
            if n.op == "call_function" and n.target == torch.quantize_per_tensor:  # 如果节点是调用函数且目标是torch.quantize_per_tensor
                quant = quant + 1  # 增加量化节点计数
        self.assertEqual(dequant, 1)  # 断言：期望反量化节点计数为1
        self.assertEqual(quant, 1)  # 断言：期望量化节点计数为1
    def test_quant_output_always_observed(self):
        """
        If the output is hardcoded to be quantized, ensure that
        there is always an observer, even if the last non-output node is not
        quantizeable.
        """
        # 定义量化配置字典，使用默认的量化训练 QConfig
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        # 准备自定义配置字典，指定输出量化的索引
        prepare_custom_config_dict = {'output_quantized_idxs': [0]}
        # 示例输入
        example_inputs = (torch.randn(4, 1, 4, 4),)

        # non-quantizeable node, quantized output
        # 定义模块 M1，包含一个不可量化节点和量化输出
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.identity = torch.nn.Identity()

            def forward(self, x):
                x = self.identity(x)
                return x

        m1 = M1()
        # 调用检查图模式操作函数，验证模型 m1 在量化训练 QAT 下的节点出现情况
        self.checkGraphModeFxOp(
            m1, example_inputs, QuantType.QAT,
            prepare_expected_node_occurrence={
                ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 2,
            },
            expected_node_occurrence={
                ns.call_function(torch.quantize_per_tensor): 1,
            },
            prepare_custom_config=prepare_custom_config_dict)

        # quantizeable node, quantized output
        # 定义模块 M2，包含一个可量化节点和量化输出
        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv(x)
                return x

        m2 = M2()
        # 调用检查图模式操作函数，验证模型 m2 在量化训练 QAT 下的节点出现情况
        self.checkGraphModeFxOp(
            m2, example_inputs, QuantType.QAT,
            prepare_expected_node_occurrence={
                # one for weights, one for activations
                ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 2,
            },
            expected_node_occurrence={
                ns.call_function(torch.quantize_per_tensor): 1,
            },
            prepare_custom_config=prepare_custom_config_dict)

        # quantizeable node, quantized dictionary output
        # 定义模块 M3，包含一个可量化节点和返回量化字典输出
        class M3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv(x)
                return {"output": x}

        m3 = M3()
        # 调用检查图模式操作函数，验证模型 m3 在量化训练 QAT 下的节点出现情况
        self.checkGraphModeFxOp(
            m3, example_inputs, QuantType.QAT,
            prepare_expected_node_occurrence={
                # one for weights, one for activations
                ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 2,
            },
            expected_node_occurrence={
                ns.call_function(torch.quantize_per_tensor): 1,
            },
            prepare_custom_config=prepare_custom_config_dict)
    def test_deepcopy_preserve_attributes(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = 3  # 设置一个属性 attr 初始值为 3

            def forward(self, x):
                return x

        m = M().eval()  # 创建一个 M 类的实例，并设置为评估模式
        m = prepare_fx(
            m,
            {"": default_qconfig},  # 使用默认的量化配置字典
            example_inputs=(torch.randn(1),),  # 提供一个示例输入
            prepare_custom_config={"preserved_attributes": ["attr"]})  # 设置保留的属性为 ["attr"]
        # preserved attributes are also stored in meta so that it doesn't get lost
        # during deepcopy
        self.assertTrue(hasattr(m, "attr"))  # 断言 m 具有属性 "attr"
        self.assertTrue("attr" in m.meta[_USER_PRESERVED_ATTRIBUTES_KEY])  # 断言 "attr" 在 m 的元数据中的保留属性键中
        m2 = copy.deepcopy(m)  # 深度复制 m，生成 m2
        self.assertTrue(hasattr(m2, "attr"))  # 断言 m2 具有属性 "attr"
        self.assertTrue("attr" in m2.meta[_USER_PRESERVED_ATTRIBUTES_KEY])  # 断言 "attr" 在 m2 的元数据中的保留属性键中
        m = convert_fx(m, convert_custom_config={"preserved_attributes": ["attr"]})  # 将 m 转换为另一种形式，并指定保留属性为 ["attr"]
        self.assertTrue(hasattr(m, "attr"))  # 断言转换后的 m 具有属性 "attr"
        self.assertTrue("attr" in m.meta[_USER_PRESERVED_ATTRIBUTES_KEY])  # 断言 "attr" 在转换后的 m 的元数据中的保留属性键中
        m2 = copy.deepcopy(m)  # 再次深度复制转换后的 m，生成 m2
        self.assertTrue(hasattr(m2, "attr"))  # 断言 m2 具有属性 "attr"
        self.assertTrue("attr" in m2.meta[_USER_PRESERVED_ATTRIBUTES_KEY])  # 断言 "attr" 在 m2 的元数据中的保留属性键中

    def test_output_lists_and_dicts(self):
        """Verify that specifying complicated output types does not crash.
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)  # 创建一个卷积层

            def forward(self, x):
                x = self.conv(x)  # 对输入 x 执行卷积操作
                return {'foo': [x]}, [{'foo': [[x]]}]  # 返回一个字典和一个嵌套列表，其中包含卷积结果

        m = M().eval()  # 创建一个 M 类的实例，并设置为评估模式
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}  # 设置量化配置字典
        mp = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1, 1, 1),))  # 准备模型以便量化
        mc = convert_fx(mp)  # 将准备好的模型转换为量化模型

    def test_shape_followed_by_quantized_op(self):
        """ Make sure that shape does not dequantize
        the Tensor before the next operator
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2)  # 创建第一个卷积层
                self.conv2 = torch.nn.Conv2d(2, 2, 2)  # 创建第二个卷积层

            def forward(self, x):
                x = self.conv1(x)  # 对输入 x 执行第一个卷积操作
                s = x.shape  # 获取卷积结果的形状
                torch._assert(s == x.shape, "")  # 断言卷积结果的形状与其本身形状相同
                x = self.conv2(x)  # 对第一个卷积结果执行第二个卷积操作
                return x  # 返回第二个卷积操作的结果

        # make sure quantization runs
        m = M().eval()  # 创建一个 M 类的实例，并设置为评估模式
        example_inputs = (torch.randn(2, 2, 4, 4),)  # 创建一个示例输入
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)  # 准备模型以便量化
        m = convert_fx(m)  # 将准备好的模型转换为量化模型
        m(*example_inputs)  # 对模型使用示例输入进行调用
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method("dequantize"): 1
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)  # 检查模型中的节点出现情况是否符合预期
    def test_trace_quantize_per_tensor(self):
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 在模型中应用卷积层
                x = self.conv(x)
                return x

        # 创建模型实例并设置为评估模式
        m = M().eval()
        # 准备模型以进行量化，并传入示例输入
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=(torch.randn(1, 1, 3, 3),))
        # 将量化后的模型转换为参考模式
        m = convert_fx(m)
        # 确保转换过程没有错误
        m = torch.fx.Transformer(m).transform()

    def test_linear_qint8_activation(self):
        """Test support for qint8 activation in reference pattern
        """
        # 定义一个包含卷积和线性层的神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 2, 2, 2)
                self.linear = torch.nn.Linear(8, 5)

            def forward(self, x):
                # 在模型中应用卷积层
                x = self.conv(x)
                x = torch.flatten(x, 1)
                # 在模型中应用线性层
                x = self.linear(x)
                return x

        # 创建模型实例并设置为评估模式
        m = M().eval()
        example_inputs = (torch.rand(2, 1, 5, 5),)
        # 准备模型以进行量化，并传入自定义的量化配置和示例输入
        m = prepare_fx(
            m,
            {"": torch.ao.quantization.QConfig(
                activation=torch.ao.quantization.HistogramObserver.with_args(
                    qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
                ), weight=torch.ao.quantization.default_per_channel_weight_observer)},
            example_inputs=example_inputs)
        # 将量化后的模型转换为参考模式
        m = convert_to_reference_fx(m)
        # 在示例输入上运行模型以确保正确性
        m(*example_inputs)

    def test_preserve_tuple(self):
        """ Test tuple input type is preserved
        """

        # 定义一个LSTM模型，输入和状态都是tuple类型
        class LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(50, 50, 1)

            def forward(self, inputs: torch.Tensor, state: List[torch.Tensor]):
                h = state[0]
                c = state[1]
                # 在LSTM模型中应用LSTM层
                return self.lstm(inputs, (h, c))

        # 创建模型实例并设置为评估模式
        m = LSTM().eval()
        example_inputs = (torch.randn(5, 3, 50), torch.randn(2, 3, 50), torch.randn(2, 3, 50))
        # 准备模型以进行量化，并传入默认的量化配置和示例输入
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        # 确保LSTM模型的第二个参数是一个tuple
        for n in m.graph.nodes:
            if n.target == "lstm":
                self.assertEqual(type(n.args[1]), tuple)
    def _test_static_lstm_helper(self, model, prepare_node_occurrence, convert_node_occurrence):
        """
        Helper method to validate the graph of a model with static LSTM.
        """
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping()
        # 准备自定义配置，设置 torch.nn.LSTM 的浮点数映射到观察值
        prepare_custom_config = PrepareCustomConfig() \
            .set_float_to_observed_mapping(torch.nn.LSTM, torch.ao.nn.quantizable.LSTM)
        # 转换自定义配置，设置观察值到量化值映射
        convert_custom_config = ConvertCustomConfig() \
            .set_observed_to_quantized_mapping(torch.ao.nn.quantizable.LSTM, torch.ao.nn.quantized.LSTM)
        # 创建示例输入
        example_inputs = (torch.rand(5, 3, 50), torch.rand(1, 3, 50), torch.randn(1, 3, 50))

        # 准备模型，应用准备阶段的量化配置
        model = prepare_fx(model, qconfig_mapping, example_inputs, prepare_custom_config=prepare_custom_config)
        # 检查图模块节点，验证预期的节点出现情况
        self.checkGraphModuleNodes(model, expected_node_occurrence=prepare_node_occurrence)
        # 对模型进行前向传播计算
        model(*example_inputs)

        # 转换模型，应用转换阶段的量化配置
        model = convert_fx(model, convert_custom_config=convert_custom_config)
        # 再次检查图模块节点，验证预期的节点出现情况
        self.checkGraphModuleNodes(model, expected_node_occurrence=convert_node_occurrence)
        # 再次对模型进行前向传播计算
        model(*example_inputs)

    def test_static_lstm(self):
        """
        Test statically quantized custom module LSTM followed by ops that consume individual
        tensors of the output tuple.
        """
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义包含一个 LSTM 层和三个线性层的自定义模型
                self.lstm = nn.LSTM(50, 50, 1)
                self.linear1 = nn.Linear(50, 10)
                self.linear2 = nn.Linear(50, 10)
                self.linear3 = nn.Linear(50, 10)

            def forward(self, inputs: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
                # LSTM 前向传播，返回输出和隐藏状态
                (out, (h0_out, c0_out)) = self.lstm(inputs, (h0, c0))
                # 分别将 LSTM 输出和隐藏状态作为输入，经过线性层处理
                out = self.linear1(out)
                h0_out = self.linear2(h0_out)
                c0_out = self.linear3(c0_out)
                return (out, (h0_out, c0_out))

        # 创建 MyModel 实例
        m = MyModel()
        # 定义预期的准备阶段节点出现情况字典
        prepare_node_occurrence = {
            ns.call_module(torch.ao.nn.quantizable.LSTM): 1,
        }
        # 定义预期的转换阶段节点出现情况字典
        convert_node_occurrence = {
            ns.call_module(torch.ao.nn.quantized.LSTM): 1,
            ns.call_function(torch.quantize_per_tensor): 3,
            # lstm[0].dequantize()
            # lstm[1][0].dequantize()
            # lstm[1][1].dequantize()
            ns.call_method("dequantize"): 3,
            # lstm[0], lstm[1], lstm[1][0], lstm[1][1]
            ns.call_function(operator.getitem): 4,
            # No tuples are consumed
            ns.call_function(tuple): 0,
        }
        # 调用辅助方法进行测试
        self._test_static_lstm_helper(m, prepare_node_occurrence, convert_node_occurrence)
    def test_static_lstm_consume_tuple(self):
        """
        Test statically quantized custom module LSTM followed by a module that consumes the
        output tuple, either as a whole or part of it.
        """
        # 定义一个测试方法，用于测试静态量化的自定义 LSTM 模块，以及后续消耗输出元组的模块（无论是整体还是部分）

        class ModuleAfterLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.identity = torch.nn.Identity()

            def forward(self, x):
                return self.identity(x)

        # 定义一个继承自 nn.Module 的类 ModuleAfterLSTM，用于在 forward 方法中对输入进行身份转换

        class ConsumeWholeTuple(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(50, 50, 1)
                self.module_after_lstm = ModuleAfterLSTM()

            def forward(self, inputs: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
                x = self.lstm(inputs, (h0, c0))
                x = self.module_after_lstm(x)  # 消耗元组 (output, (hidden0, hidden1))
                return x

        # 定义一个继承自 nn.Module 的类 ConsumeWholeTuple，包含 LSTM 和消耗整个元组的操作

        class ConsumeHiddenTuple(ConsumeWholeTuple):
            def forward(self, inputs: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
                x = self.lstm(inputs, (h0, c0))
                x = self.module_after_lstm(x[1])  # 消耗元组 (hidden0, hidden1)
                return x

        # 定义一个继承自 ConsumeWholeTuple 的类 ConsumeHiddenTuple，只消耗隐藏状态元组 (hidden0, hidden1)

        # Test consuming the whole tuple (output, (hidden0, hidden1))
        # 测试消耗整个元组 (output, (hidden0, hidden1))
        m1 = ConsumeWholeTuple()
        prepare_node_occurrence = {
            ns.call_module(torch.ao.nn.quantizable.LSTM): 1,
        }
        convert_node_occurrence1 = {
            ns.call_module(torch.ao.nn.quantized.LSTM): 1,
            ns.call_function(torch.quantize_per_tensor): 3,
            # lstm[0].dequantize()
            # lstm[1][0].dequantize()
            # lstm[1][1].dequantize()
            ns.call_method("dequantize"): 3,
            # lstm[0], lstm[1], lstm[1][0], lstm[1][1]
            ns.call_function(operator.getitem): 4,
            # tuple(output_dq, tuple(hidden0_dq, hidden1_dq))
            ns.call_function(tuple): 2,
        }
        self._test_static_lstm_helper(m1, prepare_node_occurrence, convert_node_occurrence1)

        # Test consuming just the hidden tuple (hidden0, hidden1)
        # 测试仅消耗隐藏状态元组 (hidden0, hidden1)
        m2 = ConsumeHiddenTuple()
        convert_node_occurrence2 = {
            ns.call_module(torch.ao.nn.quantized.LSTM): 1,
            ns.call_function(torch.quantize_per_tensor): 3,
            # lstm[1][0].dequantize()
            # lstm[1][1].dequantize()
            ns.call_method("dequantize"): 2,
            # lstm[1], lstm[1][0], lstm[1][1]
            ns.call_function(operator.getitem): 3,
            # tuple(hidden0_dq, hidden1_dq)
            ns.call_function(tuple): 1,
        }
        self._test_static_lstm_helper(m2, prepare_node_occurrence, convert_node_occurrence2)
    def test_reroute_tuple_getitem_patterns(self):
        """
        The following graph should redirect the output to `b`. After the transformation,
        all other nodes, including the inputs `a` and `c`, are no longer needed.

             a   b     c
             |   \\   /
             \\   tuple
              \\   /
               tuple
               /  \\
              /    \\
             |      \\
             |       \\
             |        \\
        getitem0    getitem1
             |      /     \\
             | getitem0  getitem1
             |     \\     /
             \\      tuple
              \\      /
               \\    /
                tuple
                  |
               getitem1
                  |
               getitem0
                  |
                output
        """
        # Construct graph manually because symbolic_trace does not insert tuple and getitem nodes
        graph = torch.fx.Graph()
        a = graph.create_node("placeholder", "a")
        b = graph.create_node("placeholder", "b")
        c = graph.create_node("placeholder", "c")

        # Create tuple node with `b` and `c` as inputs
        bc = graph.call_function(tuple, args=([b, c],))

        # Create tuple node with `a` and `bc` as inputs
        abc = graph.call_function(tuple, args=([a, bc],))

        # Break down tuple `abc` and reconstruct it
        a2 = graph.call_function(operator.getitem, args=(abc, 0))
        bc2 = graph.call_function(operator.getitem, args=(abc, 1))
        b2 = graph.call_function(operator.getitem, args=(bc2, 0))
        c2 = graph.call_function(operator.getitem, args=(bc2, 1))

        # Create tuple node with `b2` and `c2` as inputs
        bc3 = graph.call_function(tuple, args=([b2, c2],))

        # Create tuple node with `a2` and `bc3` as inputs
        abc2 = graph.call_function(tuple, args=([a2, bc3],))

        # Get `b3` from `abc2[1][0]`
        bc4 = graph.call_function(operator.getitem, args=(abc2, 1))
        b3 = graph.call_function(operator.getitem, args=(bc4, 0))
        output = graph.output(b3)

        # Perform rerouting transformation
        _reroute_tuple_getitem_pattern(graph)

        # Assert that the output node `output` now directly points to `b` and all other nodes are ancestors of `b`
        output_ancestors = []
        
        def gather_ancestors(current_node):
            for arg in current_node.args:
                output_ancestors.append(arg)
                gather_ancestors(arg)
        
        gather_ancestors(output)
        
        # Ensure that `b` is the only ancestor of `output`
        self.assertEqual(output_ancestors, [b])
        # Ensure that `output` directly points to `b`
        self.assertEqual(output.args[0], b)
    # 定义一个测试函数，用于测试下面的 relu 降级功能
    def test_relu_lowering(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 定义该类的前向传播方法，接受输入 x，应用 ReLU 激活函数后返回结果
            def forward(self, x):
                return torch.nn.functional.relu(x)

        # 创建 M 类的一个实例 m，并设置为评估模式
        m = M().eval()
        # 对 m 进行模型准备，使用默认的量化配置，并提供一个示例输入 torch.randn(1)
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=(torch.randn(1),))
        # 使用深度复制创建 m 的一个副本 m_copy
        m_copy = copy.deepcopy(m)
        # 将 m 转换为量化后的模型
        m = convert_fx(m)
        # 将 m_copy 转换为参考量化后的模型
        m_ref = convert_to_reference_fx(m_copy)
        # 定义节点出现次数的期望字典，其中 torch.quantize_per_tensor 预期出现 1 次，"dequantize" 方法预期出现 1 次
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method("dequantize"): 1
        }
        # 定义参考模型的节点出现次数的期望字典，torch.quantize_per_tensor 和 "dequantize" 预期各出现 2 次
        node_occurrence_ref = {
            ns.call_function(torch.quantize_per_tensor): 2,
            ns.call_method("dequantize"): 2
        }

        # 使用自定义的检查函数 checkGraphModuleNodes 检查 m 的图模块节点是否符合期望的节点出现次数
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        # 使用自定义的检查函数 checkGraphModuleNodes 检查 m_ref 的图模块节点是否符合期望的节点出现次数
        self.checkGraphModuleNodes(m_ref, expected_node_occurrence=node_occurrence_ref)

    # 如果没有安装 FBGEMM，跳过这个测试用例
    @skipIfNoFBGEMM
    def test_dynamic_with_fusion(self):
        """
        Tests that dynamic quantization APIs work with Linear + Relu fusion
        """
        # 使用 'fbgemm' 引擎覆盖当前量化引擎上下文
        with override_quantized_engine('fbgemm'):
            # 定义一个包含 Linear + Relu 融合的模块类
            class LinearRelu(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(5, 5)  # 创建一个线性层
                    self.relu = torch.nn.ReLU()  # 创建一个ReLU激活函数层

                def forward(self, x):
                    x = self.linear(x)  # 线性层的前向传播
                    return self.relu(x)  # ReLU激活函数的前向传播

            # 定义一个包含线性层的模块类
            class Linear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = torch.ones(5, 5)  # 权重初始化为全1
                    self.b = torch.zeros(5)  # 偏置初始化为全0

                def forward(self, x):
                    return torch.nn.functional.linear(x, self.w, self.b)  # 执行线性变换

            # 定义一个包含多个模块的主模块类
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mods1 = torch.nn.Sequential(LinearRelu(), LinearRelu())  # 序列模块包含两个 LinearRelu 实例
                    self.mods2 = Linear()  # 包含一个 Linear 实例
                    self.relu = F.relu  # 引用 torch.nn.functional.relu

                def forward(self, x):
                    x = self.mods1(x)  # 执行第一个序列模块的前向传播
                    x = self.mods2(x)  # 执行第二个线性模块的前向传播
                    x = self.relu(x)  # 执行ReLU激活函数的前向传播
                    return x

            # 定义动态量化操作的字典，包括两个量化配置
            dynamic_quantized_ops = {
                float16_dynamic_qconfig: torch.ops.quantized.linear_relu_dynamic_fp16,
                default_dynamic_qconfig: torch.ops.quantized.linear_relu_dynamic
            }

            # 遍历每个量化配置进行测试
            for qconfig in [float16_dynamic_qconfig, default_dynamic_qconfig]:
                model = M().eval()  # 创建并评估模型实例
                qconfig_dict = {
                    "": qconfig
                }
                example_inputs = (torch.rand(5, 5),)  # 创建一个示例输入
                m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)  # 准备量化模型
                m = convert_fx(m)  # 转换为量化模型
                m(*example_inputs)  # 执行模型的前向传播
                # 定义期望的节点列表
                node_list = [
                    ns.call_module(nniqd.LinearReLU),  # 调用 LinearReLU 模块
                    ns.call_module(nniqd.LinearReLU),  # 再次调用 LinearReLU 模块
                    ns.call_function(dynamic_quantized_ops[qconfig]),  # 调用相应的动态量化操作函数
                ]
                # 检查图模块节点是否符合期望
                self.checkGraphModuleNodes(m, expected_node_list=node_list)

    @skipIfNoFBGEMM
    def test_dynamic_with_fusion_multiple_uses(self):
        """
        Tests that dynamic quantization APIs work with Linear + Relu fusion
        """
        # 使用指定的量化引擎 'fbgemm' 来覆盖默认的量化引擎
        with override_quantized_engine('fbgemm'):
            # 定义一个包含 Linear + Relu 融合的模块类
            class LinearRelu(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 定义一个线性层
                    self.linear = torch.nn.Linear(5, 5)
                    # 定义一个ReLU激活函数
                    self.relu = torch.nn.ReLU()

                def forward(self, x):
                    # 执行线性层操作
                    x = self.linear(x)
                    # 执行ReLU激活函数
                    return self.relu(x)

            # 定义一个包含 LinearRelu 模块的主模块类
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建一个 LinearRelu 实例
                    self.linear_relu = LinearRelu()

                def forward(self, x):
                    # 对输入两次执行 LinearRelu 实例
                    x = self.linear_relu(x)
                    x = self.linear_relu(x)
                    return x

            # 针对每个量化配置进行测试
            for qconfig in [float16_dynamic_qconfig, default_dynamic_qconfig]:
                # 创建并评估模型实例
                model = M().eval()
                # 创建量化配置字典
                qconfig_dict = {
                    "": qconfig
                }
                # 准备函数式模型以进行量化
                example_inputs = (torch.randn(5, 5),)
                m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
                # 将函数式模型转换为量化版本
                m = convert_fx(m)
                # 执行量化模型以捕获量化图形结构
                m(*example_inputs)
                # 构建期望的节点列表，调用模块为 LinearReLU
                node_list = [
                    ns.call_module(nniqd.LinearReLU),
                    ns.call_module(nniqd.LinearReLU),
                ]
                # 检查图模块中的节点是否符合预期
                self.checkGraphModuleNodes(m, expected_node_list=node_list)

    @skipIfNoFBGEMM
    def test_dynamic_linear_input_multiple_use(self):
        """
        Tests input for dynamic linear being used by multiple ops
        """
        # 使用 'fbgemm' 引擎来覆盖量化引擎
        with override_quantized_engine('fbgemm'):
            # 定义一个包含线性层和ReLU激活函数的模块
            class LinearRelu(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(5, 5)
                    self.relu = torch.nn.ReLU()

                def forward(self, x):
                    # 执行线性变换
                    x = self.linear(x)
                    # 应用ReLU激活函数
                    return self.relu(x)

            # 定义一个包含两个LinearRelu模块的主模块
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mod1 = LinearRelu()
                    self.mod2 = LinearRelu()

                def forward(self, x):
                    # 分别对输入x应用两个LinearRelu模块，并返回它们的和
                    y1 = self.mod1(x)
                    y2 = self.mod2(x)
                    return y1 + y2

            # 循环遍历不同的量化配置
            for qconfig in [float16_dynamic_qconfig, default_dynamic_qconfig]:
                # 创建模型M的实例，并设为评估模式
                model = M().eval()
                # 构建量化配置字典
                qconfig_dict = {
                    "": qconfig
                }
                # 创建一个例子输入作为模型准备的输入样本
                example_inputs = (torch.rand(5, 5, 5),)
                # 准备量化模型
                m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
                # 转换为量化后的模型
                m = convert_fx(m)
                # 执行量化后的模型推断
                m(*example_inputs)
                # 定义期望的节点列表，包含两个LinearReLU模块调用
                node_list = [
                    ns.call_module(nniqd.LinearReLU),
                    ns.call_module(nniqd.LinearReLU),
                ]
                # 检查图模块节点是否符合期望
                self.checkGraphModuleNodes(m, expected_node_list=node_list)

    def test_ref_linear_module(self):
        """ Make sure the numerics for models with ref linear module
        matches models with fbgemm/qnnpack module
        """
        # 定义包含线性层的模型M1
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                # 执行线性变换并返回结果
                return self.linear(x)

        # 定义包含线性层和ReLU激活函数的模型M2
        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # 执行线性变换后应用ReLU激活函数，并返回结果
                return self.relu(self.linear(x))

        # 对每个模型M1和M2执行以下操作
        for M in [M1, M2]:
            # 创建模型M的实例，并设为评估模式
            m = M().eval()
            # 创建一个例子输入作为模型准备的输入样本
            example_inputs = (torch.randn(5, 10),)
            # 准备量化模型，使用默认的量化配置
            m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
            # 深度拷贝原始模型m
            m_copy = copy.deepcopy(m)
            # 转换为量化后的模型
            m = convert_fx(m)
            # 转换为参考模型
            m_ref = convert_to_reference_fx(m_copy)
            # 分别对量化后的模型和参考模型执行推断
            result = m(*example_inputs)
            result_ref = m_ref(*example_inputs)
            # 断言量化后的模型和参考模型输出的结果是否一致
            self.assertTrue(torch.equal(result, result_ref))
    def test_ref_conv_module(self):
        """测试具有参考卷积模块的模型数值与具有fbgemm/qnnpack模块的模型数值是否匹配"""
        convs = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }

        class M1(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化一个卷积层，根据给定维度选择不同的维度和参数
                self.conv = convs[dim](3, 3, 3)

            def forward(self, x):
                # 在前向传播中应用卷积层
                return self.conv(x)

        class M2(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化一个卷积层和ReLU激活函数
                self.conv = convs[dim](3, 3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # 在前向传播中依次应用卷积层和ReLU激活函数
                return self.relu(self.conv(x))

        # 使用itertools生成器创建M1和M2的不同维度的组合，并进行评估
        for dim, M in itertools.product([1, 2, 3], [M1, M2]):
            # 实例化模型并设为评估模式
            m = M(dim).eval()
            # 获取示例输入数据
            data = self.img_data_dict[dim][0][0]
            # 准备模型用于量化，并使用示例输入数据
            m = prepare_fx(m, {"": default_qconfig}, example_inputs=(data,))
            # 创建模型的深层副本
            m_copy = copy.deepcopy(m)
            # 转换模型为量化后的版本
            m = convert_fx(m)
            # 将深层副本的模型转换为参考版本
            m_ref = convert_to_reference_fx(m_copy)
            # 应用数据于量化后的模型，并获得结果
            result = m(data)
            # 应用数据于参考版本的模型，并获得结果
            result_ref = m_ref(data)
            # 断言两个结果张量是否完全相等
            self.assertTrue(torch.equal(result, result_ref))

    def test_sub_scalar(self):
        class M(torch.nn.Module):
            def forward(self, x):
                # 实现几个数学操作：加1、减1、加3、减4
                x = x + 1
                x = x - 1
                x = x + 3
                x = x - 4
                return x

        # 实例化模型并设为评估模式
        m = M().eval()
        # 准备模型用于量化，并使用示例输入数据(torch.rand(3))
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=(torch.rand(3),))
        # 转换模型为量化后的版本
        m = convert_fx(m)
        # 期望在图模块中找到的节点及其出现次数
        occurrence = {
            ns.call_function(torch.quantize_per_tensor): 2,
            ns.call_method("dequantize"): 2
        }
        # 检查图模块中的节点出现次数是否与期望匹配
        self.checkGraphModuleNodes(m, expected_node_occurrence=occurrence)
    def test_observer_fqn(self):
        """
        Test to make sure the observer FQN is based on the quantizable op/module that it is observing
        and uses the module's FQN to determine the observer name.
        """
        # 定义一个简单的线性模型
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)  # 定义权重矩阵为全1
                self.b = torch.zeros(5)  # 定义偏置向量为全0

            def forward(self, x):
                # 使用 torch.nn.functional.linear 执行线性操作
                return torch.nn.functional.linear(x, self.w, self.b)

        # 定义一个复合模型 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # mods1 是一个包含两个 Linear 模块的序列
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )
                # mods2 是一个单独的 Linear 模块
                self.mods2 = Linear()
                # mods3 是一个简单的线性层
                self.mods3 = torch.nn.Linear(5, 5)

            def forward(self, x):
                # 对输入 x 应用 mods1 中的模块序列
                x = self.mods1(x)
                # 对结果 x 加上常数 4
                x = torch.add(x, 4)
                # 对加上常数后的结果应用 mods2 中的模块
                x = self.mods2(x)
                # 对结果 x 加上常数 2，赋值给 y
                y = torch.add(x, 2)
                # 对结果 x 乘以常数 5，赋值给 z
                z = torch.mul(x, 5)
                # 对 y 应用 mods3 中的线性层，赋值给 a
                a = self.mods3(y)
                # 返回 mods3 输出的 a 和之前计算得到的 z
                return a, z

        # 创建 M 类的实例，并设置为评估模式
        model = M().eval()

        # 对模型进行量化准备，使用默认的量化配置和随机输入样本
        prepared = prepare_fx(model, {"": default_qconfig}, example_inputs=(torch.randn(1, 5)))

        # 初始化一个空列表，用于存储发现的 MinMaxObserver 的名称
        name_list = []
        # 遍历准备好的模块，并获取每个模块的名称和模块对象
        for name, mod in prepared.named_modules():
            # 检查模块是否为 MinMaxObserver 类型的实例
            if isinstance(mod, torch.ao.quantization.observer.MinMaxObserver):
                # 如果是 MinMaxObserver，将其名称添加到 name_list 中
                name_list.append(name)

        # 预期的 MinMaxObserver 名称列表
        expected_name_list = ['activation_post_process_0',
                              'activation_post_process_1',
                              'activation_post_process_2',
                              'activation_post_process_3',
                              'activation_post_process_4',
                              'activation_post_process_6',
                              'activation_post_process_7',
                              'activation_post_process_10']

        # 断言实际获取的名称列表与预期的名称列表相同
        assert name_list == expected_name_list
    # 定义测试方法，用于测试不同维度的卷积操作
    def test_conv_lowering(self):
        # 创建包含不同维度卷积类的字典
        convs = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        # 创建包含量化后不同维度卷积类的字典
        qconvs = {1: nn.quantized.Conv1d, 2: nn.quantized.Conv2d, 3: nn.quantized.Conv3d}

        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 初始化方法，接收维度参数 dim
            def __init__(self, dim):
                super().__init__()
                # 根据给定的维度创建相应的卷积层
                self.conv = convs[dim](3, 3, 3)

            # 前向传播方法，接收输入 x
            def forward(self, x):
                # 调用卷积层对输入 x 进行处理并返回结果
                return self.conv(x)

        # 遍历 convs 字典中的维度值
        for dim in range(1, len(convs) + 1):
            # 创建 M 类的实例，并设置为评估模式
            m = M(dim).eval()
            # 获取测试数据
            data = self.img_data_dict[dim][0][0]
            # 准备模型以适应量化配置，并传入示例输入数据
            m = prepare_fx(m, {"": default_qconfig}, example_inputs=(data,))
            # 深度复制准备好的模型 m，并转换为参考模型
            m_ref = copy.deepcopy(m)
            m_ref = convert_to_reference_fx(m_ref)
            # 转换模型 m 为量化模型
            m = convert_fx(m)
            # 使用参考模型 m_ref 对输入 data 进行处理并获取输出
            out_ref = m_ref(data)
            # 使用转换后的模型 m 对输入 data 进行处理并获取输出
            out = m(data)
            
            # 检查量化卷积模块的图形节点是否按预期融合
            expected_node_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 1,
                ns.call_module(qconvs[dim]): 1,
                ns.call_method("dequantize"): 1
            }
            self.checkGraphModuleNodes(m, expected_node_occurrence=expected_node_occurrence)
            
            # 检查两个输出张量 out_ref 和 out 是否完全相等
            self.assertTrue(torch.equal(out_ref, out))

    # 辅助方法：断言两个 FixedQParamsFakeQuantize 对象是否相等
    def _assertFixedQParamsFakeQuantizeEqual(self, fq1, fq2):
        self.assertEqual(fq1()._observer_ctr, fq2()._observer_ctr)
    # 定义测试方法，用于验证模式注册功能
    def test_register_patterns(self):
        # 定义清理函数，用于清除注册的模式和映射关系
        def cleanUp():
            # 删除默认融合模式中的假设融合项
            del _DEFAULT_FUSION_PATTERNS["dummy_fusion"]
            # 删除默认量化模式中的假设量化项
            del _DEFAULT_QUANTIZATION_PATTERNS["dummy_quant"]
            del _DEFAULT_QUANTIZATION_PATTERNS["dummy_quant2"]
            del _DEFAULT_QUANTIZATION_PATTERNS["dummy_quant3"]
            # 删除默认输出观察器映射中的假设量化项
            del _DEFAULT_OUTPUT_OBSERVER_MAP["dummy_quant2"]
            del _DEFAULT_OUTPUT_OBSERVER_MAP["dummy_quant3"]
            # 删除默认输出伪量化映射中的假设量化项
            del _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP["dummy_quant2"]
            del _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP["dummy_quant3"]
        
        # 在测试结束时执行清理函数
        self.addCleanup(cleanUp)

        # 注册一个名为 "dummy_fusion" 的融合模式类
        @_register_fusion_pattern("dummy_fusion")
        class DummyFusion:
            pass

        # 注册一个名为 "dummy_quant" 的量化模式类
        @_register_quant_pattern("dummy_quant")
        class DummyQuant:
            pass

        # 注册一个名为 "dummy_quant2" 的量化模式类，并指定默认的观察器
        @_register_quant_pattern("dummy_quant2", default_fixed_qparams_range_0to1_observer)
        class DummyQuant2:
            pass

        # 注册一个名为 "dummy_quant3" 的量化模式类，并指定默认的观察器
        @_register_quant_pattern("dummy_quant3", default_fixed_qparams_range_neg1to1_observer)
        class DummyQuant3:
            pass

        # 验证注册后的类与默认模式映射是否正确
        self.assertEqual(_DEFAULT_FUSION_PATTERNS["dummy_fusion"], DummyFusion)
        self.assertEqual(_DEFAULT_QUANTIZATION_PATTERNS["dummy_quant"], DummyQuant)
        self.assertEqual(_DEFAULT_QUANTIZATION_PATTERNS["dummy_quant2"], DummyQuant2)
        self.assertEqual(_DEFAULT_QUANTIZATION_PATTERNS["dummy_quant3"], DummyQuant3)
        self.assertEqual(_DEFAULT_OUTPUT_OBSERVER_MAP["dummy_quant2"], default_fixed_qparams_range_0to1_observer)
        self.assertEqual(_DEFAULT_OUTPUT_OBSERVER_MAP["dummy_quant3"], default_fixed_qparams_range_neg1to1_observer)
        # 验证伪量化映射是否正确
        self._assertFixedQParamsFakeQuantizeEqual(_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP["dummy_quant2"],
                                                  default_fixed_qparams_range_0to1_fake_quant)
        self._assertFixedQParamsFakeQuantizeEqual(_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP["dummy_quant3"],
                                                  default_fixed_qparams_range_neg1to1_fake_quant)

        # 获取训练状态下的默认输出激活后处理映射和非训练状态下的默认输出激活后处理映射
        output_fake_quantize_map = get_default_output_activation_post_process_map(is_training=True)
        output_observer_map = get_default_output_activation_post_process_map(is_training=False)

        # 验证输出观察器映射中的特定项是否正确
        self.assertEqual(output_observer_map.get("dummy_quant3"), default_fixed_qparams_range_neg1to1_observer)
        # 验证伪量化映射是否正确
        self._assertFixedQParamsFakeQuantizeEqual(output_fake_quantize_map.get("dummy_quant3"),
                                                  default_fixed_qparams_range_neg1to1_fake_quant)
    # 定义一个测试方法，用于验证在不同条件下使用相同输入的量化配置功能
    def test_reuse_input_qconfig(self):
        # 定义模型类 M1，包含一个具有3个输入和3个输出通道的3x3卷积层
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            # 前向传播函数，包括卷积和reshape操作
            def forward(self, x):
                x = self.conv(x)
                x = x.reshape()  # 对输出进行reshape操作
                return x

        # 定义模型类 M2，只包含一个reshape操作的前向传播函数
        class M2(torch.nn.Module):
            def forward(self, x):
                x = x.reshape()  # 对输入进行reshape操作
                return x

        # 使用itertools生成器创建所有可能的模型 M1 或 M2 与 True 或 False 的组合
        options = itertools.product([M1, M2], [True, False])
        
        # 对于每个模型类和是否量化的组合进行迭代
        for M, is_qat in options:
            # 创建一个 M1 模型的实例，并设置为评估模式
            m = M1().eval()
            example_inputs = (torch.randn(1, 3, 3, 3),)  # 创建一个示例输入张量
            # 准备模型以进行量化，使用默认的量化配置映射和示例输入
            m = prepare_fx(m, get_default_qconfig_mapping(), example_inputs=example_inputs)
            m = convert_fx(m)  # 将模型转换为量化模型表示
            # 定义预期的节点列表，包括量化、Conv2d、reshape和反量化操作
            node_list = [
                ns.call_function(torch.quantize_per_tensor),
                ns.call_module(nnq.Conv2d),
                ns.call_method("reshape"),
                ns.call_method("dequantize"),
            ]
            # 检查图模块的节点是否符合预期
            self.checkGraphModuleNodes(
                m,
                expected_node_list=node_list)

            # 创建一个 M2 模型的实例，并设置为评估模式
            m = M2().eval()
            # 准备模型以进行量化，使用默认的量化配置映射和示例输入
            m = prepare_fx(m, get_default_qconfig_mapping(), example_inputs=example_inputs)
            m = convert_fx(m)  # 将模型转换为量化模型表示
            # 定义预期的节点出现次数字典，只期望reshape和反量化操作出现
            node_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 0,
                ns.call_method("dequantize"): 0,
            }
            # 定义预期的节点列表，只包括reshape操作
            node_list = [
                ns.call_method("reshape"),
            ]
            # 检查图模块的节点出现次数和节点列表是否符合预期
            self.checkGraphModuleNodes(
                m,
                expected_node_occurrence=node_occurrence,
                expected_node_list=node_list)
    # 定义一个测试方法，用于验证堆栈跟踪在线性模块中被保留的情况
    def test_stack_trace_preserved_linear(self):
        # 定义一个简单的神经网络模块类
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                # 在前向传播中调用线性层
                x = self.linear(x)
                return x

        # 创建一个模型实例并设置为评估模式
        m = M().eval()
        # 使用 TorchScript 准备模型，并传入默认的量化配置映射和示例输入
        mp = prepare_fx(m, get_default_qconfig_mapping(), example_inputs=(torch.randn(1, 1),))

        # 在 TorchScript 图中查找节点，确认线性层调用时的堆栈跟踪信息存在
        found_stack_trace = False
        for n in mp.graph.nodes:
            if n.op == 'call_module' and n.target == 'linear':
                found_stack_trace = n.stack_trace is not None
                break
        # 使用断言验证找到了堆栈跟踪信息
        self.assertTrue(found_stack_trace)

        # 测试参考模型
        mq = convert_to_reference_fx(copy.deepcopy(mp))
        found_stack_trace = False
        for n in mq.graph.nodes:
            if n.op == 'call_module' and n.target == 'linear':
                found_stack_trace = n.stack_trace is not None
                break
        # 使用断言验证找到了堆栈跟踪信息，如果没有找到，会输出详细的节点信息和模型是否为参考模型的信息
        self.assertTrue(found_stack_trace, f"stack trace not found, node: {n.format_node()}, is_reference: True")

        # 测试量化模型
        mq = convert_fx(mp)
        found_stack_trace = False
        for n in mq.graph.nodes:
            if n.op == 'call_module' and n.target == 'linear':
                found_stack_trace = n.stack_trace is not None
                break
        # 使用断言验证找到了堆栈跟踪信息，如果没有找到，会输出详细的节点信息和模型是否为参考模型的信息
        self.assertTrue(found_stack_trace, f"stack trace not found, node: {n.format_node()}, is_reference: False")
    def test_qat_skip_untraced(self):
        # 定义一个不可追踪的模块类 UnTraceableModuleClass
        class UnTraceableModuleClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        # 定义另一个不可追踪的模块类 UnTraceableModuleName
        class UnTraceableModuleName(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        # 定义一个主模块 M，包含两个不可追踪的模块实例
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.untraceable_module_class = UnTraceableModuleClass()
                self.untraceable_module_name = UnTraceableModuleClass()

            def forward(self, x):
                x = self.untraceable_module_class(x)
                x = self.untraceable_module_name(x)
                return x

        # 创建主模块实例 mod
        mod = M()

        # 定义量化配置字典 qconfig_dict
        qconfig_dict = {"": torch.ao.quantization.get_default_qat_qconfig()}

        # 准备自定义配置字典 prepare_custom_config_dict，指定不可追踪的模块类和名称
        prepare_custom_config_dict = {
            "non_traceable_module_class": [UnTraceableModuleClass],
            "non_traceable_module_name": ["untraceable_module_name"],
        }

        # 创建示例输入 example_inputs
        example_inputs = (torch.randn(2, 2),)

        # 对模型进行量化准备，使用量化感知训练 prepare_qat_fx 方法
        mod_prep = torch.ao.quantization.quantize_fx.prepare_qat_fx(
            mod.train(), qconfig_dict, example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict
        )

        # 再次进行量化准备，用相同的方法，验证是否有变化
        mod_prep = torch.ao.quantization.quantize_fx.prepare_qat_fx(
            mod.train(), qconfig_dict, example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict
        )

        # 断言，确保不可追踪的模块类中的 linear 属性仍为 torch.nn.Linear 类型
        self.assertTrue(
            isinstance(mod_prep.untraceable_module_class.linear, torch.nn.Linear)
        )

        # 断言，确保不可追踪的模块名称中的 linear 属性仍为 torch.nn.Linear 类型
        self.assertTrue(
            isinstance(mod_prep.untraceable_module_name.linear, torch.nn.Linear)
        )

        # 断言，验证不可追踪的模块类中的 linear 属性未被转换为量化感知模块
        self.assertTrue(
            type(mod_prep.untraceable_module_class.linear)
            is not torch.ao.nn.qat.modules.linear.Linear,
            "prepare_qat_fx shold not convert anything inside untraced module classes",
        )

        # 断言，验证模块名称中的 linear 属性未被转换为量化感知模块
        self.assertTrue(
            type(mod_prep.untraceable_module_name.linear)
            is not torch.ao.nn.qat.modules.linear.Linear,
            "prepare_qat_fx shold not convert anything inside modules named in untraced_module_names",
        )
    def test_qconfig_dict_setup(self):
        # 定义一个简单的神经网络模型类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化不同类型的卷积层和线性层
                self.Conv1d = torch.nn.Conv1d(1, 1, 1)
                self.Conv2d = torch.nn.Conv2d(1, 1, 1)
                self.Conv3d = torch.nn.Conv3d(1, 1, 1)
                self.ConvTranspose1d = torch.nn.ConvTranspose1d(1, 1, 1)
                self.ConvTranspose2d = torch.nn.ConvTranspose2d(1, 1, 1)
                self.ConvTranspose3d = torch.nn.ConvTranspose3d(1, 1, 1)
                self.Linear = torch.nn.Linear(1, 1, 1)

            def forward(self, x):
                # 模型的前向传播过程，依次通过各种卷积和线性层
                x = self.Conv1d(x)
                x = self.Conv2d(x)
                x = self.Conv3d(x)
                x = self.ConvTranspose1d(x)
                x = self.ConvTranspose2d(x)
                x = self.ConvTranspose3d(x)
                x = self.Linear(x)
                # 使用函数式 API 进行其他类型的卷积和线性操作
                x = torch.nn.functional.conv1d(x, torch.rand(2, 2))
                x = torch.nn.functional.conv2d(x, torch.rand(2, 2))
                x = torch.nn.functional.conv3d(x, torch.rand(2, 2))
                x = torch.nn.functional.linear(x, torch.rand(2, 2))
                return x

        # 定义后端引擎列表
        backends = ["qnnpack", "fbgemm"]
        # 遍历两种获取量化配置字典的函数
        for func in [get_default_qconfig_mapping, get_default_qat_qconfig_mapping]:
            # 遍历后端引擎列表
            for backend in backends:
                # 创建并评估模型 m
                m = M().eval()
                # 获取当前后端引擎的量化配置字典
                qconfig_dict = func(backend)
                # 准备模型以进行量化仿真，提供示例输入
                m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1, 1, 1)))
                # 遍历模型的所有模块
                for name, mod in m.named_modules():
                    # 检查是否为激活后处理模块且数据类型为 torch.quint8
                    if _is_activation_post_process(mod) and mod.dtype == torch.quint8:
                        # 根据后端引擎设置量化的上下界
                        if backend == "fbgemm":
                            lower_bnd = 0
                            upper_bnd = 127
                        else:
                            lower_bnd = 0
                            upper_bnd = 255
                        # 如果是 FakeQuantize 类的子类，检查量化后处理的量化范围
                        if issubclass(type(mod), FakeQuantize):
                            self.assertEqual(mod.activation_post_process.quant_min, lower_bnd)
                            self.assertEqual(mod.activation_post_process.quant_max, upper_bnd)
                        else:
                            # 否则，检查模块的量化最小和最大值
                            self.assertEqual(mod.quant_min, lower_bnd)
                            self.assertEqual(mod.quant_max, upper_bnd)
    def test_prepare_mode(self):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        def _test(prepare_fn, qconfig_dict):
            m = LinearModel()
            m1 = copy.deepcopy(m)
            m1.train()
            example_inputs = (torch.randn(1, 5),)
            # 准备模型并应用量化配置，测试训练模式
            prepare_fn(m1, qconfig_dict, example_inputs=example_inputs)
            m2 = copy.deepcopy(m)
            m2.eval()
            # 准备模型并应用量化配置，测试评估模式
            prepare_fn(m2, qconfig_dict, example_inputs=example_inputs)

        # 确保 prepare_fx 和 prepare_qat_fx 在训练和评估模式下都能正常工作
        _test(prepare_fx, get_default_qconfig_mapping())
        _test(prepare_qat_fx, get_default_qat_qconfig_mapping())

    def _validate_qconfig_against_backend_config_constraints(
            self,
            model: torch.nn.Module,
            qconfig: QConfig,
            backend_config: BackendConfig,
            satisfies_constraints: bool,
            qconfig_name: Optional[str] = None):
        """
        Helper method to validate whether `qconfig` satisfies the constraints specified in `backend_config`.
        """
        # 设置 QConfig 映射，针对 torch.nn.Linear 类型使用给定的 qconfig
        qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear, qconfig)
        example_inputs = (torch.rand((1, 30), dtype=torch.float),)
        # 准备模型并应用量化配置，使用指定的 backend_config
        model = prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
        model(*example_inputs)
        # 将模型转换为量化模型，使用指定的 backend_config
        model = convert_fx(model, backend_config=backend_config)
        if satisfies_constraints:
            # 如果满足约束条件，设置期望的节点出现次数
            expected_node_occurrence = {
                ns.call_module(torch.ao.nn.quantized.Linear) : 1,
                ns.call_module(torch.nn.Linear) : 0,
            }
        else:
            # 如果不满足约束条件，设置期望的节点出现次数
            expected_node_occurrence = {
                ns.call_module(torch.ao.nn.quantized.Linear) : 0,
                ns.call_module(torch.nn.Linear) : 1,
            }
        try:
            # 检查模型的图结构节点是否符合期望
            self.checkGraphModuleNodes(model, expected_node_occurrence=expected_node_occurrence)
        except AssertionError as e:
            if qconfig_name is not None:
                print(f"ERROR: Validation for QConfig '{qconfig_name}' failed")
            raise e
    def test_qnnpack_backend_config(self):
        """
        Test whether default QNNPACK QConfigs are compatible with the QNNPACK BackendConfig.
        """
        # 定义一个简单的神经网络模型类 MyModel
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个线性层，输入维度为30，输出维度为4，数据类型为 float
                self.linear = torch.nn.Linear(30, 4).float()

            # 前向传播函数
            def forward(self, x):
                return self.linear(x)

        # 所有的 QConfig 列表，每个 QConfig 是一个元组，包含 QConfig 对象和对应的名称字符串
        all_qconfigs: List[Tuple[QConfig, str]] = [
            # 获取 QNNPACK 的默认 QConfig，版本为 0
            (get_default_qconfig("qnnpack", version=0), "default_qnnpack_qconfig_v0"),
            # 获取 QNNPACK 的默认量化训练 QConfig，版本为 0
            (get_default_qat_qconfig("qnnpack", version=0), "default_qat_qnnpack_qconfig_v0"),
            # 获取 QNNPACK 的默认量化训练 QConfig，版本为 1
            (get_default_qat_qconfig("qnnpack", version=1), "default_qat_qnnpack_qconfig_v1"),
            # 获取对称量化的 QNNPACK 默认 QConfig
            (default_symmetric_qnnpack_qconfig, "default_symmetric_qnnpack_qconfig"),
            # 获取对称量化的 QNNPACK 默认量化训练 QConfig
            (default_symmetric_qnnpack_qat_qconfig, "default_symmetric_qnnpack_qat_qconfig"),
            # TODO: 一旦修复，请测试这些 QConfig，参见 https://github.com/pytorch/pytorch/issues/85862
            # (default_per_channel_symmetric_qnnpack_qconfig, "default_per_channel_symmetric_qnnpack_qconfig"),
            # (default_per_channel_symmetric_qnnpack_qat_qconfig, "default_per_channel_symmetric_qnnpack_qat_qconfig"),
        ]
        # 获取 QNNPACK 的后端配置
        backend_config = get_qnnpack_backend_config()
        # 遍历所有的 QConfig，并验证其是否符合后端配置的约束条件
        for qconfig, qconfig_name in all_qconfigs:
            self._validate_qconfig_against_backend_config_constraints(
                MyModel(), qconfig, backend_config, satisfies_constraints=True, qconfig_name=qconfig_name)

    def test_symmetric_qnnpack_qconfig_mapping(self):
        """
        Test whether `torch.ao.quantization.qconfig_mapping._get_symmetric_qnnpack_qconfig_mapping`
        works with the QNNPACK BackendConfig.
        """
        # 如果 QNNPACK 不在支持的量化引擎列表中，则直接返回
        if "qnnpack" not in supported_qengines:
            return

        # 定义一个简单的神经网络模型类 MyModel
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个线性层，输入维度为30，输出维度为4，数据类型为 float
                self.linear = torch.nn.Linear(30, 4).float()

            # 前向传播函数
            def forward(self, x):
                return self.linear(x)

        # 使用 QNNPACK 作为量化引擎的上下文
        with override_quantized_engine("qnnpack"):
            # 获取对称量化 QNNPACK 的 QConfig 映射
            qconfig_mapping = _get_symmetric_qnnpack_qconfig_mapping()
            # 创建一个示例输入
            example_inputs = (torch.rand((1, 30), dtype=torch.float),)
            # 获取 QNNPACK 的后端配置
            backend_config = get_qnnpack_backend_config()
            # 创建模型实例
            model = MyModel()
            # 准备模型以便量化
            model = prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
            # 执行模型推理
            model(*example_inputs)
            # 将模型转换为量化后的模型
            model = convert_fx(model, backend_config=backend_config)
            # 预期的节点出现次数
            expected_node_occurrence = {
                ns.call_module(torch.ao.nn.quantized.Linear) : 1,
                ns.call_module(torch.nn.Linear) : 0,
            }
            # 检查图模块节点
            self.checkGraphModuleNodes(model, expected_node_occurrence=expected_node_occurrence)
            # 再次执行模型推理
            model(*example_inputs)
    def test_symmetric_qnnpack_qat_qconfig_mapping(self):
        """
        Test whether `torch.ao.quantization.qconfig_mapping._get_symmetric_qnnpack_qat_qconfig_mapping`
        works with the QNNPACK BackendConfig.
        """
        # 检查是否支持 QNNPACK 引擎，如果不支持则直接返回
        if "qnnpack" not in supported_qengines:
            return

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个包含输入尺寸为 30 和输出尺寸为 4 的线性层
                self.linear = torch.nn.Linear(30, 4).float()

            def forward(self, x):
                # 在前向传播中应用线性层
                return self.linear(x)

        # 使用 qnnpack 引擎来运行下面的代码块
        with override_quantized_engine("qnnpack"):
            # 获取 QNNPACK QAT（Quantization Aware Training）的对称量化配置映射
            qconfig_mapping = _get_symmetric_qnnpack_qat_qconfig_mapping()
            # 创建一个示例输入
            example_inputs = (torch.rand((1, 30), dtype=torch.float),)
            # 获取 QNNPACK 后端配置
            backend_config = get_qnnpack_backend_config()
            # 创建 MyModel 实例
            model = MyModel()
            # 对模型进行量化准备，使用给定的量化配置映射、示例输入和后端配置
            model = prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
            # 在准备好的量化模型上执行示例输入
            model(*example_inputs)
            # 将准备好的量化模型转换为 QNNPACK 后端配置
            model = convert_fx(model, backend_config=backend_config)
            # 预期节点出现的次数映射
            expected_node_occurrence = {
                ns.call_module(torch.ao.nn.quantized.Linear) : 1,
                ns.call_module(torch.nn.Linear) : 0,
            }
            # 检查图模块节点，验证预期节点出现次数
            self.checkGraphModuleNodes(model, expected_node_occurrence=expected_node_occurrence)
            # 在转换后的模型上再次执行示例输入
            model(*example_inputs)


    def test_get_executorch_backend_config(self):
        from torch.ao.quantization.backend_config import get_executorch_backend_config
        # 确保获取 ExecutorCh 后端配置函数能正常运行
        executorch_backend_config = get_executorch_backend_config()
    def test_backend_config_check_for_weight_and_bias(self):
        """ Test to make sure the backend_config check for weight and bias
        runs when the qconfig is None for the ops with weight and bias
        previously the error was not hit because we first check input, and
        the check for weight and bias are skipped.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.tensor((5, 5))  # 定义模型的权重张量
                self.bias = torch.tensor((5,))  # 定义模型的偏置张量

            def forward(self, x):
                return torch.addmm(self.bias, x, self.weight)  # 在前向传播中使用 torch.addmm 计算

        m = M().eval()  # 创建并评估模型实例
        qconfig_mapping = QConfigMapping()  # 创建 QConfigMapping 实例
        observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT  # 观察类型为输出使用不同观察器作为输入
        weighted_op_quint8_dtype_config = DTypeConfig(
            input_dtype=torch.quint8,
            output_dtype=torch.quint8,
            weight_dtype=torch.qint8,
            bias_dtype=torch.float,
        )  # 定义加权操作的数据类型配置
        dtype_configs = [weighted_op_quint8_dtype_config]  # 创建数据类型配置列表
        backend_pattern_config = BackendPatternConfig(torch.addmm) \
            .set_observation_type(observation_type) \
            .set_dtype_configs(dtype_configs) \
            ._set_input_type_to_index({"weight": 2, "bias": 0})  # 配置后端模式的模式配置

        backend_config = BackendConfig() \
            .set_backend_pattern_config(backend_pattern_config)  # 设置后端配置的模式配置

        example_inputs = (torch.rand(1, 5),)  # 创建示例输入
        # 确保此处运行正常
        m = prepare_fx(m, qconfig_mapping, example_inputs, backend_config=backend_config)  # 准备模型以进行量化
    # 定义一个测试方法，用于测试将模型转换为参考分解的量化函数表示
    def test__convert_to_reference_decomposed_fx(self):
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        # 创建模型实例，并设为评估模式
        m = M().eval()
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping("fbgemm")
        # 准备模型以便量化
        example_inputs = (torch.randn(1, 5),)
        m = prepare_fx(m, qconfig_mapping, example_inputs)
        # 深拷贝一个模型实例作为参考
        m_ref = copy.deepcopy(m)
        # 将参考模型转换为参考的量化函数表示
        m_ref = convert_to_reference_fx(m_ref)
        # 将原模型转换为参考分解的量化函数表示
        m = _convert_to_reference_decomposed_fx(m)
        # 期望的操作出现次数字典
        expected_occurrence = {
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 2,
        }
        # 检查模型节点以确保预期的操作出现次数
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence)
        # 确保模型运行正常
        res_ref = m_ref(*example_inputs)
        res = m(*example_inputs)
        # 断言结果是否相等
        self.assertEqual(res, res_ref)

    # 如果没有 QNNPACK，跳过此测试
    @skipIfNoQNNPACK
    def test__convert_to_reference_decomposed_fx_dynamic_quant(self):
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        # 设置量化引擎为 QNNPACK
        with override_quantized_engine("qnnpack"):
            # 创建模型实例，并设为评估模式
            m = M().eval()
            # 获取默认的量化配置映射，并设置动态量化配置
            qconfig_mapping = get_default_qconfig_mapping("fbgemm") \
                .set_object_type(torch.nn.Linear, default_dynamic_qconfig)
            # 准备模型以便量化
            example_inputs = (torch.randn(1, 5),)
            m = prepare_fx(m, qconfig_mapping, example_inputs)
            # 执行模型前向传播
            m(*example_inputs)
            # 深拷贝一个模型实例作为参考
            m_ref = copy.deepcopy(m)
            # 将参考模型转换为参考的量化函数表示
            m_ref = convert_to_reference_fx(m_ref)
            # 将原模型转换为参考分解的量化函数表示
            m = _convert_to_reference_decomposed_fx(m)
            # 期望的操作出现次数字典
            expected_occurrence = {
                ns.call_function(torch.ops.quantized_decomposed.choose_qparams.tensor): 1,
                ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.tensor): 1,
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.tensor): 1,
            }
            # 检查模型节点以确保预期的操作出现次数
            self.checkGraphModuleNodes(
                m,
                expected_node_occurrence=expected_occurrence)
            # 确保模型运行正常
            res_ref = m_ref(*example_inputs)
            res = m(*example_inputs)
            # 断言结果是否相等
            self.assertEqual(res, res_ref)
    def test__convert_to_reference_decomposed_fx_per_channel_quant(self):
        class M(torch.nn.Module):
            def forward(self, x, weight, bias):
                return F.linear(x, weight, bias)
        
        m = M().eval()  # 创建一个示例模型 M 并将其设置为评估模式
        
        # 获取默认的量化配置映射，设置对象类型为 F.linear，并使用默认的按通道量化配置
        qconfig_mapping = get_default_qconfig_mapping("fbgemm") \
            .set_object_type(F.linear, default_per_channel_qconfig)
        
        # 准备示例输入数据
        example_inputs = (torch.randn(1, 5), torch.randn(10, 5), torch.randn(10,))
        
        # 对模型 m 进行量化准备，使用上面配置的量化映射和示例输入数据
        m = prepare_fx(m, qconfig_mapping, example_inputs)
        
        # 运行模型 m，以确保一切正常
        m(*example_inputs)
        
        # 深度复制模型 m，并将其作为参考模型 m_ref
        m_ref = copy.deepcopy(m)
        
        # 将 m_ref 转换为参考推理模型（reference FX）
        m_ref = convert_to_reference_fx(m_ref)
        
        # 将模型 m 转换为分解的参考 FX
        m = _convert_to_reference_decomposed_fx(m)
        
        # 期望的节点出现次数字典，用于检查图模块中的节点
        expected_occurrence = {
            # 输入和输出激活的量化和反量化
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 2,
            # 权重的量化和反量化
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel.default): 1,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default): 1,
        }
        
        # 检查模型 m 的节点出现次数是否符合预期
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence)
        
        # 确保 m_ref 模型能够成功运行
        res_ref = m_ref(*example_inputs)
        
        # 确保 m 模型能够成功运行，并且其输出与 m_ref 一致
        res = m(*example_inputs)
        self.assertEqual(res, res_ref)

    def test_change_backend_config_for_fixed_qparam_ops(self):
        """ Making sure we can skip validation of qconfigs for fixedqparam ops based
        on BackendConfig
        """
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.tanh = torch.nn.Tanh()

            def forward(self, x: torch.Tensor):
                x = self.tanh(x)
                return x
        
        model = M().eval()  # 创建一个示例模型 M 并将其设置为评估模式
        
        # 创建一个空的 QConfigMapping 对象
        qconfig_mapping = QConfigMapping().set_global(default_qconfig)
        
        # 创建一个空的 BackendConfig 对象
        backend_config = BackendConfig()
        
        # 准备模型，使用上述设置的 qconfig_mapping 和 backend_config，以及示例输入数据
        model = prepare_fx(
            model,
            qconfig_mapping=qconfig_mapping,
            example_inputs=(torch.randn(1, 2, 3, 4),),
            backend_config=backend_config
        )
    def test_channel_shuffle_lowering(self):
        # Three versions of channel shuffle
        
        # 定义一个继承自 torch.nn.Module 的类 M1，用于测试通道混洗操作
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化时使用 torch.nn.ChannelShuffle 创建通道混洗操作对象
                self.op = torch.nn.ChannelShuffle(2)

            def forward(self, x):
                # 对输入 x 执行通道混洗操作，并返回操作后的结果
                return self.op(x + x) + x

        # 定义一个继承自 torch.nn.Module 的类 M2，测试 torch.channel_shuffle 函数
        class M2(torch.nn.Module):
            def forward(self, x):
                # 使用 torch.channel_shuffle 执行通道混洗操作，并返回结果
                return torch.channel_shuffle(x + x, 2) + x

        # 定义一个继承自 torch.nn.Module 的类 M3，测试 torch.nn.functional.channel_shuffle 函数
        class M3(torch.nn.Module):
            def forward(self, x):
                # 使用 torch.nn.functional.channel_shuffle 执行通道混洗操作，并返回结果
                return torch.nn.functional.channel_shuffle(x + x, 2) + x

        # 创建一个大小为 (4, 4, 4, 4) 的随机张量 x 作为示例输入
        x = torch.randn(4, 4, 4, 4)
        
        # torch.channel_shuffle 等价于 torch.nn.functional.channel_shuffle
        # 准备模型和节点对应关系列表
        model_node_pairs = [
            # 创建 M1 的实例，并使用 ns.call_module 调用 torch.nn.ChannelShuffle 函数
            (M1().eval(), ns.call_module(torch.nn.ChannelShuffle)),
            # 创建 M2 的实例，并使用 ns.call_function 调用 torch.channel_shuffle 函数
            (M2().eval(), ns.call_function(torch.channel_shuffle)),
            # 创建 M3 的实例，并使用 ns.call_function 调用 torch.channel_shuffle 函数
            (M3().eval(), ns.call_function(torch.channel_shuffle))
        ]
        
        # 遍历模型和节点对应关系列表
        for m, node in model_node_pairs:
            # 使用 prepare_fx 函数准备模型 m
            m = prepare_fx(m, {"": default_qconfig}, example_inputs=(x,))
            # 深度复制模型 m，并保存在 m_copy 中
            m_copy = copy.deepcopy(m)
            # 转换模型 m
            m = convert_fx(m)
            # 将 m_copy 转换为参考模型 m_ref
            m_ref = convert_to_reference_fx(m_copy)
            
            # 定义预期节点出现次数的字典 node_occurrence
            node_occurrence = {
                node: 1,
                ns.call_function(torch.quantize_per_tensor): 1,
                ns.call_method("dequantize"): 1
            }
            
            # 定义参考模型的预期节点出现次数的字典 node_occurrence_ref
            node_occurrence_ref = {
                node: 1,
                ns.call_function(torch.quantize_per_tensor): 4,
                ns.call_method("dequantize"): 4
            }
            
            # 调用 self.checkGraphModuleNodes 检查模型 m 的节点出现次数是否符合预期
            self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
            # 调用 self.checkGraphModuleNodes 检查参考模型 m_ref 的节点出现次数是否符合预期
            self.checkGraphModuleNodes(m_ref, expected_node_occurrence=node_occurrence_ref)
    def test_match_pattern_with_multiple_args(self):
        """ 测试能够匹配具有多个参数的模式
        模式:
                           shape \
        transpose (observed) -> reshape -> output (observed) ->

        其中 `reshape` 有两个参数
        """

        def _get_pattern_configs():
            # 初始化后端模式配置列表
            backend_pattern_configs = []
            # 观察类型设为输出共享观察器与输入
            observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
            # 创建权重操作的量化配置
            weighted_op_quint8_dtype_config = DTypeConfig(
                input_dtype=torch.quint8,
                output_dtype=torch.quint8,
                weight_dtype=torch.qint8,
                bias_dtype=torch.float,
            )
            # 将权重操作的量化配置添加到配置列表中
            dtype_configs = [weighted_op_quint8_dtype_config]

            def root_node_getter(node_pattern):
                # 从节点模式中获取根节点，这里选择第二个节点 transpose
                reshape, transpose, shape = node_pattern
                return transpose

            # 创建后端模式配置对象并设置属性
            backend_pattern_configs.append(
                BackendPatternConfig()
                ._set_pattern_complex_format((torch.reshape, torch.transpose, MatchAllNode))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(dtype_configs)
                ._set_root_node_getter(root_node_getter)
            )
            # 返回后端模式配置列表
            return backend_pattern_configs

        # 创建后端配置对象，并设置后端模式配置
        backend_config = BackendConfig().set_backend_pattern_configs(_get_pattern_configs())

        # 定义一个继承自 torch.nn.Module 的子类 M
        class M(torch.nn.Module):
            def forward(self, x):
                # 转置输入张量 x 的维度 0 和 1
                x = torch.transpose(x, 0, 1)
                # 将张量 x 重塑为一维
                x = torch.reshape(x, (-1,))
                return x

        # 创建 M 类的实例，并设置为评估模式
        m = M().eval()
        # 创建量化配置映射对象，并设置默认量化配置
        qconfig_mapping = QConfigMapping().set_global(default_qconfig)
        # 创建示例输入张量
        example_inputs = (torch.randn(1, 3, 3, 3),)
        # 准备模型以适应量化，使用给定的后端配置
        m = prepare_fx(m, qconfig_mapping, example_inputs, backend_config=backend_config)
        # 预期的节点出现次数字典，包括输入模式和输出模式的节点各一个
        node_occurrence = {
            ns.call_module(MinMaxObserver): 2
        }
        # 调用 self.checkGraphModuleNodes 方法，检查模型的节点出现次数是否与预期相符
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
    `
        # 定义测试函数 _test_linear_activation_fusion_lowering_helper，用于测试线性激活融合降低辅助函数
        def _test_linear_activation_fusion_lowering_helper(
                self, module, example_inputs, qconfig_mapping,
                backend_config, fused_module, root_module, activation_module):
            # 定义节点出现次数字典，用于检查图模块节点
            node_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 1,
                ns.call_method("dequantize"): 1,
                ns.call_module(fused_module): 1,
                ns.call_module(root_module): 0,
                ns.call_module(activation_module): 0,
            }
            # 参考节点出现次数字典
            node_occurrence_ref = {
                ns.call_function(torch.quantize_per_tensor): 2,
                ns.call_method("dequantize"): 2,
            }
            # 将模块设置为评估模式
            m = module.eval()
            # 准备 FX 模块，包括量化配置映射、示例输入和后端配置
            m = prepare_fx(m, qconfig_mapping,
                           example_inputs=example_inputs,
                           backend_config=backend_config)
            # 深拷贝模块
            m_copy = copy.deepcopy(m)
            # 将 FX 模块转换为量化表示
            m = convert_fx(m, backend_config=backend_config)
            # 将深拷贝的 FX 模块转换为参考 FX 模块
            m_ref = convert_to_reference_fx(m_copy)
    
            # 检查 FX 模块的节点是否符合预期
            self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
            # 检查参考 FX 模块的节点是否符合预期
            self.checkGraphModuleNodes(m_ref, expected_node_occurrence=node_occurrence_ref)
            # 执行模块的前向传播
            m(*example_inputs)
    
        # 使用 skipIfNoONEDNN 装饰器标记的测试函数，测试线性 LeakyReLU 降低
        @skipIfNoONEDNN
        def test_linear_leaky_relu_lowering(self):
            """ Test fusion and lowering of Linear - (bn -) LeakyReLU
                by FX. For onednn backedn only.
            """
            # 导入获取 onednn 后端配置的函数
            from torch.ao.quantization.backend_config import get_onednn_backend_config
            # 获取默认的量化配置映射
            qconfig_mapping = get_default_qconfig_mapping('onednn')
            # 使用 override_quantized_engine 函数切换到 onednn 引擎上下文
            with override_quantized_engine('onednn'):
                # 遍历是否包含批量归一化的情况
                for with_bn in [True, False]:
                    # 创建 LinearBnLeakyReluModel 模型实例
                    m = LinearBnLeakyReluModel(with_bn)
                    # 调用 _test_linear_activation_fusion_lowering_helper 辅助函数进行测试
                    self._test_linear_activation_fusion_lowering_helper(
                        m,
                        m.get_example_inputs(),
                        qconfig_mapping,
                        get_onednn_backend_config(),
                        nniq.LinearLeakyReLU,
                        nn.Linear,
                        nn.LeakyReLU)
    
        # 使用 skipIfNoONEDNN 装饰器标记的测试函数
        @skipIfNoONEDNN
    def test_linear_size_view(self):
        """
        Test function for evaluating the behavior of a model with linear layers and optional ReLU activation,
        focusing on size view transformations in the forward pass.

        This function tests the quantization and node occurrence in the prepared and converted models.

        The model `M` is defined with a linear layer of input size 16 and output size 32, followed by an optional
        ReLU activation. The forward pass includes quantization, preparation, and conversion steps to simulate
        the behavior under quantized conditions.

        For each scenario of `use_relu` being False and True, the following steps are performed:
        - Create an instance of `M` with the specified `use_relu` flag.
        - Evaluate the model on randomly generated input `x`.
        - Prepare the model for quantization using `prepare_fx`.
        - Execute the prepared model on input `x`.
        - Convert the prepared model to its quantized version.
        - Define expected node occurrences for quantized models using dictionaries.

        The `checkGraphModuleNodes` method verifies the occurrence of nodes in the quantized model against
        expected values, validating the correctness of the quantized model structure.

        Args:
            self: The instance of the test class containing this method.

        Returns:
            None
        """
        class M(torch.nn.Module):
            def __init__(self, use_relu=False):
                super().__init__()
                self.linear = torch.nn.Linear(16, 32)
                self.relu = torch.nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                x = self.linear(x)
                if self.use_relu:
                    x = self.relu(x)
                return x.view(x.size(0), 1, 4, 8)

        # Test both scenarios: without ReLU and with ReLU
        for use_relu in [False, True]:
            # Create the model instance with the current `use_relu` setting
            model_fp32 = M(use_relu).eval()

            # Determine the quantization engine in use
            qengine = torch.backends.quantized.engine

            # Retrieve default qconfig mapping based on the quantization engine
            qconfig_mapping = get_default_qconfig_mapping(qengine)

            # Generate random input tensor `x`
            x = torch.randn((5, 16))

            # Evaluate the FP32 model on input `x`
            model_fp32(x)

            # Prepare the model for quantization with the specified `qconfig_mapping`
            prepared_model = prepare_fx(model_fp32, qconfig_mapping, x)

            # Execute the prepared model on input `x`
            prepared_model(x)

            # Convert the prepared model to its quantized version
            quantized_model = convert_fx(prepared_model)

            # Define expected node occurrences in the quantized model
            node_occurrence = {
                ns.call_module(nnq.Linear): 0 if use_relu else 1,
                ns.call_module(nniq.LinearReLU): 1 if use_relu else 0,
                ns.call_function(torch.quantize_per_tensor): 1,
                ns.call_method("dequantize"): 1
            }

            # Verify the occurrence of nodes in the quantized model matches expected values
            self.checkGraphModuleNodes(quantized_model, expected_node_occurrence=node_occurrence)
    def test_linear_shape_view(self):
        # 定义一个测试函数，用于测试线性模型的形状视图
        class M(torch.nn.Module):
            def __init__(self, use_relu=False):
                super().__init__()
                # 创建一个线性层，输入维度为16，输出维度为32
                self.linear = torch.nn.Linear(16, 32)
                # 创建一个ReLU激活函数层
                self.relu = torch.nn.ReLU()
                # 是否使用ReLU激活函数的标志
                self.use_relu = use_relu

            def forward(self, x):
                # 模型的前向传播过程
                x = self.linear(x)
                if self.use_relu:
                    x = self.relu(x)
                # 返回调整后的张量形状，变为(batch_size, 1, 4, 8)
                return x.view(x.shape[0], 1, 4, 8)

        # 循环测试两种情况：use_relu为False和True
        for use_relu in [False, True]:
            # 创建一个eval模式下的M类实例
            model_fp32 = M(use_relu).eval()
            # 获取当前量化引擎
            qengine = torch.backends.quantized.engine
            # 获取默认的量化配置映射
            qconfig_mapping = get_default_qconfig_mapping(qengine)
            # 创建一个形状为(5, 16)的随机张量作为输入
            x = torch.randn((5, 16))
            # 执行模型的前向传播
            model_fp32(x)
            # 准备将模型转换为量化模型
            prepared_model = prepare_fx(model_fp32, qconfig_mapping, x)
            # 再次执行模型的前向传播
            prepared_model(x)
            # 将准备好的模型转换为量化模型
            quantized_model = convert_fx(prepared_model)
            # 定义预期的节点出现次数字典
            node_occurrence = {
                # 对于nnq.Linear模块的调用次数，如果use_relu为False则为0，否则为1
                ns.call_module(nnq.Linear): 0 if use_relu else 1,
                # 对于nniq.LinearReLU模块的调用次数，如果use_relu为True则为1，否则为0
                ns.call_module(nniq.LinearReLU): 1 if use_relu else 0,
                # 对torch.quantize_per_tensor函数的调用次数为1
                ns.call_function(torch.quantize_per_tensor): 1,
                # 对"dequantize"方法的调用次数为1
                ns.call_method("dequantize"): 1
            }
            # 调用self对象的方法，检查GraphModule的节点出现次数是否符合预期
            self.checkGraphModuleNodes(quantized_model, expected_node_occurrence=node_occurrence)
    def test_lowering_functional_conv_with_kwargs(self):
        # 定义不同维度下的普通卷积操作函数的映射
        dim_to_op = {
            1: F.conv1d,
            2: F.conv2d,
            3: F.conv3d,
        }
        # 定义不同维度下的量化卷积操作函数的映射
        dim_to_qop = {
            1: torch.ops.quantized.conv1d,
            2: torch.ops.quantized.conv2d,
            3: torch.ops.quantized.conv3d,
        }

        class Mod(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dimension):
                super().__init__()
                self.dim = dimension
                self.op = dim_to_op[dimension]  # 根据维度选择相应的卷积操作函数
                kernel_sizes = [kernel_size] * self.dim
                self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_sizes))

            def forward(self, input):
                # 调用选定的卷积操作函数进行前向传播
                return self.op(input, self.weight, bias=None, stride=[1] * self.dim,
                               padding=[0] * self.dim, dilation=[1] * self.dim, groups=1)

        for dimension in [1, 2, 3]:
            # 创建模型实例并设置为评估模式
            model = Mod(3, 16, 3, dimension)
            model.eval()
            # 获取默认的量化配置映射
            qconfig_mapping = get_default_qconfig_mapping()
            input_shape = (1, 3, *([8] * dimension))
            example_inputs = torch.randn(input_shape)
            # 准备模型以便量化
            prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
            # 使用示例输入执行准备好的模型
            prepared_model(example_inputs)
            # 将准备好的模型转换为量化模型
            quantized_model = convert_fx(prepared_model)
            # 确保量化模型中包含预期的量化操作
            node_occurrence = {
                ns.call_function(dim_to_qop[dimension]): 1,
            }
            # 检查量化模型的图模块节点
            self.checkGraphModuleNodes(quantized_model, expected_node_occurrence=node_occurrence)
    def test_lowering_functional_conv_transpose_with_kwargs(self):
        # 创建一个映射，将维度与对应的反卷积函数关联起来
        dim_to_op = {
            1: F.conv_transpose1d,
            2: F.conv_transpose2d,
            3: F.conv_transpose3d,
        }
        # 创建一个映射，将维度与对应的量化反卷积函数关联起来
        dim_to_qop = {
            1: torch.ops.quantized.conv_transpose1d,
            2: torch.ops.quantized.conv_transpose2d,
            3: torch.ops.quantized.conv_transpose3d,
        }

        class Mod(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dimension):
                super().__init__()
                self.dim = dimension
                # 根据维度选择合适的反卷积函数
                self.op = dim_to_op[dimension]
                kernel_sizes = [kernel_size] * self.dim
                # 创建一个随机初始化的权重参数
                self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *kernel_sizes))

            def forward(self, input):
                # 执行反卷积操作，使用设定的参数和默认选项
                return self.op(input, self.weight, bias=None, stride=[1] * self.dim,
                               padding=[0] * self.dim, output_padding=[0] * self.dim,
                               dilation=[1] * self.dim, groups=1)

        # 遍历维度 [1, 2, 3]，分别测试模型
        for dimension in [1, 2, 3]:
            model = Mod(3, 16, 3, dimension)
            model.eval()
            qconfig_mapping = get_default_qconfig_mapping()
            input_shape = (1, 3, *([8] * dimension))
            example_inputs = torch.randn(input_shape)
            # 准备模型进行量化仿真
            prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
            prepared_model(example_inputs)
            # 将准备好的模型转换为量化模型
            quantized_model = convert_fx(prepared_model)
            # 运行量化模型以验证通过
            quantized_model(example_inputs)
            # 确保量化模型中包含预期的操作
            node_occurrence = {
                ns.call_function(dim_to_qop[dimension]): 1,
            }
            # 检查图模块中的节点是否符合预期
            self.checkGraphModuleNodes(quantized_model, expected_node_occurrence=node_occurrence)

    def test_lowering_functional_linear_with_kwargs(self):
        class Mod(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                # 创建一个随机初始化的线性层权重参数
                self.weight = nn.Parameter(torch.randn(out_channels, in_channels))

            def forward(self, input):
                # 执行线性层操作，使用设定的参数和默认选项
                return F.linear(input, self.weight, bias=None)

        # 创建线性模型实例
        model = Mod(8, 4)
        model.eval()
        qconfig_mapping = get_default_qconfig_mapping()
        example_inputs = torch.randn(1, 8)
        # 准备模型进行量化仿真
        prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
        prepared_model(example_inputs)
        # 将准备好的模型转换为量化模型
        quantized_model = convert_fx(prepared_model)
        # 运行量化模型以验证通过
        quantized_model(example_inputs)
        # 确保量化模型中包含预期的操作
        node_occurrence = {
            ns.call_function(torch.ops.quantized.linear): 1,
        }
        # 检查图模块中的节点是否符合预期
        self.checkGraphModuleNodes(quantized_model, expected_node_occurrence=node_occurrence)
# 如果没有支持FBGEMM，则跳过该测试类
@skipIfNoFBGEMM
# 定义测试量化操作的测试类，继承自QuantizationTestCase
class TestQuantizeFxOps(QuantizationTestCase):
    # 在每个测试函数运行前执行的初始化操作
    def setUp(self):
        # 调用父类的setUp方法进行基本的测试环境设置
        super().setUp()
        # 自定义量化配置，包括激活值的直方图观察者和权重的默认通道量化观察者
        self.custom_qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.observer.HistogramObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
            ),
            weight=torch.ao.quantization.default_per_channel_weight_observer
        )
        # 常见量化模式的模式匹配字典，将不同的神经网络层映射到默认的量化处理程序
        self.common_quant_patterns = {
            torch.nn.ConvTranspose1d: DefaultNodeQuantizeHandler,
            torch.nn.ConvTranspose2d: DefaultNodeQuantizeHandler,
            torch.nn.ELU: DefaultNodeQuantizeHandler,
            torch.nn.LeakyReLU: DefaultNodeQuantizeHandler,
            torch.nn.Hardswish: DefaultNodeQuantizeHandler,
            torch.nn.InstanceNorm1d: DefaultNodeQuantizeHandler,
            torch.nn.InstanceNorm2d: DefaultNodeQuantizeHandler,
            torch.nn.InstanceNorm3d: DefaultNodeQuantizeHandler,
            torch.nn.LayerNorm: DefaultNodeQuantizeHandler,
            torch.nn.SiLU: DefaultNodeQuantizeHandler,
            torch.nn.Mish: DefaultNodeQuantizeHandler,
            torch.nn.GELU: DefaultNodeQuantizeHandler,
            torch.nn.Softmax: DefaultNodeQuantizeHandler,
            torch.nn.functional.elu: DefaultNodeQuantizeHandler,
            torch.nn.functional.hardswish: DefaultNodeQuantizeHandler,
            torch.nn.functional.instance_norm: DefaultNodeQuantizeHandler,
            torch.nn.functional.layer_norm: DefaultNodeQuantizeHandler,
            torch.nn.functional.leaky_relu: DefaultNodeQuantizeHandler,
            torch.nn.functional.silu: DefaultNodeQuantizeHandler,
            torch.nn.functional.mish: DefaultNodeQuantizeHandler,
            torch.nn.functional.gelu: DefaultNodeQuantizeHandler,
            torch.nn.functional.softmax: DefaultNodeQuantizeHandler,
            torch.sum: DefaultNodeQuantizeHandler
        }

    """Unit tests for individual ops
    """
    # 如果没有支持FBGEMM，则跳过该测试函数
    @skipIfNoFBGEMM
    # 定义一个测试线性模块的测试方法
    def test_linear_module(self):
        # 使用 'fbgemm' 引擎覆盖量化引擎上下文
        with override_quantized_engine('fbgemm'):
            # 定义一个简单的线性模型类
            class LinearModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建一个输入维度为 30，输出维度为 4 的线性层
                    self.linear = torch.nn.Linear(30, 4).float()

                def forward(self, x):
                    # 前向传播方法，返回线性层的输出
                    return self.linear(x)

            # 定义一个包含 ReLU 激活的线性模型类
            class LinearReLUModel(torch.nn.Module):
                def __init__(self, f_relu=False):
                    super().__init__()
                    # 创建一个输入维度为 30，输出维度为 4 的线性层
                    self.linear = torch.nn.Linear(30, 4).float()
                    # 根据 f_relu 参数选择使用 F.relu 函数还是 torch.nn.ReLU 类
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()

                def forward(self, x):
                    x = self.linear(x)
                    # 对线性层的输出应用激活函数（ReLU）
                    x = self.relu(x)
                    return x

            # 定义一个包含 Batch Normalization 的线性模型类
            class LinearBnModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建一个输入维度和输出维度都为 4 的线性层
                    self.linear = torch.nn.Linear(4, 4).float()
                    # 创建一个 Batch Normalization 层，输入维度为 4
                    self.bn = torch.nn.BatchNorm1d(4)

                def forward(self, x):
                    x = self.linear(x)
                    # 对线性层的输出应用 Batch Normalization
                    x = self.bn(x)
                    return x

            # 测试线性模型
            data = (torch.rand((1, 30), dtype=torch.float),)
            # 遍历所有量化类型
            for quant_type in self.all_quant_types:
                model = LinearModel()
                # 根据量化类型选择对应的量化线性层模块
                quantized_module = nnqd.Linear if quant_type == QuantType.DYNAMIC else nnq.Linear
                quantized_node = ns.call_module(quantized_module)
                # 检查图模式下的操作，验证量化输出是否等于参考量化输出
                result_dict = self.checkGraphModeFxOp(model, data, quant_type, quantized_node)
                if quant_type in self.static_quant_types:
                    self.assertEqual(result_dict["quantized_output"], result_dict["quantized_reference_output"])

            # TODO: enable test for dynamic quant
            # 测试带有 ReLU 的线性模型
            for f_relu, quant_type in itertools.product([True, False], [QuantType.STATIC, QuantType.QAT]):
                model = LinearReLUModel(f_relu)
                quantized_node = ns.call_module(nniq.LinearReLU)
                # 检查图模式下的操作，验证量化输出是否等于参考量化输出
                result_dict = self.checkGraphModeFxOp(model, data, quant_type, quantized_node)
                self.assertEqual(result_dict["quantized_output"], result_dict["quantized_reference_output"])

            # 测试带有 Batch Normalization 的线性模型
            data = (torch.rand((4, 4), dtype=torch.float),)
            # 遍历所有静态量化类型
            for quant_type in self.static_quant_types:
                model = LinearBnModel()
                quantized_node = ns.call_module(nnq.Linear)
                # 检查图模式下的操作，验证量化输出是否等于参考量化输出
                result_dict = self.checkGraphModeFxOp(model, data, quant_type, quantized_node)
                self.assertEqual(result_dict["quantized_output"], result_dict["quantized_reference_output"])
    `
        # 测试线性层的动态量化，并使用半精度浮点数（FP16），包括量化引擎的切换
        def test_linear_dynamic_fp16(self):
            # 覆盖量化引擎为 'fbgemm'
            with override_quantized_engine('fbgemm'):
                # 定义一个线性层的自定义模块类
                class FuncLinear(torch.nn.Module):
                    def __init__(self, use_bias, has_relu, f_relu):
                        # 初始化父类
                        super().__init__()
                        # 初始化权重矩阵 w，形状为 (4, 30)
                        self.w = torch.randn(4, 30)
                        # 初始化偏置 b，形状为 (4,)
                        self.b = torch.randn(4)
                        # 存储是否使用偏置项
                        self.use_bias = use_bias
                        # 根据是否使用 ReLU 激活函数和函数类型初始化激活层
                        if has_relu:
                            if f_relu:
                                self.relu = F.relu  # 使用函数式 ReLU
                            else:
                                self.relu = torch.nn.ReLU()  # 使用 torch.nn.ReLU()
                        else:
                            self.relu = torch.nn.Identity()  # 不使用激活函数
    
                    def forward(self, x):
                        # 根据是否使用偏置项执行线性变换
                        if self.use_bias:
                            x = F.linear(x, self.w, self.b)
                        else:
                            x = F.linear(x, self.w)
                        # 应用激活函数
                        x = self.relu(x)
                        return x
    
                # 准备输入数据，形状为 (1, 30)，数据类型为 float
                data = (torch.rand((1, 30), dtype=torch.float),)
                # 生成所有组合的参数选项
                options = itertools.product(
                    (True, False),  # use_bias
                    (True, False),  # has_relu
                    (True, False),  # functional relu
                    (True, False),  # is_reference
                )
                # 遍历所有参数组合
                for use_bias, has_relu, f_relu, is_reference in options:
                    # 创建 FuncLinear 模型实例
                    model = FuncLinear(use_bias, has_relu, f_relu)
                    # 根据是否是参考函数选择调用的函数
                    if is_reference:
                        qlinear_fun = ns.call_function(torch.nn.functional.linear)
                    else:
                        if has_relu:
                            qlinear_fun = ns.call_function(torch.ops.quantized.linear_relu_dynamic_fp16)
                        else:
                            qlinear_fun = ns.call_function(torch.ops.quantized.linear_dynamic_fp16)
                    # 定义准备节点的发生次数字典，包含激活和权重
                    prepare_node_occurrence = {
                        # activation 和 weight
                        ns.call_module(torch.ao.quantization.PlaceholderObserver): 2
                    }
                    # 定义转换节点的发生次数字典，包含量化函数和权重的转换
                    convert_node_occurrence = {
                        qlinear_fun: 1,
                        # weight
                        ns.call_method("to"): 1 if is_reference else 0
                    }
                    # 调用检查图模式函数，验证模型的量化转换是否正确
                    self.checkGraphModeFxOp(
                        model, data, QuantType.DYNAMIC, qlinear_fun,
                        is_reference=is_reference,
                        custom_qconfig_dict={"": float16_dynamic_qconfig},
                        prepare_expected_node_occurrence=prepare_node_occurrence,
                        expected_node_occurrence=convert_node_occurrence)
    def test_linear_static_fp16(self):
        # 定义一个名为 test_linear_static_fp16 的测试方法
        class FuncLinear(torch.nn.Module):
            # 定义一个名为 FuncLinear 的神经网络模块
            def __init__(self, use_bias, has_relu, f_relu):
                super().__init__()
                # 初始化权重矩阵 w，形状为 (4, 30)，随机填充
                self.w = torch.randn(4, 30)
                # 初始化偏置向量 b，形状为 (4)，随机填充
                self.b = torch.randn(4)
                # 记录是否使用偏置项
                self.use_bias = use_bias
                # 根据参数设定是否包含 ReLU 激活函数
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    # 如果不包含 ReLU 激活函数，则使用恒等映射作为激活函数
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                # 根据是否使用偏置项选择相应的线性层计算方式
                if self.use_bias:
                    x = F.linear(x, self.w, self.b)
                else:
                    x = F.linear(x, self.w)
                # 应用预先设定的激活函数
                x = self.relu(x)
                return x

        data = (torch.rand((1, 30), dtype=torch.float),)
        # 生成测试参数组合
        options = itertools.product(
            (True, False),  # use_bias 是否使用偏置项
            (True, False),  # has_relu 是否包含 ReLU 激活函数
            (True, False),  # functional relu 是否使用函数式的 ReLU
            (True, False),  # is_reference 是否参考模式
        )
        # 获取测试专用的原生后端配置
        backend_config = get_test_only_legacy_native_backend_config()
        # 遍历所有参数组合进行测试
        for use_bias, has_relu, f_relu, is_reference in options:
            # 创建 FuncLinear 模型实例
            model = FuncLinear(use_bias, has_relu, f_relu)
            # 获取线性函数的调用
            linear_fun = ns.call_function(torch.nn.functional.linear)
            # 设置预期的节点出现情况，用于模型准备阶段
            prepare_node_occurrence = {
                # 观察者节点的出现次数，包括激活函数、权重、偏置和输出
                ns.call_module(torch.ao.quantization.PlaceholderObserver): 3 + int(use_bias) + int(not has_relu),
            }
            # 设置预期的节点出现情况，用于模型转换阶段
            convert_node_occurrence = {
                # 由于不支持静态 fp16 运算，线性函数未融合
                linear_fun: 1,
                # 添加额外的量化-反量化操作，当 is_reference 为 True 且 has_relu 为 False 时
                # 因为当 has_relu 为 False 时，模型中存在 nn.Identity，它是一个复制节点，需要额外的量化-反量化
                ns.call_method("to"): 3 + int(use_bias) + int(not has_relu and is_reference),
                ns.call_method("dequantize"): 3 + int(use_bias) + int(not has_relu and is_reference)
            }
            # 调用检查图模式下的量化操作
            self.checkGraphModeFxOp(
                model, data, QuantType.DYNAMIC, linear_fun,
                is_reference=is_reference,
                custom_qconfig_dict={"": float16_static_qconfig},
                prepare_expected_node_occurrence=prepare_node_occurrence,
                expected_node_occurrence=convert_node_occurrence,
                backend_config=backend_config)

    @skipIfNoFBGEMM
    # 定义一个测试函数，用于测试卷积模块
    def test_conv_module(self):
        # 创建包含不同维度卷积层类的字典
        conv_module = {1 : torch.nn.Conv1d, 2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}

        # 定义一个卷积层包装类
        class ConvWrapper(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化指定维度的卷积层对象，并转换为浮点型
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                # 执行前向传播，返回卷积层处理后的结果
                return self.conv(x)

        # 生成维度和量化类型的所有组合
        options = itertools.product([1, 2, 3], self.static_quant_types)
        # 定义量化节点字典，每个维度对应一个量化卷积模块
        quantized_nodes = {
            1: ns.call_module(nnq.Conv1d),
            2: ns.call_module(nnq.Conv2d),
            3: ns.call_module(nnq.Conv3d),
        }
        # 遍历每个维度和量化类型的组合
        for dim, quant_type in options:
            # 调用检查图模式下操作的方法，传入卷积包装类实例、图像数据字典中的相应维度数据、量化类型和对应的量化节点
            self.checkGraphModeFxOp(
                ConvWrapper(dim), self.img_data_dict[dim], quant_type,
                quantized_nodes[dim])

    # 根据条件跳过 FBGEMM 不可用时的测试
    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    # 测试量化卷积与ReLU的功能
    def test_quantized_conv_relu(self):
        """tests for conv1d_relu/conv2d_relu/conv3d_relu"""
        # 创建包含不同维度卷积层类的字典
        conv_module = {1 : torch.nn.Conv1d, 2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}

        # 定义带ReLU的卷积层类
        class ConvNdRelu(torch.nn.Module):
            def __init__(self, dim, inplace):
                super().__init__()
                # 初始化指定维度的卷积层对象和ReLU层对象，并转换为浮点型
                self.conv = conv_module[dim](3, 3, 3).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                # 执行前向传播，先卷积后ReLU，并返回结果
                return self.relu(self.conv(x))

        # 定义使用函数式ReLU的卷积层类
        class ConvNdFunctionalRelu(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化指定维度的卷积层对象，并转换为浮点型
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                # 执行前向传播，卷积后使用函数式ReLU，并返回结果
                return F.relu(self.conv(x))

        # 定义使用就地操作的函数式ReLU的卷积层类
        class ConvNdInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化指定维度的卷积层对象，并转换为浮点型
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                # 执行前向传播，卷积后使用就地操作的函数式ReLU，并返回结果
                return F.relu(self.conv(x), True)

        # 生成维度和量化类型的所有组合
        options = itertools.product([1, 2, 3], self.static_quant_types)
        # 定义量化节点字典，每个维度对应一个量化ReLU卷积模块
        quantized_nodes = {
            1: ns.call_module(nniq.ConvReLU1d),
            2: ns.call_module(nniq.ConvReLU2d),
            3: ns.call_module(nniq.ConvReLU3d),
        }
        # 遍历每个维度和量化类型的组合
        for dim, quant_type in options:
            # 遍历包含不同ReLU类型的卷积层类列表
            for m in [ConvNdRelu(dim, True),
                      ConvNdRelu(dim, False),
                      ConvNdFunctionalRelu(dim),
                      ConvNdInplaceFunctionalRelu(dim)]:
                # 调用检查图模式下操作的方法，传入当前卷积层类实例、图像数据字典中的相应维度数据、量化类型和对应的量化节点
                self.checkGraphModeFxOp(
                    m, self.img_data_dict[dim], quant_type,
                    quantized_nodes[dim])
    # 定义一个测试函数，用于测试针对 int8 类型的二进制操作的实现
    def _test_binary_op_int8_impl(self, binary_op, ibinary_op, quantized_op):
        # 创建包含两个随机张量的数据元组，数据类型为 torch.float
        data = (torch.randn(1, 1, 1, 1, dtype=torch.float),
                torch.randn(1, 1, 1, 1, dtype=torch.float))
        # 创建所有可能的选项组合的迭代器，每个选项可以为 True 或 False
        options = itertools.product([True, False], [True, False], [True, False])
        # 设置量化类型为 STATIC
        quant_type = QuantType.STATIC

        # 对每一种选项组合进行测试
        for is_inplace, is_scalar, is_reference in options:
            # 如果是参考操作
            if is_reference:
                # 创建节点列表，包含反量化方法调用、二进制操作函数调用和张量量化操作函数调用
                node_list = [
                    ns.call_method("dequantize"),
                    ns.call_function(binary_op),
                    ns.call_function(torch.quantize_per_tensor)
                ]
                # 初始化量化节点为空
                quantized_node = None
            else:
                # 如果不是参考操作，节点列表设为 None，并创建量化节点
                node_list = None
                quantized_node = ns.call_function(quantized_op)

            # 调用 self.checkGraphModeFxOp 方法，检查图模式下的 FX 操作
            # 测试二进制操作是否能够正确量化，即使输入未被量化
            self.checkGraphModeFxOp(
                BinaryOp(binary_op, ibinary_op, is_inplace, is_scalar), data, quant_type,
                quantized_node, expected_node_list=node_list, is_reference=is_reference)
            # 这个测试验证即使二进制操作没有被量化输入，它也应该被量化
            self.checkGraphModeFxOp(
                BinaryOpNonQuantizedInput(binary_op, ibinary_op, is_inplace, is_scalar),
                data, quant_type, quantized_node,
                expected_node_list=node_list, is_reference=is_reference)
    # 测试二进制操作与float16实现的方法
    def _test_binary_op_float16_impl(self, binary_op, ibinary_op):
        # 创建两个随机张量数据，数据类型为torch.float
        data = (torch.randn(1, 1, 1, 1, dtype=torch.float),
                torch.randn(1, 1, 1, 1, dtype=torch.float))
        # 设置量化类型为STATIC
        quant_type = QuantType.STATIC
        # 测试fp16静态量化
        # 生成fp16模式的选项组合
        options = itertools.product([True, False], [True, False])
        # 自定义量化配置字典
        custom_qconfig_dict = {
            "object_type": [(binary_op, float16_static_qconfig)]
        }
        # 获取用于测试的后端配置
        backend_config = get_test_only_legacy_native_backend_config()
        # 遍历is_inplace和is_scalar的选项组合
        for is_inplace, is_scalar in options:
            # 确定节点出现次数的预期字典
            node_occurrence = {
                # 如果是标量，则为output_conv1, output_add1, output_add2；否则为output_conv1, output_conv2, output_add1, output_add2
                ns.call_method("to"): 3 if is_scalar else 4
            }
            # 检查图模式下的自定义二元操作
            self.checkGraphModeFxOp(
                BinaryOp(binary_op, ibinary_op, is_inplace, is_scalar), data, quant_type,
                expected_node_occurrence=node_occurrence,
                custom_qconfig_dict=custom_qconfig_dict,
                backend_config=backend_config)

            # 确定节点出现次数的预期字典
            node_occurrence = {
                # 如果是标量，则为input_add, output_add；否则为input_add1, input_add2, output_add
                ns.call_method("to"): 2 if is_scalar else 3
            }
            # 检查图模式下的非量化输入的自定义二元操作
            self.checkGraphModeFxOp(
                BinaryOpNonQuantizedInput(binary_op, ibinary_op, is_inplace, is_scalar), data, quant_type,
                expected_node_occurrence=node_occurrence,
                custom_qconfig_dict=custom_qconfig_dict,
                backend_config=backend_config)

    # 测试二进制操作与ReLU整数8位实现的方法
    def _test_binary_op_relu_int8_impl(self, binary_op, ibinary_op, quantized_op):
        # 创建两个随机张量数据，数据类型为torch.float
        data = (torch.rand((1, 1, 1, 1), dtype=torch.float),
                torch.rand((1, 1, 1, 1), dtype=torch.float))
        # 设置量化类型为STATIC
        quant_type = QuantType.STATIC
        # 创建量化节点
        quantized_node = ns.call_function(quantized_op)
        # 生成选项组合，包括is_inplace_op, relu_callable, is_scalar
        options = itertools.product(
            [True, False], [nn.ReLU, F.relu, torch.relu], [True, False])
        # 遍历选项组合
        for is_inplace_op, relu_callable, is_scalar in options:
            # 创建BinaryOpRelu模型
            model = BinaryOpRelu(
                binary_op, ibinary_op, is_inplace_op, relu_callable, is_scalar)
            # 检查图模式下的自定义操作
            self.checkGraphModeFxOp(
                model, data, quant_type, quantized_node)
    # 定义一个测试方法，用于测试二进制操作和ReLU激活函数的实现
    def _test_binary_op_relu_float16_impl(self, binary_op, ibinary_op):
        # 创建两个随机浮点数张量作为测试数据
        data = (torch.rand((1, 1, 1, 1), dtype=torch.float),
                torch.rand((1, 1, 1, 1), dtype=torch.float))
        # 设置量化类型为静态量化
        quant_type = QuantType.STATIC
        # 枚举所有可能的选项：是否原地操作、使用的ReLU函数、是否标量
        options = itertools.product(
            [True, False], [nn.ReLU, F.relu, torch.relu], [True, False])
        # 自定义量化配置字典，包含空键和"object_type"键，值为量化配置对象或者对象类型的元组
        custom_qconfig_dict = {
            "": float16_static_qconfig,
            "object_type": [(torch.nn.Conv2d, None)]
        }
        # 获取测试专用的后端配置
        backend_config = get_test_only_legacy_native_backend_config()
        # 遍历每种选项组合
        for is_inplace_op, is_functional_relu, is_scalar in options:
            # 设置节点出现次数字典，调用方法'to'出现3次或者4次（取决于是否标量）
            node_occurrence = {
                ns.call_method("to"): 3 if is_scalar else 4
            }
            # 创建BinaryOpRelu模型实例
            model = BinaryOpRelu(
                binary_op, ibinary_op, is_inplace_op, is_functional_relu, is_scalar)
            # 调用检查图模式下操作的方法，验证模型行为
            self.checkGraphModeFxOp(
                model, data, quant_type, custom_qconfig_dict=custom_qconfig_dict,
                expected_node_occurrence=node_occurrence,
                backend_config=backend_config)

    # 使用装饰器跳过如果没有FBGEMM的测试
    @skipIfNoFBGEMM
    def test_add(self):
        # 调用二进制整数加法的测试方法
        self._test_binary_op_int8_impl(
            operator.add, operator.iadd, torch.ops.quantized.add)
        # 调用二进制浮点数加法的测试方法
        self._test_binary_op_float16_impl(
            operator.add, operator.iadd)

    # 使用装饰器跳过不再需要的测试，可以通过新API稍后启用
    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_sub(self):
        # 调用二进制浮点数减法的测试方法
        self._test_binary_op_float16_impl(operator.sub, operator.isub)
        # 调用PyTorch库中的二进制浮点数减法的测试方法
        self._test_binary_op_float16_impl(torch.sub, None)

    # 使用装饰器跳过不再需要的测试，可以通过新API稍后启用
    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_div(self):
        # 调用二进制浮点数除法的测试方法
        self._test_binary_op_float16_impl(operator.truediv, operator.itruediv)
        # 调用PyTorch库中的二进制浮点数除法的测试方法
        self._test_binary_op_float16_impl(torch.div, None)

    # 使用装饰器跳过如果没有FBGEMM的测试
    @skipIfNoFBGEMM
    def test_mul(self):
        # 调用二进制整数乘法的测试方法
        self._test_binary_op_int8_impl(
            operator.mul, operator.imul, torch.ops.quantized.mul)
        # 调用二进制浮点数乘法的测试方法
        self._test_binary_op_float16_impl(operator.mul, operator.imul)

    # 使用装饰器跳过不再需要的测试，可以通过新API稍后启用
    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_sum(self):
        # 定义Sum类，计算张量的和，并返回结果
        class Sum(torch.nn.Module):
            def forward(self, x):
                x = torch.sum(x, [1], keepdim=True)
                x = torch.sum(x, [1])
                return x

        # 创建测试数据张量
        data = torch.randn(1, 2, 3, 4, dtype=torch.float)
        # 设置量化类型为静态量化
        quant_type = QuantType.STATIC
        # 自定义量化配置字典，包含'object_type'键，值为计算和操作的静态量化配置对象
        custom_qconfig_dict = {
            "object_type": [(torch.sum, float16_static_qconfig)]
        }
        # 设置节点出现次数字典，调用方法'to'出现3次
        node_occurrence = {
            # input_sum1, output_sum1, output_sum2
            ns.call_method("to"): 3
        }
        # 调用检查图模式下操作的方法，验证Sum类的行为
        self.checkGraphModeFxOp(
            Sum(), data, quant_type,
            expected_node_occurrence=node_occurrence,
            custom_qconfig_dict=custom_qconfig_dict)

    # 使用装饰器跳过不再需要的测试，可以通过新API稍后启用
    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_bmm(self):
        # 定义一个内部类 BMMMethod，继承自 torch.nn.Module
        class BMMMethod(torch.nn.Module):
            # 定义前向传播方法，执行 torch.bmm 操作
            def forward(self, x, y):
                return x.bmm(y)

        # 创建两个随机张量作为测试数据
        data = (torch.randn(1, 1, 1, dtype=torch.float),
                torch.randn(1, 1, 1, dtype=torch.float))
        # 设置量化类型为 STATIC
        quant_type = QuantType.STATIC
        # 测试 fp16 静态量化的情况
        # 定义自定义量化配置字典，针对 torch.bmm 操作进行配置
        custom_qconfig_dict = {
            "object_type": [(torch.bmm, float16_static_qconfig),
                            ("bmm", float16_static_qconfig)]
        }
        # 定义节点出现次数字典，用于期望的节点出现次数检查
        node_occurrence = {
            # 对于 call_method("to")，预期出现 3 次
            ns.call_method("to"): 3
        }
        # 调用 self.checkGraphModeFxOp 方法，检查图模式下的操作
        self.checkGraphModeFxOp(
            # 使用 BinaryOpNonQuantizedInput 对象，执行 torch.bmm 操作
            BinaryOpNonQuantizedInput(torch.bmm, None, False, False), data, quant_type,
            expected_node_occurrence=node_occurrence,
            custom_qconfig_dict=custom_qconfig_dict)

        # TODO: 支持 call_method("bmm")
        # 可以将 call_method("bmm") 转换为 call_function(torch.bmm)
        # self.checkGraphModeFxOp(
        #     BMMMethod(), data, quant_type,
        #     expected_node_occurrence=node_occurrence,
        #     custom_qconfig_dict=custom_qconfig_dict,
        #     print_debug_info=True)

    @skipIfNoFBGEMM
    def test_add_relu(self):
        # 调用 _test_binary_op_relu_int8_impl 方法，测试整型加法和 relu 操作
        self._test_binary_op_relu_int8_impl(
            operator.add, operator.iadd, torch.ops.quantized.add_relu)
        # 调用 _test_binary_op_relu_float16_impl 方法，测试浮点型加法和 relu 操作
        self._test_binary_op_relu_float16_impl(
            operator.add, operator.iadd)

    @skipIfNoFBGEMM
    def test_add_relu_multiple_uses_of_relu(self):
        # 定义内部类 Sub，继承自 torch.nn.Module，包含一个 inplace=True 的 ReLU 操作
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU(inplace=True)

        # 定义内部类 M，继承自 torch.nn.Module，包含 Sub 类的实例化对象
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = Sub()

            def forward(self, x, y):
                # 执行 x + y 操作
                x = x + y
                # 执行 self.sub.relu(x) 操作
                x = self.sub.relu(x)
                # 再次执行 x + y 操作
                x = x + y
                # 再次执行 self.sub.relu(x) 操作
                x = self.sub.relu(x)
                return x

        # 创建 M 类的实例并转换为评估模式
        m = M().eval()
        # 创建示例输入
        example_inputs = (torch.randn(3), torch.randn(3))
        # 使用默认的量化配置准备 m 模型
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        # 将 m 模型转换为图模式
        m = convert_fx(m)
        # 定义节点出现次数字典，用于期望的节点出现次数检查
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 2,
            ns.call_function(torch.ops.quantized.add_relu): 2,
            ns.call_method("dequantize"): 1,
        }
        # 调用 self.checkGraphModuleNodes 方法，检查图模块中的节点
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        # 检查模型是否可以被脚本化
        m = torch.jit.script(m)
        # 检查模型是否可运行
        m(*example_inputs)

    @skipIfNoFBGEMM
    def test_mul_relu(self):
        # 调用 _test_binary_op_relu_int8_impl 方法，测试整型乘法和 relu 操作
        self._test_binary_op_relu_int8_impl(
            operator.mul, operator.imul, torch.ops.quantized.mul_relu)
        # 调用 _test_binary_op_relu_float16_impl 方法，测试浮点型乘法和 relu 操作
        self._test_binary_op_relu_float16_impl(
            operator.mul, operator.imul)

    # TODO(future PR): make more generic
    @skipIfNoFBGEMM
    # 如果没有 FBGEMM，跳过测试
    def test_quantized_add_qat(self):
        # 定义一个测试类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 对输入张量 x 加 1.0
                x = torch.add(x, 1.0)
                x = self.conv1(x)
                # 对输出张量 x 加 1.0
                x = torch.add(x, 1.0)
                # 对张量 x 执行 ReLU 激活函数
                x = torch.relu(x)
                x = self.conv2(x)
                return x

        # 创建 M 类的实例 m
        m = M()
        # 创建一个示例输入张量的元组
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 期望的节点出现次数字典，指定了 FusedMovingAvgObsFakeQuantize 节点出现 5 次
        expected_node_occurrence = {
            ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 5,
        }
        # 调用 _test_quantized_add_mul_qat 方法进行测试
        self._test_quantized_add_mul_qat(m, example_inputs, expected_node_occurrence)

    @skipIfNoFBGEMM
    # 如果没有 FBGEMM，跳过测试
    def test_quantized_mul_qat(self):
        # 定义一个测试类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 对输入张量 x 乘以 1.0
                x = torch.mul(x, 1.0)
                x = self.conv1(x)
                # 对输出张量 x 乘以 1.0
                x = torch.mul(x, 1.0)
                # 对张量 x 执行 ReLU 激活函数
                x = torch.relu(x)
                x = self.conv2(x)
                return x

        # 创建 M 类的实例 m
        m = M()
        # 创建一个示例输入张量的元组
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 期望的节点出现次数字典，指定了 FusedMovingAvgObsFakeQuantize 节点出现 5 次
        expected_node_occurrence = {
            ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 5,
        }
        # 调用 _test_quantized_add_mul_qat 方法进行测试
        self._test_quantized_add_mul_qat(m, example_inputs, expected_node_occurrence)
    def test_int8_input_no_unnecessary_fq(self):
        """
        If the inputs to the graph are quantized and the only node
        does not need an activation observer, verifies that the
        activation observer is not inserted.
        """
        # 定义一个测试方法，用于验证输入图像经过量化后，如果唯一的节点不需要激活观察器，
        # 则确保不会插入激活观察器。

        class M(nn.Module):
            # 定义一个简单的 nn.Module 模型
            def __init__(self, scalar):
                super().__init__()
                self.scalar = scalar
                # 初始化一个 FloatFunctional 实例，用于量化操作
                self.add_func = torch.ao.nn.quantized.FloatFunctional()

            def forward(self, x):
                # 模型的前向传播方法，将输入 x 和标量 self.scalar 相加
                return self.add_func.add_scalar(x, self.scalar)

        # 创建一个 M 类的实例，标量为 0.5
        m = M(0.5)

        # 准备量化感知训练的 FX 模型
        mp = torch.ao.quantization.quantize_fx.prepare_qat_fx(
            m, {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')},
            example_inputs=(torch.randn(1),),
            prepare_custom_config={"input_quantized_idxs": [0]})
        
        # 预期的节点出现次数，其中包含一个调用了 FusedMovingAvgObsFakeQuantize 的模块
        expected_node_occurrence = {
            ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 1,
        }

        # 调用父类的方法，检查 GraphModule 中的节点是否符合预期
        self.checkGraphModuleNodes(
            mp, expected_node_occurrence=expected_node_occurrence)

    @skipIfNoFBGEMM
    def test_cat(self):
        """ quantization of the output of cat will depend on the
        input of cat. we only quantize the output of cat when its inputs are quantized.
        """
        # 定义一个名为 M 的 PyTorch 模块类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建两个二维卷积层，每层输入输出通道数均为 2
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # 对第一个输入 x 进行卷积操作
                x = self.conv1(x)
                # 对第二个输入 y 进行卷积操作
                y = self.conv2(y)
                # 将两个卷积的结果在通道维度上进行拼接
                return torch.cat([x, y], 1)

        # 示例输入，两个 1x2x5x5 的张量，数据类型为 float
        example_inputs = (torch.randn(1, 2, 5, 5, dtype=torch.float),
                          torch.randn(1, 2, 5, 5, dtype=torch.float))
        # 获取 torch.cat 函数的节点
        quantized_node = ns.call_function(torch.cat)
        # 生成静态量化类型和是否参考值的所有组合
        options = itertools.product(self.static_quant_types, [True, False])
        for quant_type, is_reference in options:
            if is_reference:
                # 如果是参考值模式，设置转换后的节点列表
                converted_node_list = [
                    ns.call_method("dequantize"),
                    ns.call_function(torch.cat),
                    ns.call_function(torch.quantize_per_tensor)
                ]
                # 设置转换后节点的出现次数
                converted_node_occurrence = {
                    # 两个卷积的输入和输出，以及 torch.cat 的输出
                    ns.call_method("dequantize"): 5,
                    ns.call_function(torch.cat): 1,
                    ns.call_function(torch.quantize_per_tensor): 5,
                }
            else:
                # 如果不是参考值模式，转换后的节点列表为空
                converted_node_list = None
                # 设置转换后节点的出现次数
                converted_node_occurrence = {
                    # torch.cat 的输出
                    ns.call_method("dequantize"): 1,
                    ns.call_function(torch.cat): 1,
                    # 两个输入的量化
                    ns.call_function(torch.quantize_per_tensor): 2,
                }

            # 检查转换后的图模式下的操作
            self.checkGraphModeFxOp(
                M(),
                example_inputs,
                quant_type,
                quantized_node,
                expected_node_list=converted_node_list,
                expected_node_occurrence=converted_node_occurrence,
                is_reference=is_reference)

        # 检查 torch.cat 是否在输入和输出上使用相同的观察器
        m = M().eval()
        # 准备转换模型，使用默认的量化配置和示例输入
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        # 获取所有模块的观察器数量，包括重复的
        all_observers = len(dict(m.named_modules(remove_duplicate=False)))
        # 获取不同模块的观察器数量，去除重复的
        distinct_observers = len(dict(m.named_modules()))
        # 断言所有观察器数量应等于不同观察器数量加上两个
        self.assertEqual(all_observers, distinct_observers + 2)
        # 确保转换后的模型可以运行
        m = convert_fx(m)
        m(*example_inputs)

    @skipIfNoFBGEMM
    def test_qbatch_norm(self):
        bn_module = {
            # TODO: quantized batchnorm 1d module is missing
            # 1 : torch.nn.BatchNorm1d,
            # 定义不同维度下的批量归一化模块
            2 : torch.nn.BatchNorm2d,
            3 : torch.nn.BatchNorm3d,
        }

        class M(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化模块时，根据给定维度选择相应的批量归一化模块
                self.bn = bn_module[dim](3to(torch.float)

            def forward(self, x):
                # 在前向传播中应用批量归一化
                return self.bn(x)

        options = itertools.product(self.static_quant_types, [2, 3], [True, False])
        quantized_nodes = {
            False: {
                # 1: ns.call_module(nnq.BatchNorm1d),
                # 非参考模式时调用量化后的二维和三维批量归一化模块
                2: ns.call_module(nnq.BatchNorm2d),
                3: ns.call_module(nnq.BatchNorm3d),
            },
            True: {
                # 1: ns.call_module(nn.BatchNorm1d),
                # 参考模式时调用非量化的二维和三维批量归一化模块
                2: ns.call_module(nn.BatchNorm2d),
                3: ns.call_module(nn.BatchNorm3d),
            }
        }
        for quant_type, dim, is_reference in options:
            # 检查图模式中的操作
            self.checkGraphModeFxOp(
                M(dim), self.img_data_dict[dim], quant_type, quantized_nodes[is_reference][dim], is_reference=is_reference)

    @skipIfNoFBGEMM
    def test_qbatch_norm_relu(self):
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}

        class BNRelu(torch.nn.Module):
            def __init__(self, dim, inplace):
                super().__init__()
                # 初始化模块时，根据给定维度选择相应的批量归一化模块和ReLU激活函数
                self.bn = bn_module[dim](3to(torch.float)
                self.relu = torch.nn.ReLU(inplace=inplace)

            def forward(self, x):
                # 在前向传播中应用批量归一化和ReLU激活函数
                return self.relu(self.bn(x))

        class BNFuncRelu(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化模块时，根据给定维度选择相应的批量归一化模块
                self.bn = bn_module[dim](3to(torch.float)

            def forward(self, x):
                # 在前向传播中应用批量归一化和ReLU激活函数（不进行原地操作）
                return F.relu(self.bn(x), False)

        class BNFuncInplaceRelu(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化模块时，根据给定维度选择相应的批量归一化模块

                self.bn = bn_module[dim](3to(torch.float)

            def forward(self, x):
                # 在前向传播中应用批量归一化和ReLU激活函数（进行原地操作）
                return F.relu(self.bn(x), True)

        options = itertools.product(self.static_quant_types, [2, 3], [True, False])
        quantized_nodes = {
            True: {
                # 选择量化后的二维和三维批量归一化与ReLU模块
                2: ns.call_module(nni.BNReLU2d),
                3: ns.call_module(nni.BNReLU3d),
            },
            False: {
                # 选择非量化的二维和三维批量归一化与ReLU模块
                2: ns.call_module(nniq.BNReLU2d),
                3: ns.call_module(nniq.BNReLU3d),
            }
        }
        for quant_type, dim, is_reference in options:
            # 遍历不同的量化类型、维度和是否参考模式的选项
            for instance in [BNRelu(dim, True), BNRelu(dim, False),
                             BNFuncRelu(dim), BNFuncInplaceRelu(dim)]:
                # 对每个实例执行图模式操作的检查
                self.checkGraphModeFxOp(
                    instance, self.img_data_dict[dim], quant_type,
                    quantized_nodes[is_reference][dim], is_reference=is_reference)
    # 定义测试激活函数操作的内部方法，接受浮点模块、浮点操作、量化模块和量化操作作为参数
    def _test_activation_impl(
            self, float_module, float_op, quantized_module, quantized_op):
        ''' Test for activation op(with inplace options), float_op can be
        torch op or functional op
        '''
        # 定义内部的模块类，用于测试激活函数操作
        class M(torch.nn.Module):
            def __init__(self, is_module, inplace):
                super().__init__()
                self.is_module = is_module  # 是否为模块
                self.inplace = inplace      # 是否为原地操作
                if self.is_module:
                    self.op = float_module(self.inplace)  # 如果是模块，则使用浮点模块
                else:
                    self.op = float_op          # 否则使用浮点操作

            def forward(self, input):
                if self.is_module:
                    return self.op(input)        # 如果是模块，则直接调用模块操作
                else:
                    return self.op(input, self.inplace)  # 否则调用浮点操作，并传入原地操作参数

        # 组合不同的选项：是否为模块、是否为原地操作、量化类型、是否为参考（引用）节点
        options = itertools.product([True, False], [True, False], self.static_quant_types, [True, False])
        # 定义量化节点字典，根据是否为模块和是否为参考节点来调用不同的函数或模块
        quantized_nodes = {
            True: {  # 如果是模块
                True: ns.call_module(float_module),    # 如果是参考节点，则调用浮点模块
                False: ns.call_module(quantized_module),  # 否则调用量化模块
            },
            False: {  # 如果不是模块
                True: ns.call_function(float_op),       # 如果是参考节点，则调用浮点操作
                False: ns.call_function(quantized_op),  # 否则调用量化操作
            }
        }

        # 遍历所有选项，并调用检查图模式下的操作方法，用于检查功能实现的正确性
        for is_module, is_inplace, quant_type, is_reference in options:
            self.checkGraphModeFxOp(
                M(is_module, is_inplace), self.img_data_2d,
                quant_type, quantized_nodes[is_module][is_reference], is_reference=is_reference)

    # 测试 Hardswish 激活函数的方法
    def test_hardswish(self):
        self._test_activation_impl(nn.Hardswish, F.hardswish, nnq.Hardswish, torch.ops.quantized.hardswish)

    # 测试 ELU 激活函数的方法
    def test_elu(self):
        self._test_activation_impl(nn.ELU, F.elu, nnq.ELU, torch.ops.quantized.elu)

    # 测试 LeakyReLU 激活函数的方法
    def test_leaky_relu(self):
        self._test_activation_impl(nn.LeakyReLU, F.leaky_relu, nnq.LeakyReLU, torch.ops.quantized.leaky_relu)

    # 测试 PReLU 激活函数的方法
    def test_prelu(self):
        # 定义内部的模块类，用于测试 PReLU 激活函数
        class M(torch.nn.Module):
            def __init__(self, num_param: int):
                super().__init__()
                self.op = torch.nn.PReLU(num_parameters=num_param)  # 使用给定参数创建 PReLU 模块

            def forward(self, input):
                return self.op(input)  # 调用 PReLU 操作

        X = [[torch.randn(4, 4, 4, 4, dtype=torch.float)]]  # 定义输入数据 X
        options = itertools.product([1, 4], self.static_quant_types, [True, False])  # 组合不同选项
        quantized_nodes = {
            True: ns.call_module(torch.nn.PReLU),  # 如果是参考节点，则调用标准 PReLU 模块
            False: ns.call_module(torch.ao.nn.quantized.PReLU),  # 否则调用量化 PReLU 模块
        }

        # 遍历所有选项，并调用检查图模式下的操作方法，用于检查功能实现的正确性
        for num_parameter, quant_type, is_reference in options:
            self.checkGraphModeFxOp(
                M(num_parameter), X, quant_type, quantized_nodes[is_reference],
                is_reference=is_reference)
    def _test_norm_impl(
            self, float_module, float_op, op_args, data, quantized_module, quantized_op,
            skip_op_arg_for_functional=False):
        ''' Test for normalization op, float_op can be torch op or functional op,
        op_args is a list of positional argument for the module/op
        '''
        # 定义内部类 M，用于构建测试模块
        class M(torch.nn.Module):
            def __init__(self, is_module):
                super().__init__()
                self.is_module = is_module
                # 根据是否是模块选择初始化操作
                if self.is_module:
                    self.op = float_module(*op_args)
                else:
                    self.op = float_op

            def forward(self, input):
                if self.is_module:
                    return self.op(input)
                else:
                    args = [input]
                    # 如果不跳过函数式操作的参数，则将参数添加到 args 列表中
                    if not skip_op_arg_for_functional:
                        args += op_args
                    return self.op(*args)

        # 生成选项组合，True/False 分别代表是否使用模块化量化和量化类型
        options = itertools.product([True, False], self.static_quant_types)
        quantized_nodes = {
            # 根据 is_module 的值选择对应的量化节点
            True: ns.call_module(quantized_module),
            False: ns.call_function(quantized_op),
        }

        # 遍历选项组合，对每个组合执行图模式下的操作检查
        for is_module, quant_type in options:
            self.checkGraphModeFxOp(
                M(is_module), data, quant_type, quantized_nodes[is_module])

    def _test_norm_float16_impl(
            self, float_module, float_op, op_args, data,
            skip_op_arg_for_functional=False):
        ''' Test for normalization op, float_op can be torch op or functional op,
        op_args is a list of positional argument for the module/op
        '''
        # 定义内部类 M，用于构建测试模块
        class M(torch.nn.Module):
            def __init__(self, is_module):
                super().__init__()
                self.is_module = is_module
                # 根据是否是模块选择初始化操作
                if self.is_module:
                    self.op = float_module(*op_args)
                else:
                    self.op = float_op

            def forward(self, input):
                if self.is_module:
                    return self.op(input)
                else:
                    args = [input]
                    # 如果不跳过函数式操作的参数，则将参数添加到 args 列表中
                    if not skip_op_arg_for_functional:
                        args += op_args
                    return self.op(*args)

        # 生成选项组合，True/False 分别代表是否使用模块化量化和量化类型
        options = itertools.product([True, False], self.static_quant_types)
        qconfig_dict = {
            "object_type": [
                (float_module, float16_static_qconfig),
                (float_op, float16_static_qconfig)
            ]
        }
        node_occurrence = {
            ns.call_method("to"): 2
        }
        # 遍历选项组合，对每个组合执行图模式下的操作检查，并使用自定义的量化配置字典和预期节点出现次数
        for is_module, quant_type in options:
            self.checkGraphModeFxOp(
                M(is_module), data, quant_type, custom_qconfig_dict=qconfig_dict, expected_node_occurrence=node_occurrence)
    # 定义测试方法，用于测试 LayerNorm 的功能
    def test_layer_norm(self):
        # 创建包含随机数据的元组，作为测试输入数据
        data = (torch.rand((1, 2, 5, 5), dtype=torch.float),)
        # 调用 _test_norm_impl 方法，测试 nn.LayerNorm 和 F.layer_norm 的实现
        self._test_norm_impl(
            nn.LayerNorm, F.layer_norm, [[2, 5, 5]], data, nnq.LayerNorm, torch.ops.quantized.layer_norm)

    # 定义测试方法，用于测试 InstanceNorm 的功能
    def test_instance_norm(self):
        # 创建不同维度数据的随机数据元组
        data_1d = (torch.rand((1, 4, 5), dtype=torch.float),)
        data_2d = (torch.rand((1, 4, 5, 1), dtype=torch.float),)
        data_3d = (torch.rand((1, 4, 5, 1, 1), dtype=torch.float),)
        # 将不同维度的数据存储在字典中
        data_dict = {1 : data_1d, 2 : data_2d, 3 : data_3d}
        # 创建不同维度的 InstanceNorm 模块字典
        instance_norm_modules = {1 : nn.InstanceNorm1d,
                                 2 : nn.InstanceNorm2d,
                                 3 : nn.InstanceNorm3d}
        # 创建相应的量化 InstanceNorm 模块字典
        quantized_instance_norm_modules = {
            1 : nnq.InstanceNorm1d,
            2 : nnq.InstanceNorm2d,
            3 : nnq.InstanceNorm3d
        }
        # 遍历维度 [1, 2, 3]
        for dim in [1, 2, 3]:
            # 获取对应维度的数据
            data = data_dict[dim]
            # 获取对应维度的 InstanceNorm 模块
            module = instance_norm_modules[dim]
            # 获取对应维度的量化 InstanceNorm 模块
            quantized_module = quantized_instance_norm_modules[dim]
            # 调用 _test_norm_impl 方法，测试 module 和 quantized_module 的实现
            self._test_norm_impl(
                module, F.instance_norm, [4], data,
                quantized_module, torch.ops.quantized.instance_norm,
                skip_op_arg_for_functional=True)

    # 定义测试方法，用于测试带权重和偏置的规范化功能
    def test_norm_weight_bias(self):
        # 定义包含线性层的类 Linear
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)  # 初始化权重为全 1
                self.b = torch.zeros(5)    # 初始化偏置为全 0

            def forward(self, x):
                return torch.nn.functional.linear(x, self.w, self.b)  # 执行线性运算

        # 定义包含模块 Linear 的类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = Linear()               # 实例化 Linear 类
                self.scale = torch.randn(5, 5)      # 随机初始化 scale
                self.bias = torch.randn(5, 5)       # 随机初始化 bias

            def forward(self, x):
                x1 = self.mods1(x)                              # 执行 mods1 的前向传播
                y = F.layer_norm(x1, [5, 5], weight=self.scale, bias=self.bias)  # 执行 LayerNorm
                return y

        model = M()  # 实例化模型 M
        expected_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,         # 预期在图模式下出现的 torch.quantize_per_tensor 调用次数为 1
            ns.call_function(torch.ops.quantized.linear): 1,       # 预期在图模式下出现的 torch.ops.quantized.linear 调用次数为 1
            ns.call_function(torch.ops.quantized.layer_norm): 1,   # 预期在图模式下出现的 torch.ops.quantized.layer_norm 调用次数为 1
            ns.call_method("dequantize"): 1,                       # 预期在图模式下出现的 dequantize 方法调用次数为 1
        }

        # 调用 checkGraphModeFxOp 方法，检查模型的图模式下操作
        self.checkGraphModeFxOp(
            model,
            (torch.rand(5, 5),),    # 输入随机数据
            QuantType.STATIC,       # 使用静态量化类型
            expected_node_occurrence=expected_occurrence,  # 预期节点出现次数
            custom_qconfig_dict=get_default_qconfig_mapping().to_dict()  # 自定义量化配置字典
        )

    # 定义测试方法，用于测试默认节点量化处理器操作
    def _test_default_node_quant_handler_ops(
            self, module, functional, qconfig, is_reference=True, node_list=None, additional_quant_pattern_dict=None
    ):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # M 类的初始化方法
            def __init__(self, mod, func):
                super().__init__()
                # 初始化 M 类的 module 属性，调用 mod 构造器
                self.module = mod()
                # 初始化 M 类的 functional 属性，使用传入的 func
                self.functional = func

            # M 类的前向传播方法
            def forward(self, x):
                # 调用 module 对象的前向传播
                x = self.module(x)
                # 调用 functional 函数处理前向传播结果
                x = self.functional(x)
                return x

        # 如果 node_list 为 None，则将其设为空列表
        if node_list is None:
            node_list = []
        # 如果 additional_quant_pattern_dict 为 None，则将其设为空字典
        if additional_quant_pattern_dict is None:
            additional_quant_pattern_dict = {}

        # 创建一个形状为 (2, 2, 2, 2) 的随机张量 data
        data = torch.randn((2, 2, 2, 2))
        # 设置量化类型为 STATIC
        quant_type = QuantType.STATIC
        # 准备自定义量化配置字典
        prepare_custom_qconfig_dict = {"additional_quant_pattern": additional_quant_pattern_dict}
        # 设置 qconfig 字典，将空字符串映射到 qconfig 变量
        qconfig_dict = {"": qconfig}

        # 创建 M 类的实例 m，并设置为评估模式
        m = M(module, functional).eval()
        # 准备量化模型 m_prep，使用给定的量化配置
        m_prep = prepare_fx(m, qconfig_dict, prepare_custom_qconfig_dict)
        # 对 m_prep 应用转换函数，根据是否参考量化进行选择
        convert_fn = convert_to_reference_fx if is_reference else convert_fx
        m_quant = convert_fn(m_prep, is_reference=is_reference)
        m_quant(data)

        # 对量化后的模型进行节点检查
        self.checkGraphModuleNodes(m_quant, expected_node_list=node_list)

    @unittest.skip("TODO: reenable with backend_config api")
    def test_gelu_normal(self):
        # 设置 module 和 functional 分别为 GELU 模块和 gelu 函数
        module = torch.nn.GELU
        functional = torch.nn.functional.gelu
        # 获取默认的量化配置 qconfig
        qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        is_reference = False
        # 设置节点列表，包括调用 module 和 functional 的信息
        node_list = [
            ns.call_module(module),
            ns.call_function(functional),
        ]
        # 调用 _test_default_node_quant_handler_ops 方法，测试默认的节点量化处理操作
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list)

    @unittest.skip("TODO: reenable with backend_config api")
    def test_softmax_normal(self):
        # 设置 module 和 functional 分别为 Softmax 模块和 softmax 函数
        module = torch.nn.Softmax
        functional = torch.nn.functional.softmax
        # 获取默认的量化配置 qconfig
        qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        is_reference = False
        # 设置节点列表，包括调用 quantized Softmax 模块和 functional 函数的信息
        node_list = [
            ns.call_module(torch.ao.nn.quantized.Softmax),
            ns.call_function(functional),
        ]
        # 调用 _test_default_node_quant_handler_ops 方法，测试默认的节点量化处理操作
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list)

    @unittest.skip("This is no longer needed right now, can enable later with new api")
    # 定义测试函数，用于验证 GELU 激活函数的参考实现
    def test_gelu_reference(self):
        # 设定模块为 torch.nn.GELU
        module = torch.nn.GELU
        # 设定函数实现为 torch.nn.functional.gelu
        functional = torch.nn.functional.gelu
        # 获取默认的量化配置 "fbgemm"
        qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 指示此处使用的是参考实现
        is_reference = True
        # 创建节点列表，用于模拟量化和反量化操作
        node_list = [
            ns.call_function(torch.quantize_per_tensor),  # 调用 torch.quantize_per_tensor 函数
            ns.call_method("dequantize"),  # 调用对象的 dequantize 方法
            ns.call_module(module),  # 调用模块（module）
            ns.call_function(torch.quantize_per_tensor),  # 再次调用 torch.quantize_per_tensor 函数
            ns.call_method('dequantize'),  # 再次调用对象的 dequantize 方法
            ns.call_function(functional),  # 调用函数 functional
            ns.call_function(torch.quantize_per_tensor),  # 再次调用 torch.quantize_per_tensor 函数
            ns.call_method('dequantize')  # 再次调用对象的 dequantize 方法
        ]
        # TODO: 使用 backend_config 替换这些配置
        # 额外的模式字典，将 GELU 模块和函数映射到默认的节点量化处理器
        additional_patterns = {torch.nn.GELU: DefaultNodeQuantizeHandler,
                               torch.nn.functional.gelu: DefaultNodeQuantizeHandler}
        # 调用测试方法，验证默认节点量化处理器的操作
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list, additional_patterns)

        # 使用自定义的量化配置继续测试
        self._test_default_node_quant_handler_ops(module, functional, self.custom_qconfig, is_reference, node_list,
                                                  additional_quant_pattern_dict=self.common_quant_patterns)

    # 标记为跳过测试，因为当前版本不需要此测试，后续版本可能会启用新的 API
    @unittest.skip("This is no longer needed right now, can enable later with new api")
    # 定义测试函数，用于验证 Softmax 函数的参考实现
    def test_softmax_reference(self):
        # 设定模块为 torch.nn.Softmax
        module = torch.nn.Softmax
        # 设定函数实现为 torch.nn.functional.softmax
        functional = torch.nn.functional.softmax
        # 获取默认的量化配置 "fbgemm"
        qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 指示此处使用的是参考实现
        is_reference = True
        # 创建节点列表，用于模拟量化和反量化操作
        node_list = [
            ns.call_function(torch.quantize_per_tensor),  # 调用 torch.quantize_per_tensor 函数
            ns.call_method("dequantize"),  # 调用对象的 dequantize 方法
            ns.call_module(module),  # 调用模块（module）
            ns.call_function(torch.quantize_per_tensor),  # 再次调用 torch.quantize_per_tensor 函数
            ns.call_method('dequantize'),  # 再次调用对象的 dequantize 方法
            ns.call_function(functional),  # 调用函数 functional
            ns.call_function(torch.quantize_per_tensor),  # 再次调用 torch.quantize_per_tensor 函数
            ns.call_method('dequantize')  # 再次调用对象的 dequantize 方法
        ]
        # 额外的模式字典，将 Softmax 模块和函数映射到默认的节点量化处理器
        additional_patterns = {torch.nn.Softmax: DefaultNodeQuantizeHandler,
                               torch.nn.functional.softmax: DefaultNodeQuantizeHandler}
        # 调用测试方法，验证默认节点量化处理器的操作
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list, additional_patterns)

        # 使用自定义的量化配置继续测试
        self._test_default_node_quant_handler_ops(module, functional, self.custom_qconfig, is_reference, node_list,
                                                  additional_quant_pattern_dict=self.common_quant_patterns)
    # 测试 SiLU 激活函数的参考实现
    def test_silu_reference(self):
        # 使用 torch.nn.SiLU 模块
        module = torch.nn.SiLU
        # 使用 torch.nn.functional.silu 函数
        functional = torch.nn.functional.silu
        # 使用 float16_static_qconfig 作为量化配置
        qconfig = float16_static_qconfig
        # 指示这是参考实现
        is_reference = True
        # 定义节点列表，包括一系列调用方法和函数的操作
        node_list = [
            ns.call_method("to"),              # 调用对象的 "to" 方法
            ns.call_method("dequantize"),      # 调用对象的 "dequantize" 方法
            ns.call_module(module),            # 调用 SiLU 模块
            ns.call_method("to"),              # 再次调用对象的 "to" 方法
            ns.call_method('dequantize'),      # 再次调用对象的 "dequantize" 方法
            ns.call_function(functional),      # 调用 torch.nn.functional.silu 函数
            ns.call_method("to"),              # 再次调用对象的 "to" 方法
            ns.call_method('dequantize')       # 再次调用对象的 "dequantize" 方法
        ]
        # 调用测试函数，验证默认节点量化处理器的操作
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list)

        # 重新定义节点列表，包括另一系列调用函数的操作
        node_list = [
            ns.call_function(torch.quantize_per_tensor),  # 调用 torch.quantize_per_tensor 函数
            ns.call_method("dequantize"),                 # 调用对象的 "dequantize" 方法
            ns.call_module(module),                       # 调用 SiLU 模块
            ns.call_function(torch.quantize_per_tensor),  # 再次调用 torch.quantize_per_tensor 函数
            ns.call_method("dequantize"),                 # 再次调用对象的 "dequantize" 方法
            ns.call_function(functional),                 # 调用 torch.nn.functional.silu 函数
            ns.call_function(torch.quantize_per_tensor),  # 再次调用 torch.quantize_per_tensor 函数
            ns.call_method("dequantize")                  # 再次调用对象的 "dequantize" 方法
        ]
        # 再次调用测试函数，验证默认节点量化处理器的操作，同时传递自定义的量化配置和常见量化模式字典
        self._test_default_node_quant_handler_ops(module, functional, self.custom_qconfig, is_reference, node_list,
                                                  additional_quant_pattern_dict=self.common_quant_patterns)

    # 跳过测试 Mish 激活函数的参考实现，因为当前不需要，可以在新的 API 下启用
    def test_mish_reference(self):
        # 使用 torch.nn.Mish 模块
        module = torch.nn.Mish
        # 使用 torch.nn.functional.mish 函数
        functional = torch.nn.functional.mish
        # 使用 float16_static_qconfig 作为量化配置
        qconfig = float16_static_qconfig
        # 指示这是参考实现
        is_reference = True
        # 定义节点列表，包括一系列调用方法和函数的操作
        node_list = [
            ns.call_method("to"),              # 调用对象的 "to" 方法
            ns.call_method("dequantize"),      # 调用对象的 "dequantize" 方法
            ns.call_module(module),            # 调用 Mish 模块
            ns.call_method("to"),              # 再次调用对象的 "to" 方法
            ns.call_method('dequantize'),      # 再次调用对象的 "dequantize" 方法
            ns.call_function(functional),      # 调用 torch.nn.functional.mish 函数
            ns.call_method("to"),              # 再次调用对象的 "to" 方法
            ns.call_method('dequantize')       # 再次调用对象的 "dequantize" 方法
        ]
        # 调用测试函数，验证默认节点量化处理器的操作
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list)

        # 重新定义节点列表，包括另一系列调用函数的操作
        node_list = [
            ns.call_function(torch.quantize_per_tensor),  # 调用 torch.quantize_per_tensor 函数
            ns.call_method("dequantize"),                 # 调用对象的 "dequantize" 方法
            ns.call_module(module),                       # 调用 Mish 模块
            ns.call_function(torch.quantize_per_tensor),  # 再次调用 torch.quantize_per_tensor 函数
            ns.call_method("dequantize"),                 # 再次调用对象的 "dequantize" 方法
            ns.call_function(functional),                 # 调用 torch.nn.functional.mish 函数
            ns.call_function(torch.quantize_per_tensor),  # 再次调用 torch.quantize_per_tensor 函数
            ns.call_method("dequantize")                  # 再次调用对象的 "dequantize" 方法
        ]
        # 再次调用测试函数，验证默认节点量化处理器的操作，同时传递自定义的量化配置和常见量化模式字典
        self._test_default_node_quant_handler_ops(module, functional, self.custom_qconfig, is_reference, node_list,
                                                  additional_quant_pattern_dict=self.common_quant_patterns)
    def test_bmm_int_reference(self):
        """ int8 is not supported for bmm so we won't produce reference
            pattern for it
        """
        # 定义一个测试函数，用于检查 int8 类型在 bmm 操作中不受支持，因此我们不会为其生成参考模式
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块，包括一个 bmm 属性，用于执行 torch.bmm 操作
                self.bmm = torch.bmm

            def forward(self, x, y):
                # 模型前向传播函数，执行 torch.bmm 操作并返回结果
                out = self.bmm(x, y)
                return out

        # 生成测试数据
        data_x = torch.randn((2, 2, 2,))
        data_y = torch.randn((2, 2, 2,))
        example_inputs = (data_x, data_y)
        # 定义量化配置字典
        qconfig_dict = {"": torch.ao.quantization.get_default_qconfig("fbgemm")}
        # 指示是否为参考模式
        is_reference = True
        # 创建节点列表，包含一个对 torch.bmm 的函数调用
        node_list = [
            ns.call_function(torch.bmm),
        ]

        # 创建并评估模型 M
        m = M().eval()
        # 准备模型以进行量化，指定量化配置和示例输入
        m_prep = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m_prep(*example_inputs)
        # 根据是否为参考模式选择转换函数
        convert_fn = convert_to_reference_fx if is_reference else convert_fx
        # 转换模型以进行量化
        m_quant = convert_fn(m_prep)
        m_quant(*example_inputs)

        # 检查量化后的图模块节点是否符合预期
        self.checkGraphModuleNodes(m_quant, expected_node_list=node_list)

    @skipIfNoFBGEMM
    def test_clamp(self):
        # 定义一个测试函数，用于检查不同的 clamp 操作和量化类型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块，包括 Conv2d、ReLU6、Hardtanh 层，每个层的初始化参数不同
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu6 = torch.nn.ReLU6()
                self.relu6_ = torch.nn.ReLU6(True)
                self.hardtanh = torch.nn.Hardtanh()
                self.hardtanh_ = torch.nn.Hardtanh(inplace=True)

            def forward(self, x):
                # 模型前向传播函数，依次执行 Conv2d、ReLU6、Hardtanh 操作并返回结果
                x = self.conv(x)
                x = self.relu6(x)
                self.relu6_(x)
                x = F.relu6(x)
                # 执行 torch.clamp 操作限制张量 x 的值在 -3 到 3 之间
                x = torch.clamp(x, -3, 3)
                # 执行 in-place torch.clamp 操作限制张量 x 的值在 -2.5 到 2.5 之间
                x = x.clamp(-2.5, 2.5)
                # 暂时禁用的 in-place torch.clamp 操作，当量化的 clamp_ 操作准备好时启用
                # x = x.clamp_(-2, 2)
                # 执行 Hardtanh 操作限制张量 x 的值在 -1 到 1 之间
                x = self.hardtanh(x)
                self.hardtanh_(x)
                x = F.hardtanh(x)
                return x

        # 创建测试数据
        data = (torch.rand((1, 2, 5, 5), dtype=torch.float),)
        # 期望的节点列表，包括 torch.quantize_per_tensor 函数调用和 Conv2d 层的量化和反量化操作
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_method('dequantize')
        ]
        # 对于每种静态量化类型，检查图模式下的操作
        for quant_type in self.static_quant_types:
            self.checkGraphModeFxOp(
                M(), data, quant_type, expected_node_list=node_list)
    # 定义一个测试方法，用于测试使用固定量化参数的操作在FP16（半精度浮点数）上的情况
    def test_fixed_qparams_ops_fp16(self):
        # 定义一个继承自torch.nn.Module的模型类M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()  # 初始化Sigmoid激活函数
                self.tanh = torch.nn.Tanh()  # 初始化Tanh激活函数

            def forward(self, x):
                x = self.sigmoid(x)  # 使用实例中的Sigmoid激活函数对输入x进行处理
                x = torch.sigmoid(x)  # 使用torch中的Sigmoid激活函数对x进行处理
                x = x.sigmoid()  # 对x使用其自身的sigmoid方法进行处理
                x = self.tanh(x)  # 使用实例中的Tanh激活函数对x进行处理
                x = torch.tanh(x)  # 使用torch中的Tanh激活函数对x进行处理
                x = x.tanh()  # 对x使用其自身的tanh方法进行处理
                return x  # 返回处理后的x

        data = (torch.randn((2, 2, 2, 2), dtype=torch.float),)  # 创建一个随机数据元组
        quant_type = QuantType.STATIC  # 设置量化类型为静态量化
        # TODO: use get_default_qconfig_mapping once it handles fp16
        # 定义量化配置映射，使用float16_static_qconfig设置全局量化配置
        qconfig_mapping = QConfigMapping().set_global(float16_static_qconfig)
        # 获取用于测试的仅限本地后端配置
        backend_config = get_test_only_legacy_native_backend_config()
        # 定义节点发生情况字典，表示调用"to"方法的节点发生了7次
        node_occurrence = {
            ns.call_method("to"): 7
        }
        # 调用测试函数checkGraphModeFxOp，验证模型M在给定数据、量化类型、自定义量化配置字典、预期节点发生情况和后端配置下的运行情况
        self.checkGraphModeFxOp(
            M(), data, quant_type, custom_qconfig_dict=qconfig_mapping,
            expected_node_occurrence=node_occurrence,
            backend_config=backend_config)

    # 定义一个测试方法，用于测试使用固定量化参数的操作在QINT8（量化整数8位）上的情况
    def test_fixed_qparams_ops_qint8(self):
        # 定义一个继承自torch.nn.Module的模型类M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()  # 初始化Sigmoid激活函数
                self.tanh = torch.nn.Tanh()  # 初始化Tanh激活函数

            def forward(self, x):
                x = self.sigmoid(x)  # 使用实例中的Sigmoid激活函数对输入x进行处理
                x = torch.sigmoid(x)  # 使用torch中的Sigmoid激活函数对x进行处理
                x = x.sigmoid()  # 对x使用其自身的sigmoid方法进行处理
                x = self.tanh(x)  # 使用实例中的Tanh激活函数对x进行处理
                x = torch.tanh(x)  # 使用torch中的Tanh激活函数对x进行处理
                x = x.tanh()  # 对x使用其自身的tanh方法进行处理
                return x  # 返回处理后的x

        data = (torch.randn((2, 2, 2, 2), dtype=torch.float),)  # 创建一个随机数据元组
        quant_type = QuantType.STATIC  # 设置量化类型为静态量化
        # 定义激活函数和权重的默认量化配置对象
        qconfig = torch.ao.quantization.QConfig(
            activation=HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
            weight=default_weight_observer)
        # 获取默认的量化配置映射，并设置全局量化配置为qconfig
        qconfig_mapping = get_default_qconfig_mapping().set_global(qconfig)
        # 定义节点发生情况字典，表示调用torch.quantize_per_tensor函数和"dequantize"方法的节点都发生了7次
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 7,
            ns.call_method("dequantize"): 7
        }
        # 调用测试函数checkGraphModeFxOp，验证模型M在给定数据、量化类型、自定义量化配置字典、预期节点发生情况和是否为参考模型的情况下的运行情况
        self.checkGraphModeFxOp(
            M(), data, quant_type, custom_qconfig_dict=qconfig_mapping,
            expected_node_occurrence=node_occurrence, is_reference=True)
    def test_fixed_qparams_ops_wrong_qconfig(self):
        """ Test that wrong qconfigs for fixed qparams ops results in the ops not being quantized.
        """
        # 定义一个测试类，用于验证固定量化参数操作中错误的量化配置不会使操作被量化
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()  # 实例化 Sigmoid 激活函数
                self.tanh = torch.nn.Tanh()  # 实例化 Tanh 激活函数

            def forward(self, x):
                x = self.sigmoid(x)  # 使用实例化的 Sigmoid 对输入进行操作
                x = torch.sigmoid(x)  # 使用 torch 自带的 sigmoid 函数对输入进行操作
                x = x.sigmoid()  # 直接使用 tensor 对象的 sigmoid 方法对输入进行操作
                x = self.tanh(x)  # 使用实例化的 Tanh 对输入进行操作
                x = torch.tanh(x)  # 使用 torch 自带的 tanh 函数对输入进行操作
                x = x.tanh()  # 直接使用 tensor 对象的 tanh 方法对输入进行操作
                return x

        data = (torch.randn((2, 2, 2, 2), dtype=torch.float),)  # 创建一个随机数据元组
        qconfig_mapping = QConfigMapping().set_global(default_qconfig)  # 设置全局量化配置映射
        m = M().eval()  # 实例化 M 类并设置为评估模式
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 0,  # 设置函数调用计数，用于验证图中是否包含指定函数调用
            ns.call_method("dequantize"): 0,  # 设置方法调用计数，用于验证图中是否包含指定方法调用
        }
        self.checkGraphModeFxOp(
            m, data, QuantType.STATIC, custom_qconfig_dict=qconfig_mapping,
            expected_node_occurrence=node_occurrence, is_reference=True)
        self.assertTrue(isinstance(m.sigmoid, torch.nn.Sigmoid))  # 验证模型中的 sigmoid 属性是否为 torch.nn.Sigmoid 类型
        self.assertTrue(isinstance(m.tanh, torch.nn.Tanh))  # 验证模型中的 tanh 属性是否为 torch.nn.Tanh 类型

    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    def test_ave_pool_with_custom_cfg(self):
        """ A test that checks correct patterns are produced for
        avg_pool2d with customized config
        """
        # 定义一个测试类，用于验证 avg_pool2d 操作在自定义配置下是否生成正确的模式
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool2d = torch.nn.AvgPool2d(3)  # 实例化 AvgPool2d 操作，kernel_size=3

            def forward(self, x):
                x = self.avg_pool2d(x)  # 对输入 x 进行 AvgPool2d 操作
                return x

        # 此模型不可执行，因为所有操作都在同一个 forward 方法中
        m = M().eval()  # 实例化 M 类并设置为评估模式
        # 没有可融合的内容，因此跳过融合步骤
        qconfig_dict = {'': default_qconfig}  # 设置量化配置字典
        example_inputs = (torch.randn(1, 3, 3, 3),)  # 创建示例输入数据
        prepared = prepare_fx(
            m, qconfig_dict, example_inputs=example_inputs,
            prepare_custom_config={"input_quantized_idxs": [0]})  # 使用量化配置准备模型

        # 不可运行
        quantized = convert_fx(prepared)  # 将准备好的模型转换为量化模型

        # 这里检查从第一个卷积层输出的反量化是否被传播到最后，以避免插入额外的观察器
        # 检查量化和反量化的确切次数
        count_check = {
            ns.call_method('dequantize') : 1
        }
        order_check = [
            ns.call_module(nn.AvgPool2d),  # 验证图中是否包含 AvgPool2d 模块调用
            ns.call_method('dequantize'),  # 验证图中是否包含 dequantize 方法调用
        ]
        self.checkGraphModuleNodes(
            quantized,
            expected_node_occurrence=count_check,
            expected_node_list=order_check)  # 验证模型图中节点的出现次数和顺序

    @skipIfNoFBGEMM
    def test_general_value_ops(self):
        """ 测试检查对所有支持的常规值操作生成正确的模式，如 aten::avg_pool2d，但实际上不检查这些操作的执行情况 """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.avg_pool1d = torch.nn.AvgPool1d(3)
                self.avg_pool2d = torch.nn.AvgPool2d(3)
                self.avg_pool3d = torch.nn.AvgPool3d(3)
                self.adaptive_avg_pool1d = torch.nn.AdaptiveAvgPool1d(1)
                self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.adaptive_avg_pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

            def forward(self, x):
                x = self.conv(x)  # 执行卷积操作
                x = self.avg_pool1d(x)  # 执行平均池化1D操作
                x = self.avg_pool2d(x)  # 执行平均池化2D操作
                x = self.avg_pool3d(x)  # 执行平均池化3D操作
                x = self.adaptive_avg_pool1d(x)  # 执行自适应平均池化1D操作
                x = self.adaptive_avg_pool2d(x)  # 执行自适应平均池化2D操作
                x = self.adaptive_avg_pool3d(x)  # 执行自适应平均池化3D操作
                x = F.avg_pool1d(x, 3)  # 使用函数式API执行平均池化1D操作
                x = F.avg_pool2d(x, 3)  # 使用函数式API执行平均池化2D操作
                x = F.avg_pool3d(x, 3)  # 使用函数式API执行平均池化3D操作
                x = F.adaptive_avg_pool1d(x, (1))  # 使用函数式API执行自适应平均池化1D操作
                x = F.adaptive_avg_pool2d(x, (1, 1))  # 使用函数式API执行自适应平均池化2D操作
                x = F.adaptive_avg_pool3d(x, (1, 1, 1))  # 使用函数式API执行自适应平均池化3D操作
                x = torch.mean(x)  # 计算张量的均值
                x = torch.mean(x, [2, 3], False)  # 计算指定维度的均值，不保留维度
                x = x.mean()  # 计算张量的全局均值
                x = x.mean([2, 3], True)  # 计算指定维度的均值，并保留维度
                x = F.interpolate(x, 4, mode='nearest')  # 使用插值函数进行插值，最近邻插值模式
                x = F.interpolate(x, 4, mode='linear')  # 使用插值函数进行插值，线性插值模式
                x = self.conv(x)  # 再次执行卷积操作
                return x

        # 这个模型不可执行，因为我们只是把所有操作放在了同一个 forward 方法里
        m = M().eval()
        # 没有融合的步骤，所以跳过融合步骤
        qconfig_dict = {'': default_qconfig}
        example_inputs = (torch.randn(1, 3, 3, 3),)
        prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 不可运行
        quantized = convert_fx(prepared)

        # 这里检查第一个卷积输出的去量化是否传播到最后，以确保我们不会插入额外的观察器
        # 检查量化和去量化的确切次数
        count_check = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method('dequantize'): 1
        }
        order_check = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_module(nnq.Conv2d),
            ns.call_method('dequantize'),
        ]
        self.checkGraphModuleNodes(
            quantized,
            expected_node_occurrence=count_check,
            expected_node_list=order_check)
    def test_copy_node_fp32_input(self):
        """ CopyNode works for both fp32 and int8 inputs, this is a test to make
        sure that a CopyNode can be successfully quantized in both cases
        """
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            def forward(self, x):
                # 使用 ReLU 激活函数处理输入
                x = x.relu()
                return x

        # 创建并设为评估模式
        m = M().eval()
        # 准备模型以便量化，指定默认重用输入的量化配置
        m = prepare_fx(m, {"": default_reuse_input_qconfig}, example_inputs=(torch.randn(1),))
        # 将准备好的模型转换为量化表示
        m = convert_fx(m)
        # 确保模型可以运行
        m(torch.rand(1))

    def test_getitem(self):
        """ Make sure we only insert observer for getitem if the following node is matched
        or needs to be quantized
        """
        # 定义一个简单的神经网络模型类，接收多个输入并返回第一个输入
        class M(torch.nn.Module):
            def forward(self, xs):
                x = xs[0]
                return x

        # 创建并设为评估模式
        m = M().eval()
        example_inputs = (torch.rand(1, 2),)
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping()
        # 准备模型以便量化，指定示例输入和量化配置映射
        m = prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
        # 检查预期的图模块节点出现情况，确保不包含 MinMaxObserver 节点
        self.checkGraphModuleNodes(m, expected_node_occurrence={
            ns.call_module(torch.ao.quantization.MinMaxObserver): 0
        })
        # 将准备好的模型转换为量化表示
        m = convert_fx(m)
        # 使用示例输入运行模型
        m(*example_inputs)

        # 定义另一个简单的神经网络模型类，接收多个输入并返回经过 sigmoid 激活后的第一个输入
        class M2(torch.nn.Module):
            def forward(self, xs):
                x = xs[0]
                x = torch.sigmoid(x)
                return x

        # 创建并设为评估模式
        m2 = M2().eval()
        example_inputs = ([torch.rand(1, 2)],)
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping()
        # 准备模型以便量化，指定示例输入和量化配置映射
        m2 = prepare_fx(m2, qconfig_mapping, example_inputs=example_inputs)
        # 检查预期的图模块节点出现情况，确保包含两个 FixedQParamsObserver 节点
        self.checkGraphModuleNodes(m2, expected_node_occurrence={
            ns.call_module(torch.ao.quantization.FixedQParamsObserver): 2
        })
        # 将准备好的模型转换为量化表示
        m2 = convert_fx(m2)
        # 检查预期的图模块节点列表，确保包含量化和反量化操作节点
        self.checkGraphModuleNodes(m2, expected_node_list=[
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize")
        ])
        # 使用示例输入运行模型
        m2(*example_inputs)

        # 定义另一个简单的神经网络模型类，接收一个输入并对其进行 sigmoid 处理后返回
        class M3(torch.nn.Module):
            def forward(self, x):
                s = x.shape
                n, c = s[:2]
                x = torch.sigmoid(x)
                return x

        # 创建并设为评估模式
        m3 = M3().eval()
        example_inputs = (torch.rand(1, 2, 3, 4),)
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping()
        # 准备模型以便量化，指定示例输入和量化配置映射
        m3 = prepare_fx(m3, qconfig_mapping, example_inputs=example_inputs)
        # 检查预期的图模块节点出现情况，确保包含两个 FixedQParamsObserver 节点
        self.checkGraphModuleNodes(m3, expected_node_occurrence={
            ns.call_module(torch.ao.quantization.FixedQParamsObserver): 2
        })
        # 将准备好的模型转换为量化表示
        m3 = convert_fx(m3)
        # 检查预期的图模块节点列表，确保包含量化和反量化操作节点
        self.checkGraphModuleNodes(m3, expected_node_list=[
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize")
        ])
        # 使用示例输入运行模型
        m3(*example_inputs)
    # 定义一个测试函数 test_embedding，用于测试嵌入模型的量化操作
    def test_embedding(self):
        # 定义一个简单的神经网络模型 M，包含一个嵌入层
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

            # 前向传播函数，接收索引值作为输入，返回嵌入层的输出
            def forward(self, indices):
                return self.emb(indices)

        # 遍历不同的量化配置类型
        for qconfig_type in [float_qparams_weight_only_qconfig, float_qparams_weight_only_qconfig_4bit]:
            # 创建一个 M 类的实例，并设为评估模式
            model = M().eval()
            # 创建包含索引张量的示例输入
            indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
            example_inputs = (indices,)
            # 创建量化后的嵌入节点
            quantized_node = ns.call_module(nnq.Embedding)

            # 检查动态量化
            self.checkGraphModeFxOp(
                model,
                example_inputs,
                QuantType.DYNAMIC,
                quantized_node,
                custom_qconfig_dict={"": qconfig_type}
            )
            # 再次创建 M 类的实例，并设为评估模式
            model = M().eval()

            # 定义多个量化配置和相应的节点
            configs = [
                (qconfig_type, ns.call_module(nnq.Embedding)),
                (None, ns.call_module(nn.Embedding)),
                (default_qconfig, ns.call_module(nn.Embedding)),
            ]

            # 检查静态量化
            for qconfig, node in configs:
                qconfig_dict = {"": qconfig}
                # 准备模型进行量化，并指定示例输入
                m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
                # 检查图模块中的预期节点出现次数
                self.checkGraphModuleNodes(m, expected_node_occurrence={
                    ns.call_module(torch.ao.quantization.MinMaxObserver): 0
                })
                # 执行转换为量化模型
                m = convert_fx(m)
                # 再次检查图模块中的预期节点
                self.checkGraphModuleNodes(m, expected_node=node)
                # 确保模型可以成功运行
                m(*example_inputs)
    def test_embedding_bag(self):
        # 定义内部测试用的模型类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 EmbeddingBag 层，设置词汇表大小为 10，嵌入维度为 12，包括最后一个偏移量
                self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True)

            # 前向传播函数，接受 indices 和 offsets 作为输入，并返回 EmbeddingBag 层的输出
            def forward(self, indices, offsets):
                return self.emb(indices, offsets)

        # 指定测试用的 indices 和 offsets 张量
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        # 调用 nnq.EmbeddingBag，这里 ns 是一个命名空间或模块的缩写，用于调用 quantized node
        quantized_node = ns.call_module(nnq.EmbeddingBag)
        example_inputs = (indices, offsets)

        # 对于每种 dtype 进行测试，torch.quint8 和 torch.quint4x2
        for dtype in [torch.quint8, torch.quint4x2]:
            # 创建并评估模型 M
            model = M().eval()
            # 使用 PerChannelMinMaxObserver 来观察浮点数的量化参数
            float_qparams_observer = PerChannelMinMaxObserver.with_args(dtype=dtype,
                                                                        qscheme=torch.per_channel_affine_float_qparams,
                                                                        ch_axis=0)
            # 定义 float_qparams_qconfig 作为 QConfig 的实例
            float_qparams_qconfig = QConfig(activation=default_placeholder_observer,
                                            weight=float_qparams_observer)
            # 检查图模式下的操作，使用动态量化类型 QuantType.DYNAMIC
            self.checkGraphModeFxOp(
                model,
                example_inputs,
                QuantType.DYNAMIC,
                quantized_node,
                custom_qconfig_dict={"": float_qparams_qconfig}
            )

        # 检查 None 和静态 qconfig 下的运行情况
        for qconfig in [None, default_qconfig]:
            qconfig_dict = {"": default_qconfig}
            m = M().eval()
            # 准备模型，使用 prepare_fx 函数进行准备，传入 qconfig_dict 和 example_inputs
            m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
            # 检查图模块节点的出现情况，预期不应有 torch.ao.quantization.MinMaxObserver 节点
            self.checkGraphModuleNodes(m, expected_node_occurrence={
                ns.call_module(torch.ao.quantization.MinMaxObserver): 0
            })
            # 转换模型为量化模型
            m = convert_fx(m)
            # 再次检查图模块节点，预期有 nn.EmbeddingBag 节点
            self.checkGraphModuleNodes(m, expected_node=ns.call_module(nn.EmbeddingBag))
            # 确保模型可以运行
            m(*example_inputs)
    # 定义测试 RNN 实现的私有方法，用于测试不同的量化配置和模块类型组合
    def _test_rnn_impl(self, qconfigs, M, module_type_strs, module_types, sample_input):
        # 生成所有量化配置和模块类型的组合
        options = itertools.product(qconfigs, module_type_strs)
        # 遍历每个配置和类型的组合
        for qconfig, module_type_str in options:
            # 创建并获取 eager 模式下的量化模型
            model_eager = M(module_type_str).eval()
            # 深度复制 eager 模型以备用于图模式
            model_graph = copy.deepcopy(model_eager)
            # 如果当前量化引擎为 qnnpack 并且配置为 fp16 dynamic qconfig，则跳过当前循环
            if torch.backends.quantized.engine == 'qnnpack' and \
               qconfig is float16_dynamic_qconfig:
                continue
                # fp16 dynamic quant is not supported for qnnpack

            # 创建一个字典，将每种模块类型关联到对应的 qconfig
            eager_qconfig_dict = dict.fromkeys(module_types, qconfig)
            # 在 eager 模式下对模型进行动态量化
            model_eager = quantize_dynamic(model_eager, qconfig_spec=eager_qconfig_dict)

            # 准备用于图模式量化的配置字典，将所有模块类型与 qconfig 元组列表关联
            graph_qconfig_dict = {
                "object_type": [
                    (x, qconfig) for x in module_types
                ]
            }
            # 使用 FX 函数库准备图模式下的模型，并提供示例输入
            model_graph = prepare_fx(model_graph, graph_qconfig_dict, example_inputs=(sample_input,))
            # 将图模式下的模型进行转换
            model_graph = convert_fx(model_graph)
            # 断言两个模型在相同输入下的输出一致
            self.assertEqual(model_eager(sample_input), model_graph(sample_input))
            # 检查图模式下的模型是否可脚本化，并使用示例输入进行测试
            self.checkScriptable(model_graph, [[sample_input]], True)

    # 装饰器：覆盖量化引擎，用于测试 RNN 单元
    @override_qengines
    def test_rnn_cell(self):
        # 如果当前量化引擎不是 'fbgemm' 或 'qnnpack'，则直接返回
        if torch.backends.quantized.engine not in ('fbgemm', 'qnnpack'):
            return
        # 定义可用的量化配置列表
        qconfigs = [per_channel_dynamic_qconfig, default_dynamic_qconfig, float16_dynamic_qconfig]
        # 定义 RNN 单元模块类型的字符串列表
        module_type_strs = ['LSTMCell', 'GRUCell', 'RNNTanh', 'RNNReLU']
        # 定义 RNN 单元模块类型列表
        module_types = [torch.nn.LSTMCell, torch.nn.GRUCell, torch.nn.RNNCell]
        # 定义示例输入张量，用于测试
        sample_input = torch.tensor([[100, -155],
                                     [-155, 100],
                                     [100, -155]], dtype=torch.float)
        # 调用 _test_rnn_impl 方法进行 RNN 单元的测试
        self._test_rnn_impl(qconfigs, RNNCellDynamicModel, module_type_strs, module_types, sample_input)

    # 装饰器：覆盖量化引擎，用于测试 RNN
    @override_qengines
    def test_rnn(self):
        # 如果当前量化引擎不是 'fbgemm' 或 'qnnpack'，则直接返回
        if torch.backends.quantized.engine not in ('fbgemm', 'qnnpack'):
            return
        # 定义可用的量化配置列表
        qconfigs = [per_channel_dynamic_qconfig, default_dynamic_qconfig, float16_dynamic_qconfig]
        # 定义 RNN 模块类型的字符串列表
        module_type_strs = ['LSTM', 'GRU']
        # 定义 RNN 模块类型列表
        module_types = [torch.nn.LSTM, torch.nn.GRU]
        # 定义迭代次数
        niter = 10
        # 定义示例输入张量，用于测试，并对其进行维度扩展和重复以满足迭代次数
        sample_input = torch.tensor([[100, -155],
                                     [-155, 100],
                                     [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1)
        # 调用 _test_rnn_impl 方法进行 RNN 的测试
        self._test_rnn_impl(qconfigs, RNNDynamicModel, module_type_strs, module_types, sample_input)
    # 定义测试函数 _test_conv_transpose_impl，用于测试转置卷积操作的实现
    def _test_conv_transpose_impl(
            self, float_cls: Callable, q_cls: Callable, data: torch.Tensor):
        # 使用 qnnpack 引擎覆盖当前量化引擎
        with override_quantized_engine('qnnpack'):
            # 创建两个 FP32 版本的 FX 和 Eager 模型
            m1 = torch.nn.Sequential(float_cls(1, 1, 1))
            m2 = torch.nn.Sequential(float_cls(1, 1, 1))
            # 从 m1 复制状态到 m2
            m2.load_state_dict(m1.state_dict())
            # 将 m2 包装成量化模型
            m2 = torch.ao.quantization.QuantWrapper(m2)
            # 对 FX 图执行操作
            result_dict = self.checkGraphModeFxOp(
                m1, (data,), QuantType.STATIC,
                expected_node_occurrence={
                    ns.call_module(q_cls): 1,
                })
            # 获取量化输出结果
            q_result1 = result_dict["quantized_output"]
            # 对 Eager 模型进行量化准备和转换
            m2.qconfig = get_default_qconfig(torch.backends.quantized.engine)
            m2.eval()
            m2p = torch.ao.quantization.prepare(m2)
            m2p(data)
            m2q = torch.ao.quantization.convert(m2p)
            q_result2 = m2q(data)
            # 验证两个量化结果是否一致
            self.assertEqual(q_result1, q_result2)

    # 当支持 qnnpack 引擎时才运行该测试，否则跳过
    @unittest.skipUnless('qnnpack' in supported_qengines,
                         "This Pytorch Build has not been built with or does not support QNNPACK")
    # 测试 ConvTranspose1d 操作
    def test_conv_transpose_1d(self):
        # 调用 _test_conv_transpose_impl 函数测试 ConvTranspose1d
        self._test_conv_transpose_impl(
            torch.nn.ConvTranspose1d, nnq.ConvTranspose1d, torch.randn(4, 1, 4))

    # 当支持 qnnpack 引擎时才运行该测试，否则跳过
    @unittest.skipUnless('qnnpack' in supported_qengines,
                         "This Pytorch Build has not been built with or does not support QNNPACK")
    # 测试 ConvTranspose2d 操作
    def test_conv_transpose_2d(self):
        # 调用 _test_conv_transpose_impl 函数测试 ConvTranspose2d
        self._test_conv_transpose_impl(
            torch.nn.ConvTranspose2d, nnq.ConvTranspose2d, torch.randn(4, 1, 4, 4))
    # 定义一个名为 test_reshape_fp16 的测试方法
    def test_reshape_fp16(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 类初始化方法，接收权重 w 和偏置 b 作为参数
            def __init__(self, w, b):
                super().__init__()
                self.w = w  # 初始化权重
                self.b = b  # 初始化偏置

            # 前向传播方法，接收输入 x 作为参数
            def forward(self, x):
                # 执行线性操作并返回结果
                x = torch.nn.functional.linear(x, self.w)
                # 对结果进行形状重塑
                x = x.reshape(-1, 4)
                # 再次执行线性操作并返回结果
                x = torch.nn.functional.linear(x, self.w)
                return x

        # 随机生成一个 4x4 的权重 w
        w = torch.randn(4, 4)
        # 随机生成一个长度为 4 的偏置 b
        b = torch.randn(4)
        # 创建 M 类的实例 m，并设置为评估模式
        m = M(w, b).eval()
        
        # 定义量化配置字典 qconfig_dict
        qconfig_dict = {
            # 空字符串对应的 qconfig 将 reshape 操作量化为 fp16
            "": float16_static_qconfig,
            # "object_type" 对应的量化配置为 [(torch.nn.functional.linear, default_qconfig)]
            "object_type": [
                (torch.nn.functional.linear, default_qconfig)
            ]
        }
        
        # 获取测试用的后端配置
        backend_config = get_test_only_legacy_native_backend_config()
        
        # 创建示例输入 example_inputs，这里是一个形状为 (1, 4) 的随机张量
        example_inputs = (torch.randn(1, 4),)
        
        # 对模型 m 应用量化准备，使用给定的 qconfig_dict 和示例输入
        m = prepare_fx(
            m, qconfig_dict, example_inputs=example_inputs,
            backend_config=backend_config)
        
        # 定义预期出现次数的字典 expected_occurrence
        expected_occurrence = {
            # call_module(torch.ao.quantization.MinMaxObserver) 应该出现 6 次
            ns.call_module(torch.ao.quantization.MinMaxObserver): 6,
            # call_module(torch.ao.quantization.PlaceholderObserver) 应该出现 2 次，用于 reshape 操作的输入和输出
            ns.call_module(torch.ao.quantization.PlaceholderObserver): 2
        }
        
        # 检查模型 m 的节点出现情况是否符合预期
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence
        )
        
        # 将模型 m 转换为量化后的表示，使用指定的后端配置
        m = convert_fx(m, backend_config=backend_config)
        
        # 更新预期出现次数的字典 expected_occurrence
        expected_occurrence = {
            # call_function(torch.quantize_per_tensor) 应该出现 2 次
            ns.call_function(torch.quantize_per_tensor): 2,
            # call_method("dequantize") 应该在第一个线性操作之后、形状重塑之前以及输出之前各出现 3 次
            ns.call_method("dequantize"): 3,
            # call_method("to") 应该在形状重塑之前出现 1 次，用于将数据转换为 fp16
            ns.call_method("to"): 1,
            # call_function(torch.ops.quantized.linear) 应该出现 2 次
            ns.call_function(torch.ops.quantized.linear): 2
        }
        
        # 再次检查模型 m 的节点出现情况是否符合更新后的预期
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence
        )
        
        # 确保模型 m 能够正常运行，传入一个形状为 (2, 4) 的随机张量作为输入
        m(torch.randn(2, 4))
    def test_multiple_qconfigs_for_single_value(self):
        """ Test multiple qconfigs for a single value"""
        # 定义一个测试用例类，用于验证多个量化配置对单个值的影响
        class M(torch.nn.Module):
            def __init__(self, w, b):
                super().__init__()
                self.w = w
                self.b = b

            def forward(self, x):
                # 执行线性函数和sigmoid激活函数
                x = torch.nn.functional.linear(x, self.w)
                x = torch.sigmoid(x)
                return x

        # 创建随机权重和偏置张量
        w = torch.randn(4, 4)
        b = torch.randn(4)
        # 实例化模型并设置为评估模式
        m = M(w, b).eval()
        # TODO: use get_default_qconfig_mapping once it handles fp16
        # 创建量化配置映射对象，并设置全局量化配置和线性函数的默认量化配置
        qconfig_mapping = QConfigMapping() \
            .set_global(float16_static_qconfig) \
            .set_object_type(torch.nn.functional.linear, default_qconfig)
        # 创建示例输入
        example_inputs = (torch.randn(1, 4),)
        # 获取测试专用的本地后端配置
        backend_config = get_test_only_legacy_native_backend_config()
        # 准备模型以进行量化仿真，并传入量化配置映射、示例输入和后端配置
        m = prepare_fx(
            m, qconfig_mapping, example_inputs=example_inputs,
            backend_config=backend_config)
        # 期望出现的节点及其次数
        expected_occurrence = {
            # 线性函数的输入、权重和输出的MinMaxObserver节点
            ns.call_module(torch.ao.quantization.MinMaxObserver): 3,
            # sigmoid函数的输入和输出的PlaceholderObserver节点
            ns.call_module(torch.ao.quantization.PlaceholderObserver): 2,
        }
        # 检查模型的节点是否符合预期
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence
        )
        # 确保模型能够成功转换
        m = convert_fx(m)
        # 更新期望的节点及其次数，用于转换后的模型
        expected_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method("dequantize"): 3,
            ns.call_method("to"): 2
        }
        # 再次检查模型的节点是否符合新的预期
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence
        )

    def test_boolean_tensor(self):
        """ Make sure we don't insert observer for boolean Tensors """
        # 定义一个测试用例类，确保我们不会为布尔张量插入观察器
        class M(torch.nn.Module):
            def forward(self, x, mask):
                # 扩展布尔掩码张量的维度
                mask = mask.unsqueeze(0)
                mask = mask.unsqueeze(1)
                # 使用掩码填充输入张量
                x = x.masked_fill(mask, 1)
                return x

        # 实例化模型并设置为评估模式
        m = M().eval()
        # 创建示例输入，包括一个随机张量和一个布尔张量
        example_inputs = (torch.rand(1, 2, 3, 4), torch.rand(3, 4).bool())
        # 准备模型以进行量化仿真，并传入默认的量化配置和示例输入
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        # 预期不应出现MinMaxObserver节点
        expected_occurrence = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 0
        }
        # 检查模型的节点是否符合预期
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence)
        # 将模型转换为量化模型
        m = convert_fx(m)
        # 执行转换后的模型的测试
        m(*example_inputs)

    def test_chunk(self):
        # 定义一个测试用例类，用于验证torch.chunk函数的行为
        class M(torch.nn.Module):
            def forward(self, x):
                # 将输入张量按维度切分为两部分
                x, y = torch.chunk(x, 2)
                # 对切分后的张量部分进行加法操作
                x = x + y
                return x

        # 实例化模型并设置为评估模式
        m = M().eval()
        # 创建示例输入
        example_inputs = (torch.rand(2, 2, 2, 2),)
        # 准备模型以进行量化仿真，并传入默认的量化配置和示例输入
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        # 执行准备后的模型的测试
        m(*example_inputs)
        # 将模型转换为量化模型
        m = convert_fx(m)
        # 执行转换后的模型的测试
        m(*example_inputs)
        # 确保一切正常运行
    def test_qmatmul(self):
        class M(torch.nn.Module):
            # 定义神经网络模型的前向传播方法
            def forward(self, x, y):
                # 使用 torch.matmul 计算输入张量 x 和 y 的矩阵乘积
                z = torch.matmul(x, y)
                # 返回计算结果张量 z
                return z

        # 创建并评估模型实例
        m = M().eval()
        # 准备用于示例输入的随机张量
        example_inputs = (torch.randn(2, 2), torch.randn(2, 2))
        # 获取指定配置的量化配置字典
        qconfig_dict = get_default_qconfig_mapping("fbgemm")
        # 准备模型以进行量化仿真
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 在量化仿真模式下运行模型的示例输入
        mp(*example_inputs)
        # 将准备好的模型转换为量化后的模型
        mq = convert_fx(mp)
        # 定义期望的节点发生次数字典，包括 torch.matmul 和 torch.ops.quantized.matmul 的调用次数
        expected_occurrence = {
            ns.call_function(torch.matmul): 0,
            ns.call_function(torch.ops.quantized.matmul): 1,
        }
        # 检查量化后模型中的节点发生情况是否符合期望
        self.checkGraphModuleNodes(
            mq,
            expected_node_occurrence=expected_occurrence)
        # 验证模型在示例输入上运行时不会崩溃
        res = mq(*example_inputs)
    def test_pixel_shuffle_module(self) -> None:
        # 定义一个包含偏置参数的简单模块
        class MyBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(8))

        # 定义包含卷积层、像素重排层和偏置模块的神经网络模型
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 一个不带偏置的1x1卷积层，输入输出通道数均为8
                self.conv = nn.Conv2d(8, 8, 1, bias=False)
                # 像素重排层，上采样因子为2
                self.ps = nn.PixelShuffle(upscale_factor=2)
                # 调用偏置模块
                self.bias = MyBias()

            def forward(self, x):
                # 卷积计算
                x = self.conv(x)
                # 像素重排
                x = self.ps(x)
                # 重新调整张量形状
                x = x.view(-1, 8, 2, 2)
                # 获取偏置参数
                bias = self.bias.bias
                # 返回卷积结果加上偏置
                return x + bias

        # 获取 QNNPACK 后端配置
        backend_config = get_qnnpack_backend_config()
        # 获取 qnnpack 的默认量化配置映射
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        # 创建模型实例
        model = MyModel()
        # 准备模型以适应 QNNPACK 的后端配置和量化配置
        m = prepare_fx(
            model,
            qconfig_mapping=qconfig_mapping,
            example_inputs=(torch.randn(1, 8, 3, 3),),
            backend_config=backend_config
        )
        # 将模型转换为量化的效果图模块
        m = convert_fx(m)
        # 预期的节点出现情况，包括两次量化函数调用和一次去量化方法调用
        expected_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 2,
            ns.call_method("dequantize"): 1,
            # 像素重排模块调用一次
            ns.call_module(nn.PixelShuffle): 1,
        }
        # 检查生成的效果图模块的节点出现情况是否符合预期
        self.checkGraphModuleNodes(m, expected_node_occurrence=expected_occurrence)
    def test_pixel_unshuffle(self):
        # 定义一个包含偏置的自定义模块
        class MyBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(64))

        # 定义一个包含卷积和偏置的自定义模型
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个不带偏置的 1x1 卷积层
                self.conv = nn.Conv2d(8, 8, 1, bias=False)
                # 实例化自定义的偏置模块
                self.bias = MyBias()

            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 对卷积输出进行像素解组操作
                x = nn.functional.pixel_unshuffle(x, 2)
                # 获取偏置值
                bias = self.bias.bias
                # 返回处理后的张量与偏置值之和
                return x + bias

        # 针对两种后端进行测试
        for backend in ["fbgemm", "qnnpack"]:
            # 根据后端类型获取后端配置
            if backend == "fbgemm":
                backend_config = get_fbgemm_backend_config()
            else:
                backend_config = get_qnnpack_backend_config()
            # 获取默认的量化配置映射
            qconfig_mapping = get_default_qconfig_mapping(backend)
            # 创建模型实例
            model = MyModel()
            # 使用准备函数将模型转换为图模块
            m = prepare_fx(
                model,
                qconfig_mapping=qconfig_mapping,
                example_inputs=(torch.randn(1, 8, 6, 6),),
                backend_config=backend_config
            )
            # 将转换后的图模块进一步转换为量化后的图模块
            m = convert_fx(m)
            # 定义期望的节点出现次数字典
            expected_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 2,
                ns.call_method("dequantize"): 1,
            }
            # 调用检查图模块节点函数，验证预期节点的出现次数
            self.checkGraphModuleNodes(m, expected_node_occurrence=expected_occurrence)

    def test_pixel_unshuffle_module(self) -> None:
        # 定义一个包含偏置的自定义模块
        class MyBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(64))

        # 定义一个包含卷积、像素解组和偏置的自定义模型
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个不带偏置的 1x1 卷积层
                self.conv = nn.Conv2d(8, 8, 1, bias=False)
                # 创建一个像素解组层，指定下采样因子为 2
                self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
                # 实例化自定义的偏置模块
                self.bias = MyBias()

            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 执行像素解组操作
                x = self.unshuffle(x)
                # 获取偏置值
                bias = self.bias.bias
                # 返回处理后的张量与偏置值之和
                return x + bias

        # 针对两种后端进行测试
        for backend in ["fbgemm", "qnnpack"]:
            # 根据后端类型获取后端配置
            if backend == "fbgemm":
                backend_config = get_fbgemm_backend_config()
            else:
                backend_config = get_qnnpack_backend_config()
            # 获取默认的量化配置映射
            qconfig_mapping = get_default_qconfig_mapping(backend)
            # 创建模型实例
            model = MyModel()
            # 使用准备函数将模型转换为图模块
            m = prepare_fx(
                model,
                qconfig_mapping=qconfig_mapping,
                example_inputs=(torch.randn(1, 8, 6, 6),),
                backend_config=backend_config
            )
            # 将转换后的图模块进一步转换为量化后的图模块
            m = convert_fx(m)
            # 定义期望的节点出现次数字典，包含了 PixelUnshuffle 模块的期望次数
            expected_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 2,
                ns.call_method("dequantize"): 1,
                ns.call_module(nn.PixelUnshuffle): 1,
            }
            # 调用检查图模块节点函数，验证预期节点的出现次数
            self.checkGraphModuleNodes(m, expected_node_occurrence=expected_occurrence)
   `
    def test_narrow(self):
        # 定义一个名为 test_narrow 的测试方法
        class MyBias(nn.Module):
            # 定义 MyBias 类，继承自 nn.Module
            def __init__(self):
                super().__init__()
                # 调用父类构造函数
                self.bias = nn.Parameter(torch.randn(4))
                # 初始化一个包含随机数的可学习参数 bias

        class MyModel(nn.Module):
            # 定义 MyModel 类，继承自 nn.Module
            def __init__(self):
                super().__init__()
                # 调用父类构造函数
                self.conv = nn.Conv2d(8, 8, 1, bias=False)
                # 初始化一个 2D 卷积层，输入通道数为 8，输出通道数为 8，卷积核大小为 1x1，无偏置
                self.bias = MyBias()
                # 初始化一个 MyBias 实例作为模型的 bias 属性

            def forward(self, x):
                # 定义模型的前向传播函数
                x = self.conv(x)
                # 对输入 x 进行卷积操作
                x = torch.narrow(x, 1, 0, 4)
                # 在第二个维度上对张量 x 进行裁剪，从索引 0 开始取长度为 4 的子张量
                bias = self.bias.bias
                # 获取 MyBias 类中的 bias 属性
                return x + bias
                # 返回裁剪后的张量 x 与 bias 的和作为输出

        # 遍历两种后端 ["fbgemm", "qnnpack"]
        for backend in ["fbgemm", "qnnpack"]:
            if backend == "fbgemm":
                backend_config = get_fbgemm_backend_config()
                # 如果后端是 fbgemm，获取对应的配置信息
            else:
                backend_config = get_qnnpack_backend_config()
                # 如果后端是 qnnpack，获取对应的配置信息

            qconfig_mapping = get_default_qconfig_mapping(backend)
            # 获取指定后端的默认量化配置映射
            model = MyModel()
            # 创建一个 MyModel 实例
            m = prepare_fx(
                model,
                qconfig_mapping=qconfig_mapping,
                example_inputs=(torch.randn(1, 8, 3, 3),),
                backend_config=backend_config
            )
            # 使用指定的配置信息对模型进行准备
            m = convert_fx(m)
            # 将准备好的模型转换为量化模型

            expected_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 2,
                ns.call_method("dequantize"): 1,
            }
            # 定义期望的节点出现次数字典

            self.checkGraphModuleNodes(m, expected_node_occurrence=expected_occurrence)
            # 调用自定义的检查函数，验证量化模型中特定节点的出现次数是否符合预期
# 定义一个测试类 TestQuantizeFxModels，继承自 QuantizationTestCase，用于量化测试
class TestQuantizeFxModels(QuantizationTestCase):
    
    # 装饰器：如果没有安装 FBGEMM 库，则跳过测试
    @skipIfNoFBGEMM
    # 装饰器：如果没有 CUDA 支持，则跳过测试
    @unittest.skipIf(not TEST_CUDA, "gpu is not available.")
    # 定义一个测试方法 test_static_gpu_convert_basic
    def test_static_gpu_convert_basic(self):
        
        # 定义一个内部类 Net，继承自 nn.Module，用于定义神经网络结构
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu1 = nn.ReLU()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.linear1 = nn.Linear(120, 1)

            # 定义前向传播函数 forward
            def forward(self, x):
                x = self.relu1(self.conv1(x))
                y = self.linear1(x.view(-1))
                return y
        
        # 生成一个随机输入张量 input，并将其移到 CUDA 设备上
        input = torch.randn((5, 1, 6, 6)).to('cuda')
        # 将 input 包装成一个元组 example_inputs
        example_inputs = (input,)
        # 创建一个 Net 类的实例 model，并将其移到 CUDA 设备上，设置为评估模式
        model = Net().to('cuda').eval()
        # 定义量化配置字典 qconfig_dict，使用默认的 fbgemm 量化配置
        qconfig_dict = {"": torch.ao.quantization.get_default_qconfig('fbgemm')}
        # 对模型进行量化准备，返回一个准备好的模型 model_prepared
        model_prepared = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
        # 调用准备好的模型，输入 example_inputs
        model_prepared(*example_inputs)
        # 将准备好的模型转换为参考固定点模型，得到 model_quantized
        model_quantized = convert_to_reference_fx(model_prepared)
        # 使用输入 example_inputs 对量化后的模型进行推理，得到输出 out
        out = model_quantized(*example_inputs)
        # 断言输出的设备类型为 CUDA
        self.assertEqual(out.device.type, 'cuda')

    # 装饰器：如果没有安装 FBGEMM 库，则跳过测试
    @skipIfNoFBGEMM
    # 装饰器：如果没有 CUDA 支持，则跳过测试
    @unittest.skipIf(not TEST_CUDA, "gpu is not available.")
    # 定义一个测试方法 test_switch_device_prepare_convert
    def test_switch_device_prepare_convert(self):

        # 定义一个内部类 Net，继承自 nn.Module，用于定义神经网络结构
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu1 = nn.ReLU()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.linear1 = nn.Linear(120, 1)

            # 定义前向传播函数 forward
            def forward(self, x):
                x = self.relu1(self.conv1(x))
                y = self.linear1(x.view(-1))
                return y

        # 遍历设备列表 ['cuda', 'cpu']
        for device in ['cuda', 'cpu']:
            # 根据当前设备确定后续设备类型
            device_after = 'cuda' if device == 'cpu' else 'cpu'
            # 生成一个随机输入张量 input，并将其移到指定设备上
            input = torch.randn((5, 1, 6, 6)).to(device)
            # 创建一个 Net 类的实例 model，并将其移到指定设备上，设置为评估模式
            model = Net().to(device).eval()
            # 定义量化配置字典 qconfig_dict，使用默认的 fbgemm 量化配置
            qconfig_dict = {"": torch.ao.quantization.get_default_qconfig('fbgemm')}
            # 对模型进行量化准备，返回一个准备好的模型 model_prepared，输入为 (input,)
            model_prepared = prepare_fx(model, qconfig_dict, example_inputs=(input,))
            # 调用准备好的模型，输入 input
            model_prepared(input)
            # 将准备好的模型转移到后续设备类型
            model_prepared.to(device_after)
            # 将准备好的模型转换为参考固定点模型，得到 model_quantized
            model_quantized = convert_to_reference_fx(model_prepared)
            # 使用后续设备类型的输入对量化后的模型进行推理，得到输出 out
            out = model_quantized(input.to(device_after))
            # 断言输出的设备类型为后续设备类型
            self.assertEqual(out.device.type, device_after)
    def test_prepare_serialize_switch_device_convert(self):
        # 定义一个名为 `test_prepare_serialize_switch_device_convert` 的测试方法
        class Net(nn.Module):
            # 定义一个名为 `Net` 的神经网络模型
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 6, 5)  # 添加一个二维卷积层，输入通道数为 1，输出通道数为 6，卷积核大小为 5x5
                self.linear1 = nn.Linear(120, 1)  # 添加一个线性层，输入特征数为 120，输出特征数为 1

            def forward(self, x):
                # 定义前向传播方法
                x = self.conv1(x)  # 执行第一层卷积操作
                y = self.linear1(x.view(-1))  # 执行线性层操作
                return y  # 返回输出

        for device in ['cuda', 'cpu']:
            # 遍历设备类型列表，分别为 'cuda' 和 'cpu'
            for device_after in ['cuda', 'cpu']:
                # 再次遍历设备类型列表，分别为 'cuda' 和 'cpu'
                input = torch.randn((5, 1, 6, 6)).to(device)  # 生成随机输入张量并将其移动到指定设备
                model = Net().to(device).eval()  # 创建 `Net` 模型实例，并将其移动到指定设备后设置为评估模式
                qconfig_dict = {"": torch.ao.quantization.get_default_qconfig('fbgemm')}  # 配置量化参数字典
                model_prepared_first = prepare_fx(model, qconfig_dict, example_inputs=(input,))
                # 准备模型第一次，使用给定的量化配置和示例输入
                model_prepared_second = prepare_fx(model, qconfig_dict, example_inputs=(input,))
                # 准备模型第二次，使用相同的量化配置和示例输入
                model_prepared_first(input)  # 对第一个准备好的模型执行前向传播
                state_dict = model_prepared_first.state_dict()  # 获取第一个模型的状态字典
                del model_prepared_first  # 删除第一个准备好的模型
                model_prepared_second.load_state_dict(state_dict)  # 加载第一个模型的状态字典到第二个模型
                model_prepared_second.to(device_after)  # 将第二个模型移动到另一个指定的设备
                model_quantized = convert_to_reference_fx(model_prepared_second)
                # 将第二个准备好的模型转换为参考模型（quantized reference model）
                out = model_quantized(input.to(device_after))  # 对输入数据执行量化参考模型的前向传播
                self.assertEqual(out.device.type, device_after)  # 断言输出结果的设备类型与预期一致

    @skipIfTorchDynamo("too slow")
    @skip_if_no_torchvision
    def test_model_dropout(self):
        # 定义一个名为 `test_model_dropout` 的测试方法，用于测试模型的 dropout 功能
        from torchvision import models  # 导入 torchvision 中的模型库
        m = models.mobilenet_v3_small()  # 创建一个 MobileNetV3 Small 模型实例
        qconfig_mapping = torch.ao.quantization.get_default_qat_qconfig_mapping('fbgemm')
        # 获取默认的量化训练配置映射
        example_inputs = (torch.randn(1, 3, 224, 224),)  # 生成一个示例输入张量元组
        mp = prepare_qat_fx(m, qconfig_mapping, example_inputs=example_inputs)
        # 准备量化训练的模型，使用给定的量化训练配置映射和示例输入
        mp(*example_inputs)  # 对准备好的量化训练模型执行前向传播
        with override_quantized_engine("qnnpack") if IS_ARM64 else contextlib.nullcontext():
            # 根据平台选择性地覆盖量化引擎为 "qnnpack"，否则使用空的上下文管理器
            mq = convert_fx(mp)  # 将量化训练模型转换为量化后的模型
        mq(*example_inputs)  # 对量化后的模型执行前向传播
    @override_qengines
    # 覆盖默认的量化引擎设置，确保在测试期间使用正确的量化引擎配置

    def test_resnet_base(self):
        # 测试 ResNet 模型的基础功能
        models = [ResNetBase]
        # 生成量化类型和模型的所有组合选项
        options = itertools.product(self.static_quant_types, models)
        for quant_type, M in options:
            # 调用 _test_building_block 方法来测试建模块
            self._test_building_block(quant_type, M)

    @skip_if_no_torchvision
    @skipIfNoFBGEMM
    @unittest.skip("skip for now since tbb failed")
    @skip_if_no_torchvision
    @skipIfNoFBGEMM
    @unittest.skip("TODO: Test is always failing - https://github.com/pytorch/pytorch/issues/54979")
    def test_resnet18_ddp(self):
        # 导入 torchvision 的模型和量化模型
        from torchvision import models
        from torchvision.models import quantization as quantized_models
        # 创建并准备用于急切量化的可量化模型
        eager_quantizable_model = quantized_models.__dict__[name](pretrained=False, quantize=False).eval().float()  # noqa: F821
        # 创建非量化模型
        model = models.__dict__[name](pretrained=False).eval().float()  # noqa: F821
        # 调用 _test_model_impl 方法来测试 ResNet18 在分布式数据并行训练下的功能
        self._test_model_impl(
            'ddp', 'resnet18', model, eager_quantizable_model)

    @override_qengines
    # 再次确认覆盖默认的量化引擎设置
    def test_qat_embeddingbag_linear(self):
        # 遍历所有支持的设备类型
        for device in get_supported_device_types():
            # 定义一个包含 EmbeddingBag 和 Linear 层的神经网络模型
            class EmbeddingBagLinear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建一个 EmbeddingBag 层，设置嵌入数量为10，嵌入维度为12，求和模式
                    self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, mode='sum')
                    # 创建一个线性层，输入维度为12，输出维度为1，转换为 float 类型
                    self.linear = torch.nn.Linear(12, 1).to(dtype=torch.float)

                def forward(self, input: torch.Tensor, offsets: Optional[torch.Tensor] = None,
                            per_sample_weights: Optional[torch.Tensor] = None):
                    # 前向传播函数，计算 EmbeddingBag 的输出
                    x = self.emb(input, offsets, per_sample_weights)
                    # 将 EmbeddingBag 的输出输入线性层
                    x = self.linear(x)
                    return x

            # 获取量化引擎
            qengine = torch.backends.quantized.engine
            # 设置量化配置字典，针对 EmbeddingBag 设置默认的量化配置
            qconfig_dict = QConfigMapping() \
                .set_global(get_default_qat_qconfig(qengine)) \
                .set_object_type(torch.nn.EmbeddingBag, default_embedding_qat_qconfig)

            # 创建训练数据集
            train_indices = [[torch.randint(0, 10, (12, 12)), torch.randn((12, 1))] for _ in range(2)]
            # 创建评估数据集
            eval_output = [[torch.randint(0, 10, (12, 1))]]

            # 创建模型实例并设置为训练模式
            model = EmbeddingBagLinear().train()
            # 准备量化训练后的模型
            prepared_fx_model = prepare_qat_fx(model, qconfig_dict, example_inputs=(train_indices[0][0],))
            # 使用仅训练函数进行测试
            test_only_train_fn(prepared_fx_model, train_indices)
            # 将准备好的量化模型转换为量化模型
            quant_model = convert_fx(prepared_fx_model,
                                     qconfig_mapping=qconfig_dict)

            def checkQuantized(model):
                # 断言 EmbeddingBag 层已经被量化
                self.assertTrue(type(model.emb), nn.quantized.EmbeddingBag)
                # 断言 Linear 层已经被量化
                self.assertTrue(type(model.linear), nnq.Linear)

                # 使用仅评估函数对模型进行测试
                test_only_eval_fn(model, eval_output)
                # 检查模型是否可脚本化
                self.checkScriptable(model, eval_output)
                # 检查模型是否没有量化配置
                self.checkNoQconfig(model)
            # 对量化后的模型进行检查
            checkQuantized(quant_model)

    @override_qengines
    # 定义一个测试函数，用于测试量化感知训练（QAT）后的线性嵌入模型
    def test_qat_embedding_linear(self):
        # 遍历支持的设备类型，对每种设备类型执行以下操作
        for device in get_supported_device_types():
            # 定义一个内部类 EmbeddingLinear，继承自 torch.nn.Module
            class EmbeddingLinear(torch.nn.Module):
                # 类的初始化方法
                def __init__(self):
                    super().__init__()
                    # 初始化一个嵌入层，包含10个嵌入向量，每个向量维度为12
                    self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)
                    # 初始化一个线性层，输入维度为12，输出维度为1，转换为 float 类型
                    self.linear = torch.nn.Linear(12, 1).to(dtype=torch.float)

                # 前向传播方法
                def forward(self, input: torch.Tensor):
                    # 计算输入张量在嵌入层的嵌入和，沿着维度1求和
                    x = torch.sum(self.emb(input), dim=1)
                    # 将求和结果传入线性层中
                    x = self.linear(x)
                    # 返回线性层的输出结果
                    return x

            # 获取当前量化后端引擎
            qengine = torch.backends.quantized.engine
            # 创建一个量化配置字典，包含默认的 QAT 量化配置
            qconfig_dict = {"": get_default_qat_qconfig(qengine),
                            "object_type": [(torch.nn.Embedding, default_embedding_qat_qconfig)]}

            # 创建训练数据索引，每个元素为包含随机整数张量和随机标准正态分布张量的列表
            train_indices = [[torch.randint(0, 10, (12, 12)), torch.randn((12, 1))] for _ in range(2)]
            # 创建评估输出数据，每个元素为随机整数张量的列表
            eval_output = [[torch.randint(0, 10, (12, 1))]]

            # 创建 EmbeddingLinear 模型实例，并设置为训练模式
            model = EmbeddingLinear().train()
            # 准备 QAT 后的模型，使用指定的量化配置字典和示例输入
            prepared_fx_model = prepare_qat_fx(model, qconfig_dict, example_inputs=(train_indices[0][0],))
            # 仅测试训练函数，对准备好的 QAT 模型和训练数据索引进行测试
            test_only_train_fn(prepared_fx_model, train_indices)
            # 将准备好的 QAT 模型转换为量化模型
            quant_model = convert_fx(prepared_fx_model,
                                     qconfig_mapping=qconfig_dict)

            # 定义一个函数 checkQuantized，用于检查模型是否正确量化
            def checkQuantized(model):
                # 断言嵌入层现在是量化的嵌入层 nn.quantized.Embedding
                self.assertTrue(type(model.emb), nn.quantized.Embedding)
                # 同时检查线性层是否被量化
                self.assertTrue(type(model.linear), nnq.Linear)

                # 仅测试评估函数，对模型和评估输出数据进行测试
                test_only_eval_fn(model, eval_output)
                # 检查模型是否可脚本化
                self.checkScriptable(model, eval_output)
                # 检查模型是否没有量化配置
                self.checkNoQconfig(model)
            
            # 使用 checkQuantized 函数检查量化后的模型
            checkQuantized(quant_model)

    # 使用 hypothesis 的 given 装饰器，设定设备参数为 CPU 或 CUDA（若可用）
    @given(
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        )
    )
    # 设置不设定超时时间的配置
    @settings(deadline=None)
    # 覆盖量化引擎的装饰器
    @override_qengines
# 如果当前脚本被直接运行（而不是被作为模块导入），则抛出运行时错误并显示提示信息
if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
```
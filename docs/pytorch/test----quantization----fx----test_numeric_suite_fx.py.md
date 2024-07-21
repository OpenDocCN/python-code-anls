# `.\pytorch\test\quantization\fx\test_numeric_suite_fx.py`

```
# 导入必要的库和模块，用于量化（quantization）相关的功能测试和操作
# 这些模块包括了复制、数学计算、操作符等标准库，以及 PyTorch 中的量化模块和测试工具

import copy  # 导入复制（copy）相关的标准库
import math  # 导入数学计算（math）相关的标准库
import operator  # 导入操作符（operator）相关的标准库
import unittest  # 导入单元测试（unittest）相关的标准库

import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块（nn）
import torch.nn.functional as F  # 导入 PyTorch 中的函数式模块（F）
from torch.ao.quantization import (  # 导入 PyTorch 中的量化模块
    default_dynamic_qconfig,
    QConfigMapping,
    get_default_qconfig_mapping,
)
import torch.ao.nn.quantized as nnq  # 导入 PyTorch 中的量化神经网络（nnq）
toq = torch.ops.quantized  # 设置 PyTorch 的量化操作符（toq）
from torch.ao.quantization.quantize_fx import (  # 导入 PyTorch 中的量化 FX（quantize_fx）模块
    convert_fx,
    convert_to_reference_fx,
    prepare_fx,
    prepare_qat_fx,
)
from torch.testing._internal.common_quantization import (  # 导入 PyTorch 测试工具中的量化相关模块
    ConvBnModel,
    ConvBnReLUModel,
    ConvModel,
    QuantizationTestCase,
    skipIfNoFBGEMM,
    skipIfNoQNNPACK,
    withQNNPACKBackend,
    SingleLayerLinearDynamicModel,
    SingleLayerLinearModel,
    LSTMwithHiddenDynamicModel,
    SparseNNModel,
    skip_if_no_torchvision,
    TwoLayerLinearModel
)
from torch.testing._internal.common_utils import skipIfTorchDynamo  # 导入 PyTorch 测试工具中的通用工具
from torch.ao.quantization.quantization_mappings import (  # 导入 PyTorch 量化映射（quantization_mappings）模块
    get_default_static_quant_module_mappings,
    get_default_dynamic_quant_module_mappings,
    get_default_float_to_quantized_operator_mappings,
)
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入 PyTorch 测试工具中的 CUDA 相关模块
from torch.testing._internal.common_quantization import NodeSpec as ns  # 导入 PyTorch 测试工具中的量化相关模块别名
from torch.ao.quantization.fx.pattern_utils import get_default_quant_patterns  # 导入 PyTorch 量化 FX 中的模式工具（pattern_utils）
import torch.ao.quantization.fx.quantize_handler as qh  # 导入 PyTorch 量化 FX 中的量化处理器（quantize_handler）
from torch.ao.ns.fx.pattern_utils import (  # 导入 PyTorch ns（namespace）中的模式工具
    get_type_a_related_to_b,
)
from torch.ao.ns.fx.graph_matcher import (  # 导入 PyTorch ns 中的图匹配器（graph_matcher）
    get_matching_subgraph_pairs,
    GraphMatchingException,
)
from torch.ao.ns.fx.utils import (  # 导入 PyTorch ns 中的工具函数
    compute_sqnr,
    compute_normalized_l2_error,
    compute_cosine_similarity,
)
from torch.ao.ns.fx.mappings import (  # 导入 PyTorch ns 中的映射
    get_node_type_to_io_type_map,
    get_unmatchable_types_map,
    get_base_name_to_sets_of_related_ops,
    get_base_name_for_op,
    add_op_to_sets_of_related_ops,
)
from torch.ao.ns.fx.weight_utils import (  # 导入 PyTorch ns 中的权重工具（weight_utils）
    get_op_to_type_to_weight_extraction_fn,
)
from torch.ao.ns._numeric_suite_fx import (  # 导入 PyTorch ns 中的数值套件 FX（numeric_suite_fx）
    extract_weights,
    _extract_weights_impl,
    add_loggers,
    _add_loggers_impl,
    OutputLogger,
    add_shadow_loggers,
    _add_shadow_loggers_impl,
    extract_logger_info,
    extract_shadow_logger_info,
    extend_logger_results_with_comparison,
    prepare_n_shadows_model,
    convert_n_shadows_model,
    extract_results_n_shadows_model,
    OutputComparisonLogger,
    print_comparisons_n_shadows_model,
    loggers_set_enabled,
    loggers_set_save_activations,
    _prepare_n_shadows_add_loggers_model,
    _n_shadows_compare_weights,
)
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping  # 导入 PyTorch ns 中的多重映射（qconfig_multi_mapping）
from torch.ao.quantization.backend_config import get_native_backend_config  # 导入 PyTorch 量化后端配置（backend_config）
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers  # 导入 PyTorch 量化 FX 中的量化处理器

# 注意：这些模型仅在本文件内部使用，不可在其他地方使用。
# 尽管重用代码很好，我们也需要能够迭代测试。
# 当调试时，通过快速调用以便在调试各个测试案例时降低其速度。
# 如果测试模型在不同文件中有大量调用点，那么在单个测试案例上进行调试的速度会降低。

# 定义一个继承自 nn.Module 的类 LinearReluFunctional
class LinearReluFunctional(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个 4x4 的参数张量 w1，并将其设为模型的参数
        self.w1 = nn.Parameter(torch.empty(4, 4))
        # 定义一个大小为 4 的偏置参数张量 b1，并将其设为模型的参数
        self.b1 = nn.Parameter(torch.zeros(4))
        # 使用 Kaiming 均匀初始化方法初始化参数 w1
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        # 对输入 x 执行线性变换，使用模型的参数 w1 和 b1
        x = F.linear(x, self.w1, self.b1)
        # 对线性变换的结果执行 ReLU 激活函数
        x = F.relu(x)
        return x


# 定义一个继承自 nn.Module 的类 LinearFunctional
class LinearFunctional(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个 4x4 的参数张量 w1，并将其设为模型的参数
        self.w1 = nn.Parameter(torch.empty(4, 4))
        # 定义一个大小为 4 的偏置参数张量 b1，并将其设为模型的参数
        self.b1 = nn.Parameter(torch.zeros(4))
        # 使用 Kaiming 均匀初始化方法初始化参数 w1
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        # 对输入 x 执行线性变换，使用模型的参数 w1 和 b1
        x = F.linear(x, self.w1, self.b1)
        return x


# 定义一个继承自 nn.Module 的类 LinearReluLinearFunctional
class LinearReluLinearFunctional(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个 4x4 的参数张量 w，并将其设为模型的参数
        self.w = nn.Parameter(torch.Tensor(4, 4))
        # 定义一个大小为 4 的偏置参数张量 b，并将其设为模型的参数
        self.b = nn.Parameter(torch.zeros(4))
        # 使用 Kaiming 均匀初始化方法初始化参数 w
        torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):
        # 对输入 x 执行第一个线性变换，使用模型的参数 w 和 b
        x = F.linear(x, self.w, self.b)
        # 对第一个线性变换的结果执行 ReLU 激活函数
        x = F.relu(x)
        # 对ReLU激活后的结果再次执行线性变换，使用相同的参数 w 和 b
        x = F.linear(x, self.w, self.b)
        return x


# 定义一个继承自 nn.Module 的类 AddMulFunctional
class AddMulFunctional(nn.Module):
    def forward(self, x, y):
        # 对输入 x 加 1.0
        x = x + 1.0
        # 对结果乘以 1.0，即保持不变
        x = x * 1.0
        # 对 1.0 加上 x
        x = 1.0 + x
        # 将结果乘以 1.0，即保持不变
        x = 1.0 * x
        # 将 x 加上 y
        x = x + y
        # 将结果乘以 y
        x = x * y
        return x


# 定义一个继承自 nn.Module 的类 AllConvAndLinearFusionModules
class AllConvAndLinearFusionModules(torch.nn.Module):
    pass
    # 初始化函数，继承父类的初始化方法
    def __init__(self):
        super().__init__()
        # 定义一个1维卷积层，输入通道数为1，输出通道数为1，卷积核大小为1
        self.conv1d_0 = nn.Conv1d(1, 1, 1)
        # 定义一个1维卷积层后接ReLU激活函数
        self.conv1d_1 = nn.Conv1d(1, 1, 1)
        self.relu_0 = nn.ReLU()
        # 定义一个1维卷积层后接Batch Normalization层（仅在量化训练时使用）
        self.conv1d_2 = nn.Conv1d(1, 1, 1)
        self.bn1d_0 = nn.BatchNorm1d(1)
        # 定义一个1维卷积层后接Batch Normalization层和ReLU激活函数（仅在量化训练时使用）
        self.conv1d_3 = nn.Conv1d(1, 1, 1)
        self.bn1d_1 = nn.BatchNorm1d(1)
        self.relu_4 = nn.ReLU()
        # 定义一个2维卷积层，输入通道数为1，输出通道数为1，卷积核大小为1
        self.conv2d_0 = nn.Conv2d(1, 1, 1)
        # 定义一个2维卷积层后接ReLU激活函数
        self.conv2d_1 = nn.Conv2d(1, 1, 1)
        self.relu_1 = nn.ReLU()
        # 定义一个2维卷积层后接Batch Normalization层（仅在量化训练时使用）
        self.conv2d_2 = nn.Conv2d(1, 1, 1)
        self.bn2d_0 = nn.BatchNorm2d(1)
        # 定义一个2维卷积层后接Batch Normalization层和ReLU激活函数（仅在量化训练时使用）
        self.conv2d_3 = nn.Conv2d(1, 1, 1)
        self.bn2d_1 = nn.BatchNorm2d(1)
        self.relu_5 = nn.ReLU()
        # 定义一个3维卷积层，输入通道数为1，输出通道数为1，卷积核大小为1
        self.conv3d_0 = nn.Conv3d(1, 1, 1)
        # 定义一个3维卷积层后接ReLU激活函数
        self.conv3d_1 = nn.Conv3d(1, 1, 1)
        self.relu_2 = nn.ReLU()
        # 定义一个3维卷积层后接Batch Normalization层（仅在量化训练时使用）
        self.conv3d_2 = nn.Conv3d(1, 1, 1)
        self.bn3d_0 = nn.BatchNorm3d(1)
        # 定义一个3维卷积层后接Batch Normalization层和ReLU激活函数（仅在量化训练时使用）
        self.conv3d_3 = nn.Conv3d(1, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(1)
        self.relu_6 = nn.ReLU()
        # 定义一个全连接层，输入特征数为1，输出特征数为1
        self.linear_0 = nn.Linear(1, 1)
        # 定义一个全连接层后接ReLU激活函数
        self.linear_1 = nn.Linear(1, 1)
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        # 1维卷积层操作
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        x = self.relu_0(x)
        x = self.conv1d_2(x)
        x = self.bn1d_0(x)
        x = self.conv1d_3(x)
        x = self.bn1d_1(x)
        x = self.relu_4(x)
        # 2维卷积层操作
        x = x.reshape(1, 1, 1, 1)
        x = self.conv2d_0(x)
        x = self.conv2d_1(x)
        x = self.relu_1(x)
        x = self.conv2d_2(x)
        x = self.bn2d_0(x)
        x = self.conv2d_3(x)
        x = self.bn2d_1(x)
        x = self.relu_5(x)
        # 3维卷积层操作
        x = x.reshape(1, 1, 1, 1, 1)
        x = self.conv3d_0(x)
        x = self.conv3d_1(x)
        x = self.relu_2(x)
        x = self.conv3d_2(x)
        x = self.bn3d_0(x)
        x = self.conv3d_3(x)
        x = self.bn3d_1(x)
        x = self.relu_6(x)
        # 全连接层操作
        x = x.reshape(1, 1)
        x = self.linear_0(x)
        x = self.linear_1(x)
        x = self.relu_3(x)
        return x
class AllConvFunctional(torch.nn.Module):
    # 定义一个包含多种卷积操作的神经网络模块
    def __init__(self, weight1d, weight2d, weight3d, bias1d, bias2d, bias3d):
        super().__init__()
        # 初始化各种权重和偏置参数为可训练的参数
        self.weight1d = torch.nn.Parameter(weight1d)
        self.weight2d = torch.nn.Parameter(weight2d)
        self.weight3d = torch.nn.Parameter(weight3d)
        self.bias1d = torch.nn.Parameter(bias1d)
        self.bias2d = torch.nn.Parameter(bias2d)
        self.bias3d = torch.nn.Parameter(bias3d)
        # 定义各种卷积操作的步长、填充和膨胀系数等参数
        self.stride1d = 1
        self.padding1d = 0
        self.dilation1d = 1
        self.stride2d = (1, 1)
        self.padding2d = (0, 0)
        self.dilation2d = (1, 1)
        self.groups = 1
        self.stride3d = (1, 1, 1)
        self.padding3d = (0, 0, 0)
        self.dilation3d = (1, 1, 1)

    def forward(self, x):
        # 执行一维卷积操作，使用指定的权重和偏置
        x = F.conv1d(
            x, self.weight1d, self.bias1d, self.stride1d, self.padding1d,
            self.dilation1d, self.groups)
        x = F.conv1d(
            x, self.weight1d, self.bias1d, self.stride1d, self.padding1d,
            self.dilation1d, self.groups)
        x = F.relu(x)  # 对卷积结果应用ReLU激活函数
        # 执行二维卷积操作，使用指定的权重和偏置
        x = F.conv2d(
            x, self.weight2d, self.bias2d, self.stride2d, self.padding2d,
            self.dilation2d, self.groups)
        x = F.conv2d(
            x, self.weight2d, self.bias2d, self.stride2d, self.padding2d,
            self.dilation2d, self.groups)
        x = F.relu(x)  # 对卷积结果应用ReLU激活函数
        # 执行三维卷积操作，使用指定的权重和偏置
        x = F.conv3d(
            x, self.weight3d, self.bias3d, self.stride3d, self.padding3d,
            self.dilation3d, self.groups)
        x = F.conv3d(
            x, self.weight3d, self.bias3d, self.stride3d, self.padding3d,
            self.dilation3d, self.groups)
        x = F.relu(x)  # 对卷积结果应用ReLU激活函数
        return x

@torch.fx.wrap
def _wrapped_hardswish(x):
    # 对输入张量应用hardswish激活函数，使用FX库的wrap装饰器包装
    return F.hardswish(x)

@torch.fx.wrap
def _wrapped_hardswish_fp16(x):
    # 将输入张量去量化，应用hardswish激活函数，再转换为float16类型，使用FX库的wrap装饰器包装
    x = x.dequantize()
    x = F.hardswish(x)
    x = x.to(torch.float16)
    return x

@torch.fx.wrap
def _wrapped_sigmoid(x):
    # 对输入张量应用sigmoid激活函数，使用FX库的wrap装饰器包装
    return F.sigmoid(x)

@torch.fx.wrap
def _wrapped_linear(x, w, b):
    # 执行线性变换，即全连接层，使用FX库的wrap装饰器包装
    return F.linear(x, w, b)

def get_all_quant_patterns():
    """ we are in the process to migrate the frontend of fx graph mode quant
    to use backend_config_dict, so some of the patterns are moved to backend_config_dict
    this function will include these patterns so that we can still have all the patterns
    """
    # 获取默认的量化模式
    all_quant_patterns = get_default_quant_patterns()
    # 一些量化模式已经移到了后端配置字典中，所以在这里需要将它们重新添加回来
    for pattern, quantize_handler in _get_pattern_to_quantize_handlers(get_native_backend_config()).items():
        all_quant_patterns[pattern] = quantize_handler
    return all_quant_patterns

class TestFXGraphMatcher(QuantizationTestCase):

    @skipIfNoFBGEMM
    def test_simple_mod(self):
        # 创建一个包含单个卷积层的序列模型，并设置为评估模式
        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        # 使用预处理函数准备模型，设置量化配置为默认配置，提供示例输入
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(1, 1, 1, 1),))
        # 深度复制预处理后的模型
        mp_copy = copy.deepcopy(mp)
        # 将混合前端模型转换为量化模型
        mq = convert_fx(mp_copy)
        # 获取匹配的子图对列表
        results = get_matching_subgraph_pairs(mp, mq)

        # 获取基本操作名称到相关操作集合的映射
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        # 获取第一个卷积操作的基本名称
        conv_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_0'

        # 预期的操作类型映射
        expected_types = {
            conv_name_0: ((nn.Conv2d, torch.ao.quantization.MinMaxObserver), (nnq.Conv2d, nnq.Conv2d)),
        }
        # 断言匹配的子图对的操作类型
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_simple_fun(self):
        # 定义一个包含线性层的简单模型类
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.empty(1, 4))
                self.b = nn.Parameter(torch.zeros(1))
                torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

            def forward(self, x):
                return F.linear(x, self.w, self.b)

        # 创建模型实例并设置为评估模式
        m = M().eval()
        # 使用预处理函数准备模型，设置量化配置为默认配置，提供示例输入
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(1, 1, 1, 1),))
        # 深度复制预处理后的模型
        mp_copy = copy.deepcopy(mp)
        # 将混合前端模型转换为量化模型
        mq = convert_fx(mp_copy)
        # 获取匹配的子图对列表
        results = get_matching_subgraph_pairs(mp, mq)

        # 获取基本操作名称到相关操作集合的映射
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        # 获取第一个线性操作的基本名称
        linear_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.linear) + '_0'

        # 预期的操作类型映射
        expected_types = {
            linear_name_0:
                ((F.linear, torch.ao.quantization.MinMaxObserver), (toq.linear, toq.linear))
        }
        # 断言匹配的子图对的操作类型
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_simple_fusion(self):
        # 创建一个包含线性-ReLU功能的模型实例，并设置为评估模式
        m = LinearReluFunctional().eval()
        # 使用预处理函数准备模型，设置量化配置为默认配置，提供示例输入
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(4, 4),))
        # 深度复制预处理后的模型
        mp_copy = copy.deepcopy(mp)
        # 将混合前端模型转换为量化模型
        mq = convert_fx(mp_copy)
        # 获取匹配的子图对列表
        results = get_matching_subgraph_pairs(mp, mq)

        # 获取基本操作名称到相关操作集合的映射
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        # 获取第一个线性操作的基本名称
        linear_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.linear) + '_0'

        # 预期的操作类型映射
        expected_types = {
            linear_name_0:
                ((F.linear, torch.ao.quantization.MinMaxObserver), (toq.linear_relu, toq.linear_relu)),
        }
        # 断言匹配的子图对的操作类型
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)
    def test_simple_mod_multi(self):
        # 创建一个包含两个嵌套的 nn.Sequential 的模型，其中包含两个 nn.Conv2d 层
        m = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 1, 1),  # 第一个嵌套的 nn.Conv2d 层
            ),
            nn.Conv2d(1, 1, 1),  # 第二个 nn.Conv2d 层
        ).eval()
        # 使用 torch.ao.quantization.default_qconfig 准备模型 mp，准备用例输入数据为 (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(1, 1, 1, 1),))
        # 深度复制 mp 以便后续转换操作
        mp_copy = copy.deepcopy(mp)
        # 将模型 mp 转换为量化后的模型 mq
        mq = convert_fx(mp_copy)
        # 假设转换成功，没有异常发生
        # 获取匹配的子图对，结果保存在 results 中
        results = get_matching_subgraph_pairs(mp, mq)

    @skipIfNoFBGEMM
    def test_simple_tensor_ops(self):
        class M(nn.Module):
            def forward(self, x, y):
                z = x + y  # 定义模型前向传播逻辑，执行张量加法
                return z

        m = M().eval()
        example_inputs = (torch.randn(1), torch.randn(1))  # 示例输入数据
        # 使用 torch.ao.quantization.default_qconfig 准备模型 mp
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        # 深度复制 mp 以便后续转换操作
        mp_copy = copy.deepcopy(mp)
        # 将模型 mp 转换为量化后的模型 mq
        mq = convert_fx(mp_copy)
        # 假设转换成功，没有异常发生
        # 获取匹配的子图对，结果保存在 results 中
        results = get_matching_subgraph_pairs(mp, mq)

    @skipIfNoFBGEMM
    def test_matching_failure_node_count(self):
        # 验证具有相同节点类型但可匹配节点数量不同的图形匹配失败
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)).eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 使用 torch.ao.quantization.default_qconfig 准备模型 mp1 和 mp2
        mp1 = prepare_fx(m1, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp2 = prepare_fx(m2, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        # 预期抛出 GraphMatchingException 异常，因为节点数量不匹配
        with self.assertRaises(GraphMatchingException) as ex:
            results = get_matching_subgraph_pairs(mp1, mp2)

    @skipIfNoFBGEMM
    def test_matching_failure_node_type(self):
        # 验证具有不匹配节点类型的图形匹配失败
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m2 = nn.Sequential(nn.Linear(1, 1)).eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 使用 torch.ao.quantization.default_qconfig 准备模型 mp1 和 mp2
        mp1 = prepare_fx(m1, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        example_inputs = (torch.randn(1, 1),)
        mp2 = prepare_fx(m2, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        # 预期抛出 GraphMatchingException 异常，因为节点类型不匹配
        with self.assertRaises(GraphMatchingException) as ex:
            results = get_matching_subgraph_pairs(mp1, mp2)
    def test_nodes_before_cat(self):
        # 验证在执行 torch.cat 之前的节点是否能够匹配到
        class M(nn.Module):
            def forward(self, x0):
                # 执行 x0 + 1.0 的操作
                x1 = torch.add(x0, 1.0)
                # 执行 x0 + 1.0 的操作
                y1 = torch.add(x0, 1.0)
                # 将 x1 和 y1 进行拼接操作
                x2 = torch.cat([x1, y1])
                return x2

        m = M().eval()
        example_inputs = (torch.randn(1),)
        # 准备 FX 表示的模型，并使用默认的量化配置
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        # 转换 FX 表示的模型
        mq = convert_fx(mp_copy)
        # 获取匹配子图对的结果
        results = get_matching_subgraph_pairs(mp, mq)

        # 获取操作符的基本名称到相关操作集的映射
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        # 为 torch.cat 操作获取基本名称
        cat_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.cat) + '_0'
        # 为第一个 torch.add 操作获取基本名称
        add_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_0'
        # 为第二个 torch.add 操作获取基本名称
        add_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_1'

        # 预期的子图对应的操作类型
        expected_types = {
            cat_name_0: ((torch.cat, torch.cat), (torch.cat, torch.cat)),
            add_name_0: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
            add_name_1: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
        }
        # 断言匹配的子图对应的操作类型
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_dict_return_type(self):
        # 验证能够遍历返回字典类型的节点
        class M(nn.Module):
            def forward(self, x0):
                # 执行 x0 + 1.0 的操作
                x1 = torch.add(x0, 1.0)
                # 执行 x0 + 1.0 的操作
                y1 = torch.add(x0, 1.0)
                # 执行 x0 + 1.0 的操作
                z1 = torch.add(x0, 1.0)
                # 创建包含多个元素的字典 a1
                a1 = {'x1': x1, 'y1': (y1,), 'z1': [{'key': (z1,)}]}
                return a1

        m = M().eval()
        example_inputs = (torch.randn(1),)
        # 准备 FX 表示的模型，并使用默认的量化配置
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        # 转换 FX 表示的模型
        mq = convert_fx(mp_copy)
        # 获取匹配子图对的结果
        results = get_matching_subgraph_pairs(mp, mq)

        # 获取操作符的基本名称到相关操作集的映射
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        # 为第一个 torch.add 操作获取基本名称
        add_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_0'
        # 为第二个 torch.add 操作获取基本名称
        add_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_1'
        # 为第三个 torch.add 操作获取基本名称
        add_name_2 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_2'

        # 预期的子图对应的操作类型
        expected_types = {
            add_name_0: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
            add_name_1: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
            add_name_2: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
        }
        # 断言匹配的子图对应的操作类型
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)
    # 根据装饰器条件跳过测试，如果没有FBGEMM支持的话
    @skipIfNoFBGEMM
    # 定义一个测试方法，用于检查具有相同类型的节点是否匹配
    def test_nodes_with_equal_types_get_matched(self):
        # 定义一个继承自nn.Module的模型类M
        class M(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加两个Conv2d层，输入、输出通道和卷积核大小均为1
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            # 前向传播方法
            def forward(self, x):
                # 对输入x应用第一个卷积层conv1
                x = self.conv1(x)
                # 对上一层输出x应用第二个卷积层conv2
                x = self.conv2(x)
                # 对上一层输出x执行元素级平方
                x = torch.mul(x, x)
                # 对上一层输出x执行sigmoid激活函数
                x = torch.sigmoid(x)
                # 对上一层输出x执行ReLU激活函数
                x = F.relu(x)
                # 返回处理后的输出x
                return x

        # 创建M类的实例，并设置为评估模式
        m = M().eval()
        
        # 防止conv2层被量化，以便测试具有相同类型的模块
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping().set_module_name("conv2", None)
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 准备模型以进行量化仿真，使用指定的量化配置映射和示例输入
        mp = prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
        # 深度复制量化模型
        mp_copy = copy.deepcopy(mp)
        # 将复制后的量化模型转换为量化模型
        mq = convert_fx(mp_copy)
        # 获取匹配子图对的结果
        results = get_matching_subgraph_pairs(mp, mq)

        # 获取基本操作名称到相关操作集合的映射
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        # 获取Conv2d操作的第一个实例的基本名称
        conv_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_0'
        # 获取Conv2d操作的第二个实例的基本名称
        conv_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_1'
        # 获取torch.mul操作的基本名称
        mul_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.mul) + '_0'
        # 获取torch.relu操作的基本名称
        relu_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.relu) + '_0'
        # 获取torch.sigmoid操作的基本名称
        sigmoid_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.sigmoid) + '_0'

        # 断言：所有这些操作应该是匹配的
        expected_types = {
            conv_name_1:
                ((nn.Conv2d, torch.ao.quantization.HistogramObserver), (nnq.Conv2d, nnq.Conv2d)),
            conv_name_0:
                ((nn.Conv2d, torch.ao.quantization.HistogramObserver), (nn.Conv2d, nn.Conv2d)),
            mul_name_0: ((torch.mul, torch.ao.quantization.HistogramObserver), (toq.mul, toq.mul)),
            relu_name_0: ((F.relu, torch.ao.quantization.FixedQParamsObserver), (F.relu, F.relu)),
            sigmoid_name_0:
                ((torch.sigmoid, torch.ao.quantization.FixedQParamsObserver), (torch.sigmoid, torch.sigmoid)),
        }
        # 断言：对于匹配的子图对，检查其类型是否符合预期
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)
    def test_methods(self):
        """
        Verify that graph matching works on methods
        """
        # 定义一个简单的 PyTorch 模块 M，包含一个 forward 方法，应用 sigmoid 激活函数
        class M(nn.Module):
            def forward(self, x):
                x = x.sigmoid()
                return x

        # 创建两个 M 类的实例，并设置为评估模式
        m1 = M().eval()
        m2 = M().eval()

        # 获取默认的量化配置映射
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
        
        # 创建一个示例输入
        example_inputs = (torch.randn(1),)

        # 对 m1 和 m2 进行函数式量化准备
        m1p = prepare_fx(m1, qconfig_mapping, example_inputs=example_inputs)
        m2p = prepare_fx(m2, qconfig_mapping, example_inputs=example_inputs)

        # 获取匹配的子图对
        results = get_matching_subgraph_pairs(m1p, m2p)

        # 获取基础操作名称到相关操作集合的映射
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()

        # 为 torch.sigmoid 添加基础操作到相关操作集合映射
        sigmoid_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.sigmoid) + '_0'

        # 期望的操作类型字典
        expected_types = {
            sigmoid_name_0:
                (('sigmoid', torch.ao.quantization.FixedQParamsObserver), ('sigmoid', torch.ao.quantization.FixedQParamsObserver)),
        }

        # 断言匹配的子图对的类型
        self.assert_types_for_matched_subgraph_pairs(
            results, expected_types, m1p, m2p)

    @skipIfNoFBGEMM
    def test_user_defined_function(self):
        """
        Verify that graph matching works on user defined functions
        """
        # 定义两个简单的 PyTorch 模块 M1 和 M2，分别使用不同的用户定义函数进行前向传播
        class M1(nn.Module):
            def forward(self, x):
                x = F.hardswish(x)
                return x

        class M2(nn.Module):
            def forward(self, x):
                x = _wrapped_hardswish(x)
                return x

        # 获取默认的量化配置映射
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()

        # 创建一个示例输入
        example_inputs = (torch.randn(1, 1, 1, 1),)

        # 对 M1 和 M2 进行函数式量化准备
        m1 = prepare_fx(M1().eval(), qconfig_mapping, example_inputs=example_inputs)
        m2 = prepare_fx(M2().eval(), qconfig_mapping, example_inputs=example_inputs)

        # 获取基础操作名称到相关操作集合的映射
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()

        # 将 _wrapped_hardswish 和 F.hardswish 添加到相关操作集合中
        add_op_to_sets_of_related_ops(
            base_name_to_sets_of_related_ops, _wrapped_hardswish, F.hardswish)

        # 获取匹配的子图对，传入基础操作到相关操作集合映射
        results = get_matching_subgraph_pairs(
            m1, m2,
            base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops)

        # 获取基础操作名称为 F.hardswish 的第一个相关操作名称
        hardswish_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.hardswish) + '_0'

        # 期望的操作类型字典
        expected_types = {
            hardswish_name_0:
                ((F.hardswish, torch.ao.quantization.HistogramObserver), (_wrapped_hardswish, _wrapped_hardswish)),
        }

        # 断言匹配的子图对的类型
        self.assert_types_for_matched_subgraph_pairs(
            results, expected_types, m1, m2)
    # 定义一个测试方法，用于验证结果顺序
    def test_results_order(self):
        # 创建一个包含卷积层和线性层的序列模型，并设置为评估模式
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Linear(1, 1),
        ).eval()
        # 准备示例输入数据
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 使用输入数据准备模型，应用默认的量化配置
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        # 深度复制模型对象
        mp_copy = copy.deepcopy(mp)
        # 将复制的模型对象转换为量化后的模型
        mq = convert_fx(mp_copy)
        # 获取匹配的子图对，返回结果为字典
        results = get_matching_subgraph_pairs(mp, mq)
        # 断言结果字典中的子图对数量为2
        self.assertTrue(len(results) == 2)
        # 获取结果字典的迭代器
        results_iter = iter(results.items())
        # 获取第一对子图的引用，并验证其起始节点名称为 '_0'
        _, (subgraph_a_0, subgraph_b_0) = next(results_iter)
        self.assertTrue(subgraph_a_0.start_node.name == '_0' and
                        subgraph_b_0.start_node.name == '_0')
        # 获取第二对子图的引用，并验证其起始节点名称为 '_1'
        _, (subgraph_a_1, subgraph_b_1) = next(results_iter)
        self.assertTrue(subgraph_a_1.start_node.name == '_1' and
                        subgraph_b_1.start_node.name == '_1')
class TestFXGraphMatcherModels(QuantizationTestCase):

    @skipIfTorchDynamo("too slow")  # 如果在Torch Dynamo环境下，跳过该测试（原因是太慢）
    @skipIfNoFBGEMM  # 如果没有FBGEMM支持，跳过该测试
    @skip_if_no_torchvision  # 如果没有安装torchvision，跳过该测试
    def test_mobilenet_v2(self):
        # 验证mobilenetv2图是否能够匹配
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).eval().float()
        example_inputs = (torch.randn(1, 3, 224, 224),)
        mp = prepare_fx(copy.deepcopy(m), {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        # 假设没有异常发生则认为成功
        results_m_mp = get_matching_subgraph_pairs(torch.fx.symbolic_trace(m), mp)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # 假设没有异常发生则认为成功
        results_mp_mq = get_matching_subgraph_pairs(mp, mq)

    @skipIfNoFBGEMM  # 如果没有FBGEMM支持，跳过该测试
    @skip_if_no_torchvision  # 如果没有安装torchvision，跳过该测试
    def test_mobilenet_v2_qat(self):
        # 验证mobilenetv2图是否能够匹配
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).float()
        example_inputs = (torch.randn(1, 3, 224, 224),)
        mp = prepare_qat_fx(
            copy.deepcopy(m),
            {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')},
            example_inputs=example_inputs)
        # 假设没有异常发生则认为成功
        results_m_mp = get_matching_subgraph_pairs(torch.fx.symbolic_trace(m), mp)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # 假设没有异常发生则认为成功
        results_mp_mq = get_matching_subgraph_pairs(mp, mq)


class FXNumericSuiteQuantizationTestCase(QuantizationTestCase):
    def _test_extract_weights(
        self, m, example_inputs, results_len=0, qconfig_dict=None, prepare_fn=prepare_fx
    ):
        # 对模型进行符号跟踪，生成FX模型
        m = torch.fx.symbolic_trace(m)
        
        # 如果未提供量化配置字典，则使用默认的量化配置
        if qconfig_dict is None:
            qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        
        # 使用深拷贝的模型m，应用指定的量化配置字典和示例输入，进行准备工作
        mp = prepare_fn(copy.deepcopy(m), qconfig_dict, example_inputs=example_inputs)
        
        # 对准备好的模型mp进行深拷贝
        mp_copy = copy.deepcopy(mp)
        
        # 将深拷贝的模型mp_copy转换为量化FX模型mq
        mq = convert_fx(mp_copy)

        # 测试公共API和内部GraphModule API
        # 对提取权重函数extract_weights_fun进行测试，包括公共API和GraphModule API
        for extract_weights_fun in (extract_weights, _extract_weights_impl):
            # 在模型m和准备后的模型mp，以及准备后的模型mp和量化后的模型mq之间进行测试
            for m1, m2 in ((m, mp), (mp, mq)):
                # 使用提取权重函数提取权重'a'和'b'，并获取结果
                results = extract_weights_fun('a', m1, 'b', m2)
                
                # 断言结果列表长度符合预期长度results_len
                self.assertTrue(
                    len(results) == results_len,
                    f"expected len {results_len}, got len {len(results)}")
                
                # 验证结果字典的有效性
                self.assert_ns_compare_dict_valid(results)
                
                # 使用比较函数compute_sqnr，将计算结果与日志结果进行比较，扩展日志结果
                extend_logger_results_with_comparison(
                    results, 'a', 'b', compute_sqnr, 'sqnr')
                
                # 使用比较函数compute_normalized_l2_error，将计算结果与日志结果进行比较，扩展日志结果
                extend_logger_results_with_comparison(
                    results, 'a', 'b', compute_normalized_l2_error, 'l2_error')
                
                # 使用比较函数compute_cosine_similarity，将计算结果与日志结果进行比较，扩展日志结果
                extend_logger_results_with_comparison(
                    results, 'a', 'b', compute_cosine_similarity,
                    'cosine_similarity')
    ):
        # 如果没有指定量化配置字典，则使用默认的量化配置映射
        if qconfig_dict is None:
            qconfig_dict = torch.ao.quantization.get_default_qconfig_mapping()
        
        # 如果准备函数是 prepare_fx，则将模型设为评估模式；否则设为训练模式
        if prepare_fn == prepare_fx:
            m.eval()
        else:
            m.train()
        
        # 使用准备函数对模型进行准备，返回准备后的模型对象 mp
        mp = prepare_fn(copy.deepcopy(m), qconfig_dict, example_inputs=data)
        
        # 在准备后的模型上执行数据示例
        mp(*data)
        
        # 深拷贝准备后的模型对象 mp，为后续量化做准备
        mp_copy = copy.deepcopy(mp)
        
        # 将深拷贝的准备后的模型对象 mp_copy 转换为量化模型 mq
        mq = convert_fx(mp_copy)

        # 向模型 m 和 mp 添加记录器，并返回修改后的模型对象 m_ns 和 mp_ns2
        m_ns, mp_ns2 = add_loggers(
            'a', m, 'b', copy.deepcopy(mp), OutputLogger,
            should_log_inputs=should_log_inputs)
        
        # 向准备后的模型对象 mp 和 mq 添加记录器，并返回修改后的模型对象 mp_ns 和 mq_ns
        mp_ns, mq_ns = add_loggers(
            'a', mp, 'b', mq, OutputLogger,
            should_log_inputs=should_log_inputs)

        # 如果预期的节点出现情况已经准备好，则检查模型 m_ns 和 mp_ns2 的节点
        if prepared_expected_node_occurrence:
            self.checkGraphModuleNodes(
                m_ns, expected_node_occurrence=prepared_expected_node_occurrence)
            self.checkGraphModuleNodes(
                mp_ns2, expected_node_occurrence=prepared_expected_node_occurrence)
        
        # 检查模型 mp_ns 和 mq_ns 的节点
        self.checkGraphModuleNodes(
            mp_ns, expected_node_occurrence=prepared_expected_node_occurrence)
        self.checkGraphModuleNodes(
            mq_ns, expected_node_occurrence=prepared_expected_node_occurrence)

        # 如果不跳过脚本化，则将模型 m_ns、mp_ns 和 mq_ns 脚本化
        if not skip_scripting:
            m_ns = torch.jit.script(m_ns)
            mp_ns = torch.jit.script(mp_ns)
            mq_ns = torch.jit.script(mq_ns)

        # 在模型 m_ns、mp_ns2、mp_ns 和 mq_ns 上执行数据示例
        m_ns(*data)
        mp_ns2(*data)
        mp_ns(*data)
        mq_ns(*data)

        # 检查激活结果的正确性
        results = []
        for m1, m2 in ((m_ns, mp_ns2), (mp_ns, mq_ns)):
            # 提取输出日志信息并比较
            act_compare_dict = extract_logger_info(
                m1, m2, OutputLogger, 'b')
            # 断言输出日志的长度符合预期
            self.assertTrue(
                len(act_compare_dict) == results_len,
                f"expected len {results_len}, got len {len(act_compare_dict)}")
            # 验证比较字典的有效性
            self.assert_ns_compare_dict_valid(act_compare_dict)
            # 使用比较函数扩展日志结果
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_sqnr, 'sqnr')
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_normalized_l2_error, 'l2_error')
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_cosine_similarity,
                'cosine_similarity')
            results.append(act_compare_dict)
        
        # 返回结果列表
        return results

    # 测试匹配阴影激活函数
    def _test_match_shadow_activations(
        self, m, data, prepared_expected_node_occurrence=None, results_len=None,
        should_log_inputs=False, qconfig_dict=None, skip_scripting=False,
        prepare_fn=prepare_fx, compare_fp32_vs_fp32_prepared=True,
        ):
        # 如果未提供量化配置字典，则使用默认的量化配置映射
        if qconfig_dict is None:
            qconfig_dict = torch.ao.quantization.get_default_qconfig_mapping()
        # 如果准备函数为 prepare_fx，则将模型设为评估模式
        if prepare_fn == prepare_fx:
            m.eval()
        else:
            # 否则将模型设为训练模式
            m.train()
        # 打印量化配置字典信息
        print("qconfig_dict:", qconfig_dict)
        # 使用准备函数准备模型，并返回准备后的模型
        mp = prepare_fn(copy.deepcopy(m), qconfig_dict, example_inputs=data)
        # 打印准备后的模型信息
        print("prepared:", mp)
        # 在准备后的模型上执行输入数据
        mp(*data)
        # 深拷贝准备后的模型，以备份
        mp_copy = copy.deepcopy(mp)
        # 将备份的模型转换为量化版本
        mq = convert_fx(mp_copy)
        # 打印量化后的模型信息
        print("quantized:", mq)

        # 如果需要比较准备前后的浮点数模型
        if compare_fp32_vs_fp32_prepared:
            # 创建一个影子模型用于记录日志，比较原始模型 m 和准备后的模型 mp
            m_shadows_mp = add_shadow_loggers(
                'a', copy.deepcopy(m), 'b', copy.deepcopy(mp),
                OutputLogger, should_log_inputs=should_log_inputs)
        # 创建一个影子模型用于记录日志，比较准备后的模型 mp 和量化后的模型 mq
        mp_shadows_mq = add_shadow_loggers(
            'a', mp, 'b', mq, OutputLogger,
            should_log_inputs=should_log_inputs)

        # 如果需要验证预期节点的出现情况
        if prepared_expected_node_occurrence:
            # 如果需要比较准备前后的浮点数模型，验证 m_shadows_mp 的节点出现情况
            if compare_fp32_vs_fp32_prepared:
                self.checkGraphModuleNodes(
                    m_shadows_mp, expected_node_occurrence=prepared_expected_node_occurrence)
            # 验证 mp_shadows_mq 的节点出现情况
            self.checkGraphModuleNodes(
                mp_shadows_mq, expected_node_occurrence=prepared_expected_node_occurrence)

        # 如果不跳过脚本化
        if not skip_scripting:
            # 如果需要比较准备前后的浮点数模型，对 m_shadows_mp 进行脚本化
            if compare_fp32_vs_fp32_prepared:
                m_shadows_mp = torch.jit.script(m_shadows_mp)
            # 对 mp_shadows_mq 进行脚本化
            mp_shadows_mq = torch.jit.script(mp_shadows_mq)

        # 进行校准
        # 如果需要比较准备前后的浮点数模型，执行 m_shadows_mp 的输入数据
        if compare_fp32_vs_fp32_prepared:
            m_shadows_mp(*data)
        # 执行 mp_shadows_mq 的输入数据
        mp_shadows_mq(*data)

        # 检查激活结果的正确性
        results = []
        models = (m_shadows_mp, mp_shadows_mq) if \
            compare_fp32_vs_fp32_prepared else (mp_shadows_mq,)
        # 对每个模型进行循环，提取影子日志信息
        for model in models:
            act_compare_dict = extract_shadow_logger_info(
                model, OutputLogger, 'b')
            # 如果结果长度不为 None，则断言结果长度与预期长度一致
            if results_len is not None:
                self.assertTrue(
                    len(act_compare_dict) == results_len,
                    f"expected len {results_len}, got len {len(act_compare_dict)}")
            # 断言比较字典有效性
            self.assert_ns_compare_dict_valid(act_compare_dict)
            # 扩展日志结果，计算和比较信噪比
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_sqnr, 'sqnr')
            # 扩展日志结果，计算和比较归一化 L2 误差
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_normalized_l2_error, 'l2_error')
            # 扩展日志结果，计算和比较余弦相似度
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_cosine_similarity,
                'cosine_similarity')
            # 将结果添加到结果列表中
            results.append(act_compare_dict)
        # 返回所有模型的比较结果
        return results
# 定义一个测试类，测试 FX 数字套件的核心 API，继承于 FXNumericSuiteQuantizationTestCase
class TestFXNumericSuiteCoreAPIs(FXNumericSuiteQuantizationTestCase):

    # 如果没有 FBGEMM，跳过此测试函数
    @skipIfNoFBGEMM
    # 测试从模型中提取加权参数，用于模型中的量化过程
    def test_extract_weights_mod_ptq(self):
        # 创建 AllConvAndLinearFusionModules 的实例，并设置为评估模式
        m = AllConvAndLinearFusionModules().eval()
        # 创建一个示例输入数据，包含一个随机生成的张量
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 调用 _test_extract_weights 方法，验证从模型 m 中提取权重，并期望结果的长度为 14
        self._test_extract_weights(m, example_inputs, results_len=14)

    @skipIfNoFBGEMM
    # 测试从模型中提取加权参数（用于量化训练），即量化训练过程中的权重提取测试
    def test_extract_weights_mod_qat(self):
        # 创建 AllConvAndLinearFusionModules 的实例，并设置为训练模式
        m = AllConvAndLinearFusionModules().train()
        # 定义量化配置字典，使用 'fbgemm' 作为默认的量化训练配置
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        # 创建一个示例输入数据，包含一个随机生成的张量
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 调用 _test_extract_weights 方法，验证从模型 m 中提取权重，并期望结果的长度为 14
        # 同时传入量化配置字典和准备函数 prepare_qat_fx
        self._test_extract_weights(
            m, example_inputs, results_len=14, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    # 测试从线性-ReLU-线性函数组合中提取加权参数（用于模型评估）
    def test_extract_weights_linear_fun_ptq(self):
        # 创建 LinearReluLinearFunctional 的实例，并设置为评估模式
        m = LinearReluLinearFunctional().eval()
        # 创建一个示例输入数据，包含一个随机生成的张量
        example_inputs = (torch.randn(1, 4),)
        # 调用 _test_extract_weights 方法，验证从模型 m 中提取权重，并期望结果的长度为 2
        self._test_extract_weights(m, example_inputs, results_len=2)

    @skipIfNoFBGEMM
    # 测试从线性-ReLU-线性函数组合中提取加权参数（用于量化训练）
    def test_extract_weights_linear_fun_qat(self):
        # 创建 LinearReluLinearFunctional 的实例，并设置为训练模式
        m = LinearReluLinearFunctional().train()
        # 定义量化配置字典，使用 'fbgemm' 作为默认的量化训练配置
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        # 创建一个示例输入数据，包含一个随机生成的张量
        example_inputs = (torch.randn(1, 4),)
        # 调用 _test_extract_weights 方法，验证从模型 m 中提取权重，并期望结果的长度为 2
        # 同时传入量化配置字典和准备函数 prepare_qat_fx
        self._test_extract_weights(
            m, example_inputs, results_len=2, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    # 测试从全部卷积功能函数中提取加权参数（用于模型评估）
    def test_extract_weights_conv_fun_ptq(self):
        # 定义一维、二维和三维的随机权重张量和偏置张量
        w1d = torch.randn(1, 1, 1)
        w2d = torch.randn(1, 1, 1, 1)
        w3d = torch.randn(1, 1, 1, 1, 1)
        b1d = torch.randn(1)
        b2d = torch.randn(1)
        b3d = torch.randn(1)
        # 创建 AllConvFunctional 的实例，并设置为评估模式，传入定义的权重和偏置
        m = AllConvFunctional(w1d, w2d, w3d, b1d, b2d, b3d).eval()
        # 创建一个示例输入数据，包含一个随机生成的张量
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 调用 _test_extract_weights 方法，验证从模型 m 中提取权重，并期望结果的长度为 6
        self._test_extract_weights(m, example_inputs, results_len=6)

    @skipIfNoFBGEMM
    # 测试从全部卷积功能函数中提取加权参数（用于量化训练）
    def test_extract_weights_conv_fun_qat(self):
        # 定义一维、二维和三维的随机权重张量和偏置张量
        w1d = torch.randn(1, 1, 1)
        w2d = torch.randn(1, 1, 1, 1)
        w3d = torch.randn(1, 1, 1, 1, 1)
        b1d = torch.randn(1)
        b2d = torch.randn(1)
        b3d = torch.randn(1)
        # 创建 AllConvFunctional 的实例，并设置为训练模式，传入定义的权重和偏置
        m = AllConvFunctional(w1d, w2d, w3d, b1d, b2d, b3d).train()
        # 定义量化配置字典，使用 'fbgemm' 作为默认的量化训练配置
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        # 创建一个示例输入数据，包含一个随机生成的张量
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 调用 _test_extract_weights 方法，验证从模型 m 中提取权重，并期望结果的长度为 6
        # 同时传入量化配置字典和准备函数 prepare_qat_fx
        self._test_extract_weights(
            m, example_inputs, results_len=6, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    # 测试动态量化情况下从模型中提取加权参数，暂时不包括线性-ReLU
    def test_extract_weights_dynamic(self):
        # 创建包含一个线性层的神经网络模型，设置为评估模式
        m = nn.Sequential(nn.Linear(1, 1)).eval()
        # 定义动态量化配置字典，对线性层使用默认的动态量化配置
        qconfig_dict = {
            'object_type': [
                (nn.Linear, default_dynamic_qconfig),
            ],
        }
        # 创建一个示例输入数据，包含一个随机生成的张量
        example_inputs = (torch.randn(1, 1),)
        # 调用 _test_extract_weights 方法，验证从模型 m 中提取权重，并期望结果的长度为 1
        # 同时传入动态量化配置字典
        self._test_extract_weights(m, example_inputs, results_len=1, qconfig_dict=qconfig_dict)
    def test_extract_weights_fqn(self):
        # 创建一个包含两个卷积层的序列模型，并将其设置为评估模式
        m = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 1, 1)),
            nn.Conv2d(1, 1, 1),
        ).eval()
        # 定义量化配置字典，此处设置默认的量化配置
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        # 创建一个示例输入，这里使用随机生成的张量
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 使用函数 prepare_fx 对模型 m 进行量化准备
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 使用函数 convert_fx 复制并转换量化后的模型 mp
        mq = convert_fx(copy.deepcopy(mp))
        # 调用 extract_weights 函数，提取模型权重信息并存储在 results 中
        results = extract_weights('a', mp, 'b', mq)
        # 获取第一个卷积层的权重全限定名（fqn）
        fqn_a_0 = results['_0_0']['weight']['a'][0]['fqn']
        fqn_b_0 = results['_0_0']['weight']['b'][0]['fqn']
        # 断言第一个卷积层的权重全限定名相等，并且与第二个卷积层的权重全限定名相等
        self.assertTrue(fqn_a_0 == '0.0' and fqn_a_0 == fqn_b_0)
        # 获取第二个卷积层的权重全限定名（fqn）
        fqn_a_1 = results['_1']['weight']['a'][0]['fqn']
        fqn_b_1 = results['_1']['weight']['b'][0]['fqn']
        # 断言第二个卷积层的权重全限定名为 '1'，并且与第二个卷积层的权重全限定名相等
        self.assertTrue(fqn_a_1 == '1' and fqn_a_1 == fqn_b_1)

    def _test_match_activations_mod_impl(self, prepare_fn=prepare_fx):
        # 创建一个包含两个卷积层的序列模型，并将其设置为评估模式
        m = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        # 初始化量化配置字典为 None
        qconfig_dict = None
        # 如果 prepare_fn 是 prepare_qat_fx 函数，则设置 QAT 量化配置字典
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        # 预期的节点出现次数字典，此处期望 OutputLogger 模块出现 2 次
        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        # 调用 _test_match_activations 函数，测试模型 m 的激活匹配情况
        self._test_match_activations(
            m, (torch.randn(2, 1, 2, 2),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=2, qconfig_dict=qconfig_dict, prepare_fn=prepare_fn)

    @skipIfNoFBGEMM
    def test_match_activations_mod_ptq(self):
        # 调用 _test_match_activations_mod_impl 函数，使用 prepare_fx 函数进行测试
        self._test_match_activations_mod_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_match_activations_mod_qat(self):
        # 调用 _test_match_activations_mod_impl 函数，使用 prepare_qat_fx 函数进行测试
        self._test_match_activations_mod_impl(prepare_fn=prepare_qat_fx)

    def _test_match_activations_fun_impl(self, prepare_fn=prepare_fx):
        # 创建一个 LinearReluLinearFunctional 类的实例，并将其设置为评估模式
        m = LinearReluLinearFunctional().eval()
        # 初始化量化配置字典为 None
        qconfig_dict = None
        # 如果 prepare_fn 是 prepare_qat_fx 函数，则设置 QAT 量化配置字典
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        # 预期的节点出现次数字典，此处期望 OutputLogger 模块出现 2 次
        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        # 调用 _test_match_activations 函数，测试模型 m 的激活匹配情况
        self._test_match_activations(
            m, (torch.randn(4, 4),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=2, prepare_fn=prepare_fn, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_match_activations_fun_ptq(self):
        # 调用 _test_match_activations_fun_impl 函数，使用 prepare_fx 函数进行测试
        self._test_match_activations_fun_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_match_activations_fun_qat(self):
        # 调用 _test_match_activations_fun_impl 函数，使用 prepare_qat_fx 函数进行测试
        self._test_match_activations_fun_impl(prepare_fn=prepare_qat_fx)
    @skipIfNoFBGEMM
    def test_match_activations_meth_ptq(self):
        """
        Verify that add_loggers works on methods
        """
        # 定义一个简单的神经网络模型 M
        class M(nn.Module):
            # 定义前向传播函数
            def forward(self, x):
                # 对输入 x 应用 sigmoid 函数
                x = x.sigmoid()
                # 返回处理后的结果 x
                return x

        # 创建一个 M 类的实例，并设置为评估模式
        m = M().eval()
        # 调用 _test_match_activations 方法，验证匹配激活函数是否正常工作
        res = self._test_match_activations(
            m, (torch.randn(4, 4),),
            results_len=1)

    @skipIfNoFBGEMM
    def test_match_activations_fqn(self):
        # 创建一个包含两个卷积层的神经网络模型并设置为评估模式
        m = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 1, 1)),
            nn.Conv2d(1, 1, 1),
        ).eval()
        # 定义量化配置字典
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        # 创建一个示例输入
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 准备模型 m，返回预处理后的模型 mp
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # 将 mp 深拷贝后转换为量化后的模型 mq
        mq = convert_fx(copy.deepcopy(mp))
        # 添加日志记录器 'a' 到 mp 和 'b' 到 mq，使用 OutputLogger
        mp_ns, mq_ns = add_loggers('a', mp, 'b', mq, OutputLogger)
        # 创建一个数据示例
        datum = torch.randn(1, 1, 1, 1)
        # 在 mp_ns 上执行前向传播
        mp_ns(datum)
        # 在 mq_ns 上执行前向传播
        mq_ns(datum)

        # 提取 mp_ns 和 mq_ns 的输出日志信息，标签为 'b'
        results = extract_logger_info(mp_ns, mq_ns, OutputLogger, 'b')
        # 获取第一个节点的输出的完全限定名 'a' 和 'b'
        fqn_a_0 = results['_0_0']['node_output']['a'][0]['fqn']
        fqn_b_0 = results['_0_0']['node_output']['b'][0]['fqn']
        # 断言第一个节点的输出完全限定名相等且为 '0.0'
        self.assertTrue(fqn_a_0 == '0.0' and fqn_a_0 == fqn_b_0)
        # 获取第二个节点的输出的完全限定名 'a' 和 'b'
        fqn_a_1 = results['_1']['node_output']['a'][0]['fqn']
        fqn_b_1 = results['_1']['node_output']['b'][0]['fqn']
        # 断言第二个节点的输出完全限定名相等且为 '1'
        self.assertTrue(fqn_a_1 == '1' and fqn_a_1 == fqn_b_1)

    def _test_add_shadow_loggers_mod_impl(self, prepare_fn=prepare_fx):
        # 创建一个包含两个卷积层的神经网络模型并设置为评估模式
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        # 初始化量化配置字典为 None
        qconfig_dict = None
        # 如果 prepare_fn 为 prepare_qat_fx，则设置 qconfig_dict
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        # 调用 _test_match_shadow_activations 方法，验证添加阴影激活函数是否正常工作
        res = self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),), results_len=2,
            prepare_fn=prepare_fn, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_mod_ptq(self):
        # 调用 _test_add_shadow_loggers_mod_impl 方法，使用 prepare_fx 函数
        self._test_add_shadow_loggers_mod_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_mod_qat(self):
        # 调用 _test_add_shadow_loggers_mod_impl 方法，使用 prepare_qat_fx 函数
        self._test_add_shadow_loggers_mod_impl(prepare_fn=prepare_qat_fx)

    def _test_add_shadow_loggers_fun_impl(self, prepare_fn=prepare_fx):
        # 创建一个 LinearReluLinearFunctional 类的实例
        m = LinearReluLinearFunctional()
        # 初始化量化配置字典为 None
        qconfig_dict = None
        # 如果 prepare_fn 为 prepare_qat_fx，则设置 qconfig_dict
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        # 调用 _test_match_shadow_activations 方法，验证添加阴影激活函数是否正常工作
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),), results_len=2, prepare_fn=prepare_fn,
            qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_fun_ptq(self):
        # 调用 _test_add_shadow_loggers_fun_impl 方法，使用 prepare_fx 函数
        self._test_add_shadow_loggers_fun_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_fun_qat(self):
        # 调用 _test_add_shadow_loggers_fun_impl 方法，使用 prepare_qat_fx 函数
        self._test_add_shadow_loggers_fun_impl(prepare_fn=prepare_qat_fx)
    def test_add_shadow_loggers_meth_ptq(self):
        """
        Verify that add_loggers works on methods
        """
        # 定义一个简单的神经网络模型
        class M(nn.Module):
            def forward(self, x):
                x = x.sigmoid()  # 对输入数据进行 sigmoid 激活
                return x

        m = M().eval()  # 创建并评估模型实例
        # 调用测试函数，验证在方法上添加日志记录器不会崩溃
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),),
            # 目前不支持对 sigmoid 方法进行影子模型跟踪，因为其数据类型推断尚未实现。
            # 因此，这里只是测试在方法调用时是否会崩溃。
            results_len=0)

    @skipIfNoFBGEMM
    def test_shadow_activations_fqn(self):
        # 定义一个包含卷积层的序列模型
        m = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 1, 1)),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
        mq = convert_fx(copy.deepcopy(mp))
        # 在模型 mp 和其深度复制 mq 上添加影子日志记录器
        mp_shadows_mq = add_shadow_loggers('a', mp, 'b', mq, OutputLogger)
        datum = torch.randn(1, 1, 1, 1)
        mp_shadows_mq(datum)

        # 提取影子日志记录器的信息
        results = extract_shadow_logger_info(mp_shadows_mq, OutputLogger, 'b')
        # 获取节点 0 的输出信息的全限定名
        fqn_a_0 = results['_0_0']['node_output']['a'][0]['fqn']
        fqn_b_0 = results['_0_0']['node_output']['b'][0]['fqn']
        self.assertTrue(fqn_a_0 == '0.0' and fqn_a_0 == fqn_b_0)
        # 获取节点 1 的输出信息的全限定名
        fqn_a_1 = results['_1']['node_output']['a'][0]['fqn']
        fqn_b_1 = results['_1']['node_output']['b'][0]['fqn']
        self.assertTrue(fqn_a_1 == '1' and fqn_a_1 == fqn_b_1)

    @skipIfNoFBGEMM
    def test_logging_inputs(self):
        """
        Verifies that logging inputs works correctly
        """
        # 定义一个包含卷积层和拼接操作的模型
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv(x)  # 使用卷积层处理输入
                x = torch.cat([x, x], dim=0)  # 在维度0上拼接输出
                return x

        m = M().eval()
        # 测试是否正确记录输入
        self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),),
            results_len=1,
            should_log_inputs=True)

    @skipIfNoFBGEMM
    def test_ops_with_same_fp32_and_int8_signature(self):
        """
        Verifies that we can match pairs of ops which have the same aten
        signature for fp32 and int8 tensors.
        """
        # 定义一个包含最大池化和 ReLU 操作的模型
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.max_pool_2d = nn.MaxPool2d(2)

            def forward(self, x):
                x = self.max_pool_2d(x)  # 最大池化操作
                x = F.relu(x)  # 使用 ReLU 激活函数
                return x

        m = M().eval()
        # 测试是否正确匹配操作的激活输出
        self._test_match_activations(
            m, (torch.randn(1, 1, 2, 2),),
            results_len=2)
    @skipIfNoFBGEMM
    def test_add_mul_inputs_activations(self):
        # 创建 AddMulFunctional 实例并设置为评估模式
        m = AddMulFunctional().eval()
        # 调用 _test_match_activations 方法测试模型 m 的行为
        # 使用两个随机生成的 2x2 的张量作为输入
        res = self._test_match_activations(
            m, (torch.randn(2, 2), torch.randn(2, 2)),
            results_len=6, should_log_inputs=True)

    @skipIfNoFBGEMM
    def test_linear_fp16_weights(self):
        # 定义包含量化配置的字典 qconfig_dict
        qconfig_dict = {'': torch.ao.quantization.float16_static_qconfig}
        # 创建 LinearReluFunctional 实例并设置为评估模式
        m = LinearReluFunctional().eval()
        # 准备一个示例输入
        example_inputs = (torch.randn(1, 4),)
        # 调用 _test_extract_weights 方法测试模型 m 的权重提取功能
        self._test_extract_weights(m, example_inputs, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_linear_fp16_activations(self):
        # 对于两种 should_log_inputs 的情况分别进行测试
        for should_log_inputs in (True, False):
            # 定义包含量化配置的字典 qconfig_dict
            qconfig_dict = {'': torch.ao.quantization.float16_static_qconfig}
            # 创建 LinearReluFunctional 实例并设置为评估模式
            m = LinearReluFunctional().eval()
            # 根据 should_log_inputs 确定期望的日志记录器数量
            num_loggers = 2 if should_log_inputs else 1
            # 定义期望的节点出现次数字典
            expected_occurrence = {
                ns.call_module(OutputLogger): num_loggers,
            }
            # 调用 _test_match_activations 方法测试模型 m 的激活匹配功能
            res = self._test_match_activations(
                m, (torch.randn(4, 4),),
                prepared_expected_node_occurrence=expected_occurrence,
                results_len=1,
                qconfig_dict=qconfig_dict,
                should_log_inputs=should_log_inputs)

    @skipIfNoFBGEMM
    def test_linear_fp16_shadow_activations(self):
        # 对于两种 should_log_inputs 的情况分别进行测试
        for should_log_inputs in (True, False):
            # 定义包含量化配置的字典 qconfig_dict
            qconfig_dict = {'': torch.ao.quantization.float16_static_qconfig}
            # 创建 LinearReluFunctional 实例并设置为评估模式
            m = LinearReluFunctional().eval()
            # 根据 should_log_inputs 确定期望的日志记录器数量
            num_loggers = 4 if should_log_inputs else 2
            # 定义期望的节点出现次数字典
            expected_occurrence = {
                ns.call_module(OutputLogger): num_loggers,
            }
            # 调用 _test_match_shadow_activations 方法测试模型 m 的影子激活匹配功能
            res2 = self._test_match_shadow_activations(
                m, (torch.randn(4, 4),),
                prepared_expected_node_occurrence=expected_occurrence,
                results_len=1,
                qconfig_dict=qconfig_dict,
                should_log_inputs=should_log_inputs)

    @skipIfNoFBGEMM
    def test_linear_fp16_vs_linear_fp16_shadow_activations(self):
        # 创建 LinearFunctional 实例并设置为评估模式
        m = LinearFunctional().eval()
        # 定义包含量化配置的字典 qconfig_dict
        qconfig_dict = {'': torch.ao.quantization.float16_static_qconfig}
        # 准备一个示例输入
        example_inputs = (torch.randn(1, 4),)
        # 准备模型 m 的量化版本
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mq1 = convert_fx(copy.deepcopy(mp))
        mq2 = convert_fx(copy.deepcopy(mp))
        # 创建 mq1 和 mq2 的影子日志记录器
        mq1_shadows_mq2 = _add_shadow_loggers_impl(
            'a', mq1, 'b', mq2, OutputLogger, should_log_inputs=False)
        # 传入随机张量以触发 mq1_shadows_mq2
        mq1_shadows_mq2(torch.randn(4, 4))
        # 提取和比较影子日志信息
        act_compare_dict = extract_shadow_logger_info(
            mq1_shadows_mq2, OutputLogger, 'b')
        # 断言影子日志信息的长度为 1
        self.assertTrue(len(act_compare_dict) == 1)
        # 断言 ns 比较字典的有效性
        self.assert_ns_compare_dict_valid(act_compare_dict)
    def test_op_with_either_fp32_or_int8_input(self):
        """
        验证对接受 fp32 或 int8 输入的操作进行阴影处理的功能。
        """
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()  # 实例化 ReLU 激活函数模块

            def forward(self, x):
                x = self.relu(x)  # 使用模块内部的 ReLU 激活函数处理输入
                x = F.relu(x)  # 使用 torch.nn.functional 中的 ReLU 函数处理输入
                return x

        m = M()  # 实例化 M 类
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),),
            # 注意：目前不支持单独对 relu 进行阴影处理，
            # 这个测试只是确保不会导致崩溃
            results_len=0)

    def _test_int8_shadows_int8_impl(self, m):
        """
        验证当两个模块都是 int8 类型时阴影处理是否有效。
        """
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(4, 1, 4, 4),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mp(*example_inputs)
        mq1 = convert_fx(copy.deepcopy(mp))
        mq2 = convert_fx(mp)
        mq1_shadows_mq2 = add_shadow_loggers('a', mq1, 'b', mq2, OutputLogger)
        mq1_shadows_mq2(torch.randn(4, 1, 4, 4))
        act_compare_dict = extract_shadow_logger_info(
            mq1_shadows_mq2, OutputLogger, 'b')
        self.assertTrue(len(act_compare_dict) == 1)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    @skipIfNoFBGEMM
    def test_int8_shadows_int8_mod(self):
        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        self._test_int8_shadows_int8_impl(m)

    @skipIfNoFBGEMM
    def test_int8_shadows_int8_fun(self):
        m = LinearFunctional().eval()
        self._test_int8_shadows_int8_impl(m)

    @skipIfNoFBGEMM
    def test_user_module_scriptable(self):
        # Logging of the output of this class is not supported, because it is
        # neither a tensor or an RNN return type.
        
        # 定义一个简单的神经网络模块 M1，实现前向传播
        class M1(nn.Module):
            def forward(self, x):
                # 计算输入张量 x 的两倍
                x1 = x * 2
                # 计算输入张量 x 的四倍
                x2 = x * 4
                return (x1, x2)

        # 定义一个包含 M1 模块的神经网络模块 M2，实现前向传播
        class M2(nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = M1()

            def forward(self, x):
                # 调用内部的 M1 模块进行前向传播计算
                x1, x2 = self.m1(x)
                return x1, x2

        # 创建 M2 的实例并设置为评估模式
        m = M2().eval()
        
        # 定义量化配置字典
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        
        # 准备自定义配置字典，指定不可追踪的模块类为 M1
        prepare_custom_config_dict = {
            'non_traceable_module_class': [M1],
        }
        
        # 创建示例输入张量
        example_inputs = (torch.randn(1),)
        
        # 使用准备函数 prepare_fx 对模型 m 进行配置
        mp1 = prepare_fx(
            m,
            qconfig_dict,
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict)
        
        # 深度复制 mp1，得到 mp2
        mp2 = copy.deepcopy(mp1)
        
        # 获取不可匹配类型映射
        unmatchable_types_map = get_unmatchable_types_map()
        
        # 将 M1 类型添加到不可匹配类型映射中的 'mods_unmatchable' 键
        unmatchable_types_map['mods_unmatchable'].add(M1)
        
        # 对 mp1 和 mp2 应用输出记录器 _add_loggers_impl，生成 mp1_ns 和 mp2_ns
        mp1_ns, mp2_ns = _add_loggers_impl(
            'a', mp1, 'b', mp2, OutputLogger, should_log_inputs=False,
            unmatchable_types_map=unmatchable_types_map)
        
        # 使用 TorchScript 对带有记录器的模型进行脚本化
        mp1_ns_scripted = torch.jit.script(mp1_ns)
        mp2_ns_scripted = torch.jit.script(mp2_ns)

    @skipIfNoFBGEMM
    def test_user_module(self):
        """
        For user defined modules,
        1. weight extraction should not crash
        2. unshadowed activations should only have loggers for known types
        3. shadowed activations should only have loggers for known types with
             known dtypes
        """
        # 定义一个测试用例，用于验证自定义模块的功能
        class UserModule(nn.Module):
            def forward(self, x):
                return x

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.user_module = UserModule()

            def forward(self, x):
                # 线性层前向传播
                x = self.linear(x)
                # 用户定义模块前向传播
                x = self.user_module(x)
                return x

        m = M().eval()

        # 在不通过 UserModule 追踪的情况下进行量化
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        prepare_custom_config_dict = {'non_traceable_module_name': ['user_module']}
        example_inputs = (torch.randn(1, 1, 1),)
        mp = prepare_fx(
            m,
            qconfig_dict,
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict)
        mp(*example_inputs)
        mq = convert_fx(copy.deepcopy(mp))

        # 从准备后的模型中提取权重，不应崩溃
        weights = _extract_weights_impl('fp32_prepared', mp, 'int8', mq)

        # 非阴影激活应有日志记录器

        # 添加日志记录器，无需重新追踪
        # 注意：需要再次转换，因为无法复制量化线性层
        mp_ns, mq_ns = _add_loggers_impl(
            'fp32_prepared', copy.deepcopy(mp), 'int8',
            convert_fx(copy.deepcopy(mp)), OutputLogger,
            should_log_inputs=True)
        # fp32 和 int8 模型各应有 2 个日志记录器，线性层的输入输出各 2 个，user_module 的输入输出为 0
        unshadowed_expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self.checkGraphModuleNodes(
            mp_ns, expected_node_occurrence=unshadowed_expected_occurrence)
        self.checkGraphModuleNodes(
            mq_ns, expected_node_occurrence=unshadowed_expected_occurrence)

        # 阴影激活应仅在已知类型且能进行 dtype 转换的节点上有日志记录器

        # 添加阴影日志记录器，无需重新追踪
        mp_shadows_mq_ns = _add_shadow_loggers_impl(
            'fp32_prepared', mp, 'int8', mq, OutputLogger,
            should_log_inputs=True)
        # 线性层的输入输出各 4 个日志记录器，user_module 的输入输出为 0
        shadowed_expected_occurrence = {
            ns.call_module(OutputLogger): 4,
        }
        self.checkGraphModuleNodes(
            mp_shadows_mq_ns, expected_node_occurrence=shadowed_expected_occurrence)

    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    # 定义一个测试方法，用于测试量化过程中神经网络层的命名是否正确
    def test_layer_names(self):
        # 创建一个包含卷积层和 Sigmoid 激活函数的序列模型，并设置为评估模式
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid(),
        ).eval()
        
        # 获取默认的量化配置映射，使用 "fbgemm" 量化引擎
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping("fbgemm")
        
        # 准备量化操作所需的示例输入
        example_inputs = (torch.randn(1, 1, 1, 1),)
        
        # 准备量化后的模型 mp
        mp = torch.ao.quantization.quantize_fx.prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
        
        # 将量化后的模型 mp 转换为量化后的模型 mq
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))

        # 提取权重信息
        results = extract_weights('fp32', mp, 'int8', mq)
        
        # 获取 mq 中的节点名称列表
        mq_node_names = [node.name for node in mq.graph.nodes]
        
        # 验证每个层的名称是否存在于 mq 的节点名称中
        for layer_name in results.keys():
            self.assertTrue(layer_name in mq_node_names)

        # 再次将 mp 转换为量化后的模型 mq
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))
        
        # 添加日志记录器并记录 mp 和 mq 的激活输出
        mp_ns, mq_ns = add_loggers(
            'fp32', copy.deepcopy(mp), 'int8', mq, OutputLogger)
        
        # 创建测试数据
        data = torch.randn(1, 1, 1, 1)
        
        # 在 mp_ns 和 mq_ns 上执行测试数据
        mp_ns(data)
        mq_ns(data)
        
        # 提取日志信息并匹配激活函数的结果
        results = extract_logger_info(mp_ns, mq_ns, OutputLogger, 'int8')
        
        # 获取 mq_ns 中的节点名称列表
        mq_node_names = [node.name for node in mq_ns.graph.nodes]
        
        # 验证每个层的名称是否存在于 mq_ns 的节点名称中
        for layer_name in results.keys():
            self.assertTrue(layer_name in mq_node_names)

        # 再次将 mp 转换为量化后的模型 mq
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))
        
        # 添加阴影日志记录器并记录 mp 和 mq 的激活输出
        mp_shadows_mq = add_shadow_loggers(
            'fp32', mp, 'int8', mq, OutputLogger)
        
        # 在 mp_shadows_mq 上执行测试数据
        mp_shadows_mq(data)
        
        # 提取阴影日志信息并匹配激活函数的结果
        results = extract_shadow_logger_info(
            mp_shadows_mq, OutputLogger, 'int8')
        
        # 获取 mp_shadows_mq 中的节点名称列表
        mq_node_names = [node.name for node in mp_shadows_mq.graph.nodes]
        
        # 验证每个层的名称是否存在于 mp_shadows_mq 的节点名称中
        for layer_name in results.keys():
            self.assertTrue(layer_name in mq_node_names)

    @skipIfNoFBGEMM
    def test_extend_logger_results_with_comparison(self):
        # 创建一个序列模型，包括两个卷积层，然后设为评估模式
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)).eval()
        
        # 定义量化配置字典，使用默认的量化配置
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        
        # 创建一个示例输入，是一个包含随机数据的元组
        example_inputs = (torch.randn(1, 1, 1, 1),)
        
        # 准备模型以进行量化
        mp = torch.ao.quantization.quantize_fx.prepare_fx(
            m, qconfig_dict, example_inputs=example_inputs)
        
        # 将准备好的量化模型转换为量化后的模型
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))

        # 提取权重信息
        results = extract_weights('fp32', mp, 'int8', mq)
        
        # 将比较结果扩展到日志记录器中，使用平方信噪比作为度量
        extend_logger_results_with_comparison(
            results, 'fp32', 'int8', compute_sqnr, 'sqnr_int8_vs_fp32')
        
        # 将比较结果扩展到日志记录器中，使用归一化 L2 误差作为度量
        extend_logger_results_with_comparison(
            results, 'fp32', 'int8', compute_normalized_l2_error, 'l2_error_int8_vs_fp32')
        
        # 将比较结果扩展到日志记录器中，使用余弦相似度作为度量
        extend_logger_results_with_comparison(
            results, 'fp32', 'int8', compute_cosine_similarity,
            'cosine_similarity_int8_vs_fp32')

        # 遍历结果中的每一层，验证度量结果是否已经正确添加到字典中
        for layer_results in results.values():
            assert 'sqnr_int8_vs_fp32' in \
                layer_results['weight']['int8'][0].keys()
            assert 'l2_error_int8_vs_fp32' in \
                layer_results['weight']['int8'][0].keys()
            assert 'cosine_similarity_int8_vs_fp32' in \
                layer_results['weight']['int8'][0].keys()

    @skipIfNoFBGEMM


这段代码是一个测试函数，用于测试量化模型在不同度量标准下的表现，并将结果扩展到日志记录器中。
    def test_int8_shadows_fp32_simple(self):
        # 创建一个包含两个卷积层和ReLU激活的神经网络模型，并设置为评估模式
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1), nn.ReLU()).eval()
        # 定义量化配置字典，使用默认的量化配置
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        # 创建示例输入数据
        example_inputs = (torch.randn(1, 1, 1, 1),)
        # 准备量化仿真
        mp = torch.ao.quantization.quantize_fx.prepare_fx(
            m, qconfig_dict, example_inputs=example_inputs)
        # 在准备好的量化仿真模型上执行一次前向传播
        mp(torch.randn(1, 1, 1, 1))
        # 将量化仿真模型转换为量化模型
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))
        # 深度复制量化仿真模型，作为参考模型
        mq_ref = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))
        # 添加阴影日志记录器，将int8量化模型（mq）与fp32仿真模型（mp）关联起来
        mp_shadows_mq = add_shadow_loggers(
            'int8', mq, 'fp32', mp, OutputLogger)

        # 验证量化参数（scale和zero point）是否正确提取

        # 对于第一个操作，scale和zero point作为模块的属性存在
        scale_0 = mp_shadows_mq._0_input_scale_0
        scale_0_ref = getattr(mq_ref, '0_input_scale_0')
        self.assertEqual(scale_0, scale_0_ref)
        zp_0 = mp_shadows_mq._0_input_zero_point_0
        zp_0_ref = getattr(mq_ref, '0_input_zero_point_0')
        self.assertEqual(zp_0, zp_0_ref)

        # 对于第二个操作，第二个操作的输入的scale和zero point
        # 必须等于第一个操作的输出的scale和zero point
        scale_1 = mp_shadows_mq._1_input_scale_0
        scale_1_ref = getattr(mq_ref, '0').scale
        self.assertEqual(scale_1, scale_1_ref)
        zp_1 = mp_shadows_mq._1_input_zero_point_0
        zp_1_ref = getattr(mq_ref, '0').zero_point
        self.assertEqual(zp_1, zp_1_ref)

        # 验证运行时数据的功能性
        mp_shadows_mq(torch.randn(1, 1, 1, 1))
        # 提取阴影日志记录器中的信息，并验证其有效性
        act_compare_dict = extract_shadow_logger_info(
            mp_shadows_mq, OutputLogger, 'fp32')
        self.assertTrue(len(act_compare_dict) == 2)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    @skipIfNoFBGEMM
    def test_int8_shadows_fp32_coverage(self):
        # 定义一个内部类M，继承自torch.nn.Module，用于测试整型8位量化与单精度浮点数的覆盖情况
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
                self.conv = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # 执行自适应平均池化操作
                x = self.adaptive_avg_pool(x)
                # 将输入量化参数应用于卷积操作的量化参数
                x = self.conv(x)
                x = torch.mul(x, x)  # 对x进行平方操作
                x = self.conv(x)  # 再次对x进行卷积操作
                x = torch.add(x, x)  # 将x与自身相加
                x = F.relu(x)  # 对x进行ReLU激活函数处理
                x = self.conv(x)  # 最后一次对x进行卷积操作
                return x

        m = M().eval()  # 创建M类的实例并设为评估模式
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)  # 使用prepare_fx准备模型
        mp(*example_inputs)  # 对示例输入进行前向传播
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))  # 将量化后的模型保存在mq中
        mq_ref = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))  # 保存量化后的模型副本在mq_ref中
        mp_shadows_mq = add_shadow_loggers(
            'int8', mq, 'fp32', mp, OutputLogger)  # 将int8和fp32版本的模型mq和mp添加为影子记录器
        mp_shadows_mq(torch.randn(1, 1, 1, 1))  # 对示例输入执行前向传播
        act_compare_dict = extract_shadow_logger_info(
            mp_shadows_mq, OutputLogger, 'fp32')  # 从影子记录器mp_shadows_mq中提取fp32信息到act_compare_dict
        self.assertTrue(len(act_compare_dict) == 3)  # 断言影子记录器中的信息条目数量为3
        self.assert_ns_compare_dict_valid(act_compare_dict)  # 断言影子记录器中的信息条目有效性

    @skipIfNoFBGEMM
    def test_loggers_preserve_qat_numerics(self):
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1))
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_qat_fx(m, qconfig_dict, example_inputs=example_inputs)  # 使用prepare_qat_fx准备量化感知训练模型
        mp(*example_inputs)  # 对示例输入进行前向传播
        mc = convert_fx(copy.deepcopy(mp))  # 将量化感知训练后的模型转换为标准量化模型
        mp.apply(torch.ao.quantization.disable_observer)  # 禁用模型中的观察器

        ref_fp32 = mp(*example_inputs)  # 获取fp32版本的模型前向传播结果
        ref_int8 = mc(*example_inputs)  # 获取int8版本的模型前向传播结果

        mp_ns, mc_ns = add_loggers('fp32', mp, 'int8', mc, OutputLogger)  # 将fp32和int8版本的模型mp和mc添加为记录器
        ref_fp32_ns = mp_ns(*example_inputs)  # 对示例输入执行fp32版本的前向传播
        ref_int8_ns = mc_ns(*example_inputs)  # 对示例输入执行int8版本的前向传播
        self.assertEqual(ref_fp32, ref_fp32_ns)  # 断言标准和记录器版本的fp32模型输出相等
        self.assertEqual(ref_int8, ref_int8_ns)  # 断言标准和记录器版本的int8模型输出相等

    @skipIfNoFBGEMM
    def test_shadow_loggers_preserve_qat_numerics(self):
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1))
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_qat_fx(m, qconfig_dict, example_inputs=example_inputs)  # 使用prepare_qat_fx准备量化感知训练模型
        mp(*example_inputs)  # 对示例输入进行前向传播
        mc = convert_fx(copy.deepcopy(mp))  # 将量化感知训练后的模型转换为标准量化模型
        mp.apply(torch.ao.quantization.disable_observer)  # 禁用模型中的观察器

        ref_fp32 = mp(*example_inputs)  # 获取fp32版本的模型前向传播结果
        ref_int8 = mc(*example_inputs)  # 获取int8版本的模型前向传播结果

        mc_shadows_mp = add_shadow_loggers('int8', mc, 'fp32', mp, OutputLogger)  # 将int8和fp32版本的模型mc和mp添加为影子记录器
        ref_shadow = mc_shadows_mp(*example_inputs)  # 对示例输入执行前向传播
        self.assertEqual(ref_fp32, ref_shadow)  # 断言影子记录器版本的前向传播结果与fp32版本模型输出相等
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 如果 CUDA 不可用，则跳过测试
    def test_extract_weights_cuda(self):
        # 注意：这里不使用量化，因为量化内核目前不支持 CUDA。
        # 创建两个在 CUDA 上运行的包含单个 Conv2d 层的神经网络模型
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        # 提取权重信息
        results = extract_weights('a', m1, 'b', m2)
        # 将结果与参考数据进行比较，并记录输出日志
        extend_logger_results_with_comparison(
            results, 'a', 'b', compute_sqnr, 'sqnr')
        # 使用断言验证结果的有效性
        self.assert_ns_compare_dict_valid(results)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 如果 CUDA 不可用，则跳过测试
    def test_add_loggers_cuda(self):
        # 注意：这里不使用量化，因为量化内核目前不支持 CUDA。
        # 创建两个在 CUDA 上运行的包含单个 Conv2d 层的神经网络模型
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        # 添加日志记录器并获取相应的命名空间函数
        m1_ns, m2_ns = add_loggers('a', m1, 'b', m2, OutputLogger)
        # 创建一个在 CUDA 上的随机数据
        datum = torch.randn(1, 1, 1, 1)
        datum = datum.cuda()

        # 对 m1 和 m2 执行数据前向传递
        m1_ns(datum)
        m2_ns(datum)

        # 提取日志信息并与参考数据进行比较
        act_compare_dict = extract_logger_info(m1_ns, m2_ns, OutputLogger, 'b')
        extend_logger_results_with_comparison(
            act_compare_dict, 'a', 'b', compute_sqnr, 'sqnr')

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 如果 CUDA 不可用，则跳过测试
    def test_add_shadow_loggers_cuda(self):
        # 注意：这里不使用量化，因为量化内核目前不支持 CUDA。
        # 创建两个在 CUDA 上运行的包含单个 Conv2d 层的神经网络模型
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        # 添加阴影日志记录器函数并获取其引用
        m1_shadows_m2 = add_shadow_loggers('a', m1, 'b', m2, OutputLogger)
        # 创建一个在 CUDA 上的随机数据
        datum = torch.randn(1, 1, 1, 1)
        datum = datum.cuda()

        # 对 m1_shadows_m2 执行数据前向传递
        m1_shadows_m2(datum)

        # 提取阴影日志信息并与参考数据进行比较
        act_compare_dict = extract_shadow_logger_info(m1_shadows_m2, OutputLogger, 'b')
        extend_logger_results_with_comparison(
            act_compare_dict, 'a', 'b', compute_sqnr, 'sqnr')

    def test_fp16_shadows_fp32(self):
        # 创建一个 LinearReluFunctional 模型的副本
        m = LinearReluFunctional().eval()
        example_inputs = (torch.randn(1, 4),)
        qconfig_dict = {"": torch.ao.quantization.float16_static_qconfig}
        # 准备和转换为参考版本的 FX 模型
        mp = prepare_fx(copy.deepcopy(m), qconfig_dict, example_inputs=example_inputs)
        mq = convert_to_reference_fx(mp)
        # 添加阴影日志记录器函数，用于比较其行为
        mq_shadows_m = add_shadow_loggers('a', mq, 'b', m, OutputLogger)

    def test_mul_add_cat_stack_skips_shadowing(self):
        class M(nn.Module):
            def forward(self, x):
                x = x * x
                x = torch.mul(x, x)
                x = x + x
                x = torch.add(x, x)
                x = torch.cat([x])
                x = torch.stack([x])
                return x

        # 创建 M 类的实例并设置为评估模式
        m = M().eval()
        # 测试匹配阴影激活函数的行为
        self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),),
            results_len=0)
    def test_linear_kwargs_shadow():

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模型参数
                self.w1 = nn.Parameter(torch.empty(4, 4))
                self.b1 = nn.Parameter(torch.zeros(4))
                # 使用 Kaiming 均匀初始化权重
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

            def forward(self, x):
                # 使用线性函数进行前向传播
                x = F.linear(input=x, weight=self.w1, bias=self.b1)
                return x

        # FX 图模式量化目前对于仅有 kwargs 的支持不是很好，
        # 所以我们传入两个未量化的模型
        m = M().eval()
        # 对模型进行符号化追踪
        mt = torch.fx.symbolic_trace(m)
        # 深度复制追踪后的模型
        mt_copy = copy.deepcopy(mt)

        # 向两个模型添加影子日志记录器
        mt_shadows_mt_copy = add_shadow_loggers(
            'a', mt, 'b', mt_copy, OutputLogger)

        # 使用输入数据调用添加了影子日志记录器的模型
        mt_shadows_mt_copy(torch.randn(4, 4))
        # 提取影子日志记录器信息用于比较
        act_compare_dict = extract_shadow_logger_info(
            mt_shadows_mt_copy, OutputLogger, 'b')
        # 断言影子日志记录器信息字典长度为 1
        self.assertTrue(len(act_compare_dict) == 1)
# 在使用 QNNPACK 后端时才运行测试类，跳过没有 QNNPACK 的情况
@skipIfNoQNNPACK
class TestFXNumericSuiteNShadows(FXNumericSuiteQuantizationTestCase):
    """
    Tests the "n shadows" workflow.
    """

    def _test_impl(self, m, example_input, qconfig_mappings):
        # 获取本地后端配置信息
        backend_config = get_native_backend_config()

        # 测试输入是否有效
        _ = m(*example_input)

        # 准备 "n shadows" 模型
        msp = prepare_n_shadows_model(
            m, example_input, qconfig_mappings, backend_config)

        # 多次执行 "n shadows" 模型
        for _ in range(2):
            msp(*example_input)

        # 将 "n shadows" 模型转换为量化后的模型
        msq = convert_n_shadows_model(msp)

        # 启用日志记录器并执行量化后的模型
        loggers_set_enabled(msq, True)
        msq(*example_input)

        # 提取 "n shadows" 模型的结果
        results = extract_results_n_shadows_model(msq)

        # 打印 "n shadows" 模型的比较结果
        print_comparisons_n_shadows_model(results)

        # 返回量化后的模型
        return msq

    # 使用 QNNPACK 后端运行线性模型测试
    @withQNNPACKBackend
    def test_linear_mod(self):
        # 定义一个简单的线性模型
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(2, 2)

            def forward(self, x):
                x = self.fc1(x)
                return x

        # 创建模型实例并设置为评估模式
        m = M().eval()

        # 创建示例输入数据
        example_input = (torch.randn(2, 2),)

        # 设置量化配置映射
        qconfig_mappings = \
            QConfigMultiMapping().set_global([torch.ao.quantization.default_qconfig])

        # 调用测试实现函数
        self._test_impl(m, example_input, qconfig_mappings)

    # 使用 QNNPACK 后端运行包含 ReLU 的线性模型测试
    @withQNNPACKBackend
    def test_linear_relu_mod(self):
        # 定义包含 ReLU 的线性模型
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(2, 2)
                self.fc2 = nn.Linear(2, 2)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                x = self.relu(x)
                return x

        # 创建模型实例并设置为评估模式
        m = M().eval()

        # 创建示例输入数据
        example_input = (torch.randn(2, 2),)

        # 设置量化配置映射，包括动态量化配置
        qconfig_mappings = (
            QConfigMultiMapping().set_global([
                torch.ao.quantization.default_qconfig,
                torch.ao.quantization.default_dynamic_qconfig
            ])
        )

        # 调用测试实现函数
        self._test_impl(m, example_input, qconfig_mappings)

    # 使用 QNNPACK 后端运行包含卷积、批归一化和 ReLU 的模型测试
    @withQNNPACKBackend
    def test_conv_bn_relu_mod(self):
        # 定义包含卷积、批归一化和 ReLU 的模型
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.bn = nn.BatchNorm2d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        # 创建模型实例并设置为评估模式
        m = M().eval()

        # 创建示例输入数据
        example_input = (torch.randn(32, 1, 16, 16),)

        # 设置量化配置映射，包括逐通道量化配置
        qconfig_mappings = QConfigMultiMapping() \
            .set_global([
                torch.ao.quantization.default_qconfig,
                torch.ao.quantization.default_per_channel_qconfig
            ])

        # 调用测试实现函数
        self._test_impl(m, example_input, qconfig_mappings)

    # 使用 QNNPACK 后端
    def test_functions(self):
        # 定义一个内部类 M，继承自 nn.Module
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化参数 w1 和 b1，并使用 Kaiming 均匀分布初始化 w1
                self.w1 = nn.Parameter(torch.randn(2, 2))
                self.b1 = nn.Parameter(torch.zeros(2))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

            def forward(self, x):
                # 使用 sigmoid 激活函数处理输入 x
                x = F.sigmoid(x)
                # 使用线性函数对 x 进行变换，使用参数 w1 和 b1
                x = F.linear(x, self.w1, self.b1)
                # 再次使用线性函数，使用 w1 的一个切片和 b1
                x = F.linear(x, self.w1[:], self.b1)
                # 使用 ReLU 激活函数
                x = F.relu(x)
                # 对 x 进行加法操作
                x = x + x
                # 将 x 沿着第一个维度进行拼接
                x = torch.cat([x])
                # 将 x 沿着第一个维度进行拼接，使用另一种语法形式
                x = torch.cat((x,))
                # 将 x 沿着第一个维度进行拼接，使用命名参数形式
                x = torch.cat(tensors=[x])
                # TODO(future PR): enable layernorm
                # 在 FX 图模式量化中插入观察者时，由于无法处理第二个参数，
                # 所以暂时阻塞 layer_norm 的启用，如果第二个参数是模块输入的话
                # x = F.layer_norm(x, x.shape)
                # x = F.layer_norm(x, x.shape[1:])
                # x = x.reshape(1, -1) * 2
                # x = F.layer_norm(x.reshape(1, -1), x.shape[1:])
                # 使用矩阵乘法计算 x 与 x 重塑后的结果的乘积
                x = torch.matmul(x, x.reshape(2, 2))
                x = torch.matmul(x.reshape(2, 2), x.reshape(2, 2))
                # TODO(future PR): enable below after FX graph mode quantization handles
                # it, currently this is not supported
                # x = F.linear(input=x, weight=self.w1, bias=self.b1)
                return x

        # 创建 M 类的实例 m，并设置为评估模式
        m = M().eval()
        # 创建一个示例输入 example_input
        example_input = (torch.randn(2, 2),)

        # 创建一个 QConfigMultiMapping 的实例 qconfig_mappings
        qconfig_mappings = QConfigMultiMapping() \
            .set_global([torch.ao.quantization.default_qconfig])
        # 调用 _test_impl 方法来测试模型 m 的实现
        self._test_impl(m, example_input, qconfig_mappings)

    # 使用 QNNPACK 后端进行测试
    @withQNNPACKBackend
    def test_partial_qconfig_mapping(self):
        # 定义一个内部类 M，继承自 nn.Module
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个线性层 fc，输入维度为 2，输出维度为 2
                self.fc = nn.Linear(2, 2)
                # 初始化参数 w1 和 b1
                self.w1 = nn.Parameter(torch.randn(2, 2))
                self.b1 = nn.Parameter(torch.randn(2))
                # 使用 Kaiming 均匀分布初始化 w1
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

            def forward(self, x):
                # 使用线性层 fc 对输入 x 进行变换
                x = self.fc(x)
                # 使用线性函数对 x 进行变换，使用参数 w1 和 b1
                x = F.linear(x, self.w1, self.b1)
                # 使用 ReLU 激活函数
                x = F.relu(x)
                # 对 x 进行加法操作
                x = x + x
                return x

        # 创建 M 类的实例 m，并设置为评估模式
        m = M().eval()
        # 创建一个示例输入 example_input
        example_input = (torch.randn(2, 2),)
        # 获取默认的量化配置 qconfig
        qconfig = torch.ao.quantization.default_qconfig

        # 创建一个 QConfigMultiMapping 的实例 qconfig_mappings，并设置线性函数和 ReLU 函数的量化配置
        qconfig_mappings = QConfigMultiMapping() \
            .set_object_type(F.linear, [qconfig]) \
            .set_object_type(F.relu, [qconfig])
        # 调用 _test_impl 方法来测试模型 m 的实现
        self._test_impl(m, example_input, qconfig_mappings)
    # 定义一个测试方法，验证日志记录器和保存激活状态标志位的行为
    def test_logger_enabled_and_save_activations_flags(self):
        # 创建一个包含单个线性层的序列模型，并设置为评估模式
        m = nn.Sequential(nn.Linear(1, 1)).eval()
        # 创建一个示例输入数据元组
        example_input = (torch.randn(1, 1),)

        # 创建量化配置的映射对象，设置全局量化配置
        qconfig_mappings = QConfigMultiMapping() \
            .set_global([torch.ao.quantization.default_qconfig])
        # 获取本地后端配置
        backend_config = get_native_backend_config()

        # 准备 N-Shadows 模型，包括模型、示例输入、量化配置映射和后端配置
        msp = prepare_n_shadows_model(
            m, example_input, qconfig_mappings, backend_config)

        # 多次运行模型，以确保准备 N-Shadows 模型正确进行
        for _ in range(2):
            msp(*example_input)

        # 定义一个内部方法，用于检查日志记录器的数量
        def _check_logger_count(model, exp_count_stats, exp_count_comparisons):
            # 遍历模型中的所有模块，检查是否是 OutputLogger 类型的模块
            for name, mod in model.named_modules():
                if isinstance(mod, OutputLogger):
                    # 断言统计数据的数量符合预期
                    self.assertTrue(
                        len(mod.stats) == exp_count_stats,
                        f'stats: expected {len(mod.stats)} to equal {exp_count_stats}')
                    # 如果是 OutputComparisonLogger 类型的模块，断言比较数据的数量符合预期
                    if isinstance(mod, OutputComparisonLogger):
                        self.assertTrue(
                            len(mod.comparisons) == exp_count_comparisons,
                            f'comparisons: expected {len(mod.comparisons)} to equal {exp_count_comparisons}')

        # 启用保存激活状态后，转换 N-Shadows 模型并设置日志记录器
        msq = convert_n_shadows_model(copy.deepcopy(msp))
        loggers_set_enabled(msq, True)
        loggers_set_save_activations(msq, True)
        # 在准备校准后但转换校准前，日志记录器不应保存任何内容
        _check_logger_count(msq, 0, 0)
        msq(*example_input)
        # 校准后，日志记录器应保存每个项目
        _check_logger_count(msq, 1, 1)

        # 启用保存激活状态前，转换 N-Shadows 模型并设置日志记录器
        msq = convert_n_shadows_model(copy.deepcopy(msp))
        loggers_set_enabled(msq, True)
        loggers_set_save_activations(msq, False)
        # 在准备校准后但转换校准前，日志记录器不应保存任何内容
        _check_logger_count(msq, 0, 0)
        msq(*example_input)
        # 统计应为空，但比较应存在
        _check_logger_count(msq, 0, 1)

    # 使用 TorchDynamo 的 skipIfTorchDynamo 装饰器，跳过此测试因为速度太慢
    @skipIfTorchDynamo("too slow")
    # 如果没有导入 torchvision，则跳过此测试
    @skip_if_no_torchvision
    # 使用 QNNPACK 后端进行测试
    @withQNNPACKBackend
    # 测试 MobileNet V2 模型
    def test_mobilenet_v2(self):
        import torchvision
        # 加载 torchvision 库
        m = torchvision.models.quantization.mobilenet_v2(
            pretrained=False, quantize=False).eval()
        # 创建一个示例输入数据元组
        example_input = (torch.randn(1, 3, 224, 224),)

        # 创建量化配置的映射对象，设置全局量化配置和默认动态量化配置
        qconfig_mappings = QConfigMultiMapping() \
            .set_global([torch.ao.quantization.default_qconfig, torch.ao.quantization.default_dynamic_qconfig])

        # 调用 _test_impl 方法，测试实现
        self._test_impl(m, example_input, qconfig_mappings)

    # 使用 QNNPACK 后端进行装饰，适用于测试前端
    @withQNNPACKBackend
    def test_qconfig_multi_mapping_deduplication(self):
        # 检查插入操作是否能够对 qconfigs 进行去重
        qconfig_multi_mapping = QConfigMultiMapping().set_global(
            [torch.ao.quantization.default_qconfig, torch.ao.quantization.default_qconfig]
        )
        # 断言：确保 qconfig_mappings_list 中的元素个数为 1
        self.assertEqual(len(qconfig_multi_mapping.qconfig_mappings_list), 1)

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_insert_padding(self):
        # 测试插入高优先级 qconfig 样式时，如果比低优先级 qconfig 元素更少，
        # 结果应该会在相同样式和键的额外 QConfigMappings 中添加 None
        qconfig_multi_mapping = (
            QConfigMultiMapping()
            .set_global(
                [
                    torch.ao.quantization.default_qconfig,
                    torch.ao.quantization.default_dynamic_qconfig,
                ]
            )
            .set_object_type(torch.nn.Linear, [torch.ao.quantization.default_qconfig])
            .set_module_name_regex("fc", [torch.ao.quantization.default_qconfig])
            .set_module_name("fc2", [torch.ao.quantization.default_qconfig])
            .set_module_name_object_type_order(
                "", nn.Linear, 0, [torch.ao.quantization.default_qconfig]
            )
        )

        # 断言：确保在 qconfig_mappings_list 的第 1 个元素中，特定对象类型的 qconfigs 为 None
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].object_type_qconfigs[
                torch.nn.Linear
            ],
            None,
        )
        # 断言：确保在 qconfig_mappings_list 的第 1 个元素中，特定模块名正则表达式的 qconfigs 为 None
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_regex_qconfigs[
                "fc"
            ],
            None,
        )
        # 断言：确保在 qconfig_mappings_list 的第 1 个元素中，特定模块名的 qconfigs 为 None
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_qconfigs["fc2"],
            None,
        )
        # 断言：确保在 qconfig_mappings_list 的第 1 个元素中，特定模块名和对象类型顺序的 qconfigs 为 None
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[
                1
            ].module_name_object_type_order_qconfigs[("", nn.Linear, 0)],
            None,
        )

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_retroactive_padding(self):
        # 测试向 QConfigMultiMapping 插入具有更多元素的低优先级 qconfig 样式
        # 会导致新的 QConfigMapping 在所有先前存在的样式和键上都有 None

        # 创建一个 QConfigMultiMapping 对象
        qconfig_multi_mapping = (
            QConfigMultiMapping()
            .set_object_type(torch.nn.Linear, [torch.ao.quantization.default_qconfig])
            # 设置对象类型为 torch.nn.Linear 的 qconfig 样式为默认 qconfig
            .set_module_name_regex("fc", [torch.ao.quantization.default_qconfig])
            # 设置模块名匹配正则表达式包含 "fc" 的 qconfig 样式为默认 qconfig
            .set_module_name("fc2", [torch.ao.quantization.default_qconfig])
            # 设置模块名为 "fc2" 的 qconfig 样式为默认 qconfig
            .set_module_name_object_type_order(
                "", nn.Linear, 0, [torch.ao.quantization.default_qconfig]
            )
            # 设置模块名为空、类型为 nn.Linear、顺序为 0 的 qconfig 样式为默认 qconfig
            .set_global(
                [
                    torch.ao.quantization.default_qconfig,
                    torch.ao.quantization.default_dynamic_qconfig,
                ]
            )
            # 设置全局 qconfig 样式为默认 qconfig 和默认动态 qconfig
        )

        # 断言检查新创建的 qconfig_mappings_list 的第二个元素的属性是否为 None
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].object_type_qconfigs[
                torch.nn.Linear
            ],
            None,
        )
        # 断言检查新创建的 qconfig_mappings_list 的第二个元素的属性是否为 None
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_regex_qconfigs[
                "fc"
            ],
            None,
        )
        # 断言检查新创建的 qconfig_mappings_list 的第二个元素的属性是否为 None
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_qconfigs["fc2"],
            None,
        )
        # 断言检查新创建的 qconfig_mappings_list 的第二个元素的属性是否为 None
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[
                1
            ].module_name_object_type_order_qconfigs[("", nn.Linear, 0)],
            None,
        )

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_end_to_end(self):
        # 测试 prepare/convert_n_shadows_model 是否按预期工作
        # 与 qconfig_multi_mapping 一起，避免不必要的匹配

        # 创建一个 TwoLayerLinearModel 对象并设置为评估模式
        m = TwoLayerLinearModel().eval()
        # 获取模型的示例输入
        example_input = m.get_example_inputs()

        # 创建一个 QConfigMultiMapping 对象
        qconfig_multi_mapping = (
            QConfigMultiMapping()
            .set_global(
                [
                    torch.ao.quantization.default_qconfig,
                    torch.ao.quantization.default_dynamic_qconfig,
                ]
            )
            # 设置全局 qconfig 样式为默认 qconfig 和默认动态 qconfig
            .set_module_name("fc2", [None, torch.ao.quantization.default_qconfig])
            # 设置模块名为 "fc2" 的 qconfig 样式为 None 和默认 qconfig
        )

        # 断言检查新创建的 qconfig_mappings_list 的第二个元素的属性是否为 None
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_qconfigs["fc2"],
            None,
        )

        # 调用 _test_impl 方法，传入模型 m、示例输入 example_input 和 qconfig_multi_mapping
        msq = self._test_impl(m, example_input, qconfig_multi_mapping)

        # 检查 msq.shadow_wrapper_0_1.mod_0 是否为量化线性模型
        self.checkQuantizedLinear(msq.shadow_wrapper_0_1.mod_0)
        # 检查 msq.shadow_wrapper_0_2.mod_0 是否为动态量化的量化线性模型，使用 torch.qint8 类型
        self.checkDynamicQuantizedLinear(msq.shadow_wrapper_0_2.mod_0, torch.qint8)
        # 检查 msq.shadow_wrapper_1_1.mod_0 是否为量化线性模型
        self.checkQuantizedLinear(msq.shadow_wrapper_1_1.mod_0)
        # 断言检查是否抛出 AttributeError 异常，异常消息包含 ".*"
        self.assertRaisesRegex(AttributeError, ".*", lambda: msq.shadow_wrapper_1_2)

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_from_list(self):
        # 测试 QConfigMultiMapping.from_list_qconfig_mapping 是否按预期工作

        # 创建一个评估模式的 TwoLayerLinearModel 实例
        m = TwoLayerLinearModel().eval()
        # 获取模型的示例输入
        example_input = m.get_example_inputs()

        # 创建 QConfigMapping 对象列表
        qconfig_mappings_list = [
            QConfigMapping().set_global(torch.ao.quantization.default_qconfig),
            # 创建一个新的 QConfigMapping 对象，并设置全局量化配置为默认动态量化配置
            QConfigMapping()
            .set_global(torch.ao.quantization.default_dynamic_qconfig)
            .set_module_name("fc2", torch.ao.quantization.default_qconfig),
        ]

        # 从 QConfigMapping 对象列表创建 QConfigMultiMapping 对象
        qconfig_multi_mapping = QConfigMultiMapping().from_list_qconfig_mapping(
            qconfig_mappings_list
        )
        
        # 断言第二个 QConfigMapping 对象中的 "fc2" 模块的量化配置为 None
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_qconfigs["fc2"],
            None,
        )

        # 使用测试实现函数 _test_impl 进行测试
        msq = self._test_impl(m, example_input, qconfig_multi_mapping)

        # 检查量化线性层是否正常
        self.checkQuantizedLinear(msq.shadow_wrapper_0_1.mod_0)
        self.checkDynamicQuantizedLinear(msq.shadow_wrapper_0_2.mod_0, torch.qint8)
        self.checkQuantizedLinear(msq.shadow_wrapper_1_1.mod_0)
        # 尝试调用未定义的属性，预期引发 AttributeError 异常
        self.assertRaisesRegex(AttributeError, ".*", lambda: msq.shadow_wrapper_1_2)

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_ordering(self):
        # 测试模块排序是否忽略 None

        # 创建一个评估模式的 TwoLayerLinearModel 实例
        m = TwoLayerLinearModel().eval()
        # 获取模型的示例输入
        example_input = m.get_example_inputs()

        # 创建 QConfigMultiMapping 对象，并设置全局量化配置列表及对应模块的量化配置列表
        qconfig_multi_mapping = (
            QConfigMultiMapping()
            .set_global(
                [
                    torch.ao.quantization.default_qconfig,
                    torch.ao.quantization.default_dynamic_qconfig,
                ]
            )
            .set_module_name(
                "fc2",
                [
                    None,
                    torch.ao.quantization.default_dynamic_qconfig,
                    torch.ao.quantization.default_qat_qconfig_v2,
                ],
            )
        )
        
        # 断言 QConfigMultiMapping 对象中 QConfigMapping 对象列表的长度为 2
        self.assertEqual(len(qconfig_multi_mapping.qconfig_mappings_list), 2)

        # 使用测试实现函数 _test_impl 进行测试
        msq = self._test_impl(m, example_input, qconfig_multi_mapping)

        # 检查量化线性层是否正常
        self.checkQuantizedLinear(msq.shadow_wrapper_0_1.mod_0)
        self.checkDynamicQuantizedLinear(msq.shadow_wrapper_0_2.mod_0, torch.qint8)
        self.checkDynamicQuantizedLinear(msq.shadow_wrapper_1_1.mod_0, torch.qint8)
        self.checkQuantizedLinear(msq.shadow_wrapper_1_2.mod_0)
    # 定义一个测试方法，用于测试 QConfigMultiMapping 的字符串表示
    def test_qconfig_multi_mapping_repr(self):
        # 创建 QConfigMultiMapping 的实例并设置全局配置和模块特定配置
        qconfig_multi_mapping = (
            QConfigMultiMapping()
            .set_global(
                [
                    torch.ao.quantization.default_qconfig,
                    torch.ao.quantization.default_dynamic_qconfig,
                ]
            )
            .set_module_name(
                "fc2",
                [
                    None,
                    torch.ao.quantization.default_dynamic_qconfig,
                    torch.ao.quantization.default_qat_qconfig_v2,
                ],
            )
        )
        # 断言 qconfig_multi_mapping 的字符串表示是一个字符串类型
        self.assertTrue(isinstance(qconfig_multi_mapping.__repr__(), str))

    # 使用 QNNPACK 后端装饰的测试方法
    @withQNNPACKBackend
    def test_custom_functions_and_tracer(self):
        # 定义一个简单的神经网络模型类 M
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(2, 2)
                self.fc2 = nn.Linear(2, 2)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        # 创建 M 类的实例，并设置为评估模式
        m = M().eval()
        # 创建一个示例输入
        example_inputs = (torch.randn(2, 2),)

        # 创建 QConfigMultiMapping 的实例并设置全局配置
        qconfig_mappings = QConfigMultiMapping().set_global(
            [torch.ao.quantization.default_qat_qconfig]
        )

        # 创建 QuantizationTracer 的实例，用于自定义量化过程追踪
        custom_tracer = torch.ao.quantization.quantize_fx.QuantizationTracer(
            ["fc2"], []
        )

        # 定义自定义的准备函数
        custom_prepare_fn = torch.ao.quantization.quantize_fx.prepare_qat_fx

        # 定义自定义的转换函数
        def custom_convert_fn(module, to_print):
            print(to_print)
            mod = torch.ao.quantization.quantize_fx.convert_fx(module)
            return mod

        # 获取本地后端配置
        backend_config = get_native_backend_config()

        # 测试输入是否有效
        _ = m(*example_inputs)

        # 准备原始模型及其阴影模型
        msp = prepare_n_shadows_model(
            m,
            example_inputs,
            qconfig_mappings,
            backend_config,
            custom_prepare_fn=custom_prepare_fn,
            custom_prepare_kwargs=None,
            custom_tracer=custom_tracer,
        )

        # 多次执行阴影模型
        for _ in range(2):
            msp(*example_inputs)

        # 转换阴影模型
        msq = convert_n_shadows_model(
            msp, custom_convert_fn=custom_convert_fn, custom_convert_kwargs=kwargs
        )
        # 打印转换后的模型
        print(msq)

        # 启用日志记录器
        loggers_set_enabled(msq, True)

        # 执行转换后的模型
        msq(*example_inputs)

        # 提取阴影模型的结果
        results = extract_results_n_shadows_model(msq)

        # 打印比较阴影模型的结果
        print_comparisons_n_shadows_model(results)

    # 内部方法，用于测试提取权重实现
    def _test_extract_weights_impl(self, m, example_input, qconfig_mapping):
        # 获取本地后端配置
        backend_config = get_native_backend_config()
        # 使用阴影模型比较权重并打印比较结果
        results = _n_shadows_compare_weights(
            m, example_input, qconfig_mapping, backend_config)
        print_comparisons_n_shadows_model(results)

    # 使用 QNNPACK 后端装饰
    @withQNNPACKBackend
    def test_extract_weights_linear(self):
        # 定义一个内部的神经网络模型类 M，继承自 nn.Module
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化第一层的权重和偏置参数
                self.w1 = nn.Parameter(torch.randn(2, 2))
                self.b1 = nn.Parameter(torch.randn(2))
                # 使用 Kaiming 均匀初始化方法初始化权重 self.w1
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
                # 初始化第二层的权重和偏置参数
                self.w2 = nn.Parameter(torch.randn(2, 2))
                self.b2 = nn.Parameter(torch.randn(2))
                # 使用 Kaiming 均匀初始化方法初始化权重 self.w2
                torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
                # 初始化第三层的权重和偏置参数
                self.w3 = nn.Parameter(torch.randn(2, 2))
                self.b3 = nn.Parameter(torch.randn(2))
                # 使用 Kaiming 均匀初始化方法初始化权重 self.w3
                torch.nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
                # 初始化第四层的权重和偏置参数
                self.w4 = nn.Parameter(torch.randn(2, 2))
                self.b4 = nn.Parameter(torch.randn(2))
                # 使用 Kaiming 均匀初始化方法初始化权重 self.w4
                torch.nn.init.kaiming_uniform_(self.w4, a=math.sqrt(5))

            # 前向传播函数定义
            def forward(self, x):
                # 第一层线性操作
                x = F.linear(x, self.w1, self.b1)
                # 第二层线性操作
                x = F.linear(x, self.w2, self.b2)
                # 使用 ReLU 激活函数
                x = F.relu(x)
                # 第三层线性操作
                x = F.linear(x, self.w3, self.b3)
                # 第四层线性操作
                x = F.linear(x, self.w4, self.b4)
                return x

        # 获取默认的量化配置
        per_tensor_qconfig = torch.ao.quantization.default_qconfig

        # 创建 M 类的实例，并设置为评估模式
        m = M().eval()
        # 定义一个示例输入
        example_input = (torch.randn(2, 2),)
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping()
        
        # 测试未量化的情况
        qconfig_mapping.set_module_name_object_type_order(
            '', F.linear, 2, None)
        # 测试每张量量化的情况
        qconfig_mapping.set_module_name_object_type_order(
            '', F.linear, 3, per_tensor_qconfig)
        # 调用内部函数 _test_extract_weights_impl 进行权重提取测试
        self._test_extract_weights_impl(m, example_input, qconfig_mapping)
    # 定义一个用于测试添加日志记录器的方法，内部实现
    def _test_add_loggers_impl(self, m, example_input, qconfig_mapping):
        # 获取本地后端的配置信息
        backend_config = get_native_backend_config()
        # 深度拷贝输入的模型对象m
        m_copy = copy.deepcopy(m)

        # 测试输入数据是否有效，调用模型m进行前向推理
        _ = m(*example_input)

        # 准备带有N个影子模型的日志记录器模型msp
        msp = _prepare_n_shadows_add_loggers_model(
            m, example_input, qconfig_mapping, backend_config)
        msp(*example_input)

        # 将带有N个影子模型的日志记录器模型msp转换为量化模型msq
        msq = convert_n_shadows_model(msp)
        loggers_set_enabled(msq, True)

        # 使用输入数据在量化模型msq上进行推理，得到FP32输出
        output_fp32 = msq(*example_input)

        # 从结果中提取带有N个影子模型的模型msq的输出
        results = extract_results_n_shadows_model(msq)

        # 获取内部结果的最后一个量化输出
        inner_results = results['model']['node_output']
        last_subgraph = list(inner_results.keys())[-1]
        output_shadow = inner_results[last_subgraph][0]['values'][-1]

        # 验证FP32和量化输出是否与参考值匹配
        output_fp32_ref = m_copy(*example_input)
        mp_ref = prepare_fx(m_copy, qconfig_mapping, example_input)
        for _ in range(2):
            mp_ref(*example_input)
        mq_ref = convert_fx(mp_ref)
        output_shadow_ref = mq_ref(*example_input)

        # 使用断言确保FP32输出与参考值相似
        self.assertTrue(
            torch.allclose(output_fp32, output_fp32_ref),
            f"fp32 comparison: {output_fp32} not close to {output_fp32_ref}")

        # 使用断言确保量化输出与参考值相似
        self.assertTrue(
            torch.allclose(output_shadow, output_shadow_ref),
            f"shadow comparison: {output_shadow} not close to {output_shadow_ref}")

        # 返回量化模型msq
        return msq

    # 使用QNNPACK后端装饰器测试线性模型添加日志记录器，输入和输出均为量化
    @withQNNPACKBackend
    def test_add_loggers_linear_mod_quant_quant(self):
        # 创建包含两个线性层的序列模型m
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        example_input = (torch.randn(2, 2),)  # 创建一个随机输入
        qconfig_mapping = get_default_qconfig_mapping()  # 获取默认的量化配置映射
        # 调用内部测试方法_test_add_loggers_impl进行测试
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    # 使用QNNPACK后端装饰器测试线性模型添加日志记录器，第一个线性层为FP32，第二个线性层为量化
    @withQNNPACKBackend
    def test_add_loggers_linear_mod_fp32_quant(self):
        # 创建包含两个线性层的序列模型m
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        example_input = (torch.randn(2, 2),)  # 创建一个随机输入
        qconfig_mapping = get_default_qconfig_mapping()  # 获取默认的量化配置映射
        qconfig_mapping.set_module_name('0', None)  # 将第一个模块设置为FP32
        # 调用内部测试方法_test_add_loggers_impl进行测试
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    # 使用QNNPACK后端装饰器测试线性模型添加日志记录器，第一个线性层为量化，第二个线性层为FP32
    @withQNNPACKBackend
    def test_add_loggers_linear_mod_quant_fp32(self):
        # 创建包含两个线性层的序列模型m
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        example_input = (torch.randn(2, 2),)  # 创建一个随机输入
        qconfig_mapping = get_default_qconfig_mapping()  # 获取默认的量化配置映射
        qconfig_mapping.set_module_name('1', None)  # 将第二个模块设置为FP32
        # 调用内部测试方法_test_add_loggers_impl进行测试
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    # 使用QNNPACK后端装饰器
    # 定义测试函数，用于测试添加记录器到线性模型的量化过程（输入输出都是32位浮点数）
    def test_add_loggers_linear_mod_fp32_fp32(self):
        # 创建一个包含两个线性层的序列模型
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        # 准备一个示例输入作为模型的输入
        example_input = (torch.randn(2, 2),)
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping()
        # 设置第一个和第二个模块的名称为 None，表示不量化
        qconfig_mapping.set_module_name('0', None)
        qconfig_mapping.set_module_name('1', None)
        # 调用 _test_add_loggers_impl 方法进行测试
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    # 使用 QNNPACK 后端测试添加记录器到卷积、批归一化和ReLU融合的量化过程
    @withQNNPACKBackend
    def test_add_loggers_conv_bn_relu_fusion_quant(self):
        # 创建一个包含卷积、批归一化和ReLU的序列模型
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
        # 将模型设置为评估模式
        m.eval()
        # 准备一个示例输入作为模型的输入
        example_input = (torch.randn(16, 1, 4, 4),)
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping()
        # 调用 _test_add_loggers_impl 方法进行测试
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    # 使用 QNNPACK 后端测试添加记录器到卷积、批归一化和ReLU融合的32位浮点数过程
    @withQNNPACKBackend
    def test_add_loggers_conv_bn_relu_fusion_fp32(self):
        # 创建一个包含卷积、批归一化和ReLU的序列模型
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
        # 将模型设置为评估模式
        m.eval()
        # 准备一个示例输入作为模型的输入
        example_input = (torch.randn(16, 1, 4, 4),)
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping()
        # 设置第一个、第二个和第三个模块的名称为 None，表示不量化
        qconfig_mapping.set_module_name('0', None)
        qconfig_mapping.set_module_name('1', None)
        qconfig_mapping.set_module_name('2', None)
        # 调用 _test_add_loggers_impl 方法进行测试
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    # 测试添加记录器到自定义函数的量化过程
    def test_add_loggers_functions(self):
        # 定义一个包含线性层和ReLU的自定义模型类
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Parameter(torch.randn(2, 2))
                self.b1 = nn.Parameter(torch.randn(2))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

            def forward(self, x):
                x = F.linear(x, self.w1, self.b1)
                x = F.relu(x)
                x = x + x
                x = x + 1
                # TODO: 未来的 PR：支持第一个参数为标量的情况
                # x = 1 + x
                x = torch.cat([x, x])
                x = torch.cat([x, x])
                x = torch.cat(tensors=[x, x])
                # 由于量化不可匹配的函数
                x = torch.nn.functional.rrelu(x)
                x = F.linear(x, self.w1, self.b1)
                return x

        # 创建 M 类的实例并设置为评估模式
        m = M().eval()
        # 准备一个示例输入作为模型的输入
        example_input = (torch.randn(16, 2),)
        # 遍历两个量化配置映射进行测试
        for qconfig_mapping in (get_default_qconfig_mapping(), QConfigMapping()):
            # 调用 _test_add_loggers_impl 方法进行测试
            self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    # 跳过 Torch Dynamo 环境下测试 MobileNet V2 模型的记录器添加，因为速度太慢
    @skipIfTorchDynamo("too slow")
    @skip_if_no_torchvision
    @withQNNPACKBackend
    def test_add_loggers_mobilenet_v2(self):
        # 导入 torchvision 库
        import torchvision
        # 创建一个未量化的 MobileNet V2 模型实例并设置为评估模式
        m = torchvision.models.quantization.mobilenet_v2(pretrained=False, quantize=False).eval()
        # 准备一个示例输入作为模型的输入
        example_input = (torch.randn(8, 3, 224, 224),)
        # 获取默认的量化配置映射
        qconfig_mapping = get_default_qconfig_mapping()
        # 调用 _test_add_loggers_impl 方法进行测试
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)
class TestFXNumericSuiteCoreAPIsModels(FXNumericSuiteQuantizationTestCase):
    """
    Tests numeric suite core APIs on non-toy models.
    """

    @skipIfNoFBGEMM
    def test_compare_weights_conv(self):
        # 定义测试用例，包含不同的卷积模型
        test_cases = (
            (ConvModel(),),
            (ConvBnModel(),),
            (ConvBnReLUModel(),),
        )
        # 遍历每个测试用例
        for m, in test_cases:
            # 将模型设置为评估模式
            m.eval()
            # 创建示例输入
            example_inputs = (torch.randn(1, 3, 5, 5),)
            # 调用测试函数，验证权重提取功能，预期结果长度为1
            self._test_extract_weights(m, example_inputs, results_len=1)

    @skipIfNoFBGEMM
    def test_compare_weights_linear(self):
        # 定义测试用例，包含不同的线性模型
        test_cases = (
            (SingleLayerLinearModel(), None),
            (
                SingleLayerLinearDynamicModel(),
                {"object_type": [(nn.Linear, default_dynamic_qconfig)]},
            ),
        )
        # 遍历每个测试用例
        for m, qconfig_dict in test_cases:
            # 将模型设置为评估模式
            m.eval()
            # 创建示例输入
            example_inputs = (torch.randn(1, 3, 5, 5),)
            # 调用测试函数，验证权重提取功能，预期结果长度为1，带有量化配置字典
            res = self._test_extract_weights(
                m, example_inputs, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_compare_weights_lstm_dynamic(self):
        # 设置量化配置字典，针对LSTM模型
        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}
        # 创建LSTM模型的示例输入
        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))
        example_inputs = (lstm_input, lstm_hidden)
        # 初始化LSTM模型，并设置为评估模式
        m = LSTMwithHiddenDynamicModel().eval()
        # 调用测试函数，验证权重提取功能，预期结果长度为1，带有量化配置字典
        res = self._test_extract_weights(
            m, example_inputs, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_compare_activations_conv(self):
        # 定义测试用例，包含不同的卷积模型
        test_cases = (
            (ConvModel(),),
            (ConvBnModel(),),
            (ConvBnReLUModel(),),
        )
        # 遍历每个测试用例
        for m, in test_cases:
            # 将模型设置为评估模式
            m.eval()
            # 调用测试函数，验证激活匹配功能，预期结果长度为1
            res = self._test_match_activations(
                m, (torch.randn(1, 3, 4, 4),), results_len=1)

    @skipIfNoFBGEMM
    def test_compare_activations_linear(self):
        # 定义测试用例，包含不同的线性模型
        test_cases = (
            (SingleLayerLinearModel(), None),
            (
                SingleLayerLinearDynamicModel(),
                {"object_type": [(nn.Linear, default_dynamic_qconfig)]},
            ),
        )
        # 遍历每个测试用例
        for m, qconfig_dict in test_cases:
            # 将模型设置为评估模式
            m.eval()
            # 调用测试函数，验证激活匹配功能，预期结果长度为1，带有量化配置字典
            res = self._test_match_activations(
                m, (torch.randn(5, 5),), results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_compare_activations_lstm_dynamic(self):
        # 设置量化配置字典，针对LSTM模型
        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}
        # 初始化LSTM模型，并设置为评估模式
        m = LSTMwithHiddenDynamicModel().eval()
        # 创建LSTM模型的示例输入
        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))
        # 调用测试函数，验证激活匹配功能，预期结果长度为1，带有量化配置字典，跳过脚本化
        res = self._test_match_activations(
            m, (lstm_input, lstm_hidden), results_len=1, qconfig_dict=qconfig_dict,
            skip_scripting=True)

    @skipIfNoFBGEMM
    @skipIfTorchDynamo("too slow")
    @skip_if_no_torchvision
    @skipIfNoFBGEMM


    # 如果运行在Torch Dynamo环境下速度过慢，则跳过此测试
    # 如果没有安装torchvision库，则跳过此测试
    # 如果系统不支持FBGEMM加速，则跳过此测试



    def test_resnet18(self):
        import torchvision
        m = torchvision.models.quantization.resnet18(pretrained=False, quantize=False).eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        self._test_match_shadow_activations(
            m, (torch.randn(1, 3, 224, 224),),
            qconfig_dict=qconfig_dict,
            should_log_inputs=False)


    # 定义一个测试函数，测试ResNet-18模型的行为
    # 导入torchvision库
    # 实例化一个ResNet-18模型，不使用预训练权重，不进行量化，并设置为评估模式
    # 定义量化配置字典，使用默认的量化配置
    # 调用内部方法 _test_match_shadow_activations 测试模型的影子激活
    # 输入为一个随机生成的1x3x224x224的张量
    # 设置不记录输入日志



    @skipIfNoFBGEMM
    def test_compare_shadow_activations_conv(self):
        test_cases = (
            (ConvModel(),),
            (ConvBnModel(),),
            (ConvBnReLUModel(),),
        )
        for m, in test_cases:
            m.eval()
            res = self._test_match_shadow_activations(
                m, (torch.randn(1, 3, 4, 4),), results_len=1)


    # 如果系统不支持FBGEMM加速，则跳过此测试
    # 定义一个测试函数，用于比较卷积模型的影子激活
    # 初始化多个测试案例，每个案例包含一个不同的卷积模型实例
    # 将每个模型设为评估模式
    # 调用内部方法 _test_match_shadow_activations 测试模型的影子激活
    # 输入为一个随机生成的1x3x4x4的张量
    # 设置期望输出结果的长度为1



    @skipIfNoFBGEMM
    def test_compare_shadow_activations_linear(self):
        test_cases = (
            (SingleLayerLinearModel(), None),
            (
                SingleLayerLinearDynamicModel(),
                {"object_type": [(nn.Linear, default_dynamic_qconfig)]},
            ),
        )
        for m, qconfig_dict in test_cases:
            m.eval()
            res = self._test_match_shadow_activations(
                m, (torch.randn(5, 5),), results_len=1, qconfig_dict=qconfig_dict)


    # 如果系统不支持FBGEMM加速，则跳过此测试
    # 定义一个测试函数，用于比较线性模型的影子激活
    # 初始化多个测试案例，每个案例包含一个不同的线性模型实例以及对应的量化配置字典
    # 将每个模型设为评估模式
    # 调用内部方法 _test_match_shadow_activations 测试模型的影子激活
    # 输入为一个随机生成的5x5的张量
    # 设置期望输出结果的长度为1，并传入相应的量化配置字典



    @skipIfNoFBGEMM
    def test_compare_shadow_activations_lstm_dynamic(self):
        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}
        m = LSTMwithHiddenDynamicModel().eval()
        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))
        # TODO(future PR): enable scripting (quant prepared LSTM not scriptable)
        res = self._test_match_shadow_activations(
            m, (lstm_input, lstm_hidden), results_len=1, qconfig_dict=qconfig_dict,
            skip_scripting=True)


    # 如果系统不支持FBGEMM加速，则跳过此测试
    # 定义一个测试函数，用于比较动态LSTM模型的影子激活
    # 定义动态量化配置字典，针对LSTM模型进行配置
    # 实例化一个动态LSTM模型，并设为评估模式
    # 生成随机的LSTM输入和隐藏状态
    # 调用内部方法 _test_match_shadow_activations 测试模型的影子激活
    # 输入为生成的LSTM输入和隐藏状态
    # 设置期望输出结果的长度为1，并传入相应的量化配置字典
    # 设置跳过脚本化为True，因为准备好的LSTM无法脚本化



    @skipIfNoFBGEMM
    def test_sparsenn_compare_activations(self):
        for should_log_inputs in (True, False):
            sparse_nn = SparseNNModel().eval()
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            self._test_match_activations(
                sparse_nn, (idx, offsets, x),
                results_len=5,
                should_log_inputs=should_log_inputs)


    # 如果系统不支持FBGEMM加速，则跳过此测试
    # 定义一个测试函数，用于比较稀疏神经网络模型的激活
    # 遍历两种日志输入的选项：True和False
    # 实例化一个评估模式的稀疏神经网络模型
    # 定义索引和偏移量张量
    # 生成一个随机的输入张量
    # 调用内部方法 _test_match_activations 测试模型的激活
    # 输入为生成的索引、偏移量和输入张量
    # 设置期望输出结果的长度为5，并传入日志输入选项



    @skipIfNoFBGEMM
    def test_sparsenn_shadow(self):
        for should_log_inputs in (True, False):
            sparse_nn = SparseNNModel().eval()
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            self._test_match_shadow_activations(
                sparse_nn, (idx, offsets, x),
                results_len=3,
                should_log_inputs=should_log_inputs)


    # 如果系统不支持FBGEMM加速，则跳过此测试
    # 定义一个测试函数，用于比较稀疏神经网络模型的影子激活
    # 遍历两种日志输入的选项：True和False
    # 实例化一个评估模式的稀疏神经网络模型
    # 定义索引和偏移量张量
    # 生成一个随机的输入张量
    # 调用内部方法 _test_match_shadow_activations 测试模型的影子激活
    # 输入为生成的索引、偏移量和输入张量
    # 设置期望输出结果的长度为3，并传入日志输入选项
    # 定义一个测试方法，用于测试 MobileNetV2 模型的功能
    def test_mobilenet_v2(self):
        # 导入 torchvision 库，这是一个用于计算机视觉任务的流行库
        import torchvision
        # 创建一个 MobileNetV2 模型实例，不使用预训练权重，并设置为评估模式
        m = torchvision.models.quantization.mobilenet_v2(pretrained=False, quantize=False).eval()
        # 创建一个量化配置字典，使用默认的量化配置
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        # 调用自定义方法 _test_match_shadow_activations 进行测试，验证阴影激活是否匹配
        self._test_match_shadow_activations(
            m, (torch.randn(1, 3, 224, 224),),  # 传入 MobileNetV2 模型和随机输入张量
            qconfig_dict=qconfig_dict,  # 使用上面定义的量化配置字典
            should_log_inputs=False  # 不记录输入日志
        )
```
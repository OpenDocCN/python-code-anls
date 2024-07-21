# `.\pytorch\torch\ao\quantization\backend_config\onednn.py`

```
# mypy: allow-untyped-defs
# 导入PyTorch库及其模块
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.nn.functional as F
import torch.ao.nn.quantized.reference as nnqr
# 导入通用操作配置工具函数
from ._common_operator_config_utils import (
    _get_conv_configs,
    _get_linear_configs,
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_ln_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
)
# 导入后端配置相关类和类型
from .backend_config import (
    BackendPatternConfig,
    BackendConfig,
    DTypeConfig,
    ObservationType,
)
# 导入融合方法映射函数
from ..fuser_method_mappings import (
    _sequential_wrapper2,
)
# 导入运算符模块
import operator
# 导入PyTorch AO量化工具模块
from torch.ao.quantization.utils import MatchAllNode
# 导入itertools模块
import itertools

# ===================
# |  DTYPE CONFIGS  |
# ===================

# 定义针对ONEDNN加权操作的int8数据类型配置
onednn_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

# 定义针对ONEDNN的quint8数据类型配置
onednn_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

# 定义针对ONEDNN动态int8数据类型配置
onednn_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

# 定义仅针对权重的qint8数据类型配置
onednn_weight_only_qint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
)

# 定义仅针对输入输出的quint8数据类型配置
onednn_input_output_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.float,
    bias_dtype=torch.float,
)

# ===================
# |  FUSER METHODS  |
# ===================

def _fuse_linear_bn_leaky_relu(is_qat, linear, bn, leaky_relu):
    r"""Given the linear, bn and leaky_relu modules, fuses them and returns the fused module
    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
                or post training quantization fusion
        linear: Module instance of type Linear
        bn: BatchNorm1d instance that needs to be fused with the linear layer
        leaky_relu: LeakyReLU instance that needs to be fused with the linear layer
    Examples::
        >>> # xdoctest: +SKIP(failing)
        >>> m1 = nn.Linear(20, 10)
        >>> b1 = nn.BatchNorm1d(10)
        >>> lr = nn.LeakyReLU(0.01)
        >>> m2 = _fuse_linear_bn_leaky_relu(m1, b1, lr)
    """
    # 断言线性层、批归一化层和LeakyReLU层均处于相同的模式（训练或评估）
    assert linear.training == bn.training and bn.training == leaky_relu.training, \
        "Linear, BN and LeakyReLU all must be in the same mode (train or eval)."

    # 如果是量化训练，则抛出未实现错误
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {(linear, bn, leaky_relu)}")
    else:
        # 定义一个映射字典，将 nn.Linear 映射到 nni.LinearLeakyReLU
        map_to_fused_module_eval = {
            nn.Linear: nni.LinearLeakyReLU,
        }
        # 根据当前 linear 对象的类型获取对应的融合模块类，如果没有则返回 None
        fused_module = map_to_fused_module_eval.get(type(linear), None)
        # 如果找到了对应的融合模块类
        if fused_module is not None:
            # 融合 linear 和 bn 到一个新的融合线性层
            fused_linear = nn.utils.fusion.fuse_linear_bn_eval(linear, bn)
            # 使用融合模块类创建一个新的融合模块实例 fm，使用 leaky_relu 作为参数
            fm = fused_module(fused_linear, leaky_relu)
            # 返回融合后的模块实例
            return fm
        else:
            # 如果没有找到对应的融合模块类，则抛出未实现的错误
            raise NotImplementedError(f"Cannot fuse eval modules: {(linear, bn, leaky_relu)}")
# ======================
# |  CONFIGS FOR CONV  |
# ======================

# 设置观察类型为输出使用不同观察者作为输入
observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT

# 使用指定的卷积数据类型配置列表
conv_dtype_configs = [onednn_weighted_op_int8_dtype_config]

# 获取卷积配置
conv_configs = _get_conv_configs(conv_dtype_configs)

# (1) Conv2d + Add

# 定义函数 _fuse_conv_add_left，将卷积层和加法融合
def _fuse_conv_add_left(is_qat, add, conv, _):
    return nni.ConvAdd2d(conv, add)

# 获取左侧卷积加法模式的根节点
def _conv_add_root_node_getter_left(pattern):
    _, conv, _ = pattern
    return conv

# 获取左侧卷积加法模式的额外输入
def _conv_add_extra_inputs_getter_left(pattern):
    """获取额外输入模式，假设根节点的输入已复制到融合节点"""
    _, conv, extra_input = pattern
    return [extra_input]

# 定义函数 _fuse_conv_bn_add_left，将卷积、批归一化和加法融合
def _fuse_conv_bn_add_left(is_qat, add, bn_conv, _):
    bn, conv = bn_conv
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, add)}")
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAdd2d(fused_conv, add)

# 获取左侧卷积批归一化加法模式的根节点
def _conv_bn_add_root_node_getter_left(add_pattern):
    _, bn_conv, _ = add_pattern
    bn, conv = bn_conv
    return conv

# 获取左侧卷积批归一化加法模式的额外输入
def _conv_bn_add_extra_inputs_getter_left(add_pattern):
    """获取额外输入模式，假设根节点的输入已复制到融合节点"""
    _, bn_conv, extra_input = add_pattern
    bn, conv = bn_conv
    return [extra_input]

# 定义卷积加法左侧选项的组合
conv_add_left_optioins = itertools.product(
    [True, False],  # with_bn
    [torch.add, operator.add],  # add_op
)

# 遍历卷积加法左侧选项组合
for with_bn, add_op in conv_add_left_optioins:
    if with_bn:
        # 如果包含批归一化，则添加相应的模式配置
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((add_op, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(conv_dtype_configs)
                .set_fuser_method(_fuse_conv_bn_add_left)
                ._set_root_node_getter(_conv_bn_add_root_node_getter_left)
                ._set_extra_inputs_getter(_conv_bn_add_extra_inputs_getter_left)
                .set_fused_module(nni.ConvAdd2d))
    else:
        # 否则，添加基本的卷积加法模式配置
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((add_op, nn.Conv2d, MatchAllNode))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(conv_dtype_configs)
                .set_fuser_method(_fuse_conv_add_left)
                ._set_root_node_getter(_conv_add_root_node_getter_left)
                ._set_extra_inputs_getter(_conv_add_extra_inputs_getter_left)
                .set_fused_module(nni.ConvAdd2d))

# conv2d
#  \
#  bn   Y
#   \   /
#    add

# 定义函数 _fuse_conv_add_right，将卷积和加法融合
def _fuse_conv_add_right(is_qat, add, _, conv):
    return nni.ConvAdd2d(conv, add)

# 获取右侧卷积加法模式的根节点
def _conv_add_root_node_getter_right(pattern):
    add, _, conv = pattern
    return conv


注释：

    # 返回变量 conv 的值作为函数的返回结果
def _conv_add_extra_inputs_getter_right(pattern):
    """ 获取额外输入的模式，假设根节点的输入已复制到融合节点 """
    _, extra_input, conv = pattern
    return [extra_input]

def _fuse_conv_bn_add_right(is_qat, add, _, bn_conv):
    """ 将卷积层和批归一化层融合，并添加加法层 """
    bn, conv = bn_conv
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, add)}")
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAdd2d(fused_conv, add)

def _conv_bn_add_root_node_getter_right(pattern):
    """ 获取根节点的卷积层 """
    add, _, bn_conv = pattern
    bn, conv = bn_conv
    return conv

def _conv_bn_add_extra_inputs_getter_right(pattern):
    """ 获取额外输入的模式，假设根节点的输入已复制到融合节点 """
    _, extra_input, bn_conv = pattern
    bn, conv = bn_conv
    return [extra_input]

conv_add_optioins = itertools.product(
    [True, False],  # with_bn
    [torch.add, operator.add],  # add_op
)

for with_bn, add_op in conv_add_optioins:
    if with_bn:
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((add_op, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d)))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(conv_dtype_configs)
                .set_fuser_method(_fuse_conv_bn_add_right)
                ._set_root_node_getter(_conv_bn_add_root_node_getter_right)
                ._set_extra_inputs_getter(_conv_bn_add_extra_inputs_getter_right)
                .set_fused_module(nni.ConvAdd2d))
    else:
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((add_op, MatchAllNode, nn.Conv2d))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(conv_dtype_configs)
                .set_fuser_method(_fuse_conv_add_right)
                ._set_root_node_getter(_conv_add_root_node_getter_right)
                ._set_extra_inputs_getter(_conv_add_extra_inputs_getter_right)
                .set_fused_module(nni.ConvAdd2d))

conv_configs.append(
    BackendPatternConfig(nni.ConvAdd2d)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(conv_dtype_configs)
        .set_root_module(nn.Conv2d)
        .set_reference_quantized_module(nnqr.Conv2d))

# (2) Conv2d + Add + Relu

def _fuse_conv_add_relu_left(is_qat, relu, add_pattern):
    """ 将卷积层、加法层和ReLU层融合 """
    add, conv, _ = add_pattern
    return nni.ConvAddReLU2d(conv, add, relu)

def _conv_add_relu_root_node_getter_left(pattern):
    """ 获取根节点的卷积层 """
    relu, add_pattern = pattern
    _, conv, _ = add_pattern
    return conv

def _conv_add_relu_extra_inputs_getter_left(pattern):
    """ 获取额外输入的模式，假设根节点的输入已复制到融合节点 """
    # 这里缺少注释的部分应该补充完整
    are assumed to be copied over from root node to the fused node
    """
    relu, add_pattern = pattern
    # 解构 pattern 元组，将第一个元素赋值给 relu，将剩余的元素赋值给 add_pattern
    _, conv, extra_input = add_pattern
    # 解构 add_pattern 元组，将第二个元素赋值给 conv，将第三个元素赋值给 extra_input
    # 返回包含 extra_input 的列表作为结果
    return [extra_input]
# conv2d
#  \
#  bn   Y
#   \   /
#    add
#     \
#     relu

def _fuse_conv_bn_add_relu_left(is_qat, relu, add_pattern):
    add, bn_conv, _ = add_pattern  # 解构 add_pattern 元组，获取 add、bn_conv 和占位符 _
    bn, conv = bn_conv  # 解构 bn_conv 元组，获取 bn 和 conv
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, add, relu)}")  # 如果是量化训练，则抛出未实现错误
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)  # 调用 PyTorch 提供的函数，将卷积层和批归一化融合
        return nni.ConvAddReLU2d(fused_conv, add, relu)  # 返回融合后的 ConvAddReLU2d 模块实例化对象

def _conv_bn_add_relu_root_node_getter_left(pattern):
    relu, add_pattern = pattern  # 解构 pattern 元组，获取 relu 和 add_pattern
    _, bn_conv, _ = add_pattern  # 解构 add_pattern 元组，获取 bn_conv 和占位符 _
    bn, conv = bn_conv  # 解构 bn_conv 元组，获取 bn 和 conv
    return conv  # 返回卷积层对象 conv

def _conv_bn_add_relu_extra_inputs_getter_left(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    relu, add_pattern = pattern  # 解构 pattern 元组，获取 relu 和 add_pattern
    _, bn_conv, extra_input = add_pattern  # 解构 add_pattern 元组，获取 bn_conv 和 extra_input
    bn, conv = bn_conv  # 解构 bn_conv 元组，获取 bn 和 conv
    return [extra_input]  # 返回额外输入的列表，这些输入假设从根节点复制到融合节点

conv_add_relu_left_options = itertools.product(
    [True, False],  # with_bn
    [torch.add, operator.add],  # add_op
)

for with_bn, add_op in conv_add_relu_left_options:
    if with_bn:
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((nn.ReLU, (add_op, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode)))  # 设置复杂模式格式，包含 nn.ReLU、add_op、(nn.BatchNorm2d, nn.Conv2d) 和 MatchAllNode
                .set_observation_type(observation_type)  # 设置观察类型
                .set_dtype_configs(conv_dtype_configs)  # 设置数据类型配置
                .set_fuser_method(_fuse_conv_bn_add_relu_left)  # 设置融合方法为 _fuse_conv_bn_add_relu_left 函数
                ._set_root_node_getter(_conv_bn_add_relu_root_node_getter_left)  # 设置根节点获取方法为 _conv_bn_add_relu_root_node_getter_left 函数
                ._set_extra_inputs_getter(_conv_bn_add_relu_extra_inputs_getter_left)  # 设置额外输入获取方法为 _conv_bn_add_relu_extra_inputs_getter_left 函数
                .set_fused_module(nni.ConvAddReLU2d))  # 设置融合后的模块为 nni.ConvAddReLU2d 类型
    else:
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((nn.ReLU, (add_op, nn.Conv2d, MatchAllNode)))  # 设置复杂模式格式，包含 nn.ReLU、add_op、nn.Conv2d 和 MatchAllNode
                .set_observation_type(observation_type)  # 设置观察类型
                .set_dtype_configs(conv_dtype_configs)  # 设置数据类型配置
                .set_fuser_method(_fuse_conv_add_relu_left)  # 设置融合方法为 _fuse_conv_add_relu_left 函数
                ._set_root_node_getter(_conv_add_relu_root_node_getter_left)  # 设置根节点获取方法为 _conv_add_relu_root_node_getter_left 函数
                ._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_left)  # 设置额外输入获取方法为 _conv_add_relu_extra_inputs_getter_left 函数
                .set_fused_module(nni.ConvAddReLU2d))  # 设置融合后的模块为 nni.ConvAddReLU2d 类型

#  Y   conv2d
#   \   /
#    add
#     \
#     relu

def _fuse_conv_add_relu_right(is_qat, relu, add_pattern):
    add, _, conv = add_pattern  # 解构 add_pattern 元组，获取 add 和 conv
    return nni.ConvAddReLU2d(conv, add, relu)  # 返回 ConvAddReLU2d 模块实例化对象，将 conv、add 和 relu 作为参数传递

def _conv_add_relu_root_node_getter_right(pattern):
    relu, add_pattern = pattern  # 解构 pattern 元组，获取 relu 和 add_pattern
    _, _, conv = add_pattern  # 解构 add_pattern 元组，获取占位符和 conv
    return conv  # 返回卷积层对象 conv

def _conv_add_relu_extra_inputs_getter_right(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    relu, add_pattern = pattern  # 解构 pattern 元组，获取 relu 和 add_pattern
    _, extra_input, conv = add_pattern  # 解构 add_pattern 元组，获取占位符、extra_input 和 conv
    return [extra_input]  # 返回额外输入的列表，这些输入假设从根节点复制到融合节点

#      conv2d
#        /
#  Y    bn
#   \   /
#    add
#     \
#     relu

def _fuse_conv_bn_add_relu_right(is_qat, relu, add_pattern):
    # 解包 add_pattern 元组，分别赋值给 add, _ 和 bn_conv
    add, _, bn_conv = add_pattern
    # 解包 bn_conv 元组，分别赋值给 bn 和 conv
    bn, conv = bn_conv
    # 如果是量化训练模式 is_qat，则抛出未实现错误，提示无法融合训练模块
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, add, relu)}")
    else:
        # 在非量化训练模式下，调用 nn.utils.fusion.fuse_conv_bn_eval 函数融合卷积和批量归一化
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        # 返回融合后的 ConvAddReLU2d 实例，传入融合后的 conv、add 和 relu 参数
        return nni.ConvAddReLU2d(fused_conv, add, relu)
# 定义函数，从模式中获取根节点的卷积层
def _conv_bn_add_relu_root_node_getter_right(pattern):
    relu, add_pattern = pattern
    _, _, bn_conv = add_pattern
    bn, conv = bn_conv
    return conv

# 定义函数，从模式中获取额外输入的模式，假设根节点的输入被复制到融合节点
def _conv_bn_add_relu_extra_inputs_getter_right(pattern):
    """获取额外输入的模式，假设根节点的输入被复制到融合节点"""
    relu, add_pattern = pattern
    _, extra_input, bn_conv = add_pattern
    bn, conv = bn_conv
    return [extra_input]

# 生成卷积-加法-ReLU操作的所有可能组合
conv_add_relu_options = itertools.product(
    [True, False],  # 是否包含批归一化
    [torch.add, operator.add],  # 加法操作
)

# 遍历卷积-加法-ReLU操作的所有可能组合
for with_bn, add_op in conv_add_relu_options:
    if with_bn:
        # 如果包含批归一化，配置BackendPatternConfig并添加到conv_configs中
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((nn.ReLU, (add_op, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d))))  # 设置复杂模式格式
                .set_observation_type(observation_type)  # 设置观察类型
                .set_dtype_configs(conv_dtype_configs)  # 设置数据类型配置
                .set_fuser_method(_fuse_conv_bn_add_relu_right)  # 设置融合方法
                ._set_root_node_getter(_conv_bn_add_relu_root_node_getter_right)  # 设置根节点获取器
                ._set_extra_inputs_getter(_conv_bn_add_relu_extra_inputs_getter_right)  # 设置额外输入获取器
                .set_fused_module(nni.ConvAddReLU2d))  # 设置融合后的模块
    else:
        # 如果不包含批归一化，配置BackendPatternConfig并添加到conv_configs中
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((nn.ReLU, (add_op, MatchAllNode, nn.Conv2d)))  # 设置复杂模式格式
                .set_observation_type(observation_type)  # 设置观察类型
                .set_dtype_configs(conv_dtype_configs)  # 设置数据类型配置
                .set_fuser_method(_fuse_conv_add_relu_right)  # 设置融合方法
                ._set_root_node_getter(_conv_add_relu_root_node_getter_right)  # 设置根节点获取器
                ._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_right)  # 设置额外输入获取器
                .set_fused_module(nni.ConvAddReLU2d))  # 设置融合后的模块

# 将针对ConvAddReLU2d的BackendPatternConfig添加到conv_configs中
conv_configs.append(
    BackendPatternConfig(nni.ConvAddReLU2d)
        .set_observation_type(observation_type)  # 设置观察类型
        .set_dtype_configs(conv_dtype_configs)  # 设置数据类型配置
        .set_root_module(nn.Conv2d)  # 设置根模块
        .set_reference_quantized_module(nnqr.Conv2d))  # 设置参考量化模块

# ========================
# |  CONFIGS FOR LINEAR  |
# ========================

# 线性层的数据类型配置
linear_dtype_configs = [
    onednn_weighted_op_int8_dtype_config,
    onednn_dynamic_int8_dtype_config,
]

# 获取线性层的配置
linear_configs = _get_linear_configs(linear_dtype_configs)

# 定义函数，添加融合配置到configs列表中
def _add_eltwise_fusion_configs(configs, root_module, root_op, post_module, post_op,
                                dtype_configs, fuser_method, fused_module, observation_type,
                                ref_quant_module):
    # 添加基础模块 + 操作模块的融合配置到configs列表中
    configs.append(
        BackendPatternConfig((root_module, post_module))
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_fuser_method(fuser_method)  # 设置融合方法
            .set_fused_module(fused_module))  # 设置融合后的模块
    # 添加基础模块 + 功能性后操作的融合配置到configs列表中
    # 将元组 (root_module, post_op) 封装成 BackendPatternConfig 对象，并添加到 configs 列表中
    configs.append(
        BackendPatternConfig((root_module, post_op))
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_fuser_method(fuser_method)    # 设置融合方法
            .set_fused_module(fused_module))  # 设置融合后的模块

    # 向 configs 列表中添加一个配置，配置为融合后的模块的 BackendPatternConfig
    # 设置观测类型 observation_type
    # 设置数据类型配置 dtype_configs
    # 设置根模块 root_module
    # 设置参考量化模块 ref_quant_module
    configs.append(
        BackendPatternConfig(fused_module)
            .set_observation_type(observation_type)  # 设置观测类型
            .set_dtype_configs(dtype_configs)        # 设置数据类型配置
            .set_root_module(root_module)            # 设置根模块
            .set_reference_quantized_module(ref_quant_module))  # 设置参考量化模块

    # 向 configs 列表中添加一个配置，配置为 (root_op, post_module) 的 BackendPatternConfig
    # 设置观测类型 observation_type
    # 设置数据类型配置 dtype_configs
    configs.append(
        BackendPatternConfig((root_op, post_module))
            .set_observation_type(observation_type)  # 设置观测类型
            .set_dtype_configs(dtype_configs))       # 设置数据类型配置

    # 向 configs 列表中添加一个配置，配置为 (root_op, post_op) 的 BackendPatternConfig
    # 设置观测类型 observation_type
    # 设置数据类型配置 dtype_configs
    configs.append(
        BackendPatternConfig((root_op, post_op))
            .set_observation_type(observation_type)  # 设置观测类型
            .set_dtype_configs(dtype_configs))       # 设置数据类型配置
# Configs for linear + leaky_relu fusion
_add_eltwise_fusion_configs(
    linear_configs,            # 将配置添加到 linear_configs 列表中
    nn.Linear,                 # 使用 nn.Linear 作为线性层
    F.linear,                  # 使用 F.linear 函数
    nn.LeakyReLU,              # 使用 nn.LeakyReLU 作为激活函数
    F.leaky_relu,              # 使用 F.leaky_relu 函数
    linear_dtype_configs,      # 线性层的数据类型配置
    _sequential_wrapper2(nni.LinearLeakyReLU),  # 使用 nni.LinearLeakyReLU 的顺序包装器
    nni.LinearLeakyReLU,       # 使用 nni.LinearLeakyReLU 模块
    observation_type,          # 观察类型参数
    nnqr.Linear                # 使用 nnqr.Linear
)

# Configs for linear module + batchnorm + leaky_relu
linear_configs.append(
    BackendPatternConfig((nn.Linear, nn.BatchNorm1d, nn.LeakyReLU))  # 配置线性层、批归一化和 LeakyReLU 的模式
        .set_dtype_configs(linear_dtype_configs)  # 设置数据类型配置
        .set_fuser_method(_fuse_linear_bn_leaky_relu)  # 设置融合方法为 _fuse_linear_bn_leaky_relu 函数
        .set_fused_module(nni.LinearLeakyReLU)  # 设置融合后的模块为 nni.LinearLeakyReLU
)

# Configs for linear + tanh fusion
_add_eltwise_fusion_configs(
    linear_configs,            # 将配置添加到 linear_configs 列表中
    nn.Linear,                 # 使用 nn.Linear 作为线性层
    F.linear,                  # 使用 F.linear 函数
    nn.Tanh,                   # 使用 nn.Tanh 作为激活函数
    torch.tanh,                # 使用 torch.tanh 函数
    linear_dtype_configs,      # 线性层的数据类型配置
    _sequential_wrapper2(nni.LinearTanh),  # 使用 nni.LinearTanh 的顺序包装器
    nni.LinearTanh,            # 使用 nni.LinearTanh 模块
    observation_type,          # 观察类型参数
    nnqr.Linear                # 使用 nnqr.Linear
)

# ===========================
# |  CONFIGS FOR OTHER OPS  |
# ===========================

binary_op_dtype_configs = [onednn_op_quint8_dtype_config]
default_op_dtype_configs = [onednn_op_quint8_dtype_config]
fixed_qparams_op_dtype_configs = [onednn_op_quint8_dtype_config]
share_qparams_op_dtype_configs = [onednn_op_quint8_dtype_config]
rnn_op_dtype_configs = [onednn_dynamic_int8_dtype_config]
embedding_op_dtype_configs = [onednn_weight_only_qint8_dtype_config]
layer_norm_op_dtype_configs = [onednn_input_output_only_quint8_dtype_config]

# =====================
# |  BACKEND CONFIGS  |
# =====================

def get_onednn_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native ONEDNN backend.
    """
    return BackendConfig("onednn") \  # 创建名为 "onednn" 的 BackendConfig 对象
        .set_backend_pattern_configs(conv_configs)  # 设置卷积模式配置
        .set_backend_pattern_configs(linear_configs)  # 设置线性模式配置
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))  # 设置二进制操作模式配置
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs))  # 设置拼接操作模式配置
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs))  # 设置默认操作模式配置
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs))  # 设置固定量化参数操作模式配置
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs))  # 设置共享量化参数操作模式配置
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs))  # 设置批归一化操作模式配置
        .set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs))  # 设置层归一化操作模式配置
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))  # 设置循环神经网络操作模式配置
        .set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))  # 设置嵌入操作模式配置

__all__ = [
    "get_onednn_backend_config",  # 导出 get_onednn_backend_config 函数
]
```
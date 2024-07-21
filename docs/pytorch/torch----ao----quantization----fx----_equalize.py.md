# `.\pytorch\torch\ao\quantization\fx\_equalize.py`

```
# mypy: allow-untyped-defs
# 导入警告模块，用于处理警告信息
import warnings

# 导入命名元组类和类型相关的模块
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr

# 导入本地自定义的模块
from ..observer import _with_args, ObserverBase, PerChannelMinMaxObserver
from ..utils import _parent_name, check_min_max_valid

# 导入本地工具函数模块
from .utils import (
    get_new_attr_name_with_prefix,
    maybe_get_next_module,
    node_arg_is_weight,
)
# 导入操作符模块
import operator

# 自定义模块支持列表初始化为空列表
CUSTOM_MODULE_SUPP_LIST: List[Any] = []

def reshape_scale(scale: torch.Tensor, axis: int, input: torch.Tensor) -> torch.Tensor:
    """Reshapes the scale so that we can multiply it to the input by the given axis.
    """
    # 创建一个新的形状列表，使得比例张量可以按给定的轴乘以输入张量
    new_shape = [1] * input.ndim
    new_shape[axis] = input.size(axis)
    return scale.view(new_shape)

# 定义从每个张量量化方案到每通道量化方案的映射字典
qsheme_mapping_per_tensor_to_per_channel = {
    torch.per_tensor_affine: torch.per_channel_affine,
    torch.per_tensor_symmetric: torch.per_channel_symmetric,
}

class _InputEqualizationObserver(nn.Module):
    r"""Observer for tracking the running min/max values of input columns, and
    computing the quantization parameters for the overall min/max input values.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    The running minimum/maximum :math:`x_\text{min/max}` are computed in the
    same way as :class:`~torch.ao.quantization.observer.PerChannelMinMaxObserver`,
    with the difference that the running min/max values are stored per column.
    This observer is intended to be used along with a WeightEqualizationObserver
    to calculate the equalization scale.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 quant_min=None, quant_max=None, factory_kwargs=None) -> None:
        super().__init__()

        # 检查量化方案是否为每张量，并设置数据类型和量化方案
        if qscheme not in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            raise TypeError("Input qscheme must be per-tensor")

        self.dtype = dtype
        self.qscheme = qscheme

        # 根据输入的量化方案选择对应的每通道最小/最大观察器
        per_channel_qscheme = qsheme_mapping_per_tensor_to_per_channel[qscheme]
        self.input_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=dtype,
                                                  qscheme=per_channel_qscheme,
                                                  quant_min=quant_min,
                                                  quant_max=quant_max,
                                                  factory_kwargs=factory_kwargs)

        # 初始化均衡化比例和形状
        self.equalization_scale = torch.tensor(1)
        self.equalization_shape: List[int] = []
    # 定义一个方法，用于向前传播输入数据
    def forward(self, x_orig):
        # 检查输入数据的维度是否在2到5之间，否则抛出数值错误异常
        if not (x_orig.ndim >= 2 and x_orig.ndim <= 5):
            raise ValueError("InputEqualizationObserver only supports Linear and Conv layers")
        
        # 计算用于后续等化比例重塑的形状（对于Conv层是必需的）
        self.equalization_shape = [1] * x_orig.ndim
        self.equalization_shape[1] = x_orig.size(1)
        
        # 调用input_obs方法处理输入数据并返回结果
        return self.input_obs(x_orig)

    # 获取输入观察器中的最小和最大值
    def get_input_minmax(self):
        return (self.input_obs.min_val, self.input_obs.max_val)

    # 设置等化比例
    def set_equalization_scale(self, equalization_scale):
        # 将等化比例沿axis=1重塑，以便与输入沿axis=1相乘
        if equalization_scale.nelement() == 1 and equalization_scale == torch.tensor(1):
            return
        self.equalization_scale = torch.reshape(equalization_scale, self.equalization_shape)

    # 计算经过缩放后的最小和最大输入
    def calculate_scaled_minmax(self):
        r""" 返回经过缩放的最小/最大输入 """
        if self.equalization_scale.nelement() == 1 and self.equalization_scale == torch.tensor(1):
            # 如果等化比例为1且只有一个元素，则发出警告并返回None
            warnings.warn(
                "Must call calculate_equalization_scale before calling calculate_scaled_minmax. " +
                "Will not scale the next quantization observer."
            )
            return None, None
        
        # 计算经过缩放后的最小和最大输入的量化参数
        # 将输入按位等化比例缩放，该比例位于相同列索引处
        (min_inputs, max_inputs) = self.get_input_minmax()
        equalization_scale_reshaped = reshape_scale(self.equalization_scale, 0, min_inputs)
        min_input_scaled = torch.min(torch.mul(min_inputs, equalization_scale_reshaped))
        max_input_scaled = torch.max(torch.mul(max_inputs, equalization_scale_reshaped))
        
        return min_input_scaled, max_input_scaled

    # 使用_with_args方法设置带有参数的classmethod
    with_args = classmethod(_with_args)
class _WeightEqualizationObserver(nn.Module):
    r"""Observer for tracking the running min/max values of weight columns and
    rows, and computing the quantization parameters for the weight rows.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    This observer is made up of 1 PerChannelMinMaxObserver `weight_col_obs` used
    to record the running minimum and maximum of columns of incoming weight
    tensors. This observer is intended to be used along with an
    InputEqualizationObserver to calculate the equalization scale.

    The running minimum/maximum :math:`w_\text{min/max}` are computed in the
    same way as :class:`~torch.ao.quantization.observer.PerChannelMinMaxObserver`.
    """

    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, quant_min=None,
                 quant_max=None, factory_kwargs=None) -> None:
        super().__init__()

        self.dtype = dtype  # 设置量化后的数据类型
        self.qscheme = qscheme  # 设置量化方案
        self.ch_axis = 1  # 通道轴，默认为1

        per_channel_qscheme = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            per_channel_qscheme = qsheme_mapping_per_tensor_to_per_channel[qscheme]
            # 如果是按张量或对称张量，将其映射到按通道量化方案
        self.weight_col_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=dtype,
                                                       qscheme=per_channel_qscheme,
                                                       quant_min=quant_min,
                                                       quant_max=quant_max,
                                                       factory_kwargs=factory_kwargs)
        # 创建按通道记录权重列最小值和最大值的观察器

        self.equalization_scale = torch.tensor(1)  # 初始化均衡化比例为1

    def forward(self, w_orig):
        if not (w_orig.ndim >= 2 and w_orig.ndim <= 5):
            raise ValueError("InputEqualizationObserver only supports Linear and Conv layers")
        # 检查输入张量维度是否支持线性和卷积层

        return self.weight_col_obs(w_orig)  # 返回权重列观察器处理后的结果

    def get_weight_col_minmax(self):
        return (self.weight_col_obs.min_val, self.weight_col_obs.max_val)
        # 返回记录的权重列的最小值和最大值

    def set_equalization_scale(self, equalization_scale):
        self.equalization_scale = equalization_scale
        # 设置均衡化比例

    with_args = classmethod(_with_args)


def calculate_equalization_scale(input_obs: _InputEqualizationObserver,
                                 weight_obs: _WeightEqualizationObserver) -> torch.Tensor:
    r""" Calculates the equalization scale and sets the equalization_scale value
    in the observers.

    Args:
        input_obs: Observer that tracks the ranges for the input columns
        weight_obs: Observer that tracks the ranges for the weight columns
    """

    (min_inputs, max_inputs) = input_obs.get_input_minmax()
    (min_weights, max_weights) = weight_obs.get_weight_col_minmax()
    # 获取输入观察器记录的输入列的最小值和最大值，以及权重观察器记录的权重列的最小值和最大值
    # 检查最小和最大输入值以及权重值的有效性，如果任何一个无效，则发出警告并返回默认的等化比例值为1的张量
    if not (check_min_max_valid(min_inputs, max_inputs) and check_min_max_valid(min_weights, max_weights)):
        warnings.warn(
            "Must run observer before calling calculate_equalization_scale. " +
            "Returning default equalization scale torch.tensor(1)."
        )
        return torch.tensor(1)

    # 检查输入值和权重值的列维度是否相同，如果不同则抛出值错误异常
    if not (min_inputs.shape == min_weights.shape):
        raise ValueError(
            "Input and Weight must have the same column dimension. " +
            f"Found {min_inputs.shape} and {min_weights.shape} shapes instead."
        )

    # 计算等化比例尺度，用于调整输入值和权重值的范围
    equalization_scale = torch.sqrt((max_weights - min_weights) / (max_inputs - min_inputs))
    # 将所有的 'inf'、'nan' 和 0 替换为 1，以防止出现错误
    equalization_scale[equalization_scale == 0.] = 1
    equalization_scale = torch.nan_to_num(equalization_scale, nan=1, posinf=1, neginf=1)
    # 返回计算得到的等化比例尺度
    return equalization_scale
class EqualizationQConfig(namedtuple('EqualizationQConfig', ['input_activation', 'weight'])):
    """
    Describes how to quantize a layer or a part of the network specifically for
    input-weight equalization by providing settings (observer classes) for
    inputs, outputs, and weights.

    Note that EqualizationQConfig needs to contain observer **classes** (like
    MinMaxObserver) or a callable that returns instances on invocation, not the
    concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of
    the layers.

    Observer classes have usually reasonable default arguments, but they can be
    overwritten with `with_args` method (that behaves like functools.partial):

    my_qconfig = EqualizationQConfig(input_activation=_InputEqualizationObserver.with_args(dtype=torch.qint8),
                                    weight=_WeightEqualizationObserver.with_args(dtype=torch.qint8))
    """
    def __new__(cls, input_activation=torch.nn.Identity, weight=torch.nn.Identity):
        # 检查输入的观察器是否为实例，应该传入观察器类而不是实例
        if isinstance(input_activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError("EqualizationQConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        # 调用父类的构造方法创建新的实例
        self = super().__new__(cls, input_activation, weight)
        return self


input_equalization_observer = _InputEqualizationObserver.with_args(
    dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_equalization_observer = _WeightEqualizationObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
default_equalization_qconfig = EqualizationQConfig(input_activation=input_equalization_observer,
                                                   weight=weight_equalization_observer)


def fused_module_supports_equalization(module) -> bool:
    """ Checks if the fused node supports equalization. """
    # 检查融合模块是否支持均衡化
    return type(module) in [nni.LinearReLU, nni.ConvReLU1d, nni.ConvReLU2d, nni.ConvReLU3d]

def nn_module_supports_equalization(module) -> bool:
    """ Checks if the torch.nn node supports equalization. """
    # 检查torch.nn模块是否支持均衡化
    return type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]

def custom_module_supports_equalization(module) -> bool:
    """ Checks if the custom node supports equalization. """
    # 检查自定义模块是否支持均衡化
    return type(module) in CUSTOM_MODULE_SUPP_LIST


def node_supports_equalization(node: Node, modules) -> bool:
    """ Checks if the current node supports equalization
    Currently we only support nn.Linear/F.Linear and nn.Conv/F.conv layers
    """
    # 检查当前节点是否支持均衡化，目前只支持 nn.Linear/F.Linear 和 nn.Conv/F.conv 层
    if node.op == 'call_module':
        return nn_module_supports_equalization(modules[str(node.target)]) or \
            fused_module_supports_equalization(modules[str(node.target)]) or \
            custom_module_supports_equalization(modules[str(node.target)])
    # 如果节点操作为 'call_function'，则检查节点的目标函数是否为线性函数或卷积函数之一
    elif node.op == 'call_function':
        # 返回目标函数是否在给定的线性函数或卷积函数列表中
        return node.target in [F.linear, F.conv1d, F.conv2d, F.conv3d]
    # 如果节点操作不是 'call_function'，返回 False
    return False
def is_equalization_observer(observer: nn.Module) -> bool:
    # 检查给定的观察器是否为输入均衡观察器或权重均衡观察器之一
    return (isinstance(observer, (_InputEqualizationObserver, _WeightEqualizationObserver)))


###############################################################################
# Functions for equalization during convert                                   #
###############################################################################

def get_op_node_and_weight_eq_obs(
    input_eq_obs_node: Node,
    model: GraphModule,
    modules: Dict[str, nn.Module]
) -> Tuple[Optional[Node], Optional[_WeightEqualizationObserver]]:
    """ Gets the following weight equalization observer. There should always
    exist a weight equalization observer after an input equalization observer.

    Returns the operation node that follows the input equalization observer node
    and the weight equalization observer
    """

    # 查找紧随输入均衡观察器之后的操作节点
    op_node = None
    for user in input_eq_obs_node.users.keys():
        if node_supports_equalization(user, modules):
            op_node = user
            break

    assert op_node is not None
    if op_node.op == 'call_module':
        # 如果操作节点是 nn.Linear 层，则必须具有 WeightEqualizationObserver 配置
        maybe_equalization_node_name_to_config = _get_observed_graph_module_attr(model, "equalization_node_name_to_qconfig")
        assert maybe_equalization_node_name_to_config is not None
        equalization_node_name_to_qconfig: Dict[str, Any] = maybe_equalization_node_name_to_config  # type: ignore[assignment]
        assert equalization_node_name_to_qconfig.get(op_node.name, None) is not None
        weight_eq_obs = equalization_node_name_to_qconfig.get(op_node.name, None).weight()

        assert isinstance(weight_eq_obs, _WeightEqualizationObserver)
        return op_node, weight_eq_obs

    elif op_node.op == 'call_function':
        # 如果操作节点是函数调用，则尝试获取其关联的权重均衡观察器节点
        weight_node = maybe_get_weight_eq_obs_node(op_node, modules)
        if weight_node is not None:
            weight_eq_obs = modules[str(weight_node.target)]
            assert isinstance(weight_eq_obs, _WeightEqualizationObserver)
            return op_node, weight_eq_obs

    return None, None

def maybe_get_weight_eq_obs_node(op_node: Node, modules: Dict[str, nn.Module]) -> Optional[Node]:
    """ Gets the weight equalization observer node if it exists.
    """
    assert op_node.op == 'call_function'
    for node_arg in op_node.args:
        if node_arg_is_weight(op_node, node_arg):
            assert (isinstance(node_arg, Node) and node_arg.op == 'call_module' and
                   isinstance(modules[str(node_arg.target)], _WeightEqualizationObserver))
            return node_arg
    return None

def maybe_get_next_input_eq_obs(node: Node, modules: Dict[str, nn.Module]) -> Optional[_InputEqualizationObserver]:
    """ Gets the following input equalization observer if it exists.
    """
    """
    For example, in the case of connecting linear layers:
        x -> inp_obs1 -> eq_obs1 -> linear1 -> out_obs1 -> eq_obs2 -> linear2 -> out_obs2
    If the node being passed in is the linear1 node, then we want to return eq_obs2,
    the following equalization observer for linear2.

    However, if there are no connecting layers:
        x -> inp_obs1 -> eq_obs1 -> linear1 -> out_obs1 -> add
    Then we want to return None.

    In the case of an unfused linear-relu layer with a connecting linear layer:
        linear1 -> relu -> out_obs1 -> eq_obs2 -> linear2 -> out_obs2
    Since it is unfused, we want to skip over the relu layer and return eq_obs2,
    the following equalization observer for linear2.
    """

    # Ensure that the node supports equalization, based on its type and context
    assert node_supports_equalization(node, modules)

    # Locate the following nn.ReLU or F.relu node if it exists
    maybe_relu_node = maybe_get_next_module(node, modules, nn.ReLU)
    if maybe_relu_node is None:
        maybe_relu_node = maybe_get_next_module(node, modules, target_functional_type=F.relu)

    # Locate the following output observer if it exists.
    # We will skip the relu node if it exists.
    maybe_obs_node = (
        maybe_get_next_module(node, modules, ObserverBase)
        if maybe_relu_node is None
        else maybe_get_next_module(maybe_relu_node, modules, ObserverBase)
    )

    # If no output observer node is found, return None
    if maybe_obs_node is None:
        return None

    # Try to find the _InputEqualizationObserver node following the output observer node
    maybe_eq_obs_node = maybe_get_next_module(maybe_obs_node, modules, _InputEqualizationObserver)

    # If no _InputEqualizationObserver node is found, return None
    if maybe_eq_obs_node is None:
        return None

    # Retrieve the module associated with the _InputEqualizationObserver node
    maybe_eq_obs = modules[str(maybe_eq_obs_node)]

    # Ensure that the retrieved module is indeed an instance of _InputEqualizationObserver
    assert isinstance(maybe_eq_obs, _InputEqualizationObserver)

    # Return the _InputEqualizationObserver instance
    return maybe_eq_obs
def maybe_get_next_equalization_scale(node: Node, modules: Dict[str, nn.Module]) -> Optional[torch.Tensor]:
    """ 如果下一个节点是输入均衡观察器，则返回其均衡比例，否则返回1
    
    当存在两个相连的线性层时使用：
        linear1 -> LinearOutObs -> InputEqObs -> linear2
    在这种情况下，给定的节点是linear1，我们要找到InputEqObs。
    """
    # 获取可能的下一个输入均衡观察器节点
    next_inp_eq_obs = maybe_get_next_input_eq_obs(node, modules)
    if next_inp_eq_obs:
        # 如果均衡比例只有一个元素且值为1，则返回None
        if next_inp_eq_obs.equalization_scale.nelement() == 1 and \
           next_inp_eq_obs.equalization_scale == torch.tensor(1):
            return None
        # 否则返回均衡比例
        return next_inp_eq_obs.equalization_scale
    return None

def scale_input_observer(node: Node, modules: Dict[str, nn.Module]) -> None:
    """ 通过更新输入均衡观察器计算得到的缩放后的最小/最大值，来缩放后续输入量化观察器的最小/最大值 """
    # 获取当前节点目标对应的输入均衡观察器
    input_eq_obs = modules[str(node.target)]
    assert isinstance(input_eq_obs, _InputEqualizationObserver)

    # 确定输入量化观察器节点并验证其类型
    input_quant_obs_node = node.args[0]
    assert isinstance(input_quant_obs_node, Node)

    # 获取输入量化观察器模块
    input_quant_obs = modules[str(input_quant_obs_node.target)]
    if not isinstance(input_quant_obs, ObserverBase):
        return

    # 计算缩放后的最小/最大值并更新输入量化观察器
    min_input_scaled, max_input_scaled = input_eq_obs.calculate_scaled_minmax()
    if min_input_scaled is None and max_input_scaled is None:
        return
    input_quant_obs.min_val = min_input_scaled
    input_quant_obs.max_val = max_input_scaled

def scale_weight_node(
    node: Node,
    modules: Dict[str, nn.Module],
    equalization_scale: torch.Tensor,
    next_equalization_scale: Optional[torch.Tensor],
) -> None:
    """ 通过将权重乘以均衡比例的倒数来缩放输入-权重均衡 """
    if equalization_scale is None:
        return

    # 确定操作模块是否支持均衡化
    if fused_module_supports_equalization(modules[str(node.target)]):
        op_module = modules[str(node.target)][0]    # type: ignore[index]
    else:
        op_module = modules[str(node.target)]
    assert nn_module_supports_equalization(op_module) or custom_module_supports_equalization(op_module)

    # 获取操作模块的权重张量并验证其类型
    weight = op_module.weight
    assert isinstance(weight, torch.Tensor)

    # 通过均衡比例的倒数来缩放权重
    # 重新塑造均衡比例以便沿轴1与权重相乘
    # 将等化尺度重塑为与权重的相同形状
    equalization_scale_reshaped = reshape_scale(equalization_scale, 1, weight)
    
    # 计算缩放后的权重，通过将权重与等化尺度的倒数逐元素相乘得到
    scaled_weight = torch.mul(weight, torch.reciprocal(equalization_scale_reshaped))
    
    # 如果下一个等化尺度为空，则将操作模块的权重设置为缩放后的权重，并返回
    if next_equalization_scale is None:
        op_module.weight = nn.Parameter(scaled_weight)
        return
    
    # 将权重按行乘以下一个等化尺度
    # 重塑下一个等化尺度，使其与权重的第一个维度（行数）相匹配，以便按行乘法
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, weight)
    scaled_weight = torch.mul(scaled_weight, next_equalization_scale_reshaped)
    
    # 将操作模块的权重更新为缩放后的权重
    op_module.weight = nn.Parameter(scaled_weight)
    
    # 如果操作模块没有偏置，则直接返回
    bias = op_module.bias
    if bias is None:
        return
    
    # 确保偏置是一个 Tensor 类型
    assert isinstance(bias, torch.Tensor)
    
    # 将下一个等化尺度重塑为与偏置相同的形状，以便进行逐元素乘法
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, bias)
    scaled_bias = torch.mul(bias, next_equalization_scale_reshaped)
    
    # 更新操作模块的偏置为缩放后的偏置
    op_module.bias = nn.Parameter(scaled_bias)
def scale_weight_functional(
    op_node: Node,
    model: GraphModule,
    modules: Dict[str, nn.Module],
    equalization_scale: torch.Tensor,
    next_equalization_scale: Optional[torch.Tensor],
) -> None:
    """ Scales the weight value for functional layers
    """

    # 如果 equalization_scale 为 None，则直接返回，不进行后续处理
    if equalization_scale is None:
        return

    # 从 op_node 开始，路径为:
    #   get_attr(weight) -> weight_quant_obs -> weight_eq_obs -> op_node
    # 因此我们需要追溯到 equalization observer 节点，然后到 quantization observer 节点，
    # 最后到包含权重值的节点。

    # 获取 equalization observer 节点
    weight_eq_obs_node = maybe_get_weight_eq_obs_node(op_node, modules)
    if weight_eq_obs_node is None:
        return

    # 获取 quantization observer 节点
    weight_quant_obs_node = weight_eq_obs_node.args[0]
    if weight_quant_obs_node is None:
        return
    assert (isinstance(weight_quant_obs_node, Node) and
           isinstance(modules[str(weight_quant_obs_node.target)], ObserverBase))

    # 获取 get_attr(weight) 节点
    weight_node = weight_quant_obs_node.args[0]
    if weight_node is None:
        return
    assert isinstance(weight_node, Node) and weight_node.op == 'get_attr'

    # 从 weight_node 的 target 中获取父模块名称和权重名称
    weight_parent_name, weight_name = _parent_name(weight_node.target)
    weight = getattr(modules[weight_parent_name], weight_name)

    # 对输入权重进行等化处理
    # 如果下一层需要等化，则将其尺度乘以等化尺度
    # 重塑等化尺度，以便可以沿轴向1将其与权重相乘
    equalization_scale_reshaped = reshape_scale(equalization_scale, 1, weight)
    scaled_weight = torch.mul(weight, torch.reciprocal(equalization_scale_reshaped))

    if next_equalization_scale is None:
        # 如果下一层没有等化尺度，则直接设置 scaled_weight 到模块中的权重，并返回
        setattr(modules[weight_parent_name], weight_name, scaled_weight)
        return

    # 将权重按行乘以下一个等化尺度
    # 重塑等化尺度，以便可以沿轴0将其与 scaled_weight 相乘
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, scaled_weight)
    scaled_weight = torch.mul(scaled_weight, next_equalization_scale_reshaped)

    # 将处理后的权重设置回模块中的权重
    setattr(modules[weight_parent_name], weight_name, scaled_weight)

    # 断言模型缓冲区中的权重与 scaled_weight 相似
    assert torch.allclose(model.get_buffer(str(weight_node.target)), scaled_weight)

    # 逐元素地按下一个等化尺度乘以偏置
    bias_node = None
    for node in op_node.args:
        # 查找包含偏置值的节点
        if isinstance(node, Node) and node.op == 'get_attr' and 'bias' in node.name:
            bias_node = node
            break
    if bias_node is None:
        return

    # 从 bias_node 的 target 中获取父模块名称和偏置名称
    bias_parent_name, bias_name = _parent_name(bias_node.target)
    bias = getattr(modules[bias_parent_name], bias_name)
    # 重新整形均衡化比例，以便可以对偏置进行逐元素乘法
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, bias)
    # 将偏置与重新整形后的均衡化比例进行逐元素乘法，得到缩放后的偏置
    scaled_bias = torch.mul(bias, next_equalization_scale_reshaped)
    # 将缩放后的偏置设置回模块的指定偏置名称处
    setattr(modules[bias_parent_name], bias_name, scaled_bias)
def clear_weight_quant_obs_node(op_node: Node, modules: Dict[str, nn.Module]) -> None:
    """ 给定操作节点，查找对应的量化观察器并重置其最小/最大值 """

    # 获取操作节点对应的权重均衡观察器节点
    weight_eq_obs_node = maybe_get_weight_eq_obs_node(op_node, modules)
    if weight_eq_obs_node is None:
        return

    # 获取权重量化观察器节点
    weight_quant_obs_node = weight_eq_obs_node.args[0]
    if weight_quant_obs_node is None:
        return
    assert isinstance(weight_quant_obs_node, Node)

    # 获取权重量化观察器对象并断言其为ObserverBase类型
    weight_quant_obs = modules[str(weight_quant_obs_node.target)]
    assert isinstance(modules[str(weight_quant_obs_node.target)], ObserverBase)
    
    # 重置权重量化观察器的最小/最大值
    weight_quant_obs.reset_min_max_vals()   # type: ignore[operator]

def remove_node(model: GraphModule, node: Node, prev_node: Node):
    """ 通过用给定的前一个节点替换所有使用当前节点的节点来从模型中删除给定的节点 """
    
    # 对于当前节点的所有用户节点，用输入的前一个节点替换当前节点
    orig_users = list(node.users.keys())
    for user_node in orig_users:
        user_node.replace_input_with(node, prev_node)

    # 擦除输入均衡观察器节点
    model.graph.erase_node(node)

def update_obs_for_equalization(model: GraphModule, modules: Dict[str, nn.Module]) -> Dict[str, _WeightEqualizationObserver]:
    """ 更新所有观察器的均衡化尺度。对于每个输入均衡观察器，我们将找到下一个权重均衡观察器的位置，
    创建它，并基于这两个观察器计算均衡化尺度。

    然后，我们将返回一个字典，将操作节点名称映射到相应的权重均衡观察器。
    """
    weight_eq_obs_dict = {}
    # 遍历模型图中的所有节点
    for node in model.graph.nodes:
        # 检查节点操作是否为 'call_module'，且对应模块是 _InputEqualizationObserver 类型的实例
        if node.op == 'call_module' and isinstance(modules[node.target], _InputEqualizationObserver):
            # 获取输入均衡观察器对象
            input_eq_obs = modules[node.target]
            # 确保 input_eq_obs 是 _InputEqualizationObserver 的实例
            assert isinstance(input_eq_obs, _InputEqualizationObserver)
            # 获取操作节点和权重均衡观察器对象
            op_node, weight_eq_obs = get_op_node_and_weight_eq_obs(node, model, modules)

            # 如果操作节点或者权重均衡观察器对象为空，则跳过当前循环
            if op_node is None or weight_eq_obs is None:
                continue

            # 如果操作节点的操作为 'call_module'
            if op_node.op == 'call_module':
                # 校准权重均衡观察器，因为它刚刚被创建
                if fused_module_supports_equalization(modules[str(op_node.target)]):
                    # 获取模块对象，确保它支持均衡化
                    module = modules[str(op_node.target)][0]   # type: ignore[index]
                    assert nn_module_supports_equalization(module)
                    # 对模块的权重进行均衡化
                    weight_eq_obs(module.weight)
                else:
                    # 对模块的权重进行均衡化
                    weight_eq_obs(modules[str(op_node.target)].weight)

            # 计算并设置均衡化尺度值
            equalization_scale = calculate_equalization_scale(input_eq_obs, weight_eq_obs)
            # 设置输入均衡观察器的均衡化尺度值
            input_eq_obs.set_equalization_scale(equalization_scale)
            # 设置权重均衡观察器的均衡化尺度值
            weight_eq_obs.set_equalization_scale(equalization_scale)

            # 将权重均衡观察器对象存入字典，键为操作节点的名称
            weight_eq_obs_dict[op_node.name] = weight_eq_obs

    # 返回存储了权重均衡观察器对象的字典
    return weight_eq_obs_dict
# 定义一个函数，用于将等化操作转换并更新模型中的其他节点
def convert_eq_obs(
    model: GraphModule,
    modules: Dict[str, nn.Module],
    weight_eq_obs_dict: Dict[str, _WeightEqualizationObserver],
) -> None:
    """ Converts the equalization operations and updates the other nodes in the
    following way:
        - Removes the input equalization observers and inserts a mul operator
          along with an equalization scale node wherever applicable (we do not
          want to insert a mul operator between connecting linear layers).
        - Updates the input quantization observers with the scaled input min/max
          values.
        - Scales the weights by the current and next equalization scales.
        - Removes the weight equalization observer node if it exists.

    Before (after prepare):
                                    weight values
                                          |
                                    WeightQuantObs
                                          |
                                      WeightEqObs
                                          |
        x -> InpQuantObs -> InpEqObs -> linear -> OutQuantObs

    After this function:
                                              scaled weight values
                                                      |
       equalization scale                       WeightQuantObs
              |                                       |
        x -> mul -> InpQuantObs (scaled min/max) -> linear -> OutQuantObs

    After convert:
       equalization scale                 scaled weight values
              |                                    |
        x -> mul -> quantize_per_tensor -> quantized::linear

    Note that although the equalization observer appeared after the quantization
    observer after prepare_fx, the mul node appears before the quantization node
    after convert_fx. This is because placing the equalization observer after
    the quantization observer in prepare_fx would allow us to keep the invariant
    that the graph before the current node inserts its observers is not
    modified.

    Having the equalization observer before the quantization observer would also
    cause some inconsistences between the ordering of the quantization and
    equalization observers.
    For example, a single linear layer would look like:
        x -> InpEqObs1 -> InpQuantObs1 -> linear1 -> OutQuantObs1
    But between two connected linear layers, it would look like:
        linear1 -> OutQuantObs1 -> InpEqObs2 -> linear2 -> OutQuantObs2
    """
    # 获取模型中的所有模块，包括重复的模块名
    modules = dict(model.named_modules(remove_duplicate=False))

    # 计算等化比例尺度，使用缩放后的输入更新观察者，并缩放权重
    weight_eq_obs_dict = update_obs_for_equalization(model, modules)
    
    # 调用实际的转换函数，处理模型中的等化操作
    convert_eq_obs(model, modules, weight_eq_obs_dict)



def _convert_equalization_ref(model: GraphModule):
    """ Reference function which applies changes needed for equalization, but
    does not quantize the nodes
    """
    # 获取模型中的所有模块，不包括重复的模块名
    modules = dict(model.named_modules(remove_duplicate=False))

    # 计算等化比例尺度，更新观察者
    weight_eq_obs_dict = update_obs_for_equalization(model, modules)
    
    # 调用等化转换函数，对模型进行等化操作的修改
    convert_eq_obs(model, modules, weight_eq_obs_dict)
    # 返回一个包含模型和模型图形的图形模块
    return GraphModule(model, model.graph)
###############################################################################
# Functions for running the equalized model on the Numeric Suite              #
###############################################################################

def get_layer_sqnr_dict(model_a: nn.Module, model_b: nn.Module, x: torch.Tensor) -> Dict[str, float]:
    """ Runs the Numeric Suite on model_a and model_b and returns a dictionary
    containing the SQNR between layers in model_a and model_b.

    Note: In order to support equalized models, this function has a hacky fix in
    which we do not match any torch.mul operators. This is because equalized
    models contain extra mul operators to scale the input by the equalization
    scale, but this edge case has not been resolved yet within the numeric suite code.

    Args:
        model_a: A float model
        model_b: A quantized model
        x: Inputs to use during calibration
    """
    import torch.ao.ns._numeric_suite_fx as ns  # 导入数值套件的函数
    from torch.ao.ns.fx.mappings import get_unmatchable_types_map  # 导入获取不可匹配类型映射的函数

    unmatchable_types_map = get_unmatchable_types_map()  # 获取不可匹配类型的映射
    unmatchable_types_map["funs_unmatchable"].add(torch.mul)  # 将 torch.mul 函数添加到不可匹配类型映射中

    # 为 model_a 和 model_b 创建数值套件日志记录器
    model_a_ns, model_b_ns = ns.add_loggers(
        'fp32', model_a,
        'int8', model_b,
        ns.OutputLogger,
        unmatchable_types_map=unmatchable_types_map
    )

    model_a_ns(x)  # 在 model_a 上执行数值套件
    model_b_ns(x)  # 在 model_b 上执行数值套件

    # 提取激活比较信息的字典
    activation_comparison_dict = ns.extract_logger_info(
        model_a_ns,
        model_b_ns,
        ns.OutputLogger,
        'int8')

    # 使用 compute_sqnr 计算 SQNR，扩展日志记录结果
    ns.extend_logger_results_with_comparison(
        activation_comparison_dict,
        'fp32', 'int8',
        torch.ao.ns.fx.utils.compute_sqnr, 'sqnr'
    )

    # 构造一个字典，将层名映射到 SQNR 值
    layer_sqnr_dict = {}
    for key in activation_comparison_dict:
        layer = activation_comparison_dict[key]['node_output']['int8'][0]['fqn']  # 获取层的完全限定名
        sqnr = activation_comparison_dict[key]['node_output']['int8'][0]['sqnr'][0]  # 获取层的 SQNR 值
        layer_sqnr_dict[layer] = sqnr  # 将层名和对应的 SQNR 值添加到字典中

    return layer_sqnr_dict

def get_equalization_qconfig_dict(
    layer_sqnr_dict: Dict[str, float],
    num_layers_to_equalize: int
) -> Any:
    """ Given the layer to SQNR dictionary, find the layers with the highest
    quantization errors, and return an equalization_qconfig_dict
    specifying to only equalize those top layers.

    Args:
        layer_sqnr_dict: Dictionary mapping layer names to SQNR values (found
            when comparing an equalized model against a float model)
        num_layers_to_equalize: Number of layers with the highest quantization
           errors to equalize
    """

    # 对 layer_sqnr_dict 根据 SQNR 值进行排序，并获取具有最低 SQNR 值（即最高量化误差）的层
    layer_sqnr_sorted = sorted(layer_sqnr_dict.items(), key=operator.itemgetter(1))
    layers_to_equalize = layer_sqnr_sorted[:num_layers_to_equalize]
    # 构建 equalization_qconfig_dict，指定仅对量化误差最高的层进行均衡化配置
    module_to_qconfig_list = [(item[0], default_equalization_qconfig) for item in layers_to_equalize]
    # 创建包含模块名称及其对应的均衡化配置的字典
    equalization_qconfig_dict = {"module_name": module_to_qconfig_list}
    # 返回均衡化配置字典作为函数的结果
    return equalization_qconfig_dict
```
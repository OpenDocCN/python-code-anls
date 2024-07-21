# `.\pytorch\torch\ao\quantization\fx\_lower_to_native_backend.py`

```py
# mypy: allow-untyped-defs
# 导入 PyTorch 库中的相关模块和函数
import torch
# 从 torch.fx 模块中导入 map_arg 和 Node 类
from torch.fx import map_arg, Node
# 从 torch.fx.graph 模块中导入 Graph 类
from torch.fx.graph import Graph
# 从 torch.nn 模块中导入 nn 类，用于定义神经网络模型
import torch.nn as nn
# 从 torch.nn.functional 模块中导入 F，用于包含各种功能函数
import torch.nn.functional as F
# 导入 Torch AO（加速运算）库中的特定模块
import torch.ao.nn.intrinsic as nni
# 导入 Torch AO 库中的量化模块
import torch.ao.nn.intrinsic.quantized as nniq
# 导入 Torch AO 库中的动态量化模块
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
# 导入 Torch AO 库中的量化模块
import torch.ao.nn.quantized as nnq
# 导入 Torch AO 库中的动态量化模块
import torch.ao.nn.quantized.dynamic as nnqd
# 导入 Torch AO 库中的参考量化模块
import torch.ao.nn.quantized.reference as nnqr
# 从 Torch AO 库中的量化模块工具中导入 WeightedQuantizedModule 类
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
# 从 torch.fx 模块中导入 GraphModule 类
from torch.fx import GraphModule
# 从当前目录下的 utils 文件中导入若干函数
from .utils import (
    collect_producer_nodes,
    get_linear_prepack_op_for_dtype,
    get_new_attr_name_with_prefix,
    get_qconv_prepack_op,
    graph_module_from_producer_nodes,
)
# 从上级目录的 utils 文件中导入 _parent_name 函数
from ..utils import _parent_name
# 从当前目录的 qconfig 文件中导入 QConfigAny 类
from ..qconfig import QConfigAny
# 从当前目录的 quantization_mappings 文件中导入 get_quantized_operator 函数
from ..quantization_mappings import get_quantized_operator
# 从当前目录的 utils 文件中导入 create_node_from_old_node_preserve_meta 函数
from .utils import create_node_from_old_node_preserve_meta
# 导入若干类型定义
from typing import Dict, Tuple, Type, List, Callable, Any, Union, Set, Optional
# 导入 operator 模块
import operator

# 定义字典，存储需要跳过的量化操作和对应的参数名
QOP_TO_ARG_NAMES_TO_SKIP = {
    torch._ops.ops.quantized.hardswish: ['inplace'],
    torch._ops.ops.quantized.elu: ['inplace'],
    torch._ops.ops.quantized.dropout: ['inplace'],
    torch._ops.ops.quantized.instance_norm:
    ['running_mean', 'running_var', 'use_input_stats', 'momentum'],
}

# 定义函数，判断节点是否在给定的模块列表中
def _is_node_in_list(node, modules, func_list, method_list, module_type_list):
    # 判断节点是否为调用函数，并且目标函数在 func_list 中
    is_call_function = node.op == "call_function" and node.target in func_list
    # 判断节点是否为调用方法，并且目标方法在 method_list 中
    is_call_method = node.op == "call_method" and node.target in method_list
    # 判断节点是否为调用模块，并且对应模块类型在 module_type_list 中
    is_call_module = node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    return is_call_function, is_call_method, is_call_module

# 定义函数，判断节点是否为固定量化参数节点
def is_fixed_qparams_node(node, modules):
    # 定义函数列表，包含多种激活函数
    func_list = [
        torch.nn.functional.hardsigmoid,
        torch.nn.functional.sigmoid,
        torch.sigmoid,
        torch.tanh,
    ]
    # 定义方法列表，包含多种激活函数的方法名
    method_list = [
        "hardsigmoid",
        "hardsigmoid_",
        "sigmoid",
        "sigmoid_",
        "tanh",
        "tanh_",
    ]
    # 定义模块类型列表，包含多种激活函数的模块类型
    module_type_list = [
        torch.nn.Hardsigmoid,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
        torch.nn.Softmax,
    ]
    # 调用 _is_node_in_list 函数判断节点是否在指定列表中
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)

# 定义函数，判断节点是否为默认节点
def is_default_node(node, modules):
    # 定义函数列表，包含多种非固定量化参数的函数
    func_list = [
        torch.nn.functional.elu,
        torch.nn.functional.hardswish,
        torch.nn.functional.instance_norm,
        torch.nn.functional.layer_norm,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.dropout,
    ]
    # 方法列表为空
    method_list: List[Any] = []
    # 定义了一个模块类型列表，包含了多个不同的神经网络模块和规范化方法
    module_type_list = [
        nnqr.ConvTranspose1d,                 # 一维转置卷积层
        nnqr.ConvTranspose2d,                 # 二维转置卷积层
        nnqr.ConvTranspose3d,                 # 三维转置卷积层
        torch.nn.ELU,                         # ELU激活函数
        torch.nn.LeakyReLU,                   # LeakyReLU激活函数
        torch.nn.Hardswish,                   # Hardswish激活函数
        torch.nn.InstanceNorm1d,              # 一维实例归一化层
        torch.nn.InstanceNorm2d,              # 二维实例归一化层
        torch.nn.InstanceNorm3d,              # 三维实例归一化层
        torch.nn.LayerNorm,                   # 层归一化层
        torch.nn.Dropout,                     # Dropout层
        torch.nn.PReLU,                       # PReLU激活函数
        torch.nn.BatchNorm2d,                 # 二维批归一化层
        torch.nn.BatchNorm3d,                 # 三维批归一化层
        torch.ao.nn.intrinsic.BNReLU2d,       # AO库中的二维BNReLU层
        torch.ao.nn.intrinsic.BNReLU3d,       # AO库中的三维BNReLU层
    ]
    
    # 调用_is_node_in_list函数，传入参数node, modules, func_list, method_list, module_type_list，进行判断
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)
# 判断给定节点是否属于复制节点类型
def is_copy_node(node, modules):
    # 定义包含复制节点函数的列表
    func_list = [
        torch.adaptive_avg_pool1d,
        torch.nn.functional.adaptive_avg_pool2d,
        torch.nn.functional.adaptive_avg_pool3d,
        torch.nn.functional.hardtanh,
        torch.nn.functional.hardtanh_,
        torch.nn.functional.interpolate,
        torch.nn.functional.max_pool1d,
        torch.nn.functional.max_pool2d,
        torch.nn.functional.max_pool3d,
        torch.nn.functional.relu,
        torch.nn.functional.relu6,
        torch.avg_pool1d,
        torch._C._nn.avg_pool2d,
        torch._C._nn.avg_pool3d,
        torch.clamp,
        torch.flatten,
        torch.mean,
        operator.floordiv,
        # 下面的函数 F.channel_shuffle 和 torch.channel_shuffle 本质上是相同的
        # 所以我们只需要在这里放置一个即可
        torch.channel_shuffle,
    ]
    # 定义包含复制节点方法的列表
    method_list = [
        "clamp",
        "mean",
        "relu",
        "relu_",
    ]
    # 定义包含复制节点模块类型的列表
    module_type_list = [
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveAvgPool3d,
        torch.nn.AvgPool1d,
        torch.nn.AvgPool2d,
        torch.nn.AvgPool3d,
        torch.nn.Hardtanh,
        torch.nn.MaxPool1d,
        torch.nn.MaxPool2d,
        torch.nn.MaxPool3d,
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.nn.ChannelShuffle,
    ]
    # 调用内部函数 _is_node_in_list 判断节点是否在以上列表中的任一类中
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)

# 判断给定节点是否属于一般张量形状节点类型
def is_general_tensor_shape_node(node, modules):
    # 定义包含一般张量形状节点函数的列表
    func_list = [
        torch.narrow,
        torch.transpose,
        torch.repeat_interleave,
        torch.squeeze,
        torch.stack,
        torch.unsqueeze,
        torch.nn.functional.pixel_shuffle,
        torch.nn.functional.pixel_unshuffle,
    ]
    # 定义包含一般张量形状节点方法的列表
    method_list = [
        "contiguous",
        "detach",
        "detach_",
        "permute",
        "repeat",
        "repeat_interleave",
        "reshape",
        "resize_",
        "shape",
        "size",
        "squeeze",
        "squeeze_",
        "transpose",
        "unsqueeze",
        "unsqueeze_",
        "view",
    ]
    # 定义包含一般张量形状节点模块类型的列表
    module_type_list = [
        torch.nn.Identity,
        torch.nn.PixelShuffle,
        torch.nn.PixelUnshuffle,
    ]
    # 调用内部函数 _is_node_in_list 判断节点是否在以上列表中的任一类中
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)

# 判断给定节点是否属于其他节点类型
def is_other_node(node, modules):
    # 定义包含其他节点函数的列表
    func_list = [
        torch.cat,
    ]
    # 空的其他节点方法列表
    method_list: List[Any] = []
    # 空的其他节点模块类型列表
    module_type_list: List[Any] = []
    # 调用内部函数 _is_node_in_list 判断节点是否在以上列表中的任一类中
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)

# 判断给定节点是否属于特殊模式节点类型
def is_special_pattern_node(node, modules):
    # 初始化三个结果变量为 False
    res_function, res_method, res_module = False, False, False
    # 遍历特殊模式检测函数的列表
    for checker in [is_fixed_qparams_node, is_default_node, is_copy_node, is_general_tensor_shape_node, is_other_node]:
        # 调用每个特殊模式检测函数，获取其返回值
        is_call_function, is_call_method, is_call_module = checker(node, modules)
        # 将结果与当前迭代的结果变量做或操作，更新总体结果
        res_function = res_function or is_call_function
        res_method = res_method or is_call_method
        res_module = res_module or is_call_module
    # 返回三个变量 res_function、res_method 和 res_module
    return res_function, res_method, res_module
# 检查节点是否为"dequantize"方法调用节点
def is_dequantize_node(node):
    return isinstance(node, Node) and node.op == "call_method" and node.target == "dequantize"

# 检查节点是否为调用getattr函数，目标为"shape"的节点
def is_getattr_tensor_metadata_node(node):
    return node.op == "call_function" and \
        node.target == getattr and \
        node.args[1] in ["shape"]

# 检查节点是否为调用类方法，目标为"shape"或"size"的节点
def is_get_tensor_info_node(node):
    return node.op == "call_method" and \
        node.target in ["shape", "size"]

# 判断是否应跳过降级操作，如果op的名称在qconfig_map中，并且对应的QConfig为None，则返回True
# 这里的QConfig是指量化配置
def should_skip_lowering(op: torch.fx.node.Node, qconfig_map: Dict[str, QConfigAny]):
    """
    Return True if the op is configured with a None qconfig, False otherwise.
    Note: maybe need to generalize this to also check for the dtype, and we
    only lower when dtype matches, but right now fbgemm/qnnpack only support
    a single dtype, so it is OK for now.
    """
    return op.name in qconfig_map and qconfig_map[op.name] is None

# 静态量化模块的映射，将参考模块类映射到相应的静态量化模块类进行降级
STATIC_LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[WeightedQuantizedModule]] = {
    nnqr.Linear: nnq.Linear,
    nnqr.Conv1d: nnq.Conv1d,
    nnqr.Conv2d: nnq.Conv2d,
    nnqr.Conv3d: nnq.Conv3d,
}

# 动态量化模块的映射，将参考模块类映射到相应的动态量化模块类进行降级
DYNAMIC_LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[nn.Module]] = {
    nnqr.Linear: nnqd.Linear,
    nnqr.GRUCell: nnqd.GRUCell,
    nnqr.LSTMCell: nnqd.LSTMCell,
    nnqr.RNNCell: nnqd.RNNCell,
    nnqr.LSTM: nnqd.LSTM,
    nnqr.GRU: nnqd.GRU,
}

# 仅权重量化模块的映射，将参考模块类映射到相应的仅权重量化模块类进行降级
WEIGHT_ONLY_LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[nn.Module]] = {
    nnqr.Embedding: nnq.Embedding,
    nnqr.EmbeddingBag: nnq.EmbeddingBag,
}

# 特殊模式降级模块映射，将特定的参考模块类映射到相应的量化模块类进行降级
# TODO: 修正这些模块的命名空间
SPECIAL_PATTERN_LOWER_MODULE_MAP = {
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nnqr.ConvTranspose1d: nnq.ConvTranspose1d,
    nnqr.ConvTranspose2d: nnq.ConvTranspose2d,
    nnqr.ConvTranspose3d: nnq.ConvTranspose3d,
    nn.ELU: nnq.ELU,
    nn.LeakyReLU: nnq.LeakyReLU,
    nn.Hardswish: nnq.Hardswish,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    nn.Dropout: nnq.Dropout,
    nn.Softmax: nnq.Softmax,
    nn.PReLU: nnq.PReLU,
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
}

# 静态量化融合模块映射，将融合模块类映射到其内部参考模块类和相应的静态量化模块类进行降级
STATIC_LOWER_FUSED_MODULE_MAP: Dict[Type[nn.Module], Tuple[Type[nn.Module], Type[WeightedQuantizedModule]]] = {
    nni.LinearReLU: (nnqr.Linear, nniq.LinearReLU),
    # TODO: LinearLeakyReLU is registered as global but it is only fused and
}
    # 当使用ondnn后端配置时进行降级。可能需要将不同后端的注册和降级函数分开来。
    # 将nni.LinearLeakyReLU映射为(nnqr.Linear, nniq.LinearLeakyReLU)
    # 将nni.LinearTanh映射为(nnqr.Linear, nniq.LinearTanh)
    # 将nni.ConvReLU1d映射为(nnqr.Conv1d, nniq.ConvReLU1d)
    # 将nni.ConvReLU2d映射为(nnqr.Conv2d, nniq.ConvReLU2d)
    # 将nni.ConvReLU3d映射为(nnqr.Conv3d, nniq.ConvReLU3d)
}

# STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP 和 STATIC_LOWER_FUSED_MODULE_MAP 的区别：
# STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP 中的参考节点有两个输入。
# 将融合模块类映射到一个包含两个元素的元组：
#   1) 内部参考模块类
#   2) 用于降低的替换静态量化模块类
STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP: Dict[Type[nn.Module], Tuple[Type[nn.Module], Type[WeightedQuantizedModule]]] = {
    nni.ConvAdd2d: (nnqr.Conv2d, nniq.ConvAdd2d),
    nni.ConvAddReLU2d: (nnqr.Conv2d, nniq.ConvAddReLU2d),
}

# 将融合模块类映射到一个包含两个元素的元组：
#   1) 内部参考模块类
#   2) 用于降低的替换动态量化模块类
DYNAMIC_LOWER_FUSED_MODULE_MAP: Dict[Type[nn.Module], Tuple[Type[nn.Module], Type[nn.Module]]] = {
    nni.LinearReLU: (nnqr.Linear, nniqd.LinearReLU),
}

# 将一个功能函数映射到一个包含两个元素的元组：
#   1) 操作的量化版本
#   2) 如果存在的话，与ReLU融合的量化操作版本；否则为None
STATIC_LOWER_FUNCTIONAL_MAP: Dict[Callable, Tuple[Callable, Optional[Callable]]] = {
    F.linear: (torch.ops.quantized.linear, torch.ops.quantized.linear_relu),
    F.conv1d: (torch.ops.quantized.conv1d, torch.ops.quantized.conv1d_relu),
    F.conv2d: (torch.ops.quantized.conv2d, torch.ops.quantized.conv2d_relu),
    F.conv3d: (torch.ops.quantized.conv3d, torch.ops.quantized.conv3d_relu),
    F.conv_transpose1d: (torch.ops.quantized.conv_transpose1d, None),
    F.conv_transpose2d: (torch.ops.quantized.conv_transpose2d, None),
    F.conv_transpose3d: (torch.ops.quantized.conv_transpose3d, None),
}

# WEIGHT_PREPACK_OPS 是一个集合，包含如下功能函数：
#   torch._ops.ops.quantized.linear_prepack
#   torch._ops.ops.quantized.linear_prepack_fp16
#   torch._ops.ops.quantized.conv1d_prepack
#   torch._ops.ops.quantized.conv2d_prepack
#   torch._ops.ops.quantized.conv3d_prepack
#   torch.ops.quantized.conv_transpose1d_prepack
#   torch.ops.quantized.conv_transpose2d_prepack
#   torch.ops.quantized.conv_transpose3d_prepack
WEIGHT_PREPACK_OPS: Set[Callable] = {
    torch._ops.ops.quantized.linear_prepack,
    torch._ops.ops.quantized.linear_prepack_fp16,
    torch._ops.ops.quantized.conv1d_prepack,
    torch._ops.ops.quantized.conv2d_prepack,
    torch._ops.ops.quantized.conv3d_prepack,
    torch.ops.quantized.conv_transpose1d_prepack,
    torch.ops.quantized.conv_transpose2d_prepack,
    torch.ops.quantized.conv_transpose3d_prepack,
}

# 将一个功能函数映射到一个字典，字典的键是一个包含两个元素的元组 (input_activation_dtype, weight_dtype)，
# 值是一个包含两个元素的元组：
#   1) 动态量化版本的操作
#   2) 如果存在的话，与ReLU融合的动态量化版本的操作；否则为None
DYNAMIC_LOWER_FUNCTIONAL_MAP: Dict[Callable, Dict[Tuple[torch.dtype, torch.dtype], Tuple[Callable, Optional[Callable]]]] = {
    F.linear: {
        (torch.quint8, torch.qint8): (torch.ops.quantized.linear_dynamic,
                                      torch.ops.quantized.linear_relu_dynamic),
        (torch.float16, torch.float16): (torch.ops.quantized.linear_dynamic_fp16,
                                         torch.ops.quantized.linear_relu_dynamic_fp16)
    },
    # 对于 F.conv1d，动态量化版本 + ReLU 的组合目前不可用
    F.conv1d: {
        (torch.quint8, torch.qint8): (torch.ops.quantized.conv1d_dynamic, None),
    },
}
    F.conv2d: {
        # 当输入和权重是量化类型 (torch.quint8, torch.qint8) 时，使用动态量化的二维卷积操作
        (torch.quint8, torch.qint8): (torch.ops.quantized.conv2d_dynamic, None),
    },
    F.conv3d: {
        # 当输入和权重是量化类型 (torch.quint8, torch.qint8) 时，使用动态量化的三维卷积操作
        (torch.quint8, torch.qint8): (torch.ops.quantized.conv3d_dynamic, None),
    },
}

CONV_FUNCTIONAL_OPS: Set[Callable] = {
    F.conv1d,  # 添加一维卷积函数到集合中
    F.conv2d,  # 添加二维卷积函数到集合中
    F.conv3d,  # 添加三维卷积函数到集合中
}

CONV_TRANSPOSE_FUNCTIONAL_OPS: Set[Callable] = {
    F.conv_transpose1d,  # 添加一维转置卷积函数到集合中
    F.conv_transpose2d,  # 添加二维转置卷积函数到集合中
    F.conv_transpose3d,  # 添加三维转置卷积函数到集合中
}

# TODO: add tests for lowering these ops
QBIN_OP_MAPPING: Dict[Union[Callable, str], Callable] = {
    operator.add: torch.ops.quantized.add,  # 映射加法操作到量化加法操作
    torch.add: torch.ops.quantized.add,     # 映射torch加法操作到量化加法操作
    operator.mul: torch.ops.quantized.mul,  # 映射乘法操作到量化乘法操作
    operator.matmul: torch.ops.quantized.matmul,  # 映射矩阵乘法操作到量化矩阵乘法操作
    torch.mul: torch.ops.quantized.mul,     # 映射torch乘法操作到量化乘法操作
    torch.matmul: torch.ops.quantized.matmul,  # 映射torch矩阵乘法操作到量化矩阵乘法操作
}
QBIN_RELU_OP_MAPPING: Dict[Union[Callable, str], Callable] = {
    operator.add: torch.ops.quantized.add_relu,    # 映射加法操作到带ReLU的量化加法操作
    torch.add: torch.ops.quantized.add_relu,       # 映射torch加法操作到带ReLU的量化加法操作
    operator.mul: torch.ops.quantized.mul_relu,    # 映射乘法操作到带ReLU的量化乘法操作
    torch.mul: torch.ops.quantized.mul_relu,       # 映射torch乘法操作到带ReLU的量化乘法操作
}

def _save_packed_weight(self, destination, prefix, keep_vars):
    for attr_name in dir(self):
        if "_packed_weight" in attr_name and \
           isinstance(getattr(self, attr_name), torch._C.ScriptObject):  # type: ignore[attr-defined]
            packed_weight = getattr(self, attr_name)
            destination[prefix + attr_name] = packed_weight

def _load_packed_weight(self, state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs):
    attrs_to_pop = []
    for attr_name in state_dict:
        if attr_name.startswith("_packed_weight") and isinstance(state_dict[attr_name], torch._C.ScriptObject):  # type: ignore[attr-defined] # noqa: B950
            setattr(self, attr_name, state_dict[attr_name])
            attrs_to_pop.append(attr_name)

    # pop the packed param attributesn
    for attr_name in attrs_to_pop:
        state_dict.pop(attr_name)

def fold_weight(
    quantized_model: GraphModule,
    node_name_to_scope: Dict[str, Tuple[str, type]]
) -> GraphModule:
    """
    Trace back from the weight node util we hit getattr, reconstruct the
    graph module with the traced nodes and run the graph module to pack the
    weight. then replace the original chain of ops with the packed weight.
    """
    packed_weights = {}
    # map from folded node name to the prepacked weight name
    folded_nodes = {}
    # get packed weights
    for node in quantized_model.graph.nodes:
        if node.op == 'call_function' and node.target in WEIGHT_PREPACK_OPS:
            nodes_to_fold = collect_producer_nodes(node)
            if nodes_to_fold is not None:
                for node_to_fold in nodes_to_fold:
                    folded_nodes[node_to_fold.name] = node

                prepacking_module = graph_module_from_producer_nodes(
                    quantized_model, nodes_to_fold)
                packed_weight = prepacking_module()
                packed_weights[node.name] = packed_weight

    # remove folded nodes and replace the prepacking node with getattr
    folded_graph = Graph()
    env: Dict[Any, Any] = {}
    # 定义函数 `load_arg`，接受参数 `a`，通过调用 `map_arg` 函数获取 `env` 中对应节点的值并返回
    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    # 遍历量化模型的图中的每个节点
    for node in quantized_model.graph.nodes:
        # 获取与当前节点相关的折叠节点（如果存在）
        prepack_node = folded_nodes.get(node.name, None)
        # 如果折叠节点与当前节点相同
        if prepack_node is node:
            # 获取预打包的权重
            packed_weight = packed_weights[node.name]
            # 在根节点上添加一个预打包的属性
            op_node = next(iter(prepack_node.users))
            # 获取模块路径和名称
            module_path, _ = node_name_to_scope[op_node.name]
            # 创建一个新的预打包权重名称
            get_new_packed_weight_name = \
                get_new_attr_name_with_prefix(module_path + '_packed_weight_')
            packed_weight_name = get_new_packed_weight_name(quantized_model)
            # 将预打包权重设置为量化模型的属性
            setattr(quantized_model, packed_weight_name, packed_weight)
            # 将预打包节点替换为一个 `getattr` 节点
            env[node.name] = folded_graph.create_node(
                'get_attr', packed_weight_name, (), {})
        # 如果折叠节点存在但与当前节点不同
        elif prepack_node is not None:
            # 移除折叠节点
            continue
        # 如果折叠节点不存在
        else:
            # 复制其他节点到折叠图中，并使用 `load_arg` 函数获取参数
            env[node.name] = folded_graph.node_copy(node, load_arg)

    # 创建一个新的 `GraphModule` 对象，将量化模型和折叠图作为参数
    quantized_model = GraphModule(quantized_model, folded_graph)
    # 注册状态字典钩子 `_save_packed_weight` 到量化模型上
    quantized_model._register_state_dict_hook(_save_packed_weight)
    # 注册加载状态字典前钩子 `_load_packed_weight` 到量化模型上，同时包括模块信息
    quantized_model._register_load_state_dict_pre_hook(_load_packed_weight, with_module=True)
    # 返回量化后的模型 `quantized_model`
    return quantized_model
def _get_module(node: Node, modules: Dict[str, nn.Module]) -> Optional[nn.Module]:
    """
    Return the `torch.nn.Module` that corresponds to the specified node's target.
    If no such node exists, return None.
    """
    # 如果节点的操作是调用模块，并且目标模块存在于给定的模块字典中，则返回该模块
    if node.op == "call_module" and str(node.target) in modules:
        return modules[str(node.target)]
    else:
        return None

def _match_static_pattern(
    node: Node,
    modules: Dict[str, nn.Module],
    qconfig_map: Dict[str, QConfigAny],
    matching_modules_or_ops: List[Callable],
    dequantize_node_arg_indices: List[int]
) -> Union[Tuple[Node, Node, Node], Tuple[None, None, None]]:
    """
    Match the pattern (dequantize - ref node - quantize) against the node provided.

    If there is a match, return a 3-tuple of:
      1) q_node: the quantize node,
      2) relu_node: a relu node wrapping the ref_node, and
      3) ref_node: a reference module or functional node to replace with its quantized counterpart
    Otherwise, if there is no match, return a 3-tuple of (None, None, None).

    Parameters:
      node: The `torch.fx.Node` to match against.
      modules: A mapping from node names to modules in the model graph, used for module lookup.
      qconfig_map: A mapping from node names to the qconfigs associated with the nodes.
          If the corresponding qconfig for the reference node is None, then return no match.
      matching_modules_or_ops: Either a list of functions or a list of `torch.nn.Module`s.
          If the reference node is not in this list, then return no match.
      dequantize_node_arg_indices: A list of indices in the reference node args where dequantize
          nodes may be present. An empty list means skipping the check for dequantize nodes.
    """
    SKIP_LOWERING_VALUE = (None, None, None)

    # 如果节点的操作不是调用函数或者目标不是 torch.quantize_per_tensor 函数，则返回跳过的值
    if node.op != "call_function" or node.target != torch.quantize_per_tensor:
        return SKIP_LOWERING_VALUE
    q_node = node
    ref_node = q_node.args[0]
    assert isinstance(ref_node, Node)

    # 处理节点被 ReLU 包装的情况
    if (ref_node.op == "call_function" and ref_node.target in (F.relu, torch.relu)) or\
            (ref_node.op == "call_module" and type(_get_module(ref_node, modules)) == nn.ReLU):
        relu_node = ref_node
        ref_node = relu_node.args[0]
        assert isinstance(ref_node, Node)
    else:
        relu_node = None
    if should_skip_lowering(ref_node, qconfig_map):
        return SKIP_LOWERING_VALUE

    # 匹配参考模块或者函数
    if isinstance(matching_modules_or_ops[0], type) and issubclass(matching_modules_or_ops[0], nn.Module):
        expected_op = "call_module"
        match_key = type(_get_module(ref_node, modules))
    else:
        expected_op = "call_function"
        match_key = ref_node.target
    if ref_node.op != expected_op or match_key not in matching_modules_or_ops:
        return SKIP_LOWERING_VALUE
    # 匹配去量化节点。必须同时满足以下两个条件：
    # (1) 所有匹配索引处的 `torch.fx.Node` 必须是一个去量化节点
    # (2) 必须至少有一个去量化节点存在
    matched_dequantize = False  # 初始化匹配去量化节点的标志为 False
    for i in dequantize_node_arg_indices:
        assert i < len(ref_node.args), \
            f"Dequantize index {i} exceeded reference node's arg length {len(ref_node.args)}"
        arg = ref_node.args[i]  # 获取参考节点中指定索引处的参数
        if is_dequantize_node(arg):  # 检查参数是否是去量化节点
            matched_dequantize = True  # 如果是去量化节点，则将匹配标志设置为 True
        elif isinstance(arg, Node):  # 如果参数是一个节点对象
            return SKIP_LOWERING_VALUE  # 如果不是去量化节点且是节点对象，则跳过降低操作并返回指定值
    if not matched_dequantize:
        return SKIP_LOWERING_VALUE  # 如果没有匹配到任何去量化节点，则跳过降低操作并返回指定值

    return (q_node, relu_node, ref_node)  # 返回匹配到的节点元组
# 定义一个函数 `_match_static_pattern_with_two_inputs`，用于匹配静态模式 `(dequantize - ref node - quantize)` 并返回匹配结果
def _match_static_pattern_with_two_inputs(
    node: Node,  # 输入参数 node，表示要匹配的 torch.fx.Node
    modules: Dict[str, nn.Module],  # 输入参数 modules，字典，将节点名称映射到模型中的模块
    qconfig_map: Dict[str, QConfigAny],  # 输入参数 qconfig_map，字典，将节点名称映射到与之关联的量化配置
    matching_modules_or_ops: List[Callable]  # 输入参数 matching_modules_or_ops，列表，包含要匹配的模块或操作
) -> Union[Tuple[Node, Node], Tuple[None, None]]:
    """
    Match the pattern (dequantize - ref node - quantize) against the node provided.

    If there is a match, return a 2-tuple of:
      1) q_node: the quantize node,
      2) ref_node: a reference module or functional node to replace with its quantized counterpart
    Otherwise, if there is no match, return a 2-tuple of (None, None).

    Parameters:
      node: The `torch.fx.Node` to match against.
      modules: A mapping from node names to modules in the model graph, used for module lookup.
      qconfig_map: A mapping from node names to the qconfigs associated with the nodes.
          If the corresponding qconfig for the reference node is None, then return no match.
      matching_modules_or_ops: Either a list of functions or a list of `torch.nn.Module`s.
          If the reference node is not in this list, then return no match.
    """
    SKIP_LOWERING_VALUE = (None, None)  # 定义一个常量元组，表示匹配失败时的返回值

    # Match quantize node
    if node.op != "call_function" or node.target != torch.quantize_per_tensor:
        return SKIP_LOWERING_VALUE  # 如果节点不是 quantize 操作，则返回匹配失败的常量元组
    q_node = node  # 将当前节点标记为 quantize 节点
    ref_node = q_node.args[0]  # 获取 quantize 节点的第一个参数作为参考节点
    assert isinstance(ref_node, Node)  # 断言参考节点是 Node 类型的对象

    if should_skip_lowering(ref_node, qconfig_map):
        return SKIP_LOWERING_VALUE  # 如果需要跳过降低操作，则返回匹配失败的常量元组

    # Match reference module or functional
    if isinstance(matching_modules_or_ops[0], type) and issubclass(matching_modules_or_ops[0], nn.Module):
        expected_op = "call_module"
        match_key = type(_get_module(ref_node, modules))  # 获取参考节点对应的模块类型
    else:
        # This pass only support op of "call_module"
        return SKIP_LOWERING_VALUE  # 如果不支持当前操作类型，则返回匹配失败的常量元组

    if ref_node.op != expected_op or match_key not in matching_modules_or_ops:
        return SKIP_LOWERING_VALUE  # 如果参考节点的操作不符合预期或者类型不在匹配列表中，则返回匹配失败的常量元组

    # Check ref_node has 2 input nodes, both are dq node.
    if len(ref_node.args) != 2:
        return SKIP_LOWERING_VALUE  # 如果参考节点的输入参数不为2，则返回匹配失败的常量元组
    for i in range(len(ref_node.args)):
        arg = ref_node.args[i]
        if not is_dequantize_node(arg):
            return SKIP_LOWERING_VALUE  # 如果参考节点的某个输入参数不是 dequantize 节点，则返回匹配失败的常量元组

    return (q_node, ref_node)  # 如果以上条件都满足，则返回 quantize 节点和参考节点的元组

# 定义一个函数 `_lower_static_weighted_ref_module`，用于替换模型中的 `dequantize - ref module - quantize` 模式为其量化版本
def _lower_static_weighted_ref_module(
        model: GraphModule,
        qconfig_map: Dict[str, QConfigAny]):
    """
    Traverse the graph and find dequantize - ref module - quantize patterns
    and replace them with the quantized version of the ref module.
    """
    modules = dict(model.named_modules(remove_duplicate=False))  # 获取模型中所有模块的字典表示，包括重复模块
    nodes = list(model.graph.nodes)  # 获取模型图中所有节点的列表表示
    for n in model.graph.nodes:
        # Step 0: Find nodes that match this pattern (dequantize - ref module - quantize)
        # 定义匹配的模块列表，包括静态下降的模块和静态融合模块
        matching_modules = list(STATIC_LOWER_MODULE_MAP.keys()) + list(STATIC_LOWER_FUSED_MODULE_MAP.keys())
        # 使用_match_static_pattern函数查找符合指定模式的节点
        (q_node, relu_node, ref_node) = _match_static_pattern(
            n, modules, qconfig_map, matching_modules, dequantize_node_arg_indices=[0])  # type: ignore[arg-type]
        if q_node is None:
            continue
        assert ref_node is not None
        (_, scale_node, zero_point_node, _) = q_node.args
        ref_module = _get_module(ref_node, modules)
        ref_class = type(ref_module)
        assert isinstance(scale_node, Node)
        assert isinstance(zero_point_node, Node)
        assert issubclass(ref_class, nn.Module)

        # Step 1: Change this pattern to use the corresponding quantized module
        # 对于融合模块，还需检查内部模块是否是参考模块，如果是，则替换整个融合模块为对应的量化模块
        if ref_class in STATIC_LOWER_FUSED_MODULE_MAP:
            inner_ref_class, q_class = STATIC_LOWER_FUSED_MODULE_MAP[ref_class]
            if type(ref_module[0]) != inner_ref_class:  # type: ignore[index]
                continue
        else:
            q_class = STATIC_LOWER_MODULE_MAP[ref_class]
        # 从model中获取输出的量化参数
        output_scale = getattr(model, scale_node.target)
        output_zero_point = getattr(model, zero_point_node.target)
        # 使用参考模块创建对应的量化模块
        q_module = q_class.from_reference(ref_module, output_scale, output_zero_point)
        # 替换参考模块为量化模块
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(modules[parent_name], module_name, q_module)

        # Step 2: Reroute around dq_node, and remove q_node and its args
        assert len(ref_node.args) == 1
        # 获取dequantize节点，并断言它是Node类型
        dq_node = ref_node.args[0]
        assert isinstance(dq_node, Node)
        # 替换ref_node的输入为dq_node的第一个参数
        ref_node.replace_input_with(dq_node, dq_node.args[0])
        # 用ref_node替换q_node的所有使用
        q_node.replace_all_uses_with(ref_node)
        # 从模型图中删除q_node, scale_node, zero_point_node节点
        model.graph.erase_node(q_node)
        model.graph.erase_node(scale_node)
        model.graph.erase_node(zero_point_node)
def _lower_static_weighted_ref_module_with_two_inputs(
        model: GraphModule,
        qconfig_map: Dict[str, QConfigAny]):
    """
    Traverse the graph and find patterns
    dequantize   dequantize
       \\         //
        ref module
            \\
          quantize
    and replace them with the quantized version of the ref module.
    """
    modules = dict(model.named_modules(remove_duplicate=False))  # 获取模型中所有模块的字典
    nodes = list(model.graph.nodes)  # 获取模型图中所有节点的列表
    for n in model.graph.nodes:
        # Step 0: Find nodes that match this pattern (dequantize - ref module - quantize)
        matching_modules = list(STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP.keys())  # 获取静态融合模块映射的键列表
        (q_node, ref_node) = _match_static_pattern_with_two_inputs(
            n, modules, qconfig_map, matching_modules)  # 匹配具有两个输入的静态模式，返回量化节点和参考模块节点
        if q_node is None:
            continue
        assert ref_node is not None
        (_, scale_node, zero_point_node, _) = q_node.args  # 获取量化节点的参数：量化输出标度、零点
        ref_module = _get_module(ref_node, modules)  # 获取参考模块节点对应的模块
        ref_class = type(ref_module)
        assert isinstance(scale_node, Node)
        assert isinstance(zero_point_node, Node)
        assert issubclass(ref_class, nn.Module)

        # Step 1: Change this pattern to use the corresponding quantized module
        # For fused modules, we also check whether the inner module is a reference module
        # If so, we replace the entire fused module with the corresponding quantized module
        if ref_class in STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP:
            inner_ref_class, q_class = STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP[ref_class]
            if type(ref_module[0]) != inner_ref_class:  # 检查内部模块是否是参考模块类别，如果不是则跳过
                continue
        else:
            continue
        output_scale = getattr(model, scale_node.target)  # 获取模型中标度节点的属性
        output_zero_point = getattr(model, zero_point_node.target)  # 获取模型中零点节点的属性
        q_module = q_class.from_reference(ref_module, output_scale, output_zero_point)  # 基于参考模块创建量化模块
        # replace reference module with quantized module
        parent_name, module_name = _parent_name(ref_node.target)  # 获取参考模块的父模块名称和模块名称
        setattr(modules[parent_name], module_name, q_module)  # 将量化模块设置为模型中相应位置的模块

        # Step 2: Reroute around dq_node, and remove q_node and its args
        assert len(ref_node.args) == 2
        for arg in ref_node.args:
            if not is_dequantize_node(arg):  # 检查参数是否为去量化节点
                continue
            dq_node = arg
            assert isinstance(dq_node, Node)
            ref_node.replace_input_with(dq_node, dq_node.args[0])  # 用去量化节点的输入替换参考节点的输入

        q_node.replace_all_uses_with(ref_node)  # 用参考节点替换量化节点的所有使用
        model.graph.erase_node(q_node)  # 从模型图中移除量化节点
        model.graph.erase_node(scale_node)  # 从模型图中移除标度节点
        model.graph.erase_node(zero_point_node)  # 从模型图中移除零点节点
    # 使用 model.named_modules() 返回的迭代器创建命名模块的字典
    named_modules = dict(model.named_modules(remove_duplicate=False))

    # 遍历模型图中的节点
    for n in model.graph.nodes:
        # 如果节点不是 "call_module" 操作或者命名模块类型不在动态量化模块映射的键中，则跳过此节点
        if n.op != "call_module" or \
           type(named_modules[str(n.target)]) not in \
           set(DYNAMIC_LOWER_MODULE_MAP.keys()).union(
               set(DYNAMIC_LOWER_FUSED_MODULE_MAP.keys())):
            continue

        # 将当前节点设为参考节点
        ref_node = n
        # 获取 dequantize 方法的调用节点
        dq_node = ref_node.args[0]
        
        # 如果 dequantize 节点不是 "call_method" 操作或者目标不是 "dequantize"，则跳过此节点
        if dq_node.op != "call_method" or dq_node.target != "dequantize":
            continue

        # 获取 dequantize 方法的输入动态量化节点
        input_dynamic_q_node = dq_node.args[0]

        # 如果输入动态量化节点不是 "call_function" 操作或者目标不是 torch.quantize_per_tensor_dynamic，则跳过此节点
        if input_dynamic_q_node.op != "call_function" or \
           input_dynamic_q_node.target != torch.quantize_per_tensor_dynamic:
            continue

        # 获取激活函数的数据类型
        activation_dtype = input_dynamic_q_node.args[1]
        # 判断是否为 float16 类型
        is_fp16 = activation_dtype == torch.float16
        # 判断是否为 int8 类型
        is_int8 = activation_dtype in [torch.quint8, torch.qint8]
        
        # 如果既不是 int8 也不是 float16 类型，则跳过此节点
        if not is_int8 and not is_fp16:
            continue

        # 获取参考节点对应的命名模块
        ref_module = named_modules[str(ref_node.target)]
        # 获取参考模块的类类型
        ref_class = type(ref_module)

        # 如果参考模块的类类型在 DYNAMIC_LOWER_FUSED_MODULE_MAP 中
        if ref_class in DYNAMIC_LOWER_FUSED_MODULE_MAP:
            # 获取内部参考类和量化类
            inner_ref_class, q_class = DYNAMIC_LOWER_FUSED_MODULE_MAP[ref_class]
            # 如果参考模块的第一个子模块不是内部参考类，则跳过此节点
            if type(ref_module[0]) != inner_ref_class:
                continue
        else:
            # 否则，获取 ref_class 对应的量化类
            q_class = DYNAMIC_LOWER_MODULE_MAP.get(ref_class)  # type: ignore[assignment]

        # TODO: 可能需要定义一个 WeightedDynamicallyQuantizedModule

        # 使用 q_class 创建一个从参考模块 ref_module 实例化的量化模块 q_module
        q_module = q_class.from_reference(ref_module)  # type: ignore[attr-defined]

        # 将参考模块替换为动态量化模块 q_module
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(named_modules[parent_name], module_name, q_module)

        # 替换参考节点的输入为 dequantize 节点的输入动态量化节点的第一个参数
        ref_node.replace_input_with(dq_node, input_dynamic_q_node.args[0])
# 遍历模型的图节点，查找并替换 ref_module 模式为其仅权重量化版本的模块
def _lower_weight_only_weighted_ref_module(model: GraphModule):
    """
    Traverse the graph and find ref_module patterns
    and replace them with the weight only quantized version of the ref module.
    """
    # 获取模型中所有命名模块的字典，包括重复模块
    named_modules = dict(model.named_modules(remove_duplicate=False))
    # 遍历模型的图节点
    for n in model.graph.nodes:
        # 如果节点操作不是 "call_module" 或者对应的模块类型不在 WEIGHT_ONLY_LOWER_MODULE_MAP 的键集合中，则跳过
        if n.op != "call_module" or type(named_modules[str(n.target)]) not in set(WEIGHT_ONLY_LOWER_MODULE_MAP.keys()):
            continue
        # 获取参考节点和参考模块
        ref_node = n
        ref_module = named_modules[str(ref_node.target)]
        ref_class = type(ref_module)
        # 根据参考模块的类获取其对应的仅权重量化版本的类
        q_class = WEIGHT_ONLY_LOWER_MODULE_MAP.get(ref_class)
        # TODO: WeightedQuantizedModule is currently assuming static quant apis
        # with output_scale, output_zero_point in from_reference, we may want to
        # relax that, or rename this
        # TODO: maybe define a WeightedWeightOnlyQuantizedModule
        # 使用参考模块创建仅权重量化的模块
        q_module = q_class.from_reference(ref_module)  # type: ignore[union-attr]

        # 将参考模块替换为动态量化模块
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(named_modules[parent_name], module_name, q_module)

# 遍历模型的图节点，将功能性参考模式替换为其量化版本
def _lower_static_weighted_ref_functional(
        model: GraphModule,
        qconfig_map: Dict[str, QConfigAny]):
    """
    Traverse the graph and replace functional reference patterns with their quantized versions.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    nodes = list(model.graph.nodes)

# 遍历模型的图节点，将功能性参考模式替换为其动态量化版本
def _lower_dynamic_weighted_ref_functional(
        model: GraphModule,
        qconfig_map: Dict[str, QConfigAny]):
    """
    Traverse the graph and replace functional reference patterns with their dynamically
    quantized versions.
    Examples:
    quantize_per_tensor_dynamic - dequantize - functional linear --> linear_dynamic
    to(torch.float16) - dequantize - functional linear --> linear_dynamic_fp16
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    nodes = list(model.graph.nodes)
    # 按照保留的顺序搜索，以便首先匹配较大的模式
    # 例如，我们希望在 linear - relu 之前匹配 linear。
    # we want to search in reserved order so that we can match the larger patterns first
    # e.g. we want to match linear - relu before linear.

# 遍历模型的图节点，将量化二元操作降级
def _lower_quantized_binary_op(
        model: GraphModule,
        qconfig_map: Dict[str, QConfigAny]):
    binary_ops_to_lower: List[Callable] = [operator.add, torch.add, operator.mul, torch.mul, torch.matmul]
    modules = dict(model.named_modules(remove_duplicate=False))
    # 对模型的图中的每个节点进行遍历
    for n in model.graph.nodes:
        # Step 0: 找到符合特定模式的节点序列 (dequantize - ref module - quantize)
        (q_node, relu_node, bop_node) = _match_static_pattern(
            n, modules, qconfig_map, binary_ops_to_lower, dequantize_node_arg_indices=[0, 1])
        # 如果没有找到符合条件的节点序列，则继续下一个节点的处理
        if q_node is None:
            continue
        # 确保找到了 quantize 节点和对应的二进制操作节点
        assert bop_node is not None
        # 获取 quantize 节点的相关参数 (scale_node, zero_point_node)
        (_, scale_node, zero_point_node, _) = q_node.args

        # Step 1: 移除 dequantize 节点
        num_dq_nodes = 0
        # 遍历二进制操作节点的参数
        for arg in bop_node.args:
            # 如果参数不是 dequantize 节点，则跳过
            if not is_dequantize_node(arg):
                continue
            # 获取 dequantize 节点
            dq_node = arg
            assert isinstance(dq_node, Node)
            # 获取 dequantize 节点的输入节点
            dn_input = dq_node.args[0]
            # 将二进制操作节点中的 dequantize 节点替换为其输入节点
            bop_node.replace_input_with(dq_node, dn_input)
            # 计数已移除的 dequantize 节点数量
            num_dq_nodes += 1
        # 确保至少移除了一个 dequantize 节点
        assert num_dq_nodes > 0

        # Step 2: 将二进制操作节点替换为量化二进制操作节点
        assert bop_node.target in QBIN_OP_MAPPING
        # 根据是否存在 relu 节点选择不同的量化二进制操作映射
        binop_to_qbinop = QBIN_OP_MAPPING if relu_node is None else QBIN_RELU_OP_MAPPING
        # 获取对应的量化二进制操作符
        qbin_op = binop_to_qbinop[bop_node.target]
        # 准备用于量化二进制操作的参数列表 (x, y)
        qop_node_args = list(bop_node.args)
        # 如果存在两个 dequantize 节点，则添加 scale 和 zero_point 参数
        if num_dq_nodes == 2:
            qop_node_args.extend([scale_node, zero_point_node])
        # 在 quantize 节点之后插入量化二进制操作的调用，并移除原始的二进制操作节点
        with model.graph.inserting_after(q_node):
            qop_node = create_node_from_old_node_preserve_meta(
                model.graph,
                ("call_function", qbin_op, tuple(qop_node_args), {}),
                bop_node)
            # 替换所有使用原始 quantize 节点的地方为新的量化二进制操作节点
            q_node.replace_all_uses_with(qop_node)

        # Step 3: 移除 quantize 节点、二进制操作节点以及可能的 relu 节点
        # 从模型的图中移除 quantize 节点
        model.graph.erase_node(q_node)
        # 如果存在 relu 节点，则从图中移除
        if relu_node is not None:
            model.graph.erase_node(relu_node)
        # 从图中移除二进制操作节点
        model.graph.erase_node(bop_node)
# 降低一个量化参考模型（具有参考量化操作模式）到PyTorch的本地后端（如fbgemm/qnnpack）的函数
# 参数：
# - model: 要转换的图模块
# - qconfig_map: 包含量化配置的字典映射
# - node_name_to_scope: 节点名称到作用域和类型的字典映射
# 返回值：
# - GraphModule: 降低后的模型对象
def _lower_to_native_backend(
    model: GraphModule,
    qconfig_map: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]]
) -> GraphModule:
    # 使用各种函数降低静态权重参考模块
    _lower_static_weighted_ref_module(model, qconfig_map)
    # 使用两个输入降低静态权重参考模块
    _lower_static_weighted_ref_module_with_two_inputs(model, qconfig_map)
    # 降低动态权重参考模块
    _lower_dynamic_weighted_ref_module(model)
    # 仅降低权重的参考模块
    _lower_weight_only_weighted_ref_module(model)
    # 使用量化配置映射降低静态权重的函数式参考模块
    _lower_static_weighted_ref_functional(model, qconfig_map)
    # 使用量化配置映射降低动态权重的函数式参考模块
    _lower_dynamic_weighted_ref_functional(model, qconfig_map)
    # 降低量化二元操作
    _lower_quantized_binary_op(model, qconfig_map)
    # 降低获取属性张量元数据操作
    _lower_getattr_tensor_metadta_op(model)
    # 降低获取张量信息操作
    _lower_get_tensor_info_op(model)
    # 替换特定模式的操作
    special_pattern_replacement(model)
    # 消除死代码
    model.graph.eliminate_dead_code()
    # 折叠权重
    model = fold_weight(model, node_name_to_scope)
    # 再次消除死代码
    model.graph.eliminate_dead_code()
    # 重新编译模型
    model.recompile()
    # 对模型进行静态检查
    model.graph.lint()
    # 返回处理后的模型对象
    return model
```
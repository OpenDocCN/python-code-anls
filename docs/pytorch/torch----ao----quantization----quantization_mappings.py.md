# `.\pytorch\torch\ao\quantization\quantization_mappings.py`

```py
# 导入copy模块，用于对象的复制操作
import copy

# 导入PyTorch库
import torch
from torch import nn

# 导入PyTorch函数库，提供各种神经网络函数，如激活函数等
import torch.nn.functional as F

# 导入PyTorch AO模块下的内置量化相关库
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.reference as nnqr
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.qat as nnqat
import torch.ao.nn.qat.dynamic as nnqatd

# 导入类型注解相关的模块
from typing import Optional, Union, Dict, Set, Callable, Any

# 由于torch.ao.nn模块使用了延迟导入，需要在此处显式导入其内容
import torch.ao.nn.sparse
import torch.ao.nn as ao_nn

# 导入量化相关的存根函数：QuantStub和DeQuantStub
from torch.ao.quantization.stubs import QuantStub, DeQuantStub

# 导入量化相关的假量化函数
from torch.ao.quantization.fake_quantize import (
    default_fixed_qparams_range_0to1_fake_quant,
    default_fixed_qparams_range_neg1to1_fake_quant,
)

# 导入量化相关的工具函数：get_combined_dict
from torch.ao.quantization.utils import get_combined_dict

# 导入神经网络参数化相关函数：type_before_parametrizations
from torch.nn.utils.parametrize import type_before_parametrizations

# 定义公开的导出列表，列出了模块中的一些关键成员
__all__ = [
    "DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_STATIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_QAT_MODULE_MAPPINGS",
    "DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS",
    "DEFAULT_MODULE_TO_ACT_POST_PROCESS",
    "DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS",
    "DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS",
    "no_observer_set",
    "get_default_static_quant_module_mappings",
    "get_default_static_quant_reference_module_mappings",
    "get_embedding_static_quant_module_mappings",
    "get_default_static_sparse_quant_module_mappings",
    "get_static_quant_module_class",
    "get_dynamic_quant_module_class",
    "get_default_qat_module_mappings",
    "get_embedding_qat_module_mappings",
    "get_default_dynamic_quant_module_mappings",
    "get_default_dynamic_sparse_quant_module_mappings",
    "get_default_qconfig_propagation_list",
    "get_default_compare_output_module_list",
    "get_default_float_to_quantized_operator_mappings",
    "get_quantized_operator",
]

# 默认映射字典，将浮点数模块替换为参考量化模块
DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS: Dict[Callable, Any] = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    nn.Linear: nnqr.Linear,
    nn.Conv1d: nnqr.Conv1d,
    nn.Conv2d: nnqr.Conv2d,
    nn.Conv3d: nnqr.Conv3d,
    nn.ConvTranspose1d: nnqr.ConvTranspose1d,
    nn.ConvTranspose2d: nnqr.ConvTranspose2d,
    nn.ConvTranspose3d: nnqr.ConvTranspose3d,
    nn.Embedding: nnqr.Embedding,
    nn.EmbeddingBag: nnqr.EmbeddingBag,
    nn.GRUCell: nnqr.GRUCell,
    nn.LSTMCell: nnqr.LSTMCell,
    nn.RNNCell: nnqr.RNNCell,
    nn.LSTM: nnqr.LSTM,
}

# 默认映射字典，将浮点数模块替换为量化模块
DEFAULT_STATIC_QUANT_MODULE_MAPPINGS: Dict[Callable, Any] = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    # 以下还有其他模块的映射关系，未完待续...
}
    # 将原始的 nn.Dropout 替换为量化后的 nnq.Dropout
    nn.Dropout: nnq.Dropout,
    # 将原始的 nn.Conv1d 替换为量化后的 nnq.Conv1d
    nn.Conv1d: nnq.Conv1d,
    # 将原始的 nn.Conv2d 替换为量化后的 nnq.Conv2d
    nn.Conv2d: nnq.Conv2d,
    # 将原始的 nn.Conv3d 替换为量化后的 nnq.Conv3d
    nn.Conv3d: nnq.Conv3d,
    # 将原始的 nn.ConvTranspose1d 替换为量化后的 nnq.ConvTranspose1d
    nn.ConvTranspose1d: nnq.ConvTranspose1d,
    # 将原始的 nn.ConvTranspose2d 替换为量化后的 nnq.ConvTranspose2d
    nn.ConvTranspose2d: nnq.ConvTranspose2d,
    # 将原始的 nn.ConvTranspose3d 替换为量化后的 nnq.ConvTranspose3d
    nn.ConvTranspose3d: nnq.ConvTranspose3d,
    # 将原始的 nn.ELU 替换为量化后的 nnq.ELU
    nn.ELU: nnq.ELU,
    # 将原始的 nn.Embedding 替换为量化后的 nnq.Embedding
    nn.Embedding: nnq.Embedding,
    # 将原始的 nn.EmbeddingBag 替换为量化后的 nnq.EmbeddingBag
    nn.EmbeddingBag: nnq.EmbeddingBag,
    # 将原始的 nn.GroupNorm 替换为量化后的 nnq.GroupNorm
    nn.GroupNorm: nnq.GroupNorm,
    # 将原始的 nn.Hardswish 替换为量化后的 nnq.Hardswish
    nn.Hardswish: nnq.Hardswish,
    # 将原始的 nn.InstanceNorm1d 替换为量化后的 nnq.InstanceNorm1d
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    # 将原始的 nn.InstanceNorm2d 替换为量化后的 nnq.InstanceNorm2d
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    # 将原始的 nn.InstanceNorm3d 替换为量化后的 nnq.InstanceNorm3d
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    # 将原始的 nn.LayerNorm 替换为量化后的 nnq.LayerNorm
    nn.LayerNorm: nnq.LayerNorm,
    # 将原始的 nn.LeakyReLU 替换为量化后的 nnq.LeakyReLU
    nn.LeakyReLU: nnq.LeakyReLU,
    # 将原始的 nn.modules.linear.NonDynamicallyQuantizableLinear 替换为量化后的 nnq.Linear
    nn.modules.linear.NonDynamicallyQuantizableLinear: nnq.Linear,
    # 将原始的 nn.Linear 替换为量化后的 nnq.Linear
    nn.Linear: nnq.Linear,
    # 将原始的 nn.ReLU6 替换为量化后的 nnq.ReLU6
    nn.ReLU6: nnq.ReLU6,
    # 再次将原始的 nn.Dropout 替换为量化后的 nnq.Dropout（重复了）
    nn.Dropout: nnq.Dropout,
    # 将原始的 nn.PReLU 替换为量化后的 nnq.PReLU
    nn.PReLU: nnq.PReLU,
    # Wrapper Modules（包装模块）:
    # 将原始的 nnq.FloatFunctional 替换为量化后的 nnq.QFunctional
    nnq.FloatFunctional: nnq.QFunctional,
    # Intrinsic modules（内置模块）:
    # 将原始的 nni.BNReLU2d 替换为量化后的 nniq.BNReLU2d
    nni.BNReLU2d: nniq.BNReLU2d,
    # 将原始的 nni.BNReLU3d 替换为量化后的 nniq.BNReLU3d
    nni.BNReLU3d: nniq.BNReLU3d,
    # 将原始的 nni.ConvReLU1d 替换为量化后的 nniq.ConvReLU1d
    nni.ConvReLU1d: nniq.ConvReLU1d,
    # 将原始的 nni.ConvReLU2d 替换为量化后的 nniq.ConvReLU2d
    nni.ConvReLU2d: nniq.ConvReLU2d,
    # 将原始的 nni.ConvReLU3d 替换为量化后的 nniq.ConvReLU3d
    nni.ConvReLU3d: nniq.ConvReLU3d,
    # 将原始的 nni.ConvAdd2d 替换为量化后的 nniq.ConvAdd2d
    nni.ConvAdd2d: nniq.ConvAdd2d,
    # 将原始的 nni.ConvAddReLU2d 替换为量化后的 nniq.ConvAddReLU2d
    nni.ConvAddReLU2d: nniq.ConvAddReLU2d,
    # 将原始的 nni.LinearReLU 替换为量化后的 nniq.LinearReLU
    nni.LinearReLU: nniq.LinearReLU,
    # 将原始的 nni.LinearLeakyReLU 替换为量化后的 nniq.LinearLeakyReLU
    nni.LinearLeakyReLU: nniq.LinearLeakyReLU,
    # 将原始的 nni.LinearTanh 替换为量化后的 nniq.LinearTanh
    nni.LinearTanh: nniq.LinearTanh,
    # 将原始的 nniqat.ConvBn1d 替换为量化后的 nnq.Conv1d
    nniqat.ConvBn1d: nnq.Conv1d,
    # 将原始的 nniqat.ConvBn2d 替换为量化后的 nnq.Conv2d
    nniqat.ConvBn2d: nnq.Conv2d,
    # 将原始的 nniqat.ConvBn3d 替换为量化后的 nnq.Conv3d
    nniqat.ConvBn3d: nnq.Conv3d,
    # 将原始的 nniqat.ConvBnReLU1d 替换为量化后的 nniq.ConvReLU1d
    nniqat.ConvBnReLU1d: nniq.ConvReLU1d,
    # 将原始的 nniqat.ConvBnReLU2d 替换为量化后的 nniq.ConvReLU2d
    nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
    # 将原始的 nniqat.ConvBnReLU3d 替换为量化后的 nniq.ConvReLU3d
    nniqat.ConvBnReLU3d: nniq.ConvReLU3d,
    # 将原始的 nniqat.ConvReLU2d 替换为量化后的 nniq.ConvReLU2d
    nniqat.ConvReLU2d: nniq.ConvReLU2d,
    # 将原始的 nniqat.ConvReLU3d 替换为量化后的 nniq.ConvReLU3d
    nniqat.ConvReLU3d: nniq.ConvReLU3d,
    # 将原始的 nniqat.LinearReLU 替换为量化后的 nniq.LinearReLU
    nniqat.LinearReLU: nniq.LinearReLU,
    # 将原始的 nniqat.LinearBn1d 替换为量化后的 nnq.Linear
    nniqat.LinearBn1d: nnq.Linear,
    # QAT modules（量化感知训练模块）:
    # 将原始的 nnqat.Linear 替换为量化后的 nnq.Linear
    nnqat.Linear: nnq.Linear,
    # 将原始的 nnqat.Conv2d 替换为量化后的 nnq.Conv2d
    nnqat.Conv2d: nnq.Conv2d,
    # 将原始的 nnqat.Conv3d 替换为量化后的 nnq.Conv3d
    nnqat.Conv3d: nnq.Conv3d,
}

# 默认映射，用于将浮点模块交换为量化训练模块
DEFAULT_QAT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    nn.Conv2d: nnqat.Conv2d,  # 将 nn.Conv2d 映射到 nnqat.Conv2d
    nn.Conv3d: nnqat.Conv3d,  # 将 nn.Conv3d 映射到 nnqat.Conv3d
    nn.Linear: nnqat.Linear,  # 将 nn.Linear 映射到 nnqat.Linear
    nn.modules.linear.NonDynamicallyQuantizableLinear: nnqat.Linear,  # 将非动态量化线性层映射到 nnqat.Linear
    # 内置模块:
    nni.ConvBn1d: nniqat.ConvBn1d,
    nni.ConvBn2d: nniqat.ConvBn2d,
    nni.ConvBn3d: nniqat.ConvBn3d,
    nni.ConvBnReLU1d: nniqat.ConvBnReLU1d,
    nni.ConvBnReLU2d: nniqat.ConvBnReLU2d,
    nni.ConvBnReLU3d: nniqat.ConvBnReLU3d,
    nni.ConvReLU2d: nniqat.ConvReLU2d,
    nni.ConvReLU3d: nniqat.ConvReLU3d,
    nni.LinearReLU: nniqat.LinearReLU,
    nni.LinearBn1d: nniqat.LinearBn1d,
}

# 默认映射，用于将动态模块交换
DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    nn.GRUCell: nnqd.GRUCell,
    nn.Linear: nnqd.Linear,
    nnqatd.Linear: nnqd.Linear,
    nn.modules.linear.NonDynamicallyQuantizableLinear: nnqd.Linear,
    nn.LSTM: nnqd.LSTM,
    nn.GRU: nnqd.GRU,
    nn.LSTMCell: nnqd.LSTMCell,
    nn.RNNCell: nnqd.RNNCell,
    nni.LinearReLU: nniqd.LinearReLU,
    nn.EmbeddingBag: nnq.EmbeddingBag,
    nn.Embedding: nnq.Embedding,
    # 不默认启用以下操作，因为数值精度比其他动态操作差
    # nn.Conv1d: nnqd.Conv1d,
    # nn.Conv2d: nnqd.Conv2d,
    # nn.Conv3d: nnqd.Conv3d,
    # nn.ConvTranspose1d: nnqd.ConvTranspose1d,
    # nn.ConvTranspose2d: nnqd.ConvTranspose2d,
    # nn.ConvTranspose3d: nnqd.ConvTranspose3d,
}

# QConfig传播允许列表
_INCLUDE_QCONFIG_PROPAGATE_LIST : Set[Callable] = {
    nn.Sequential,  # 包含 nn.Sequential 在内，允许QConfig传播
}

# 默认从浮点函数或torch操作映射到量化操作的映射
# TODO: 与默认静态映射合并
DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS : Dict[Union[Callable, str], Callable] = {
    F.elu: torch.ops.quantized.elu,
    F.hardswish: torch.ops.quantized.hardswish,
    F.instance_norm: torch.ops.quantized.instance_norm,
    F.layer_norm: torch.ops.quantized.layer_norm,
    F.leaky_relu: torch.ops.quantized.leaky_relu,
    F.dropout: torch.ops.quantized.dropout,
}

# 模块到输出激活后处理类的默认映射
DEFAULT_MODULE_TO_ACT_POST_PROCESS : Dict[Callable, Callable] = {
    nn.Hardsigmoid: default_fixed_qparams_range_0to1_fake_quant,
    nn.Sigmoid: default_fixed_qparams_range_0to1_fake_quant,
    nn.Softmax: default_fixed_qparams_range_0to1_fake_quant,
    nn.Tanh: default_fixed_qparams_range_neg1to1_fake_quant,
}

# 默认映射，将浮点模块交换为静态稀疏量化模块
DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    nn.Linear: ao_nn.sparse.quantized.Linear
}

# 默认映射，将浮点模块交换为动态稀疏量化模块
DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    nn.Linear: ao_nn.sparse.quantized.dynamic.Linear
}

def no_observer_set() -> Set[Any]:
    r"""These modules cannot have observers inserted by default."""
    # 定义一个字符串文档，描述以下代码的作用
    no_observers = {
        nn.quantizable.LSTM,
        nn.quantizable.MultiheadAttention
    }
    # 创建一个集合，包含不支持默认插入观察器的模块类
    return no_observers
# 返回默认的后训练静态量化模块映射字典
def get_default_static_quant_module_mappings() -> Dict[Callable, Any]:
    ''' Get module mapping for post training static quantization
    '''
    return copy.deepcopy(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)

# 返回默认的后训练静态量化参考模块映射字典
def get_default_static_quant_reference_module_mappings() -> Dict[Callable, Any]:
    ''' Get reference module mapping for post training static quantization
    '''
    return copy.deepcopy(DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS)

# 返回包括嵌入层 QAT 映射的模块映射字典
def get_embedding_static_quant_module_mappings() -> Dict[Callable, Any]:
    ''' Get module mapping, including mapping for embedding QAT
    '''
    mapping = copy.deepcopy(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)
    mapping[nnqat.EmbeddingBag] = nnq.EmbeddingBag
    mapping[nnqat.Embedding] = nnq.Embedding
    return mapping

# 返回默认的后训练静态稀疏量化模块映射字典
def get_default_static_sparse_quant_module_mappings() -> Dict[Callable, Any]:
    ''' Get module mapping for post training static sparse quantization
    '''
    return copy.deepcopy(DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS)

# 根据浮点数模块类获取相应的静态量化模块类
def get_static_quant_module_class(
        float_module_class: Callable,
        additional_static_quant_mapping: Optional[Dict[Callable, Any]] = None,
        is_reference: bool = False) -> Any:
    r"""n Get the statically quantized module class corresponding to
    the floating point module class
    """
    if additional_static_quant_mapping is None:
        additional_static_quant_mapping = {}
    # 获取合并后的映射字典
    all_mappings = get_combined_dict(
        DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS if is_reference
        else DEFAULT_STATIC_QUANT_MODULE_MAPPINGS, additional_static_quant_mapping)
    # 获取静态量化模块类并进行深拷贝
    static_quant_module_class = all_mappings.get(float_module_class, None)
    assert static_quant_module_class is not None, \
        f"Floating point module class {str(float_module_class)}" + \
        " does not have a corresponding quantized module class"
    return copy.deepcopy(static_quant_module_class)

# 根据浮点数模块类获取相应的动态量化模块类
def get_dynamic_quant_module_class(
        float_module_class: Callable,
        additional_dynamic_quant_mapping: Optional[Dict[Callable, Any]] = None) -> Any:
    r"""n Get the dynamically quantized module class corresponding to
    the floating point module class
    """
    if additional_dynamic_quant_mapping is None:
        additional_dynamic_quant_mapping = {}
    # 获取合并后的映射字典
    all_mappings = get_combined_dict(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS, additional_dynamic_quant_mapping)
    # 获取动态量化模块类并进行深拷贝
    dynamic_quant_module_class = all_mappings.get(float_module_class, None)
    assert dynamic_quant_module_class is not None, \
        f"Floating point module class {str(float_module_class)}" + \
        " does not have a corresponding quantized module class"
    return copy.deepcopy(dynamic_quant_module_class)

# 返回用于量化感知训练的默认模块映射字典
def get_default_qat_module_mappings() -> Dict[Callable, Any]:
    ''' Get default module mapping for quantization aware training
    '''
    return copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)

# 返回包括嵌入层 QAT 映射的模块映射字典
def get_embedding_qat_module_mappings() -> Dict[Callable, Any]:
    '''获取量化感知训练的模块映射
        这包括默认值以及
        为嵌入层启用量化感知训练。
    '''
    # 深度复制默认的量化感知训练模块映射
    mapping = copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)
    # 将标准库中的 nn.EmbeddingBag 映射到量化感知训练库中的 nnqat.EmbeddingBag
    mapping[nn.EmbeddingBag] = nnqat.EmbeddingBag
    # 将标准库中的 nn.Embedding 映射到量化感知训练库中的 nnqat.Embedding
    mapping[nn.Embedding] = nnqat.Embedding
    # 返回更新后的模块映射
    return mapping
# 返回默认的动态量化模块映射字典
def get_default_dynamic_quant_module_mappings() -> Dict[Callable, Any]:
    return DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS

# 返回默认的动态稀疏量化模块映射字典
def get_default_dynamic_sparse_quant_module_mappings() -> Dict[Callable, Any]:
    return DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS

# 获取默认的 qconfig 传播列表，用于在准备过程中附加 qconfig 属性到模块类型
def get_default_qconfig_propagation_list() -> Set[Callable]:
    QCONFIG_PROPAGATE_MODULE_CLASS_LIST = (
        set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys()) |
        set(DEFAULT_QAT_MODULE_MAPPINGS.keys()) |
        set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.keys()) |
        _INCLUDE_QCONFIG_PROPAGATE_LIST
    )
    return copy.deepcopy(QCONFIG_PROPAGATE_MODULE_CLASS_LIST)

# 获取默认的比较输出模块列表，用于在数值套件中记录输出
def get_default_compare_output_module_list() -> Set[Callable]:
    NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_MODULE_LIST = (
        set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.values())
        | set(DEFAULT_QAT_MODULE_MAPPINGS.values())
        | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.values())
        | set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_QAT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.keys())
        | _INCLUDE_QCONFIG_PROPAGATE_LIST
    )
    return copy.deepcopy(NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_MODULE_LIST)

# 返回默认的浮点到量化操作符映射字典的深层复制
def get_default_float_to_quantized_operator_mappings() -> Dict[Union[Callable, str], Callable]:
    return copy.deepcopy(DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS)

# TODO: merge with get_static_quant_module_class
# 根据浮点运算符获取相应的量化运算符
def get_quantized_operator(float_op: Union[Callable, str]) -> Callable:
    quantized_op = DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS.get(float_op, None)
    assert quantized_op is not None, \
        f'Operator {str(float_op)} does not have corresponding quantized op'
    return quantized_op

# 获取模块的特殊激活后处理器，优先级高于 qconfig 中的激活后处理器
def _get_special_act_post_process(module: torch.nn.Module) -> Optional[Callable]:
    r""" Get the special activation post process for `module`, this has
    higher priority than the activation post process in `qconfig`
    e.g.
    input: torch.nn.Sigmoid
    output: default_affine_fixed_qparam_fake_quant
    """
    return DEFAULT_MODULE_TO_ACT_POST_PROCESS.get(type_before_parametrizations(module), None)

# 判断模块是否具有特殊激活后处理器，用于训练且模块类型存在于 DEFAULT_MODULE_TO_ACT_POST_PROCESS 中
def _has_special_act_post_process(module: torch.nn.Module) -> bool:
    return module.training and type(module) in DEFAULT_MODULE_TO_ACT_POST_PROCESS
```
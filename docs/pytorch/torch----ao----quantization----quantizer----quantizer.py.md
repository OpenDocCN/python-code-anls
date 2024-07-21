# `.\pytorch\torch\ao\quantization\quantizer\quantizer.py`

```
# mypy: allow-untyped-defs  # 允许未类型化的定义，用于类型检查时的设置
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from dataclasses import dataclass, field  # 导入用于数据类装饰器和字段装饰器
from typing import Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入 PyTorch 库
from torch import Tensor  # 导入 Tensor 类型
from torch.ao.quantization import ObserverOrFakeQuantize  # 导入观察器或伪量化器
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor  # 导入观察器或伪量化器的构造函数
from torch.fx import Node  # 导入 Torch FX 框架中的节点类

__all__ = [  # 指定模块的公开接口列表
    "Quantizer",  # 量化器类
    "QuantizationSpecBase",  # 量化规范的基类
    "QuantizationSpec",  # 量化规范类
    "FixedQParamsQuantizationSpec",  # 固定量化参数的量化规范类
    "EdgeOrNode",  # 表示量化图中的边或节点的类型
    "SharedQuantizationSpec",  # 共享的量化规范类
    "DerivedQuantizationSpec",  # 派生的量化规范类
    "QuantizationAnnotation",  # 量化注释类
]


class QuantizationSpecBase(ABC):  # 基于 ABC 的量化规范基类
    """Base class for different types of quantization specs that allows users to
    specify how to quantize a Tensor (input/output of a Node) in the model
    """

    pass  # 占位符，暂无额外方法或属性


@dataclass(eq=True, frozen=True)
class QuantizationSpec(QuantizationSpecBase):  # 量化规范数据类
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, quant_min, quant_max etc.
    """

    dtype: torch.dtype  # 数据类型
    # 观察器或伪量化器的构造函数，如 MinMaxObserver、PerChannelHistogramObserver 等
    # 或者可以附加一些自定义参数给它们
    # 例如 MinMaxObserver.with_args(eps=eps)
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor
    quant_min: Optional[int] = None  # 最小量化值
    quant_max: Optional[int] = None  # 最大量化值
    qscheme: Optional[torch.qscheme] = None  # 量化方案
    ch_axis: Optional[int] = None  # 通道轴
    is_dynamic: bool = False  # 是否为动态量化

    def __post_init__(self):
        # 初始化函数的后处理方法
        # TODO: add init for quant_min/quant_max
        # quant_min 必须小于等于 quant_max
        if (
            self.quant_min is not None
            and self.quant_max is not None
            and self.quant_min > self.quant_max
        ):
            raise ValueError(
                f"quant_min {self.quant_min} must be <= quant_max {self.quant_max}."
            )

        # ch_axis 必须小于通道数，但此处无法检查，仅检查其是否小于 0
        if self.ch_axis is not None and self.ch_axis < 0:
            raise ValueError("Ch_axis is < 0.")


@dataclass(eq=True, frozen=True)
class FixedQParamsQuantizationSpec(QuantizationSpecBase):  # 固定量化参数的量化规范数据类
    dtype: torch.dtype  # 数据类型
    scale: float  # 缩放因子
    zero_point: int  # 零点
    quant_min: Optional[int] = None  # 最小量化值
    quant_max: Optional[int] = None  # 最大量化值
    qscheme: Optional[torch.qscheme] = None  # 量化方案
    is_dynamic: bool = False  # 是否为动态量化


"""
The way we refer to other points of quantization in the graph will be either
an input edge or an output value
input edge is the connection between input node and the node consuming the input, so it's a Tuple[Node, Node]
output value is an fx Node
"""
EdgeOrNode = Union[Tuple[Node, Node], Node]  # 量化图中的边或节点类型
EdgeOrNode.__module__ = "torch.ao.quantization.quantizer.quantizer"  # 设置模块名称


@dataclass(eq=True, frozen=True)
class SharedQuantizationSpec(QuantizationSpecBase):  # 共享的量化规范数据类
    """
    Placeholder class for shared quantization specifications.
    """
    # 定义了用于共享量化参数的张量的量化规范
    """
    
    # 指定要与其共享观察器或仿真量化实例的边缘或节点
    edge_or_node: EdgeOrNode
@dataclass(eq=True, frozen=True)
class DerivedQuantizationSpec(QuantizationSpecBase):
    """Quantization spec for the Tensors whose quantization parameters are derived from other Tensors"""

    # List of nodes or edges from which quantization parameters are derived
    derived_from: List[EdgeOrNode]

    # Function to derive quantization parameters from a list of Observers or FakeQuantize modules
    derive_qparams_fn: Callable[[List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]]

    # Data type of the tensors
    dtype: torch.dtype

    # Optional minimum quantization value
    quant_min: Optional[int] = None

    # Optional maximum quantization value
    quant_max: Optional[int] = None

    # Optional quantization scheme
    qscheme: Optional[torch.qscheme] = None

    # Optional channel axis for quantization
    ch_axis: Optional[int] = None

    # Boolean indicating if quantization is dynamic
    is_dynamic: bool = False


@dataclass
class QuantizationAnnotation:
    """How are input arguments or output should be quantized,
    expressed as QuantizationSpec, this corresponds to how a Tensor in the
    operator Graph is observed (PTQ) or fake quantized (QAT)
    """

    # Mapping from torch.fx.Node to optional QuantizationSpecBase describing input quantization
    input_qspec_map: Dict[Node, Optional[QuantizationSpecBase]] = field(
        default_factory=dict
    )

    # Quantization specification for the output of this node
    output_qspec: Optional[QuantizationSpecBase] = None

    # Flag to control implicit sharing of observers for nodes observing the same tensor
    allow_implicit_sharing: bool = True

    # Boolean indicating if the node is annotated or not
    _annotated: bool = False


class Quantizer(ABC):
    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Allows for user defined transforms to run before annotating the graph.
        This allows quantizer to allow quantizing part of the model that are otherwise not quantizable.
        For example quantizer can
        a) decompose a compound operator like scaled dot product attention,
        into bmm and softmax if quantizer knows how to quantize bmm/softmax but not sdpa
        or b) transform scalars to tensor to allow quantizing scalars.

        Note: this is an optional method
        """
        return model

    @abstractmethod
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Annotates nodes in the graph with observer or fake quant constructors
        to convey the desired way of quantization
        """
        pass

    @abstractmethod
    def validate(self, model: torch.fx.GraphModule) -> None:
        """Validates that the annotated graph is supported by the backend"""
        pass
```
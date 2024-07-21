# `.\pytorch\torch\ao\quantization\quantizer\embedding_quantizer.py`

```py
# mypy: allow-untyped-defs
# 引入未类型化的函数定义允许声明
from __future__ import annotations

# 引入标准库和第三方库
import copy
from typing import List, Set

# 引入PyTorch相关模块
import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OperatorConfig,
    OperatorPatternType,
    QuantizationConfig,
)

# 模块的公开接口列表
__all__ = [
    "get_embedding_operators_config",
    "EmbeddingQuantizer",
]

# 返回嵌入操作配置的函数
def get_embedding_operators_config() -> OperatorConfig:
    # 权重量化的规范
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,  # 使用无符号8位整数作为数据类型
        qscheme=torch.per_channel_affine_float_qparams,  # 使用每通道浮点参数的量化方案
        ch_axis=0,  # 量化通道的轴为0
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(eps=2**-12),  # 设置每通道最小-最大观察器
    )
    quantization_config = QuantizationConfig(None, None, weight_quantization_spec, None)
    ops: List[OperatorPatternType] = [[torch.nn.Embedding]]  # 嵌入操作列表
    ops.append([F.embedding])  # 添加F.embedding到操作列表
    supported_config_and_operators = OperatorConfig(
        config=quantization_config, operators=ops  # 创建操作配置对象
    )
    return copy.deepcopy(supported_config_and_operators)  # 返回深拷贝的配置对象


# 嵌入量化器类，继承自Quantizer类
class EmbeddingQuantizer(Quantizer):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法

    # 获取支持的量化配置列表的类方法
    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        op_configs: Set[QuantizationConfig] = {
            spec for spec, _ in cls.get_supported_operators()  # 使用支持的操作集合初始化量化配置
        }
        return list(op_configs)  # 返回量化配置列表

    # 根据量化配置获取支持的操作列表的类方法
    @classmethod
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: QuantizationConfig
    ) -> List[OperatorPatternType]:
        for config, ops in cls.get_supported_operators():  # 遍历支持的操作配置和操作列表
            # 注意：这假设cls.supported_spec_and_operators中的每个条目
            # 对应于一个规范，例如，我们没有
            # [(spec1, op_list1), (spec1, op_list2), (spec2, op_list3)]
            # 其中第一个和第二个条目具有相同的规范但未合并op列表
            if config == quantization_config:  # 如果找到匹配的配置
                return ops  # 返回操作列表
        return []  # 如果找不到匹配的配置，返回空列表

    # 对模型进行注释的方法
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        self._annotate_embedding_ops(model.graph)  # 对嵌入操作进行注释处理
        return model  # 返回注释后的模型对象
    # 为图中的嵌入操作节点添加注释
    def _annotate_embedding_ops(self, graph: torch.fx.Graph) -> None:
        # 获取嵌入操作的配置信息
        embedding_config: OperatorConfig = get_embedding_operators_config()
        # 遍历图中的每个节点
        for node in graph.nodes:
            # 仅保留基于节点解析的注释，而不使用模块分区器
            # 这里展示了另一种注释方式的示例
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.embedding.default
            ):
                # 如果嵌入配置中的权重为None，则抛出值错误异常
                if embedding_config.config.weight is None:
                    raise ValueError(
                        "Embedding config must have a valid weight quantization spec."
                    )
                # 为节点的元数据添加量化注释
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={
                        node.args[0]: embedding_config.config.weight,
                    }
                )

    # 用于验证模型，但在当前情况下未实现功能
    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    # 类方法，返回支持的操作配置列表
    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return [get_embedding_operators_config()]
```
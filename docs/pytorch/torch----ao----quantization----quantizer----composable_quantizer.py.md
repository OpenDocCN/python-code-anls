# `.\pytorch\torch\ao\quantization\quantizer\composable_quantizer.py`

```py
# 从未来模块中导入注解支持
from __future__ import annotations

# 导入类型提示
from typing import Dict, List, TYPE_CHECKING

# 导入量化器相关模块
from .quantizer import QuantizationAnnotation, Quantizer

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 torch 模块
    import torch
    # 导入 torch.fx 模块中的 Node 类
    from torch.fx import Node

# 声明模块中公开的类
__all__ = [
    "ComposableQuantizer",
]

# 定义一个可组合的量化器类，继承自 Quantizer 类
class ComposableQuantizer(Quantizer):
    """
    ComposableQuantizer 允许用户将多个量化器组合成单个量化器。
    这使用户能够使用多个量化器对模型进行量化。例如，嵌入量化可以由一个量化器支持，
    而线性层和其他操作可能由另一个量化器支持。

    ComposableQuantizer 初始化时接收一个 Quantizer 实例的列表。
    组合的顺序很重要，因为这是量化器将应用的顺序。
    示例：
    ```
    embedding_quantizer = EmbeddingQuantizer()
    linear_quantizer = MyLinearQuantizer()
    xnnpack_quantizer = XNNPackQuantizer() # to handle ops not quantized by previous two quantizers
    composed_quantizer = ComposableQuantizer([embedding_quantizer, linear_quantizer, xnnpack_quantizer])
    prepared_m = prepare_pt2e(model, composed_quantizer)
    ```py
    """

    def __init__(self, quantizers: List[Quantizer]):
        super().__init__()
        # 初始化成员变量，存储传入的量化器列表
        self.quantizers = quantizers
        # 初始化图形注解字典
        self._graph_annotations: Dict[Node, QuantizationAnnotation] = {}

    def _record_and_validate_annotations(
        self, gm: torch.fx.GraphModule, quantizer: Quantizer
    ) -> None:
        # 遍历图模块中的每个节点
        for n in gm.graph.nodes:
            # 如果节点的元数据中包含量化注解
            if "quantization_annotation" in n.meta:
                # 检查注解是否已被更改，通过比较 QuantizationAnnotation 对象的 id
                if n in self._graph_annotations and (
                    id(self._graph_annotations[n])
                    != id(n.meta["quantization_annotation"])
                ):
                    # 抛出运行时错误，指示量化器已更改节点的注解
                    raise RuntimeError(
                        f"Quantizer {quantizer.__class__.__name__} has changed annotations on node {n}"
                    )
                else:
                    # 将节点的注解记录到图形注解字典中
                    self._graph_annotations[n] = n.meta["quantization_annotation"]
            else:
                # 如果节点在图形注解字典中存在，但在节点的元数据中没有找到量化注解，则抛出运行时错误
                if n in self._graph_annotations:
                    raise RuntimeError(
                        f"Quantizer {quantizer.__class__.__name__} has removed annotations on node {n}"
                    )

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """处理全局规格"""
        # 遍历量化器列表中的每个量化器，并对模型进行注解
        for quantizer in self.quantizers:
            quantizer.annotate(model)
            # 记录并验证每个量化器的注解
            self._record_and_validate_annotations(model, quantizer)
        return model

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        # 遍历量化器列表中的每个量化器，并对模型进行转换以进行注解
        for quantizer in self.quantizers:
            model = quantizer.transform_for_annotation(model)
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        # 暂时不执行验证逻辑
        pass
```
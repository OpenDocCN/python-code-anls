# `.\YOLO-World\yolo_world\models\backbones\__init__.py`

```py
# 版权声明，版权归腾讯公司所有
# YOLO 多模态骨干网络（视觉语言）
# 视觉部分：YOLOv8 CSPDarknet
# 语言部分：CLIP 文本编码器（12层transformer）
# 导入多模态骨干网络相关模块
from .mm_backbone import (
    MultiModalYOLOBackbone,
    HuggingVisionBackbone,
    HuggingCLIPLanguageBackbone,
    PseudoLanguageBackbone)

# 导出的模块列表
__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'PseudoLanguageBackbone'
]
```
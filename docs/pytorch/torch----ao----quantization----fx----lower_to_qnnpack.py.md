# `.\pytorch\torch\ao\quantization\fx\lower_to_qnnpack.py`

```
from ._lower_to_native_backend import _lower_to_native_backend
from ..qconfig import QConfigAny
from torch.fx import GraphModule
from typing import Dict, Tuple

__all__ = [
    "lower_to_qnnpack"
]

def lower_to_qnnpack(
    model: GraphModule,
    qconfig_map: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]]
) -> GraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to qnnpack
    """
    # 调用内部函数 _lower_to_native_backend，将给定的量化参考模型转换到 qnnpack 后端
    return _lower_to_native_backend(model, qconfig_map, node_name_to_scope)
```
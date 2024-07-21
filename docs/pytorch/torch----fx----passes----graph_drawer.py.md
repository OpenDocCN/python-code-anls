# `.\pytorch\torch\fx\passes\graph_drawer.py`

```py
# mypy: allow-untyped-defs

# 导入哈希函数和一些类型定义工具
import hashlib
from itertools import chain
from typing import Any, Dict, Optional, TYPE_CHECKING

# 导入PyTorch及其相关模块
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import _parse_stack_trace
from torch.fx.node import _format_arg, _get_qualified_name
from torch.fx.passes.shape_prop import TensorMetadata

# 尝试导入pydot模块，如果不存在则标记为无法使用
try:
    import pydot
    HAS_PYDOT = True
except ModuleNotFoundError:
    HAS_PYDOT = False
    pydot = None

# 公开的类和函数列表
__all__ = ["FxGraphDrawer"]

# 不同类型节点对应的颜色映射
_COLOR_MAP = {
    "placeholder": '"AliceBlue"',
    "call_module": "LemonChiffon1",
    "get_param": "Yellow2",
    "get_attr": "LightGrey",
    "output": "PowderBlue",
}

# 节点哈希值对应的颜色列表
_HASH_COLOR_MAP = [
    "CadetBlue1",
    "Coral",
    "DarkOliveGreen1",
    "DarkSeaGreen1",
    "GhostWhite",
    "Khaki1",
    "LavenderBlush1",
    "LightSkyBlue",
    "MistyRose1",
    "MistyRose2",
    "PaleTurquoise2",
    "PeachPuff1",
    "Salmon",
    "Thistle1",
    "Thistle3",
    "Wheat1",
]

# 节点权重的模板样式
_WEIGHT_TEMPLATE = {
    "fillcolor": "Salmon",
    "style": '"filled,rounded"',
    "fontcolor": "#000000",
}

# 如果pydot可用，则定义兼容性修饰器
if HAS_PYDOT:
    @compatibility(is_backward_compatible=False)
# 如果pydot不可用且不是类型检查环境，则定义FxGraphDrawer类并抛出运行时错误
else:
    if not TYPE_CHECKING:
        @compatibility(is_backward_compatible=False)
        class FxGraphDrawer:
            def __init__(
                self,
                graph_module: torch.fx.GraphModule,
                name: str,
                ignore_getattr: bool = False,
                ignore_parameters_and_buffers: bool = False,
                skip_node_names_in_args: bool = True,
                parse_stack_trace: bool = False,
                dot_graph_shape: Optional[str] = None,
            ):
                raise RuntimeError('FXGraphDrawer requires the pydot package to be installed. Please install '
                                   'pydot through your favorite Python package manager.')
```
# `.\pytorch\torch\fx\__init__.pyi`

```
# 从 torch.fx._symbolic_trace 模块导入以下函数和类：
# symbolic_trace - 用于符号化跟踪的函数
# Tracer - 符号化跟踪器的类
# wrap - 封装器函数
from torch.fx._symbolic_trace import (
    symbolic_trace as symbolic_trace,
    Tracer as Tracer,
    wrap as wrap,
)

# 从 torch.fx.graph 模块导入以下类：
# Graph - 图的类
from torch.fx.graph import Graph as Graph

# 从 torch.fx.graph_module 模块导入以下类：
# GraphModule - 图模块的类
from torch.fx.graph_module import GraphModule as GraphModule

# 从 torch.fx.interpreter 模块导入以下类：
# Interpreter - 解释器的类
# Transformer - 转换器的类
from torch.fx.interpreter import Interpreter as Interpreter, Transformer as Transformer

# 从 torch.fx.node 模块导入以下函数和类：
# has_side_effect - 判断节点是否有副作用的函数
# map_arg - 对节点参数进行映射的函数
# Node - 节点的类
from torch.fx.node import (
    has_side_effect as has_side_effect,
    map_arg as map_arg,
    Node as Node,
)

# 从 torch.fx.proxy 模块导入以下类：
# Proxy - 代理的类
from torch.fx.proxy import Proxy as Proxy

# 从 torch.fx.subgraph_rewriter 模块导入以下函数：
# replace_pattern - 替换模式的函数
from torch.fx.subgraph_rewriter import replace_pattern as replace_pattern
```
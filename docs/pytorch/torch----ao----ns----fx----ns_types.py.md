# `.\pytorch\torch\ao\ns\fx\ns_types.py`

```
# 导入枚举模块 enum
import enum
# 导入命名元组模块 NamedTuple
from typing import NamedTuple
# 导入 torch.fx.graph 模块中的 Node 类
from torch.fx.graph import Node
# 导入类型提示模块 Dict, Any, List, Union, Callable
from typing import Dict, Any, List, Union, Callable

# 定义枚举 NSSingleResultValuesType，表示单个结果的类型
class NSSingleResultValuesType(str, enum.Enum):
    WEIGHT = 'weight'
    NODE_OUTPUT = 'node_output'
    NODE_INPUT = 'node_input'

# 定义命名元组 NSSubgraph，表示子图的结构
class NSSubgraph(NamedTuple):
    start_node: Node  # 开始节点
    end_node: Node    # 结束节点
    base_op_node: Node  # 基本操作节点

# TODO(future PR): see if we can use typing_extensions's TypedDict instead
# to properly type the various keys

# 定义 NSSingleResultType，表示单个结果的数据结构，是一个字典
# 该字典包含多个键值对，描述了单个结果的详细信息
NSSingleResultType = Dict[str, Any]

# 定义 NSResultsType，表示结果集的数据结构，是一个嵌套的字典
# 该字典结构描述了不同子图下不同结果类型的数据，按模型名称组织
NSResultsType = Dict[str, Dict[str, Dict[str, List[NSSingleResultType]]]]

# 定义 NSNodeTargetType，表示节点的底层目标类型
# 可以是函数或字符串，用于描述节点的底层功能
NSNodeTargetType = Union[Callable, str]
```
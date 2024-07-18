# `.\graphrag\graphrag\index\utils\topological_sort.py`

```py
# 引入图形库中的拓扑排序器模块
from graphlib import TopologicalSorter

# 定义拓扑排序函数，接受一个字典类型的图作为参数，返回排序后的节点列表
def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Topological sort."""
    # 使用图形库中的拓扑排序器，传入图作为参数
    ts = TopologicalSorter(graph)
    # 调用拓扑排序器的静态顺序方法，并将结果转换为列表返回
    return list(ts.static_order())
```
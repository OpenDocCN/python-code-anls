# `.\pytorch\torch\ao\quantization\pt2e\generate_numeric_debug_handle.py`

```py
# 导入所需模块中的特定类和函数
from torch.fx import GraphModule, Node

# 定义可以导出的函数名
__all__ = ["generate_numeric_debug_handle"]

# 定义生成数值调试句柄的函数，接受一个图模块作为参数
def generate_numeric_debug_handle(graph_module: GraphModule) -> None:
    # 初始化唯一 ID
    unique_id = 0
    # 遍历图中的每一个节点
    for node in graph_module.graph.nodes:
        # 检查节点操作是否为函数调用
        if node.op == "call_function":
            # 为节点的元数据添加数值调试句柄字典
            node.meta["numeric_debug_handle"] = {}
            # 遍历节点的参数
            for arg in node.args:
                # 如果参数是节点对象，则将其加入到数值调试句柄字典中，并分配唯一 ID
                if isinstance(arg, Node):
                    node.meta["numeric_debug_handle"][arg] = unique_id
                    unique_id += 1

            # 将节点的输出加入数值调试句柄字典中，并分配唯一 ID
            node.meta["numeric_debug_handle"]["output"] = unique_id
            unique_id += 1
```
# `.\pytorch\torch\_export\passes\remove_runtime_assertions.py`

```py
# 添加类型提示 `mypy: allow-untyped-defs`，允许在不明确类型的情况下定义函数
import torch  # 导入 PyTorch 库
from torch.fx.passes.infra.pass_base import PassBase, PassResult  # 导入 PassBase 和 PassResult 类


class _RemoveRuntimeAssertionsPass(PassBase):
    """
    Remove runtime assertions inserted by the
    _AddRuntimeAssertionsForInlineConstraintsPass.
    """
    
    def call(self, graph_module) -> PassResult:
        # 定义一个标志，表示是否修改了图模块
        modified = False
        # 遍历图模块中的每一个模块
        for module in graph_module.modules():
            # 如果模块不是 torch.fx.GraphModule 类型，则跳过
            if not isinstance(module, torch.fx.GraphModule):
                continue
            # 遍历模块中的每一个节点
            for node in module.graph.nodes:
                # 如果节点的目标是 torch.ops.aten._assert_async.msg
                if node.target == torch.ops.aten._assert_async.msg:
                    # 找到断言异步节点
                    assert_async_node = node
                    # 如果断言异步节点有用户，则继续下一个节点
                    if len(assert_async_node.users) > 0:
                        continue
                    # 从图中删除断言异步节点
                    module.graph.erase_node(assert_async_node)
                    # 标记为修改了图模块
                    modified = True
        # 返回修改后的图模块和是否有修改的标志
        return PassResult(graph_module, modified)
```
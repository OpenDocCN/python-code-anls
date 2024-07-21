# `.\pytorch\torch\_export\passes\_node_metadata_hook.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块
import contextlib

# 导入 PyTorch 库
import torch
from torch.fx.graph_module import GraphModule

# 用于表示空的神经网络模块堆栈的键
_EMPTY_NN_MODULE_STACK_KEY = "_empty_nn_module_stack_from_metadata_hook"


def _node_metadata_hook(node: torch.fx.Node, stack_trace: str) -> None:
    """
    用于在创建节点时向节点添加适当的元数据的钩子函数，通过 graph.create_node 进行使用。
    一个使用示例:

    ```
    with _set_node_metadata_hook(gm,
        functools.partial(_node_metadata_hook, stack_trace="file")
    ):
        pass(gm)
    ```py

    该钩子函数仅适用于特定情况 -- 假设被添加的节点只是 call_function 节点，并复制第一个参数节点的 nn_module_stack。
    """
    assert node.op == "call_function" and callable(node.target)

    # 收集所有参数节点的元数据
    arg_meta = [arg.meta for arg in node.args if isinstance(arg, torch.fx.Node)]
    assert len(arg_meta) >= 1
    arg_meta = arg_meta[0]

    # 根据节点的目标调用情况，设置节点的值和元数据
    if (
        isinstance(node.target, torch._ops.OpOverload)
        and len(node.target._schema.returns) == 0
    ):
        node.meta["val"] = None
    else:
        fake_args = [
            arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg
            for arg in node.args
        ]
        fake_res = node.target(*fake_args)
        node.meta["val"] = fake_res

    # 设置节点的堆栈跟踪信息、神经网络模块堆栈和 Torch 函数信息
    node.meta["stack_trace"] = stack_trace
    node.meta["nn_module_stack"] = arg_meta.get(
        "nn_module_stack",
        {
            _EMPTY_NN_MODULE_STACK_KEY: (
                _EMPTY_NN_MODULE_STACK_KEY,
                _EMPTY_NN_MODULE_STACK_KEY,
            )
        },
    )
    node.meta["torch_fn"] = (
        f"{node.target.__name__}_0",
        f"{node.target.__class__.__name__}.{node.target.__name__}",
    )


@contextlib.contextmanager
def _set_node_metadata_hook(gm: torch.fx.GraphModule, f):
    """
    接受一个可调用对象作为参数，该对象在创建新节点后被调用。该对象接受新创建的节点作为输入并返回 None。
    """
    assert callable(f), "node_metadata_hook must be a callable."

    # 将钩子函数注册到所有子模块中
    for m in gm.modules():
        if isinstance(m, GraphModule):
            m._register_create_node_hook(f)
    try:
        yield
    finally:
        # 恢复所有子模块的钩子函数
        for m in gm.modules():
            if isinstance(m, GraphModule):
                m._unregister_create_node_hook(f)
```
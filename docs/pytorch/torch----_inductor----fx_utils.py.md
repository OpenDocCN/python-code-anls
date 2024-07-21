# `.\pytorch\torch\_inductor\fx_utils.py`

```
# mypy: allow-untyped-defs
# 引入操作符模块
import operator
# 引入默认字典模块
from collections import defaultdict
# 引入类型提示相关模块
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Type

# 引入 sympy 符号计算库
import sympy

# 引入 PyTorch 主模块及其相关子模块
import torch
import torch.fx
# 从 torch.fx.experimental.symbolic_shapes 中引入多个函数
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    rebind_unbacked,
    statically_known_true,
    sym_eq,
)
# 从 torch.utils 中引入 _pytree 模块
from torch.utils import _pytree as pytree
# 从 torch.utils._pytree 中引入 tree_map 函数
from torch.utils._pytree import tree_map
# 从当前包中的 virtualized 模块中引入 V 类
from .virtualized import V


# 检查模式匹配函数：(nn.module, F.function/torch.Tensor.method) 匹配。
# 适用于长度为 2 的模式，包含一个模块和一个函数/方法。
def matches_module_function_pattern(
    pattern: Tuple[Type[torch.nn.modules.Module], Callable[..., Any]],
    node: torch.fx.node.Node,
    modules: Dict[str, torch.nn.modules.Module],
) -> bool:
    # 如果节点的参数数量为 0，则返回 False
    if len(node.args) == 0:
        return False
    # 如果节点的第一个参数不是 torch.fx.Node 的实例，或者 node 本身不是 torch.fx.Node 的实例，则返回 False
    if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
        node, torch.fx.Node
    ):
        return False
    # 第一个节点是 call_module
    if node.args[0].op != "call_module":
        return False
    # 如果 node.args[0].target 不是字符串，则返回 False
    if not isinstance(node.args[0].target, str):
        return False
    # 如果 node.args[0].target 不在 modules 字典中，则返回 False
    if node.args[0].target not in modules:
        return False
    # 如果 modules[node.args[0].target] 的类型不是 pattern[0] 所指定的类型，则返回 False
    if type(modules[node.args[0].target]) is not pattern[0]:
        return False
    # 第二个节点是 call_function 或者 call_method
    if node.op != "call_function" and node.op != "call_method":
        return False
    # 如果 node.target 不等于 pattern[1] 所指定的函数或方法，则返回 False
    if node.target != pattern[1]:
        return False
    # 确保 node.args[0] 的输出只被当前节点使用
    if len(node.args[0].users) > 1:
        return False
    # 符合所有条件，返回 True
    return True


class FakeTensorUpdater:
    """
    这里的主要思想是，在我们转换图形时，很难为每个节点准确地维护虚拟张量（我们的主要元数据）。

    最可靠的方法是通过重新运行虚拟张量传播来获取此信息。但是，一般来说，虚拟张量传播成本相对较高。
    因此，我们希望仅在发生更改的节点上重新运行虚拟张量传播。

    为了检测哪些节点发生了更改，我们首先对其节点、目标和参数列表（在 FX 中是不可变的）进行哈希。

    然后，每当调用 incremental_update 时，我们检查哪些 FX 节点具有新的哈希，并重新计算该节点的虚拟张量元数据。
    然后，我们继续递归地计算所有用户的虚拟张量，直到虚拟张量不再更改。
    """

    def __init__(self, graph: torch.fx.Graph):
        # 存储已处理节点的哈希集合
        self.processed_hashes = set()
        # 存储图形对象
        self.graph = graph

        # 对于图中的每个节点，计算其哈希并加入到已处理哈希集合中
        for node in self.graph.nodes:
            self.processed_hashes.add(self.hash_node(node))

    def hash_node(self, node: torch.fx.Node):
        # todo(chilli): 不是一个很好的哈希函数
        # 返回节点、目标和参数列表的哈希值
        return (node, node.target, id(node.args), id(node.kwargs))


def get_storage(t: torch.Tensor) -> int:
    # 返回张量 t 的未类型化存储的 _cdata
    return t.untyped_storage()._cdata


def get_node_storage(node: torch.fx.Node) -> Optional[int]:
    # 检查节点的元数据中是否包含键名为 "val"，如果没有，则返回空
    if "val" not in node.meta:
        return None
    
    # 检查节点的元数据中名为 "val" 的值是否为 torch.Tensor 类型，如果不是，则返回空
    if not isinstance(node.meta["val"], torch.Tensor):
        return None
    
    # 使用 torch._C._has_storage() 函数检查节点的元数据中名为 "val" 的 torch.Tensor 对象是否具有存储空间，
    # 如果没有存储空间，则返回空
    if not torch._C._has_storage(node.meta["val"]):
        return None
    
    # 调用 get_storage() 函数，返回节点元数据中名为 "val" 的 torch.Tensor 对象的存储空间
    return get_storage(node.meta["val"])
def get_fake(x):
    # 如果 x 是 torch.fx.Node 类型的对象
    if isinstance(x, torch.fx.Node):
        # 如果 x 的 meta 字典中没有 "val" 键
        if "val" not in x.meta:
            return x  # 返回 x 本身
        # 否则返回 x 的 meta 字典中 "val" 键对应的值
        return x.meta["val"]
    return x  # 如果 x 不是 torch.fx.Node 类型的对象，则直接返回 x


def get_fake_args_kwargs(x: torch.fx.Node) -> Tuple[bool, Tuple[Any], Dict[str, Any]]:
    """
    First value returns a boolean if any of the input nodes don't have a faketensor.
    """
    # 对 x 的 args 和 kwargs 应用 get_fake 函数，得到替换后的 args 和 kwargs
    args, kwargs = tree_map(get_fake, (x.args, x.kwargs))
    # 如果 args 或 kwargs 中有任何一个元素是 torch.fx.Node 类型的对象
    if any(
        isinstance(a, torch.fx.Node) for a in pytree.arg_tree_leaves(*args, **kwargs)
    ):
        # 返回 False，表示有未经替换的节点
        return False, args, kwargs
    # 否则返回 True，表示所有节点都已经替换完成
    return True, args, kwargs


def is_node_realized(node: torch.fx.Node) -> bool:
    """Returns true if a node is always realized when lowered to inductor IR.

    NOTE: This may return some false negatives. e.g. it doesn't
    handle buffers realized heuristically during lowering, or
    buffers realized indirectly through view ops.
    """
    from torch._inductor.lowering import fallbacks, needs_realized_inputs

    # 判断节点是否为缓冲区（buffer）
    def is_buffer(node: torch.fx.Node) -> bool:
        # 如果节点的操作为 "call_function"，且目标函数为 operator.getitem
        if node.op == "call_function" and node.target is operator.getitem:
            # 对于有多个输出的节点，我们获得 fx 图：
            #     foo = torch.ops.aten.foo(...)
            #     getitem = foo[0]
            #     getitem_1 = foo[1]
            # 我们需要检查 foo 是否为一个回退（fallback）内核
            return is_buffer(node.args[0])  # type: ignore[arg-type]
        # 否则判断节点的操作是否为 "placeholder" 或 "output"，或者目标函数是否在 fallbacks 中
        return node.op in ("placeholder", "output") or node.target in fallbacks

    # 如果节点是缓冲区，则返回 True
    if is_buffer(node):
        return True

    # 判断节点是否实现了输入
    def realizes_inputs(node: torch.fx.Node) -> bool:
        return node.op == "output" or node.target in needs_realized_inputs

    # 如果节点的任何用户实现了输入，则返回 True
    if any(realizes_inputs(user) for user in node.users):
        return True

    # 否则假设节点未实现
    return False
```
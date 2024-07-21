# `.\pytorch\torch\_functorch\compile_utils.py`

```py
# mypy: ignore-errors

# 导入所需模块和类
from typing import Callable

import torch
import torch.fx as fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten

# 设置 torch 的 aten 命名空间为 aten，简化后续代码的调用
aten = torch.ops.aten

# 定义一个函数，接收一个 fx.Node 对象，返回一个可调用对象（Callable）
def get_aten_target(node: fx.Node) -> Callable:
    # 检查节点的 target 属性是否有 overloadpacket 属性，若有则返回其值，否则返回 target 本身
    if hasattr(node.target, "overloadpacket"):
        return node.target.overloadpacket
    return node.target

# 定义一个包含多个 torch 的 aten 命名空间内函数的列表
rand_ops = [
    aten.dropout,
    aten._fused_dropout,
    aten._standard_gamma,
    aten.bernoulli,
    aten.multinomial,
    aten.native_dropout,
    aten.normal,
    aten.poisson,
    aten.binomial,
    aten.rrelu,
    aten.rand_like,
    aten.rand,
    aten.randint,
    aten.randn,
    aten.randperm,
]

# 定义函数 fx_graph_cse，接收一个 torch.fx.graph.Graph 对象，并返回一个应用了 CSE 的新图
def fx_graph_cse(fx_g: torch.fx.graph.Graph):
    # 创建一个新的空白图对象
    new_graph = fx.Graph()
    # 创建一个空字典，用于将旧图中的节点映射到新图中的节点
    env = {}  # map from node in the old graph to node in the new graph
    # 创建一个空字典，用于将哈希值映射到新图中的节点
    hash_env = {}  # map from hash to a node in the new graph
    # 创建一个空字典，用于将哈希值映射到令牌
    token_map = {}  # map from hash to token
    # 遍历fx_g中的所有节点
    for n in fx_g.nodes:
        # 如果节点是占位符、输出、获取属性节点或者是随机操作，则直接复制到新图中，不做共享子表达式消除（CSE）
        if (
            n.op == "placeholder"
            or n.op == "output"
            or n.op == "get_attr"
            or get_aten_target(n) in rand_ops
        ):
            # 复制节点到新图中，并使用env字典中的映射
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else:  # 否则，节点的操作应为'call_function'，不应该出现'call_module'或'call_method'
            # 替换节点的args和kwargs成员，如果存在则使用env中的映射
            def substitute(arg_list):
                # 将args展平并获取规范化结构
                arg_list, spec = tree_flatten(arg_list)
                for i in range(len(arg_list)):
                    v = arg_list[i]
                    if isinstance(v, torch.fx.node.Node) and v in env:
                        arg_list[i] = env[v]
                    if isinstance(v, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                        arg_list[i] = v.node
                return tuple(arg_list), spec

            # 替换args和kwargs
            args, args_spec = substitute(n.args)
            kwargs, kwargs_spec = substitute(n.kwargs)

            # 每个令牌对应一个唯一节点
            # 具有相同令牌的节点可以被替换
            token = {
                "target": n.target,
                "args": args,
                "args_spec": args_spec,
                "kwargs": kwargs,
                "kwargs_spec": kwargs_spec,
            }

            # 对替换后的args进行哈希，但不对specs进行哈希，因为specs不可哈希
            # 需要将类型信息包含在哈希中，以避免如下情况的发生：
            # hash((primals_2, 1.0)) == hash((primals_2, 1))
            hash_arg = hash(
                (tuple((a, type(a)) for a in args), tuple((a, type(a)) for a in kwargs))
            )
            hash_val = (n.target, hash_arg)

            # 检查节点是否有替换并且可以被消除
            hash_val_in_hash_env = hash_val in hash_env
            if hash_val_in_hash_env and token_map[hash_val] == token:
                env[n] = hash_env[hash_val]
                continue

            # 复制节点到新图中，并使用env字典中的映射
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
            if not hash_val_in_hash_env:
                hash_env[hash_val] = new_node
                token_map[hash_val] = token

    # 返回更新后的新图
    return new_graph
# 修改图节点中的目标，用于去除重载（overloads）的影响
def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.

    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    # 遍历图中的每个节点
    for node in gm.graph.nodes:
        # 检查节点的目标是否为 torch._ops.OpOverload 类型
        if isinstance(node.target, torch._ops.OpOverload):
            # 如果是重载类型，则将节点的目标修改为重载包中的目标
            node.target = node.target.overloadpacket
    # 重新编译修改后的图模块
    gm.recompile()


# 查找并返回图中所有占位符节点
def get_placeholders(graph):
    return graph.find_nodes(op="placeholder")


# 获取图中的输出节点，并返回其中包含的所有子节点的叶子节点
def get_outputs(graph):
    for node in graph.find_nodes(op="output"):
        # 返回输出节点的第一个参数的所有叶子节点
        return pytree.tree_leaves(node.args[0])
    # 如果没有找到输出节点，则抛出断言错误
    raise AssertionError("No output node found")
```
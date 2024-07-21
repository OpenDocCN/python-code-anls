# `.\pytorch\torch\export\_remove_auto_functionalized_pass.py`

```py
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator  # 导入 operator 模块，用于操作符的函数形式
from typing import List  # 导入 List 类型提示，用于声明列表类型

import torch  # 导入 PyTorch 库
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,  # 导入 auto_functionalized 函数，用于函数式自动化
    get_mutable_arg_names,  # 导入 get_mutable_arg_names 函数，用于获取可变参数名
)
from torch.export import ExportedProgram  # 导入 ExportedProgram 类，用于表示导出的程序对象


def _remove_auto_functionalization_from_graph_helper(ep, auto_functionalize_nodes):
    # Update every use of the HOP
    for node in reversed(auto_functionalize_nodes):
        func = node.args[0]  # 获取自动功能化节点的第一个参数作为函数对象
        original_kwargs = node.kwargs  # 获取自动功能化节点的关键字参数

        assert isinstance(func, torch._ops.OpOverload)  # 断言 func 是 torch._ops.OpOverload 类型

        with ep.graph.inserting_before(node):
            # 在节点之前插入新节点，调用 func 函数，使用关键字参数
            new_node = ep.graph.call_function(func, kwargs=node.kwargs)

        for k, v in node.meta.items():
            new_node.meta[k] = v  # 将原节点的元数据复制到新节点中

        # Replace auto_functionalize(func, args) with just func(args)
        node.replace_all_uses_with(new_node)  # 替换所有使用自动功能化节点的地方为新节点

        mutable_args_names = get_mutable_arg_names(new_node.target)  # 获取新节点目标函数的可变参数名列表

        # update the users of the auto_func node (the getitem nodes)
        for user in list(new_node.users.keys()):
            assert user.target == operator.getitem  # 断言用户节点的目标为 operator.getitem

            # getitem corresponding to a mutated input, just replace all uses with the original input
            if user.args[1] >= len(func._schema.returns):
                assert user.args[1] <= len(func._schema.returns) + len(
                    mutable_args_names
                )

                # If the result of getitem was used in an output node, update the output spec with the correct name
                adjusted_index = user.args[1] - len(func._schema.returns)
                original_arg = original_kwargs[mutable_args_names[adjusted_index]]

                # Replace all uses of the user node with the original input argument
                user.replace_all_uses_with(original_arg)

        if len(func._schema.returns) == 1:
            # If the function has only one return value, replace all getitem nodes with the function result itself
            for user in list(new_node.users.keys()):
                if user.args[1] == 0:
                    user.replace_all_uses_with(new_node)

        new_node.meta["val"] = node.meta["val"][: len(func._schema.returns)]  # 更新新节点的元数据
        ep.graph.erase_node(node)  # 删除原节点

    ep.graph.eliminate_dead_code()  # 清除无用代码


def unsafe_remove_auto_functionalized_pass(
    ep: ExportedProgram,
) -> ExportedProgram:
    """
    This pass removes instances of the higher order op 'auto_functionalized',
    """
    """
    and modifies the calling EP inplace to have the original mutator op.
    This pass doesn't perform safety checks to make sure that this inplace mutation is safe.
    """
    # 创建一个空列表，用于存储所有需要自动功能化的节点
    auto_functionalize_nodes: List[torch.fx.Node] = []
    # 遍历图模块中的每个模块
    for module in ep.graph_module.modules():
        # 如果模块不是 torch.fx.GraphModule 类型，则跳过
        if not isinstance(module, torch.fx.GraphModule):
            continue
        # 遍历 EP 图中的每个节点
        for node in ep.graph.nodes:
            # 如果节点的操作是 "call_function" 并且目标是 auto_functionalized
            if node.op == "call_function" and node.target is auto_functionalized:
                # 将此节点添加到自动功能化节点列表中
                auto_functionalize_nodes.append(node)

    # 使用图模块中的替换钩子，以便在进行删除自动功能化过程时能够替换相关内容
    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        # 调用辅助函数，从图中删除自动功能化节点
        _remove_auto_functionalization_from_graph_helper(ep, auto_functionalize_nodes)

    # 返回修改后的 EP 对象
    return ep
```
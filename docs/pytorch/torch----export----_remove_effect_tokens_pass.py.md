# `.\pytorch\torch\export\_remove_effect_tokens_pass.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和类
import operator
from typing import List

import torch
from torch._higher_order_ops.effects import _get_schema, with_effects
from .exported_program import ExportedProgram
from .graph_signature import (
    CustomObjArgument,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TokenArgument,
)

# 定义一个辅助函数，用于从图中移除效果相关的令牌
def _remove_effect_tokens_from_graph_helper(
    ep, num_tokens, input_token_names, output_token_names
):
    # 获取图中输入到提升定制对象的映射关系
    inputs_to_lifted_custom_objs = ep.graph_signature.inputs_to_lifted_custom_objs

    # 初始化输出节点和具有效果的节点列表
    output_node = None
    with_effect_nodes: List[torch.fx.Node] = []

    # 查找顶层输出节点以验证其参数是否与输出令牌名匹配
    output_node = next(reversed(ep.graph_module.graph.find_nodes(op="output")))
    
    # 遍历导出程序的模块，查找具有效果的节点
    for module in ep.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        for node in module.graph.nodes:
            # 判断节点是否为调用函数且目标为 with_effects
            if not (node.op == "call_function" and node.target is with_effects):
                continue

            # 将具有效果的节点添加到列表中
            with_effect_nodes.append(node)

    # 从输出中移除令牌
    assert output_node is not None
    output_args = output_node.args[0]
    assert len(output_args) >= num_tokens
    out_token_nodes = output_args[:num_tokens]
    # 更新输出节点的参数，移除令牌节点
    output_node.args = (tuple(output_args[num_tokens:]),)
    for out_token in out_token_nodes:
        # 确保移除的令牌名称在输出令牌名列表中
        assert out_token.name in output_token_names
        # 清空令牌节点的用户使用列表，并从图中擦除节点
        out_token.users.clear()
        ep.graph.erase_node(out_token)

    # 将 with_effects(token, func, args) 替换为 func(args)
    # 对每个具有效果节点的逆序遍历
    for node in reversed(with_effect_nodes):
        # 获取节点的第二个参数，即函数对象
        func = node.args[1]
        # 断言函数对象是 OpOverload 或 HigherOrderOperator 类型
        assert isinstance(func, (torch._ops.OpOverload, torch._ops.HigherOrderOperator))

        # 如果函数是 torch.ops.higher_order.call_torchbind
        if func == torch.ops.higher_order.call_torchbind:
            # 获取自定义对象的元信息
            custom_obj_meta = node.args[2].meta["val"]
            assert isinstance(custom_obj_meta, CustomObjArgument)
            # 如果自定义对象元信息中有伪值，则使用伪值
            if custom_obj_meta.fake_val:
                custom_obj = custom_obj_meta.fake_val
            # 否则根据节点的第三个参数名查找对应的输入自定义对象
            elif node.args[2].name in inputs_to_lifted_custom_objs:
                custom_obj = ep.constants[
                    inputs_to_lifted_custom_objs[node.args[2].name]
                ]
            else:
                # 如果找不到自定义对象，则抛出运行时错误
                raise RuntimeError(f"Unable to find custom obj for node {node}")
            # 获取函数的 schema
            schema = _get_schema(func, (custom_obj,) + node.args[3:])
        else:
            # 对于其他函数，获取函数的 schema
            schema = _get_schema(func, node.args[2:])

        # 在节点之前插入新节点
        with ep.graph.inserting_before(node):
            new_node = ep.graph.call_function(func, node.args[2:])
        # 复制节点的元数据到新节点
        for k, v in node.meta.items():
            new_node.meta[k] = v

        # 用新节点替换所有旧节点的使用
        node.replace_all_uses_with(new_node)

        # 更新用户的 getitem 节点
        for user in list(new_node.users.keys()):
            assert user.target == operator.getitem
            # 如果用户的第二个参数为 0，则删除该节点
            if user.args[1] == 0:
                ep.graph.erase_node(user)

        # 如果函数的返回值只有一个，则直接用结果替换 getitem(with_effects, 1) 的调用
        if len(schema.returns) == 1:
            for user in list(new_node.users.keys()):
                assert user.args[1] == 1
                user.replace_all_uses_with(new_node)
            # 更新新节点的值为节点的第二个返回值
            new_node.meta["val"] = node.meta["val"][1]
        # 如果函数有多个返回值，则将所有的 getitem 调用下移一个位置
        elif len(schema.returns) > 1:
            for user in list(new_node.users.keys()):
                assert user.args[1] >= 1
                user.args = (user.args[0], user.args[1] - 1)
            # 更新新节点的值为节点的从第二个返回值开始的所有返回值
            new_node.meta["val"] = node.meta["val"][1:]
        else:
            # 断言函数没有返回值，且新节点没有用户使用
            assert len(schema.returns) == 0
            assert len(new_node.users) == 0
            # 更新新节点的值为空
            new_node.meta["val"] = None

        # 删除旧节点
        ep.graph.erase_node(node)

    # 移除输入中的 tokens
    placeholders = [node for node in ep.graph.nodes if node.op == "placeholder"]
    assert len(placeholders) >= num_tokens
    inp_token_nodes = placeholders[:num_tokens]
    for inp_token in inp_token_nodes:
        # 断言 token 名称在输入 token 名称中
        assert inp_token.name in input_token_names
        # 删除节点
        ep.graph.erase_node(inp_token)

    # 消除死代码
    ep.graph.eliminate_dead_code()
    # 使 graph_module.code 与图形一致
    ep.graph_module.recompile()
# 从导出的程序中移除所有的 token，包括：
# - 移除输入和输出的 token
# - 将带有 with_effects(token, func, args) 替换为 func(args)

# 这个函数对给定的 ExportedProgram 进行就地修改。

def _remove_effect_tokens(ep: ExportedProgram) -> ExportedProgram:
    # 初始化计数器和存储输入 token 相关信息的列表
    num_tokens: int = 0
    input_token_names: List[str] = []
    new_input_specs: List[InputSpec] = []

    # 遍历导出程序的输入规范
    for inp in ep.graph_signature.input_specs:
        if inp.kind == InputKind.TOKEN:
            num_tokens += 1
            assert isinstance(inp.arg, TokenArgument)
            input_token_names.append(inp.arg.name)
        else:
            new_input_specs.append(inp)

    # 初始化计数器和存储输出 token 相关信息的列表
    num_out_tokens: int = 0
    new_output_specs: List[OutputSpec] = []
    output_token_names: List[OutputSpec] = []

    # 遍历导出程序的输出规范
    for out in ep.graph_signature.output_specs:
        if out.kind == OutputKind.TOKEN:
            num_out_tokens += 1
            output_token_names.append(out.arg.name)
        else:
            new_output_specs.append(out)

    # 更新导出程序的图形签名
    ep.graph_signature.input_specs = new_input_specs
    ep.graph_signature.output_specs = new_output_specs

    # 断言输入 token 数量与输出 token 数量相等
    assert num_tokens == num_out_tokens

    # 使用图模块的替换钩子，移除图中的效果 token
    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        _remove_effect_tokens_from_graph_helper(
            ep, num_tokens, input_token_names, output_token_names
        )

    # 返回修改后的导出程序
    return ep
```
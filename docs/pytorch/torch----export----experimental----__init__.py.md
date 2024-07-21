# `.\pytorch\torch\export\experimental\__init__.py`

```
import copy  # 导入copy模块，用于对象的深拷贝操作

import torch  # 导入PyTorch库
from torch.export.exported_program import _decompose_exported_program  # 导入_exported_program模块中的_decompose_exported_program函数


def _remove_detach_pass(
    gm: torch.fx.GraphModule, sig: torch.export.graph_signature.ExportGraphSignature
) -> None:
    # 设置gm对象的替换钩子，使用sig对象提供的替换钩子
    with gm._set_replace_hook(sig.get_replace_hook()):
        # 反向遍历图中的节点列表
        for node in list(reversed(gm.graph.nodes)):
            # 如果节点的操作不是"call_function"，则继续下一个节点的处理
            if node.op != "call_function":
                continue
            # 如果节点的目标是torch.ops.aten.detach.default，并且仅有一个用户且该用户的目标也是torch.ops.aten.detach.default
            if (
                node.target == torch.ops.aten.detach.default
                and len(node.users) == 1
                and next(iter(node.users)).target == torch.ops.aten.detach.default
            ):
                # 将该节点的用户替换为当前节点
                next(iter(node.users)).replace_all_uses_with(node)

    # 删除无用的代码
    gm.graph.eliminate_dead_code()
    # 重新编译图模块
    gm.recompile()


def _export_forward_backward(
    ep: torch.export.ExportedProgram, joint_loss_index: int = 0
) -> torch.export.ExportedProgram:
    """
    WARNING: This API is highly unstable and will be subject to change in the future.
    """
    from torch._decomp import core_aten_decompositions  # 导入core_aten_decompositions函数

    # 使用_decompose_exported_program函数对ep进行分解操作
    ep = _decompose_exported_program(
        ep,
        decomp_table=core_aten_decompositions(),  # 使用core_aten_decompositions函数生成的分解表
        _preserve_ops=(),  # 忽略保留的操作，类型为ignore[arg-type]
        joint_loss_index=joint_loss_index,  # 关节损失的索引
    )
    # 对图模块进行深拷贝操作
    gm = copy.deepcopy(ep.graph_module)
    # 对图签名进行深拷贝操作
    new_graph_signature = copy.deepcopy(ep.graph_signature)
    # 调用_remove_detach_pass函数处理gm对象和new_graph_signature对象
    _remove_detach_pass(gm, new_graph_signature)

    # 更新ep对象，使用gm和new_graph_signature
    return ep._update(gm, new_graph_signature)
```
# `.\pytorch\torch\onnx\_internal\fx\passes\readability.py`

```py
# mypy: allow-untyped-defs
# 从未来导入注释允许未类型化的定义

from typing import Dict, List, Sequence, Tuple, Union
# 导入类型提示模块

import torch
# 导入 PyTorch 库

from torch.onnx._internal import _beartype
# 从 PyTorch 的 ONNX 内部模块中导入 _beartype

from torch.onnx._internal.fx import _pass, diagnostics
# 从 PyTorch 的 ONNX 内部 FX 模块中导入 _pass 和 diagnostics

class RestoreParameterAndBufferNames(_pass.Transform):
    """从原始 nn.module 恢复参数和缓冲区名称。

    这个 pass 对于导出的 ONNX 图的可读性很有用。它从原始 nn.module 恢复参数和缓冲区名称。
    例如，如果原始 nn.module 有一个名为 `root.linear.0.weight` 的参数，
    而 FX 将该参数重命名为 `_param_constant9`，这个 pass 将把它改回来。

    这个 pass 必须在 `Decompose` pass 之后运行。因为这个 pass 预期在 `proxy_tensor.make_fx` 生成的
    `fx.GraphModule` 上调用，其中所有参数和缓冲区都在根级别注册。
    """

    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        fx_module: torch.fx.GraphModule,
        original_nn_module: torch.nn.Module,
    ):
        super().__init__(diagnostic_context, fx_module)
        self.original_nn_module = original_nn_module
        # 初始化方法，设置原始 nn.module

    @_beartype.beartype
    def _rename_param_and_buffer(
        self,
        diagnostic: diagnostics.Diagnostic,
        nodes: Sequence[torch.fx.Node],
        new_name: str,
    ) -> None:
        """重命名参数/缓冲区并用更新的目标替换相应节点。"""
        assert len(nodes) > 0, "`nodes` 不能为空"
        assert (
            len({node.target for node in nodes}) == 1
        ), "`nodes` 必须具有相同的 `target`"
        old_name = nodes[0].target
        assert isinstance(old_name, str), f"预期 str 类型，得到类型({old_name})"
        # 参数/缓冲区名称不能包含 "."
        normalized_name = new_name.replace(".", "/")
        attr_value = getattr(self.module, old_name)
        setattr(self.module, normalized_name, attr_value)
        delattr(self.module, old_name)
        for node in nodes:
            with self.module.graph.inserting_before(node):
                new_node = self.module.graph.get_attr(normalized_name)
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                self.module.graph.erase_node(node)
        diagnostic.info(
            "将 'self.%s' 重命名为 'self.%s'，从原始参数名称 '%s' 规范化。",
            old_name,
            normalized_name,
            new_name,
        )
```
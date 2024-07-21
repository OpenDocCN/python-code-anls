# `.\pytorch\torch\onnx\_internal\fx\passes\decomp.py`

```py
# mypy: allow-untyped-defs
# 导入允许未类型化定义的标志
from __future__ import annotations

# 引入上下文管理工具
import contextlib

# 引入类型提示
from typing import Callable, Mapping, Optional

# 引入 PyTorch 库
import torch
import torch._ops
import torch.fx

# 从 torch._dispatch 模块导入 python 别名
from torch._dispatch import python as python_dispatch

# 从 torch._subclasses 模块导入 fake_tensor
from torch._subclasses import fake_tensor

# 从 torch.fx.experimental 模块导入 proxy_tensor
from torch.fx.experimental import proxy_tensor

# 从 torch.onnx._internal 模块导入 _beartype
from torch.onnx._internal import _beartype

# 从 torch.onnx._internal.fx 模块导入 _pass 和 diagnostics
from torch.onnx._internal.fx import _pass, diagnostics

# 从 torch.onnx._internal.fx.passes 模块导入 _utils
from torch.onnx._internal.fx.passes import _utils


class Decompose(_pass.Transform):
    # 定义 Decompose 类，继承自 _pass.Transform
    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        decomposition_table: Mapping[torch._ops.OpOverload, Callable],
        enable_dynamic_axes: bool,
        allow_fake_constant: Optional[bool] = False,
    ):
        # 初始化方法，接受诊断上下文、图模块、分解表、动态轴开关和允许虚假常量的参数
        super().__init__(diagnostic_context, module)
        self.decomposition_table = decomposition_table
        self.enable_dynamic_axes = enable_dynamic_axes
        self.allow_fake_constant = allow_fake_constant

    @_beartype.beartype
    # 应用 beartype 装饰器的方法
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        # 禁止使用 kwargs，因为在 Decompose 中不支持
        assert not kwargs, "kwargs is not supported in Decompose."

        # 用于保留 `make_fx` 后的堆栈跟踪信息
        module = _utils.wrap_graph_module_for_node_meta_preservation(self.module)

        # fake 模式使用静态大小跟踪张量的大小。symbolic 模式则生成 aten::sym_size 来动态跟踪张量的大小。

        # 例如，fake 模式：
        #  view: f32[3, 5, 20] = torch.ops.aten.view.default(x, [3, 5, 20])

        # 例如，symbolic 模式：
        #  sym_size = torch.ops.aten.sym_size(x, 0)
        #  sym_size_1 = torch.ops.aten.sym_size(x, 1)
        #  sym_size_2 = torch.ops.aten.sym_size(x, 2)
        #  sym_size_3 = torch.ops.aten.sym_size(x, 3)
        #  mul = sym_size_2 * sym_size_3;  sym_size_2 = sym_size_3 = None
        #  view: f32[3, 5, 20] = torch.ops.aten.view.default(x, [sym_size, sym_size_1, mul])

        # 模仿 `torch._dynamo.export(aten_graph=True)` 调用 `make_fx` 的行为。
        # TODO: 可能需要重新审视用户 fake 模式导出 + 动态形状场景。
        fake_mode: Optional[fake_tensor.FakeTensorMode] = self.fake_mode
        maybe_fake_args = self._maybe_fakefy_args(fake_mode, *args)
        if fake_mode is not None:
            # 使用现有的 fake 模式作为上下文，告知 `make_fx` 不需要通过将 tracing_mode 设置为 "real" 创建新的 fake 模式。
            tracing_mode = "real"
        else:
            # 找不到现有的 fake 模式，告知 `make_fx` 创建一个新的 fake 模式。
            fake_mode = contextlib.nullcontext()  # type: ignore[assignment]
            tracing_mode = "symbolic" if self.enable_dynamic_axes else "fake"

        # 将分解表应用于输入图。
        assert fake_mode is not None  # for mypy
        with proxy_tensor.maybe_disable_fake_tensor_mode(), python_dispatch.enable_python_dispatcher(), (
            fake_mode
        ):
            decomposed_module = proxy_tensor.make_fx(
                module,
                decomposition_table=self.decomposition_table,
                tracing_mode=tracing_mode,
                _allow_non_fake_inputs=True,
                _allow_fake_constant=self.allow_fake_constant,
            )(*maybe_fake_args)

        # 重命名占位符目标，以匹配原始模块的签名，因为我们不希望将 forward(x, y, z) 映射到 forward(arg0, arg1, arg2)。
        _utils.replace_placeholder_name_and_target(decomposed_module, self.module)

        # 返回分解后的模块
        return decomposed_module
```
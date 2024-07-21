# `.\pytorch\torch\onnx\_internal\fx\passes\functionalization.py`

```
# mypy: allow-untyped-defs
# 从 __future__ 模块导入 annotations 功能，允许使用类型注解

import contextlib  # 导入 contextlib 模块，用于创建和管理上下文管理器

from typing import Callable, Optional  # 导入类型提示模块，包括 Callable 和 Optional 类型

import torch  # 导入 PyTorch 模块
import torch._ops  # 导入 PyTorch 内部操作模块
import torch.func  # 导入 PyTorch func 模块，功能不明确
import torch.fx  # 导入 PyTorch fx 模块，用于特效处理
from torch._subclasses import fake_tensor  # 从 PyTorch _subclasses 模块导入 fake_tensor
from torch.fx.experimental import proxy_tensor  # 从 PyTorch fx.experimental 模块导入 proxy_tensor
from torch.onnx._internal import _beartype  # 从 PyTorch onnx._internal 模块导入 _beartype
from torch.onnx._internal.fx import _pass, diagnostics  # 从 PyTorch onnx._internal.fx 模块导入 _pass 和 diagnostics
from torch.onnx._internal.fx.passes import _utils  # 从 PyTorch onnx._internal.fx.passes 模块导入 _utils
from torch.utils import _pytree as pytree  # 从 PyTorch utils 模块导入 _pytree 别名为 pytree

class Functionalize(_pass.Transform):
    """Functionalize a GraphModule.

    This pass utilizes ``functionalization`` utility of ``torch._functorch`` to convert
    a GraphModule into a functional form. The two main functionalities are (copied from
    its documentations):

    * ``functionalization`` removes (intermediate) mutations and aliasing from a
    function, while preserving the function's semantics.

    * ``functionalization`` also removes mutations (and views) that were performed
    on function inputs. However to preserve semantics, functionalize will "fix up" the
    mutations after the transform has finished running, by detecting if any tensor inputs
    "should have" been mutated, and copying the new data back to the inputs if necessary.
    For example, consider::

        def fn(a, b):
            a.add_(b)
            return a

      For a call like `fn(x, y)`, the variable `x` outside is also mutated. Hence just
      functionalizing is not enough for preserving the original semantics. A "special"
      input mutation step needs to be inserted at the end.::

        # After functionalization, without input mutation "fix up".
        # This is not semantically the same. The variable outside the function call that
        # was passed in as `a` is not mutated.
        def fn(a, b):
            new_a = a + b
            return new_a

        # Functionalization with input mutation "fix up" that preserves semantics.
        def fn(a, b):
            new_a = a + b

            # Copying the new data back to the inputs
            a.copy_(new_a)

            return new_a

    For ONNX inference, it is recommended to run ``RemoveInputMutation`` after this pass.
    ``RemoveInputMutation`` removes the "fix up" nodes that were added by ``Functionalize``,
    which are not needed for ONNX inference.
    """

    @_beartype.beartype
    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        enable_dynamic_axes: bool,
        allow_fake_constant: Optional[bool] = False,
    ):
        # 调用父类的构造函数，初始化 Functionalize 实例
        super().__init__(diagnostic_context, module)
        # 设置是否启用动态轴
        self.enable_dynamic_axes = enable_dynamic_axes
        # 是否允许使用虚假常数
        self.allow_fake_constant = allow_fake_constant
    def _functionalize(self, function: Callable) -> Callable:
        # 解决使用 `torch.func.functionalize` 与 `make_fx` 时的分发器问题
        # 参考：https://github.com/pytorch/pytorch/issues/99774#issuecomment-1527949391
        
        # 定义一个包装函数，接受任意数量的输入
        def wrapped(*inputs):
            # 将所有输入中的 torch.Tensor 转换为 functional tensor
            inputs_functional = pytree.tree_map_only(
                torch.Tensor, torch._to_functional_tensor, inputs
            )
            # 启用 functionalization 模式，确保视图重新应用
            torch._enable_functionalization(reapply_views=True)
            try:
                # 执行传入的 function，并传入 functional tensors 作为参数
                out = function(*inputs_functional)
            finally:
                # 禁用 functionalization 模式
                torch._disable_functionalization()
            
            # 将输入展平为列表
            flat_inputs = pytree.tree_leaves(inputs)
            # 将 functional tensors 也展平为列表
            flat_inputs_functional = pytree.tree_leaves(inputs_functional)
            # 将 functional tensors 转换回普通 tensors，并同步数据
            for inpt, input_functional in zip(flat_inputs, flat_inputs_functional):
                if isinstance(input_functional, torch.Tensor):
                    torch._sync(input_functional)
                    inpt_new = torch._from_functional_tensor(input_functional)
            
            # 同步输出中的所有 tensors
            pytree.tree_map(torch._sync, out)
            # 将输出中的 functional tensors 转换回普通 tensors
            out_unwrapped = pytree.tree_map(torch._from_functional_tensor, out)
            return out_unwrapped

        return wrapped

    @_beartype.beartype
    # 定义一个方法 `_run`，返回类型为 `torch.fx.GraphModule`
    def _run(self, *args) -> torch.fx.GraphModule:
        # 为了在 `make_fx` 后保留堆栈跟踪信息，对模块进行包装
        module = _utils.wrap_graph_module_for_node_meta_preservation(self.module)

        # 将模块功能化，以便后续处理
        functionalized_callable = self._functionalize(module)

        # 模仿 `torch._dynamo.export(aten_graph=True)` 中 `make_fx` 的行为
        # TODO: 可能需要重新考虑用户假模式导出 + 动态形状场景。
        # 设置假模式（fake_mode），用于控制是否使用假张量
        fake_mode: Optional[fake_tensor.FakeTensorMode] = self.fake_mode
        # 可能对参数进行假化处理
        maybe_fake_args = self._maybe_fakefy_args(fake_mode, *args)
        if fake_mode is not None:
            # 如果存在假模式，将跟踪模式设置为 "real"，告知 `make_fx` 不需要创建新的假模式
            tracing_mode = "real"
        else:
            # 如果不存在假模式，需要让 `make_fx` 创建一个新的假模式
            fake_mode = contextlib.nullcontext()  # type: ignore[assignment]
            tracing_mode = "symbolic" if self.enable_dynamic_axes else "fake"

        assert fake_mode is not None  # for mypy

        # 在 `make_fx` 的上下文中执行以下操作：
        # 1. 可能禁用假张量模式
        # 2. 使用当前的假模式上下文
        graph_module = proxy_tensor.make_fx(
            functionalized_callable,
            decomposition_table={},
            tracing_mode=tracing_mode,
            _allow_non_fake_inputs=True,
            _allow_fake_constant=self.allow_fake_constant,
        )(*maybe_fake_args)

        # 将占位符目标重命名以匹配原始模块的签名，确保不将 forward(x, y, z) 映射为 forward(arg0, arg1, arg2)
        _utils.replace_placeholder_name_and_target(graph_module, self.module)

        # 返回经过处理的图模块
        return graph_module
class RemoveInputMutation(_pass.Transform):
    """Remove `aten.copy_.default` nodes that mutate module inputs.

    This pass is recommended to be used after ``Functionalization`` pass.
    ``Functionalization`` pass adds `aten.copy_.default` nodes to the graph
    when it detects mutations to inputs. These nodes are not needed for ONNX export
    for inference. They could be useful for training.
    """

    @_beartype.beartype
    def _run(self, *args) -> torch.fx.GraphModule:
        # 遍历反转后的图中的每个节点
        for node in reversed(self.module.graph.nodes):
            # 检查节点是否为函数调用且目标为 torch.ops.aten.copy_.default
            # 同时节点没有被使用，并且第一个参数是占位符节点
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.copy_.default
                and len(node.users) == 0
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].op == "placeholder"
            ):
                # 在图中擦除当前节点
                self.module.graph.erase_node(node)
        # 返回修改后的模块
        return self.module
```
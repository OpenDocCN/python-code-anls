# `.\pytorch\torch\jit\_freeze.py`

```py
# mypy: allow-untyped-defs
"""Freezing.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

# 导入需要的模块和类
from typing import List, Optional
import torch
from torch.jit._script import RecursiveScriptModule, ScriptModule

# 定义 freeze 函数，用于冻结 ScriptModule，并将子模块和属性作为常量内联
def freeze(
    mod, preserved_attrs: Optional[List[str]] = None, optimize_numerics: bool = True
):
    r"""Freeze ScriptModule, inline submodules, and attributes as constants.

    Freezing a :class:`ScriptModule` will clone it and attempt to inline the cloned
    module's submodules, parameters, and attributes as constants in the TorchScript IR Graph.
    By default, `forward` will be preserved, as well as attributes & methods specified in
    `preserved_attrs`. Additionally, any attribute that is modified within a preserved
    method will be preserved.

    Freezing currently only accepts ScriptModules that are in eval mode.

    Freezing applies generic optimization that will speed up your model regardless of machine.
    To further optimize using server-specific settings, run `optimize_for_inference` after
    freezing.

    Args:
        mod (:class:`ScriptModule`): a module to be frozen
        preserved_attrs (Optional[List[str]]): a list of attributes to preserve in addition to the forward method.
            Attributes modified in preserved methods will also be preserved.
        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly
            preserve numerics. Full details of optimization can be found at `torch.jit.run_frozen_optimizations`.

    Returns:
        Frozen :class:`ScriptModule`.

    Example (Freezing a simple module with a Parameter):

    .. testcode::
        import torch
        class MyModule(torch.nn.Module):
            def __init__(self, N, M):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(N, M))
                self.linear = torch.nn.Linear(N, M)

            def forward(self, input):
                output = self.weight.mm(input)
                output = self.linear(output)
                return output

        scripted_module = torch.jit.script(MyModule(2, 3).eval())
        frozen_module = torch.jit.freeze(scripted_module)
        # parameters have been removed and inlined into the Graph as constants
        assert len(list(frozen_module.named_parameters())) == 0
        # See the compiled graph as Python code
        print(frozen_module.code)

    Example (Freezing a module with preserved attributes)
    # 如果输入的模块不是 ScriptModule 类型，则抛出运行时错误
    if not isinstance(mod, ScriptModule):
        raise RuntimeError(
            "Freezing expects a ScriptModule as input. "
            "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'."
        )

    # 如果模块处于训练模式下，则抛出运行时错误，因为冻结只支持在评估模式下进行
    if mod.training:
        raise RuntimeError(
            "Freezing is currently only implemented for modules in eval mode. "
            "Please call .eval() on your module before freezing."
        )

    # 如果 preserved_attrs 参数为 None，则将其设为空列表
    preserved_attrs = preserved_attrs if preserved_attrs is not None else []

    # 调用底层的 C++ 函数 `_freeze_module` 来冻结模块，并用结果创建一个 RecursiveScriptModule
    out = RecursiveScriptModule(torch._C._freeze_module(mod._c, preserved_attrs))

    # 对冻结后的模块进行最终的脚本化处理
    RecursiveScriptModule._finalize_scriptmodule(out)

    # 根据 preserved_attrs 和 mod._c._has_method(x) 来确定哪些方法需要保留
    preserved_methods = [x for x in preserved_attrs if mod._c._has_method(x)]

    # 运行冻结优化，包括数值优化，并保留指定的方法
    run_frozen_optimizations(out, optimize_numerics, preserved_methods)

    # 返回最终冻结的模块
    return out
def run_frozen_optimizations(
    mod, optimize_numerics: bool = True, preserved_methods: Optional[List[str]] = None
):
    r"""
    Run a series of optimizations looking for patterns that occur in frozen graphs.

    The current set of optimizations includes:
        - Dropout Removal
        - Pretranspose Linear Layers
        - Concat Linear Layers with same input Tensor
        - Conv -> Batchnorm folding
        - Conv -> Add/Sub folding
        - Conv -> Mul/Div folding

    Args:
        mod (:class:`ScriptModule`): a frozen module to be optimized

        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly
        preserve numerics. These optimizations preserve default rtol and atol of `torch.testing.assert_close`
        when applied on a single transformation, however in a module where many transformations are applied
        the rtol or atol may no longer fall within the default `assert_close` tolerance. Conv -> Batchnorm folding,
        Conv-Add/Sub, and Conv -> Mul/Div folding all may alter numerics.

    Returns:
        None

    Note:
        In rare occassions, this can result in slower execution.

    Example (Freezing a module with Conv->Batchnorm)
    .. code-block:: python
        import torch
        in_channels, out_channels = 3, 32
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=True)
        bn = torch.nn.BatchNorm2d(out_channels, eps=.001)
        mod = torch.nn.Sequential(conv, bn)
        # set optimize to False here, by default freezing runs run_frozen_optimizations
        frozen_mod = torch.jit.freeze(torch.jit.script(mod.eval()), optimize=False)
        # inspect frozen mod
        assert "batch_norm" in str(frozen_mod.graph)
        torch.jit.run_frozen_optimizations(frozen_mod)
        assert "batch_norm" not in str(frozen_mod.graph)

    """
    # 如果模型包含 forward 方法，则对整个模型图进行冻结优化
    if mod._c._has_method("forward"):
        torch._C._jit_pass_optimize_frozen_graph(mod.graph, optimize_numerics)

    # 如果指定了需要保留的方法列表，则逐个对这些方法的图进行冻结优化
    if preserved_methods is None:
        preserved_methods = []

    for method in preserved_methods:
        torch._C._jit_pass_optimize_frozen_graph(
            mod.__getattr__(method).graph, optimize_numerics
        )
    """
    Optimize a TorchScript module for inference by applying specific optimizations.

    Args:
        mod (ScriptModule): The TorchScript module to be optimized.
        other_methods (list, optional): List of other methods to preserve during optimization.

    Returns:
        ScriptModule: The optimized TorchScript module.

    Raises:
        RuntimeError: If `mod` is not an instance of ScriptModule.

    Notes:
        This function optimizes the TorchScript module `mod` for inference. It freezes the module
        if it has a `training` attribute and then applies optimizations using `torch._C._jit_pass_optimize_for_inference`.

        Example:
            import torch
            in_channels, out_channels = 3, 32
            conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=True)
            bn = torch.nn.BatchNorm2d(out_channels, eps=.001)
            mod = torch.nn.Sequential(conv, bn)
            frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(mod.eval()))
            assert "batch_norm" not in str(frozen_mod.graph)
            # if built with MKLDNN, convolution will be run with MKLDNN weights
            assert "MKLDNN" in frozen_mod.graph
    """

    # Check if mod is a ScriptModule; raise an error if not
    if not isinstance(mod, ScriptModule):
        raise RuntimeError(
            "optimize_for_inference expects a ScriptModule as input. "
            "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'."
        )

    # If other_methods is None, initialize it as an empty list
    if other_methods is None:
        other_methods = []

    # If mod has a 'training' attribute, freeze it and preserve other_methods
    if hasattr(mod, "training"):
        mod = freeze(mod.eval(), preserved_attrs=other_methods)

    # Apply optimizations for inference to the underlying C++ representation of mod
    torch._C._jit_pass_optimize_for_inference(mod._c, other_methods)

    # Return the optimized ScriptModule
    return mod
```
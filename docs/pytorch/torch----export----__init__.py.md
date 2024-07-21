# `.\pytorch\torch\export\__init__.py`

```py
import builtins
import copy
import dataclasses
import inspect
import io
import os
import sys
import typing
import warnings
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import torch
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager

from torch.utils._pytree import (
    FlattenFunc,
    FromDumpableContextFn,
    ToDumpableContextFn,
    UnflattenFunc,
)

if TYPE_CHECKING:
    # 在类型检查期间导入以下模块以启用代码智能特性，
    # 不要无条件导入，因为导入 sympy 非常慢
    from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint


__all__ = [
    "Constraint",
    "Dim",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "ExportedProgram",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "dims",
    "dynamic_dim",
    "export",
    "load",
    "register_dataclass",
    "save",
    "unflatten",
    "FlatArgsAdapter",
    "UnflattenedModule",
]


from .dynamic_shapes import Constraint, Dim, dims, dynamic_dim, ShapesCollection
from .exported_program import ExportedProgram, ModuleCallEntry, ModuleCallSignature
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
from .unflatten import FlatArgsAdapter, unflatten, UnflattenedModule


PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


def export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    strict: bool = True,
    preserve_module_call_signature: Tuple[str, ...] = (),
) -> ExportedProgram:
    """
    :func:`export` takes an arbitrary Python callable (an nn.Module, a function or
    a method) along with example inputs, and produces a traced graph representing
    only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion,
    which can subsequently be executed with different inputs or serialized.  The
    traced graph (1) produces normalized operators in the functional ATen operator set
    (as well as any user-specified custom operators), (2) has eliminated all Python control
    flow and data structures (with certain exceptions), and (3) records the set of
    shape constraints needed to show that this normalization and control-flow elimination
    is sound for future inputs.

    **Soundness Guarantee**

    While tracing, :func:`export()` takes note of shape-related assumptions
    made by the user program and the underlying PyTorch operator kernels.
    The output :class:`ExportedProgram` is considered valid only when these
    assumptions hold true.
    """
    # 以下是关于张量输入形状的假设
    # 在 `export` 函数成功之前，这些假设必须在图捕获时进行验证。
    # 具体而言：
    
    # - 输入张量静态形状的假设会自动进行验证，无需额外操作。
    # - 输入张量动态形状的假设需要通过使用 `Dim` API 显式指定，
    #   通过 `dynamic_shapes` 参数将其与示例输入关联起来。
    
    # 如果任何假设无法验证，将引发致命错误。错误消息会包含建议的修复方案，
    # 用于验证这些假设。例如，`export` 可能会建议以下修复动态维度 `dim0_x` 的定义，
    # 如果它出现在与输入 `x` 相关联的形状中，先前定义为 `Dim("dim0_x")`：
    
    #     dim = Dim("dim0_x", max=5)
    
    # 这个例子意味着生成的代码要求输入 `x` 的第0维度小于或等于5才有效。
    # 您可以查看动态维度定义的建议修复方案，然后将其逐字复制到您的代码中，
    # 而不需要更改对 `export` 调用的 `dynamic_shapes` 参数。
    """
    Trace and export a Torch module for deployment.
    
    Args:
        mod: We will trace the forward method of this module.
             The module whose forward method will be traced.
    
        args: Example positional inputs.
              Example positional arguments to be used during tracing.
    
        kwargs: Optional example keyword inputs.
                Optional keyword arguments to be used during tracing.
    
        dynamic_shapes:
            An optional argument where the type should either be:
            1) a dict from argument names of ``f`` to their dynamic shape specifications,
            2) a tuple that specifies dynamic shape specifications for each input in original order.
            If you are specifying dynamism on keyword args, you will need to pass them in the order that
            is defined in the original function signature.
    
            The dynamic shape of a tensor argument can be specified as either
            (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
            not required to include static dimension indices in this dict, but when they are,
            they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
            where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
            are denoted by None. Arguments that are dicts or tuples / lists of tensors are
            recursively specified by using mappings or sequences of contained specifications.
    
        strict: When enabled (default), the export function will trace the program through
                TorchDynamo which will ensure the soundness of the resulting graph. Otherwise, the
                exported program will not validate the implicit assumptions baked into the graph and
                may cause behavior divergence between the original model and the exported one. This is
                useful when users need to workaround bugs in the tracer, or simply want incrementally
                enable safety in their models. Note that this does not affect the resulting IR spec
                to be different and the model will be serialized in the same way regardless of what value
                is passed here.
                WARNING: This option is experimental and use this at your own risk.
    
    Returns:
        An :class:`ExportedProgram` containing the traced callable.
        Returns an ExportedProgram object which contains the traced callable function.
    
    **Acceptable input/output types**
    
    Acceptable types of inputs (for ``args`` and ``kwargs``) and outputs include:
    
    - Primitive types, i.e. ``torch.Tensor``, ``int``, ``float``, ``bool`` and ``str``.
    - Dataclasses, but they must be registered by calling :func:`register_dataclass` first.
    - (Nested) Data structures comprising of ``dict``, ``list``, ``tuple``, ``namedtuple`` and
      ``OrderedDict`` containing all above types.
    
    """
    from ._trace import _export
    
    if not isinstance(mod, torch.nn.Module):
        raise ValueError(
            f"Expected `mod` to be an instance of `torch.nn.Module`, got {type(mod)}."
        )
    return _export(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        strict=strict,
        preserve_module_call_signature=preserve_module_call_signature,
        pre_dispatch=True,
    )
def save(
    ep: ExportedProgram,
    f: Union[str, os.PathLike, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    opset_version: Optional[Dict[str, int]] = None,
) -> None:
    """
    保存一个 :class:`ExportedProgram` 到一个类文件对象中。可以使用 Python API
    :func:`torch.export.load <torch.export.load>` 加载它。

    Args:
        ep (ExportedProgram): 要保存的导出程序对象。

        f (Union[str, os.PathLike, io.BytesIO): 文件类对象（必须实现 write 和 flush 方法）或者包含文件名的字符串。

        extra_files (Optional[Dict[str, Any]]): 文件名到内容的映射，作为 f 的一部分存储。

        opset_version (Optional[Dict[str, int]]): 操作集名称到该操作集版本的映射


    Example::

        import torch
        import io

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        ep = torch.export.export(MyModule(), (torch.randn(5),))

        # 保存到文件
        torch.export.save(ep, 'exported_program.pt2')

        # 保存到 io.BytesIO 缓冲区
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)

        # 保存带有额外文件
        extra_files = {'foo.txt': b'bar'.decode('utf-8')}
        torch.export.save(ep, 'exported_program.pt2', extra_files=extra_files)

    """
    from torch._export import save

    if not isinstance(ep, ExportedProgram):
        raise TypeError(
            f"The 'ep' parameter must be an instance of 'ExportedProgram', got '{type(ep).__name__}' instead."
        )

    save(ep, f, extra_files=extra_files, opset_version=opset_version)


def load(
    f: Union[str, os.PathLike, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> ExportedProgram:
    """
    加载之前使用 :func:`torch.export.save <torch.export.save>` 保存的 :class:`ExportedProgram`。

    Args:
        f (Union[str, os.PathLike, io.BytesIO): 文件类对象（必须实现 write 和 flush 方法）或者包含文件名的字符串。

        extra_files (Optional[Dict[str, Any]]): 给定的文件名映射，将加载并将其内容存储在提供的映射中。

        expected_opset_version (Optional[Dict[str, int]]): 操作集名称到期望的操作集版本的映射

    Returns:
        返回一个 :class:`ExportedProgram` 对象

    """
    # 导入 torch 和 io 模块
    import torch
    import io

    # 从文件中加载 ExportedProgram 对象
    ep = torch.export.load('exported_program.pt2')

    # 从 io.BytesIO 对象中加载 ExportedProgram 对象
    # 首先将文件内容读取到 BytesIO 缓冲区中
    with open('exported_program.pt2', 'rb') as f:
        buffer = io.BytesIO(f.read())
    buffer.seek(0)
    ep = torch.export.load(buffer)

    # 加载时传入额外的文件
    extra_files = {'foo.txt': ''}  # 值将被替换为文件数据
    ep = torch.export.load('exported_program.pt2', extra_files=extra_files)

    # 输出加载后的 foo.txt 文件内容
    print(extra_files['foo.txt'])

    # 对加载后的 ExportedProgram 对象进行调用，传入随机生成的张量数据
    print(ep(torch.randn(5)))
def register_dataclass(
    cls: Type[Any],
    *,
    serialized_type_name: Optional[str] = None,
) -> None:
    """
    Registers a dataclass as a valid input/output type for :func:`torch.export.export`.

    Args:
        cls: the dataclass type to register
        serialized_type_name: The serialized name for the dataclass. This is
        required if you want to serialize the pytree TreeSpec containing this
        dataclass.

    Example::

        @dataclass
        class InputDataClass:
            feature: torch.Tensor
            bias: int

        class OutputDataClass:
            res: torch.Tensor

        torch.export.register_dataclass(InputDataClass)
        torch.export.register_dataclass(OutputDataClass)

        def fn(o: InputDataClass) -> torch.Tensor:
            res = res=o.feature + o.bias
            return OutputDataClass(res=res)

        ep = torch.export.export(fn, (InputDataClass(torch.ones(2, 2), 1), ))
        print(ep)

    """

    # 导入注册数据类为 pytree 节点的实用函数
    from torch._export.utils import register_dataclass_as_pytree_node

    # 调用注册函数，将数据类注册为 pytree 节点
    return register_dataclass_as_pytree_node(
        cls, serialized_type_name=serialized_type_name
    )
```
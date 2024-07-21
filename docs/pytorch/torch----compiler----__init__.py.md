# `.\pytorch\torch\compiler\__init__.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 导入 List 类型提示
from typing import List

# 暴露的函数名列表
__all__ = [
    "compile",
    "assume_constant_result",
    "reset",
    "allow_in_graph",
    "list_backends",
    "disable",
    "cudagraph_mark_step_begin",
    "wrap_numpy",
    "is_compiling",
    "is_dynamo_compiling",
]

# 编译函数，调用 torch.compile 函数并返回结果
def compile(*args, **kwargs):
    """
    See :func:`torch.compile` for details on the arguments for this function.
    """
    return torch.compile(*args, **kwargs)

# 重置函数，清除所有编译缓存并恢复系统到初始状态
def reset() -> None:
    """
    This function clears all compilation caches and restores the system to its initial state.
    It is recommended to call this function, especially after using operations like `torch.compile(...)`
    to ensure a clean state before another unrelated compilation
    """
    # 导入 torch._dynamo 模块并调用 reset 函数
    import torch._dynamo
    torch._dynamo.reset()

# allow_in_graph 装饰器函数，指示编译器前端 (Dynamo) 跳过函数的符号内省
# 直接在遇到时将其写入图中
def allow_in_graph(fn):
    """
    Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
    and instead directly write it to the graph when encountered.

    If you are using :func:`torch.compile` (with backend="inductor" (the default)), or
    :func:`torch.export.export`, and trying to black-box a Python function throughout
    all tracing, do not use this API.
    Instead, please create a custom operator (see :ref:`custom-ops-landing-page`)

    .. warning::

        If you're a typical torch.compile user (e.g. you're applying torch.compile to
        a model to make it run faster), you probably don't want to use this function.
        :func:`allow_in_graph` is a footgun because it skips the compiler frontend
        (Dynamo) that is responsible for doing safety checks (graph breaks, handling
        closures, etc). Incorrect usage will lead to difficult-to-debug silent
        incorrectness issues.

    Given a Python function with no allow_in_graph decorator, regular execution
    of torch.compile traces through the function. :func:`allow_in_graph` changes
    it so that the frontend does not trace inside the function, but the compiler
    backend still traces through it. Compare this to custom operators, which
    treats a function as a black box throughout the torch.compile stack. The following
    table compares these mechanisms.

    +------------------------+-----------------------+--------------------------------+
    | Mechanism              | Frontend (Dynamo)     | Backend (AOTAutograd+Inductor) |
    +========================+=======================+================================+
    | no decorator           | trace inside          | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | allow_in_graph         | opaque callable       | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | custom op              | opaque callable       | opaque callable                |
    +------------------------+-----------------------+--------------------------------+
    # 导入 torch._dynamo 模块，这是用于动态图控制的内部模块
    import torch._dynamo
    # 调用 torch._dynamo.allow_in_graph 函数，将函数 fn 添加到动态图中以绕过 Dynamo 的限制
    return torch._dynamo.allow_in_graph(fn)
def list_backends(exclude_tags=("debug", "experimental")) -> List[str]:
    """
    Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Args:
        exclude_tags(optional): A tuple of strings representing tags to exclude.
    """
    # 导入 torch._dynamo 模块，用于获取后端列表
    import torch._dynamo

    # 调用 torch._dynamo.list_backends 函数，返回可用的后端列表
    return torch._dynamo.list_backends(exclude_tags)


def assume_constant_result(fn):
    """
    This function is used to mark a function `fn` as having a constant result.
    This allows the compiler to optimize away your function
    Returns The same function `fn`

    Args:
        fn: The function to be marked as having a constant result.

    .. warning::
        `assume_constant_result` can if invalid cause safety and soundness issues, :func:`torch.compile`
        will not attempt to validate whether the constant assumption is true or not

    """
    # 导入 torch._dynamo 模块，用于标记函数 fn 具有常量结果
    import torch._dynamo

    # 调用 torch._dynamo.assume_constant_result 函数，标记函数 fn 为具有常量结果
    return torch._dynamo.assume_constant_result(fn)


def disable(fn=None, recursive=True):
    """
    This function provides both a decorator and a context manager to disable compilation on a function
    It also provides the option of recursively disabling called functions

    Args:
        fn (optional): The function to disable
        recursive (optional): A boolean value indicating whether the disabling should be recursive.
    """
    # 导入 torch._dynamo 模块，用于禁用函数的编译
    import torch._dynamo

    # 调用 torch._dynamo.disable 函数，返回禁用编译后的函数或上下文管理器
    return torch._dynamo.disable(fn, recursive)


def cudagraph_mark_step_begin():
    """
    Indicates that a new iteration of inference or training is about to begin.

    CUDA Graphs will free tensors of a prior iteration. A new iteration is started on each invocation of
    torch.compile, so long as there is not a pending backward that has not been called.

    If that heuristic is wrong, such as in the following example, manually mark it with this api.

    .. code-block:: python

        @torch.compile(mode="reduce-overhead")
        def rand_foo():
            return torch.rand([4], device="cuda")

        for _ in range(5):
            torch.compiler.cudagraph_mark_step_begin()
            rand_foo() + rand_foo()

    For more details, see `torch.compiler_cudagraph_trees <https://pytorch.org/docs/main/torch.compiler_cudagraph_trees.html>`__
    """
    # 导入 torch._inductor.cudagraph_trees 模块，用于标记 CUDA 图的新迭代开始
    from torch._inductor import cudagraph_trees

    # 调用 cudagraph_trees.mark_step_begin 函数，标记 CUDA 图的新迭代开始
    cudagraph_trees.mark_step_begin()


def wrap_numpy(fn):
    r"""Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function
    from ``torch.Tensor``s to ``torch.Tensor``s.

    It is designed to be used with :func:`torch.compile` with ``fullgraph=True``. It allows to
    compile a NumPy function as if it were a PyTorch function. This allows you to run NumPy code
    on CUDA or compute its gradients.

    .. note::

        This decorator does not work without :func:`torch.compile`.

    """
    # 这里不需要导入模块，仅为 NumPy 函数到 PyTorch 函数的装饰器定义

    # 返回装饰后的函数 fn，用于将 NumPy 数组作为输入转换为 PyTorch 张量
    return fn
    # 导入 wrap_numpy 函数，该函数来自 torch._dynamo.external_utils 模块
    from torch._dynamo.external_utils import wrap_numpy as wrap
    # 返回 wrap 函数对 fn 函数的包装结果
    return wrap(fn)
# 定义全局变量 _is_compiling_flag，表示编译状态的标志，默认为 False
_is_compiling_flag: bool = False

# 函数 is_compiling() 用于判断当前是否处于 Torch 的编译或导出过程中
def is_compiling() -> bool:
    """
    Indicates whether a graph is executed/traced as part of torch.compile() or torch.export().

    Note that there are 2 other related flags that should deprecated eventually:
      * torch._dynamo.external_utils.is_compiling()
      * torch._utils.is_compiling()

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_compiling():
        >>>        pass # ...logic that is not needed in a compiled/traced graph...
        >>>
        >>>     # ...rest of the function...
    """
    # 如果当前正在执行 Torch 脚本化（scripting）过程，则返回 False
    if torch.jit.is_scripting():
        return False
    else:
        # 否则返回全局的 _is_compiling_flag，表示是否处于编译状态
        return _is_compiling_flag

# 函数 is_dynamo_compiling() 用于判断当前是否处于 TorchDynamo 的图追踪过程中
def is_dynamo_compiling() -> bool:
    """
    Indicates whether a graph is traced via TorchDynamo.

    It's stricter than is_compiling() flag, as it would only be set to True when
    TorchDynamo is used.

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_dynamo_compiling():
        >>>        pass # ...logic that is not needed in a TorchDynamo-traced graph...
        >>>
        >>>     # ...rest of the function...
    """
    # 直接返回 False，表示不处于 TorchDynamo 的图追踪过程中
    return False
```
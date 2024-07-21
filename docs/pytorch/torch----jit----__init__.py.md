# `.\pytorch\torch\jit\__init__.py`

```py
# mypy: allow-untyped-defs
# 引入警告模块，用于处理警告信息
import warnings

# 引入上下文管理器模块，用于定义上下文管理器
from contextlib import contextmanager
# 引入 Any 和 Iterator 类型提示，用于类型声明
from typing import Any, Iterator

# 引入 torch._C 模块，作为 torch 的 C++ 前端入口
import torch._C

# 下面的模块从 torch._jit_internal 中导入，以便用户可以从 torch.jit 模块访问它们

# 引入 _Await 类，用于表示异步等待
from torch._jit_internal import (
    _Await,
    # 引入 _drop，用于在 JIT 编译中丢弃变量
    _drop,
    # 引入 _IgnoreContextManager，用于在 JIT 编译中忽略上下文管理器
    _IgnoreContextManager,
    # 引入 _isinstance，用于 JIT 编译中的类型检查
    _isinstance,
    # 引入 _overload，用于 JIT 编译中的重载函数
    _overload,
    # 引入 _overload_method，用于 JIT 编译中的方法重载
    _overload_method,
    # 引入 export，用于导出 JIT 编译的函数或模块
    export,
    # 引入 Final，用于 JIT 编译中声明不可变类型
    Final,
    # 引入 Future，用于 JIT 编译中表示异步操作的未来结果
    Future,
    # 引入 ignore，用于在 JIT 编译中忽略警告
    ignore,
    # 引入 is_scripting，用于判断当前是否在 JIT 脚本模式下
    is_scripting,
    # 引入 unused，用于在 JIT 编译中表示未使用的变量或参数
    unused,
)
# 从 torch.jit._async 中导入 fork 和 wait 函数，用于异步编程
from torch.jit._async import fork, wait
# 从 torch.jit._await 中导入 _awaitable, _awaitable_nowait, _awaitable_wait 函数，用于处理异步等待
from torch.jit._await import _awaitable, _awaitable_nowait, _awaitable_wait
# 从 torch.jit._decomposition_utils 中导入 _register_decomposition 函数，用于注册分解操作
from torch.jit._decomposition_utils import _register_decomposition
# 从 torch.jit._freeze 中导入 freeze, optimize_for_inference, run_frozen_optimizations 函数，用于模型冻结和推理优化
from torch.jit._freeze import freeze, optimize_for_inference, run_frozen_optimizations
# 从 torch.jit._fuser 中导入 fuser, last_executed_optimized_graph, optimized_execution, set_fusion_strategy 函数，用于运算融合
from torch.jit._fuser import (
    fuser,
    last_executed_optimized_graph,
    optimized_execution,
    set_fusion_strategy,
)
# 从 torch.jit._ir_utils 中导入 _InsertPoint 类，用于插入点操作
from torch.jit._ir_utils import _InsertPoint
# 从 torch.jit._script 中导入各种与脚本相关的类和函数，用于 JIT 脚本编译
from torch.jit._script import (
    _ScriptProfile,
    _unwrap_optional,
    Attribute,
    CompilationUnit,
    interface,
    RecursiveScriptClass,
    RecursiveScriptModule,
    script,
    script_method,
    ScriptFunction,
    ScriptModule,
    ScriptWarning,
)
# 从 torch.jit._serialization 中导入与模型序列化相关的函数，用于模型的保存与加载
from torch.jit._serialization import (
    jit_module_from_flatbuffer,
    load,
    save,
    save_jit_module_to_flatbuffer,
)
# 从 torch.jit._trace 中导入与模型追踪相关的函数，用于模型追踪与分析
from torch.jit._trace import (
    _flatten,
    _get_trace_graph,
    _script_if_tracing,
    _unique_state_dict,
    is_tracing,
    ONNXTracedModule,
    TopLevelTracedModule,
    trace,
    trace_module,
    TracedModule,
    TracerWarning,
    TracingCheckError,
)
# 从 torch.utils 中导入 set_module 函数，用于设置模块信息

# 设置公开的接口列表，以供外部访问的函数和类
__all__ = [
    "Attribute",
    "CompilationUnit",
    "Error",  # 错误类
    "Future",
    "ScriptFunction",
    "ScriptModule",
    "annotate",
    "enable_onednn_fusion",
    "export",
    "export_opnames",  # 导出操作名列表
    "fork",
    "freeze",
    "interface",
    "ignore",
    "isinstance",
    "load",
    "onednn_fusion_enabled",
    "optimize_for_inference",
    "save",
    "script",
    "script_if_tracing",
    "set_fusion_strategy",
    "strict_fusion",
    "trace",
    "trace_module",
    "unused",
    "wait",
]

# 为了向后兼容性而导入的旧版本别名
_fork = fork
_wait = wait
_set_fusion_strategy = set_fusion_strategy


# 定义函数 export_opnames，用于生成 Script 模块的新字节码
def export_opnames(m):
    r"""
    生成脚本模块的新字节码。

    根据当前代码库返回脚本模块的操作列表。

    如果您有 LiteScriptModule 并希望获取当前存在的操作列表，请调用 _export_operator_list。
    """
    return torch._C._export_opnames(m._c)


# 定义别名 Error，表示 JIT 异常
# 设置模块为 torch.jit
Error = torch._C.JITException
set_module(Error, "torch.jit")
# 这不是完美的，但在通常情况下可行
# 设置类名为 "Error"，限定名称为 "Error"
Error.__name__ = "Error"
Error.__qualname__ = "Error"


# 用于 Python 中使用 annotate 函数
def annotate(the_type, the_value):
    """用于在 TorchScript 编译器中给定 `the_value` 的类型。

    这是一个简单的传递函数，返回 `the_value`，用于提示 TorchScript
    """
    # 将 `the_value` 的类型注释为 `the_type`。在 TorchScript 外部运行时，这是一个空操作。
    #
    # 虽然 TorchScript 可以推断大多数 Python 表达式的正确类型，但也有一些情况下推断可能会出错，包括：
    # - 空容器，如 `[]` 和 `{}`，TorchScript 假设它们是 `Tensor` 的容器
    # - 可选类型，如 `Optional[T]`，但赋予了类型 `T` 的有效值，TorchScript 会认为其类型是 `T` 而不是 `Optional[T]`
    #
    # 注意，在 `torch.nn.Module` 的子类的 `__init__` 方法中，`annotate()` 无法帮助类型注解，因为它在 eager 模式下执行。
    # 要为 `torch.nn.Module` 的属性注解类型，应使用 :meth:`~torch.jit.Attribute`。
    #
    # 示例：
    # .. testcode::
    #
    #     import torch
    #     from typing import Dict
    #
    #     @torch.jit.script
    #     def fn():
    #         # 告诉 TorchScript 这个空字典是一个 (str -> int) 字典，而不是默认的 (str -> Tensor) 字典类型。
    #         d = torch.jit.annotate(Dict[str, int], {})
    #
    #         # 没有上面的 `torch.jit.annotate`，以下语句会因类型不匹配而失败。
    #         d["name"] = 20
    #
    # .. testcleanup::
    #
    #     del fn
    #
    # Args:
    #     the_type: 应传递给 TorchScript 编译器作为 `the_value` 类型提示的 Python 类型。
    #     the_value: 用于提示类型的值或表达式。
    #
    # Returns:
    #     返回 `the_value` 作为返回值。
    ```
# 定义一个装饰器函数，用于在跟踪期间首次调用时编译函数 ``fn``。
# ``torch.jit.script`` 在首次调用时具有非常显著的启动时间，因为它延迟初始化了许多编译器内建函数。
# 因此，不应在库代码中使用它。但是，如果希望库的某些部分在使用控制流时也能工作在跟踪模式下，
# 可以使用 ``@torch.jit.script_if_tracing`` 替代 ``torch.jit.script``。

def script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing.

    ``torch.jit.script`` has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Args:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `torch.jit.script` is returned.
        Otherwise, the original function `fn` is returned.
    """
    return _script_if_tracing(fn)


# 用于在 TorchScript 中提供容器类型的精细化。

def isinstance(obj, target_type):
    """
    Provide container type refinement in TorchScript.

    It can refine parameterized containers of the List, Dict, Tuple, and Optional types. E.g. ``List[str]``,
    ``Dict[str, List[torch.Tensor]]``, ``Optional[Tuple[int,str,int]]``. It can also
    refine basic types such as bools and ints that are available in TorchScript.

    Args:
        obj: object to refine the type of
        target_type: type to try to refine obj to
    Returns:
        ``bool``: True if obj was successfully refined to the type of target_type,
            False otherwise with no new type refinement


    Example (using ``torch.jit.isinstance`` for type refinement):
    .. testcode::

        import torch
        from typing import Any, Dict, List

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: Any): # note the Any type
                if torch.jit.isinstance(input, List[torch.Tensor]):
                    for t in input:
                        y = t.clamp(0, 0.5)
                elif torch.jit.isinstance(input, Dict[str, str]):
                    for val in input.values():
                        print(val)

        m = torch.jit.script(MyModule())
        x = [torch.rand(3,3), torch.rand(4,3)]
        m(x)
        y = {"key1":"val1","key2":"val2"}
        m(y)
    """
    return _isinstance(obj, target_type)


class strict_fusion:
    """
    Give errors if not all nodes have been fused in inference, or symbolically differentiated in training.

    Example:
    Forcing fusion of additions.

    .. code-block:: python

        @torch.jit.script
        def foo(x):
            with torch.jit.strict_fusion():
                return x + x + x

    """

    def __init__(self):
        if not torch._jit_internal.is_scripting():
            warnings.warn("Only works in script mode")
        pass

    def __enter__(self):
        pass

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        pass

# 用于在打印图形时全局隐藏源范围的上下文管理器。
# 定义一个上下文管理器函数 `_hide_source_ranges`，用于临时禁用 Torch 的源代码范围打印功能
@contextmanager
def _hide_source_ranges() -> Iterator[None]:
    # 获取当前全局的源代码范围打印状态，并保存为旧状态
    old_enable_source_ranges = torch._C.Graph.global_print_source_ranges  # type: ignore[attr-defined]
    try:
        # 设置全局的源代码范围打印状态为 False，禁用打印源代码范围
        torch._C.Graph.set_global_print_source_ranges(False)  # type: ignore[attr-defined]
        # 执行 yield 之前的代码块，此处是进入上下文管理器的执行代码
        yield
    finally:
        # 恢复之前保存的源代码范围打印状态
        torch._C.Graph.set_global_print_source_ranges(old_enable_source_ranges)  # type: ignore[attr-defined]


# 启用或禁用 OneDNN JIT 融合，根据参数 `enabled` 决定
def enable_onednn_fusion(enabled: bool):
    # 调用 Torch 库函数设置 LLGA（低级图优化）的启用状态
    torch._C._jit_set_llga_enabled(enabled)


# 返回当前 OneDNN JIT 融合是否启用的状态
def onednn_fusion_enabled():
    return torch._C._jit_llga_enabled()


# 删除 Any 类型的定义，通常是为了清理全局命名空间
del Any

# 如果 Torch 的 JIT 初始化失败，则抛出运行时错误
if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
```
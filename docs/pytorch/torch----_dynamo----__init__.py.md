# `.\pytorch\torch\_dynamo\__init__.py`

```
# 导入PyTorch库
import torch
# 导入本地模块中的函数和类
from . import convert_frame, eval_frame, resume_execution
# 导入注册表模块中的函数和类
from .backends.registry import list_backends, lookup_backend, register_backend
# 导入回调处理模块中的函数和类
from .callback import callback_handler, on_compile_end, on_compile_start
# 导入代码上下文模块
from .code_context import code_context
# 导入重播函数
from .convert_frame import replay
# 导入装饰器函数
from .decorators import (
    allow_in_graph,
    assume_constant_result,
    disable,
    disallow_in_graph,
    forbid_in_graph,
    graph_break,
    mark_dynamic,
    mark_static,
    mark_static_address,
    maybe_mark_dynamic,
    run,
)
# 导入评估帧模块中的函数和类
from .eval_frame import (
    _reset_guarded_backend_cache,
    explain,
    export,
    is_dynamo_supported,
    is_inductor_supported,
    optimize,
    optimize_assert,
    OptimizedModule,
    reset_code,
)
# 导入外部工具函数
from .external_utils import is_compiling
# 导入变异保护模块中的类
from .mutation_guard import GenerationTracker
# 导入实用工具函数
from .utils import graph_break_reasons, guard_failures, orig_code_map, reset_frame_count

# 声明模块中可导出的符号列表
__all__ = [
    "allow_in_graph",
    "assume_constant_result",
    "disallow_in_graph",
    "forbid_in_graph",
    "graph_break",
    "mark_dynamic",
    "maybe_mark_dynamic",
    "mark_static",
    "mark_static_address",
    "optimize",
    "optimize_assert",
    "export",
    "explain",
    "run",
    "replay",
    "disable",
    "reset",
    "OptimizedModule",
    "is_compiling",
    "register_backend",
    "list_backends",
    "lookup_backend",
]

# 如果torch.manual_seed等于torch.random.manual_seed，则执行以下操作
if torch.manual_seed is torch.random.manual_seed:
    # 导入torch.jit._builtins模块
    import torch.jit._builtins

    # 使用disable装饰器包装manual_seed函数
    # 由于依赖问题，无法在其实现时执行此操作
    torch.manual_seed = torch._disable_dynamo(torch.manual_seed)
    # 将新的manual_seed注册到内置注册表中
    torch.jit._builtins._register_builtin(torch.manual_seed, "aten::manual_seed")

# 定义函数reset，无返回值
def reset() -> None:
    """清除所有编译缓存并恢复初始状态"""
    # 使用convert_frame.compile_lock进行同步
    with convert_frame.compile_lock:
        # 清除重置编码缓存
        reset_code_caches()
        # 清空convert_frame模块中的输入和输出代码
        convert_frame.input_codes.clear()
        convert_frame.output_codes.clear()
        # 清空原始代码映射
        orig_code_map.clear()
        # 清空守护失败记录
        guard_failures.clear()
        # 清空图断裂原因列表
        graph_break_reasons.clear()
        # 清空恢复执行缓存
        resume_execution.ContinueExecutionCache.cache.clear()
        # 重置受保护后端缓存
        _reset_guarded_backend_cache()
        # 重置帧计数
        reset_frame_count()
        # 清空torch._C._dynamo.compiled_autograd缓存
        torch._C._dynamo.compiled_autograd.clear_cache()
        # 重置帧计数器
        convert_frame.FRAME_COUNTER = 0
        # 清空帧编译计数器
        convert_frame.FRAME_COMPILE_COUNTER.clear()
        # 清空回调处理程序
        callback_handler.clear()
        # 清空生成追踪器
        GenerationTracker.clear()
        # 清空torch._dynamo.utils.warn_once_cache缓存
        torch._dynamo.utils.warn_once_cache.clear()
        # 停止跟踪保存张量钩子的追踪
        torch._C._autograd._saved_tensors_hooks_set_tracing(False)

# 定义函数reset_code_caches，无返回值
def reset_code_caches() -> None:
    """清除按代码对象键控制的编译缓存"""
    # 使用convert_frame.compile_lock进行同步
    with convert_frame.compile_lock:
        # 遍历已见过的输入和输出代码的弱引用
        for weak_code in (
            convert_frame.input_codes.seen + convert_frame.output_codes.seen
        ):
            # 尝试获取代码对象
            code = weak_code()
            # 如果获取到了代码对象
            if code:
                # 调用reset_code函数重置该代码对象
                reset_code(code)
        # 清空代码上下文
        code_context.clear()
```
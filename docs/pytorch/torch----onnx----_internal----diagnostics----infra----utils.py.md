# `.\pytorch\torch\onnx\_internal\diagnostics\infra\utils.py`

```
# 引入 future 模块中的 annotations 特性，用于函数注解
from __future__ import annotations

# 引入 functools 模块，提供了操作函数对象的工具
import functools

# 引入 inspect 模块，提供了获取对象信息的函数
import inspect

# 引入 traceback 模块，用于提取和操作异常的堆栈信息
import traceback

# 引入 typing 模块中的类型提示工具
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

# 引入 torch.onnx._internal 模块中的 _beartype 功能
from torch.onnx._internal import _beartype

# 引入 torch.onnx._internal.diagnostics.infra 模块中的 _infra 和 formatter
from torch.onnx._internal.diagnostics.infra import _infra, formatter


@_beartype.beartype
def python_frame(frame: traceback.FrameSummary) -> _infra.StackFrame:
    """Returns a StackFrame for the given traceback.FrameSummary."""
    # 提取堆栈帧的代码行内容作为片段
    snippet = frame.line

    # 创建并返回 StackFrame 对象，包含位置信息和片段
    return _infra.StackFrame(
        location=_infra.Location(
            uri=frame.filename,  # 文件名
            line=frame.lineno,    # 行号
            snippet=snippet,      # 代码片段
            function=frame.name,  # 函数名
            message=snippet,      # 消息
        )
    )


@_beartype.beartype
def python_call_stack(frames_to_skip: int = 0, frames_to_log: int = 16) -> _infra.Stack:
    """Returns the current Python call stack."""
    # 检查参数 frames_to_skip 和 frames_to_log 的合法性
    if frames_to_skip < 0:
        raise ValueError("frames_to_skip must be non-negative")
    if frames_to_log < 0:
        raise ValueError("frames_to_log must be non-negative")

    # 跳过当前函数和 beartype 函数，设置实际需要的 frames_to_skip
    frames_to_skip += 2

    # 创建 Stack 对象用于存储堆栈信息
    stack = _infra.Stack()

    # 提取堆栈信息，限制数量为 frames_to_skip + frames_to_log
    frames = traceback.extract_stack(limit=frames_to_skip + frames_to_log)

    # 反转顺序，使得最老的帧排在前面
    frames.reverse()

    # 生成 StackFrame 对象列表，用于构建堆栈
    stack.frames = [python_frame(frame) for frame in frames[frames_to_skip:]]

    # 设置堆栈的消息
    stack.message = "Python call stack"

    # 返回构建好的 Stack 对象
    return stack


@functools.lru_cache
def _function_source_info(fn: Callable) -> Tuple[Sequence[str], int, Optional[str]]:
    """Returns the source lines, line number, and source file path for the given function.

    Essentially, inspect.getsourcelines() and inspect.getsourcefile() combined.
    Caching is applied to reduce the performance impact of this function.
    """
    # 获取函数的源代码行和起始行号
    source_lines, lineno = inspect.getsourcelines(fn)

    # 返回源代码行、起始行号和源文件路径
    return source_lines, lineno, inspect.getsourcefile(fn)


@_beartype.beartype
def function_location(fn: Callable) -> _infra.Location:
    """Returns a Location for the given function."""
    # 获取函数的源代码信息
    source_lines, lineno, uri = _function_source_info(fn)

    # 提取第一行代码作为片段
    snippet = source_lines[0].strip() if len(source_lines) > 0 else "<unknown>"

    # 创建并返回 Location 对象，包含源文件路径、起始行号、片段和消息
    return _infra.Location(
        uri=uri,                           # 源文件路径
        line=lineno,                       # 起始行号
        snippet=snippet,                   # 代码片段
        message=formatter.display_name(fn)  # 函数名
    )


@_beartype.beartype
def function_state(
    fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Mapping[str, Any]:
    # 使用函数签名绑定参数和关键字参数
    bind = inspect.signature(fn).bind(*args, **kwargs)

    # 返回参数绑定后的结果，以映射形式返回
    return bind.arguments
```
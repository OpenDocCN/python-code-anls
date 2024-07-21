# `.\pytorch\torch\utils\cpp_backtrace.py`

```py
# mypy: allow-untyped-defs
# 从 torch._C 模块导入 _get_cpp_backtrace 函数
from torch._C import _get_cpp_backtrace

# 定义函数 get_cpp_backtrace，返回当前线程的 C++ 栈跟踪字符串
def get_cpp_backtrace(frames_to_skip=0, maximum_number_of_frames=64) -> str:
    """
    Return a string containing the C++ stack trace of the current thread.

    Args:
        frames_to_skip (int): the number of frames to skip from the top of the stack
        maximum_number_of_frames (int): the maximum number of frames to return
    """
    # 调用 _get_cpp_backtrace 函数，传入 frames_to_skip 和 maximum_number_of_frames 参数
    return _get_cpp_backtrace(frames_to_skip, maximum_number_of_frames)
```
# `.\pytorch\torch\utils\_traceback.py`

```
# mypy: allow-untyped-defs
# 引入所需的模块和类型
from types import TracebackType
from typing import List, Optional
import tempfile  # 用于临时文件操作
import traceback  # 提供异常追踪信息的操作
import contextlib  # 提供上下文管理器的工具
import inspect  # 用于检查和获取源码信息
import os.path  # 提供关于路径操作的函数

# This file contains utilities for ensuring dynamically compile()'d
# code fragments display their line numbers in backtraces.
#
# The constraints:
#
# - We don't have control over the user exception printer (in particular,
#   we cannot assume the linecache trick will work, c.f.
#   https://stackoverflow.com/q/50515651/23845 )
#
# - We don't want to create temporary files every time we compile()
#   some code; file creation should happen lazily only at exception
#   time.  Arguably, you *should* be willing to write out your
#   generated Python code to file system, but in some situations
#   (esp. library code) it would violate user expectation to write
#   to the file system, so we try to avoid it.  In particular, we'd
#   like to keep the files around, so users can open up the files
#   mentioned in the trace; if the file is invisible, we want to
#   avoid clogging up the filesystem.
#
#   If this is not a constraint for you, there is a substantially simpler
#   way to implement the functionality in this PR: instead of using
#   eval/exec directly, just always write a Python file to filesystem
#   and compile that.
#
# - You have control over a context where the compiled code will get
#   executed, so that we can interpose while the stack is unwinding
#   (otherwise, we have no way to interpose on the exception printing
#   process.)
#
# There are two things you have to do to make use of the utilities here:
#
# - When you compile your source code, you must save its string source
#   in its f_globals under the magic name "__compile_source__"
#
# - Before running the compiled code, enter the
#   report_compile_source_on_error() context manager.

@contextlib.contextmanager
def report_compile_source_on_error():
    try:
        yield
    # 提供一个上下文管理器，用于捕获可能发生的异常
    # 在异常时，通过此上下文管理器能够获取编译的源码信息
    # 用户可以在异常时查看编译源码的行号和文件名
    # 实现动态编译代码显示行号在回溯中的位置

def shorten_filename(fn, *, base=None):
    """Shorten a source filepath, with the assumption that torch/ subdirectories don't need to be shown to user."""
    # 缩短源文件路径名，假设 torch/ 子目录不需要向用户显示
    if base is None:
        base = os.path.dirname(os.path.dirname(__file__))  # 默认基础路径为当前文件的上级目录
    try:
        prefix = os.path.commonpath([fn, base])  # 获取文件路径和基础路径的共同部分
    except ValueError:
        return fn  # 如果无法找到共同路径，直接返回原始文件名
    else:
        return fn[len(prefix) + 1:]  # 返回缩短后的文件名

def format_frame(frame, *, base=None, line=False):
    """
    Format a FrameSummary in a short way, without printing full absolute path or code.

    The idea is the result fits on a single line.
    """
    extra_line = ""
    if line:
        extra_line = f"{frame.line}  # "  # 如果需要打印行号，添加在格式化字符串后面
    return f"{extra_line}{shorten_filename(frame.filename, base=base)}:{frame.lineno} in {frame.name}"
    # 格式化给定的帧信息，返回简短的帧摘要信息，包括文件名、行号和函数名

def format_traceback_short(tb):
    """Format a TracebackType in a short way, printing only the inner-most frame."""
    return format_frame(traceback.extract_tb(tb)[-1])
    # 格式化给定的追踪对象，返回内部最深层帧的简短摘要信息

class CapturedTraceback:
    __slots__ = ['tb', 'skip']
    def __init__(self, tb, skip=0):
        # 初始化方法，接收一个 traceback 对象 tb 和一个可选的跳过参数 skip
        self.tb = tb
        self.skip = skip

    def cleanup(self):
        # 清理方法，将 traceback 对象设为 None
        self.tb = None

    def summary(self):
        import torch._C._profiler

        if self.tb is None:
            # 如果 traceback 对象为 None，则返回一个表示堆栈跟踪被省略的 StackSummary 对象
            return traceback.StackSummary()

        # 使用 Torch profiler 中的函数将 traceback 符号化，并提取出指定的堆栈信息
        return _extract_symbolized_tb(
            torch._C._profiler.symbolize_tracebacks([self.tb])[0],
            self.skip
        )

    def __getstate__(self):
        # 将对象转换为可序列化的状态，tb 设置为 None 是因为 traceback 对象不可 pickle
        return (None, {
            'tb': None,  # TB is not pickleable
            'skip': self.skip,
        })

    @staticmethod
    def extract(*, script=False, cpp=False, skip=0):
        """
        类似于 traceback.extract_stack()，但速度更快（约快 20 倍）；足够快以至于可以无条件地在正常执行中记录堆栈。
        返回一个 torch._C._profiler.CapturedTraceback 对象，必须使用 format_captured_tb 进行特殊格式化。
        
        默认只报告 Python 的回溯（类似 extract_stack）。可以设置 script/cpp 参数来同时报告 TorchScript/C++ 的跟踪信息。
        """
        import torch._C._profiler

        if script or cpp:
            assert skip == 0, "skip with script/cpp NYI"

        # 使用 Torch profiler 收集堆栈跟踪信息，根据参数决定是否包含 Python、TorchScript 或 C++ 的堆栈信息
        return CapturedTraceback(
            torch._C._profiler.gather_traceback(python=True, script=script, cpp=cpp),
            # 如果没有 script/cpp 帧，则跳过 extract() 帧；如果有，则强制设置为零
            0 if script or cpp else skip + 1
        )

    def format(self):
        """
        将单个 torch._C._profiler.CapturedTraceback 格式化为等同于 traceback.format_list 输出的字符串列表。
        注意，如果传入包含 C++ 跟踪信息的 CapturedTraceback，最好不要使用此函数，而是使用批处理格式化 API format_captured_tbs 来分摊符号化的成本。
        """
        return traceback.format_list(self.summary())

    @staticmethod
    def format_all(tbs):
        """
        CapturedTraceback.format 的批量版本。返回一个字符串列表的列表。
        """
        import torch._C._profiler

        # 直接填充已缓存摘要的 traceback
        rs: List[Optional[List[str]]] = []
        delayed_idxs = []
        for i, tb in enumerate(tbs):
            if tb.tb is None:
                rs.append([])
            else:
                rs.append(None)
                delayed_idxs.append(i)

        # 使用 Torch profiler 符号化延迟索引中的 traceback，并格式化摘要信息
        stbs = torch._C._profiler.symbolize_tracebacks([tbs[i].tb for i in delayed_idxs])
        for i, stb in zip(delayed_idxs, stbs):
            rs[i] = traceback.format_list(tbs[i].summary())

        return rs
# 给定一个符号化的回溯信息(tb)，从 symbolize_tracebacks 函数返回的结果中，
# 构建并返回一个经过预处理的 StackSummary 对象，表示堆栈跟踪条目。
def _extract_symbolized_tb(tb, skip):
    # 创建一个空的 StackSummary 对象，用于存储堆栈跟踪信息
    stack = traceback.StackSummary()
    # 从后向前遍历符号化的回溯信息中的每个条目，跳过前面的 skip 个条目
    for f in reversed(tb[skip:]):
        # 将每个回溯信息中的文件名、行号和函数名创建为 FrameSummary 对象，并加入 stack 中
        stack.append(traceback.FrameSummary(f['filename'], f['line'], f['name']))
    # 返回存储了预处理后的堆栈跟踪信息的 StackSummary 对象
    return stack
```
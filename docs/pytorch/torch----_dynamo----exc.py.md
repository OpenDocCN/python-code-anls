# `.\pytorch\torch\_dynamo\exc.py`

```py
#`
# 允许未指定类型的函数定义
# 使 mypy 忽略未指定类型的函数定义
# 这里指定了 `allow-untyped-defs`，以避免类型检查器报错
# 详细解释可参考 mypy 文档
# 例如，允许未指定类型的函数定义的方式
# import 标准库和自定义库
import os
import textwrap
from enum import auto, Enum
from traceback import extract_stack, format_exc, format_list, StackSummary
from typing import Any, cast, NoReturn, Optional

# 导入 torch._guards 模块
import torch._guards

# 从当前模块导入 config 和 utils.counters
from . import config
from .utils import counters

# 定义一个函数，生成导出数据库错误消息
# 返回一个字符串，包含有关错误的更多信息的 URL
def exportdb_error_message(case_name):
    return (
        "For more information about this error, see: "
        + "https://pytorch.org/docs/main/generated/exportdb/index.html#"
        + case_name.replace("_", "-")
    )

# 导入 logging 模块，用于记录日志
import logging

# 创建一个 logger 对象，命名为 __name__，用于记录日志
log = logging.getLogger(__name__)
# 使用 torch 的 logging 功能创建一个名为 graph_breaks 的日志记录器
graph_breaks_log = torch._logging.getArtifactLogger(__name__, "graph_breaks")

# 定义 TorchDynamoException 类，继承自 RuntimeError
class TorchDynamoException(RuntimeError):
    pass

# 定义 InternalTorchDynamoError 类，继承自 TorchDynamoException
class InternalTorchDynamoError(TorchDynamoException):
    pass

# 定义 RestartAnalysis 类，继承自 TorchDynamoException
# 添加一个名为 restart_reason 的属性
class RestartAnalysis(TorchDynamoException):
    restart_reason: str

    # 初始化函数，接受额外的参数 restart_reason
    def __init__(self, *args, restart_reason=None):
        self.restart_reason = restart_reason
        super().__init__(*args)

# 定义 SpeculationRestartAnalysis 类，继承自 RestartAnalysis
class SpeculationRestartAnalysis(RestartAnalysis):
    pass

# 定义 UnspecializeRestartAnalysis 类，继承自 RestartAnalysis
class UnspecializeRestartAnalysis(RestartAnalysis):
    pass

# 定义 SkipFrame 类，继承自 TorchDynamoException
class SkipFrame(TorchDynamoException):
    pass

# 定义 TorchRuntimeError 类，继承自 TorchDynamoException
class TorchRuntimeError(TorchDynamoException):
    pass

# 定义 InvalidBackend 类，继承自 TorchDynamoException
# 在初始化函数中，接收一个参数 name，并构造错误消息
class InvalidBackend(TorchDynamoException):
    def __init__(self, name):
        super().__init__(
            f"Invalid backend: {name!r}, see `torch._dynamo.list_backends()` for available backends."
        )

# 定义 ResetRequired 类，继承自 TorchDynamoException
# 在初始化函数中，构造错误消息，说明需要重置
class ResetRequired(TorchDynamoException):
    def __init__(self):
        super().__init__(
            textwrap.dedent(
                """
                Must call `torch._dynamo.reset()` before changing backends.  Detected two calls to
                `torch.compile()` with a different backend compiler arguments.
                """
            )
        )

# 定义 BackendCompilerFailed 类，继承自 TorchDynamoException
# 在初始化函数中，接收 backend_fn 和 inner_exception 参数，构造错误消息
class BackendCompilerFailed(TorchDynamoException):
    def __init__(self, backend_fn, inner_exception):
        self.backend_name = getattr(backend_fn, "__name__", "?")
        self.inner_exception = inner_exception
        msg = f"backend={self.backend_name!r} raised:\n{type(inner_exception).__name__}: {inner_exception}"
        super().__init__(msg)

# 定义 Unsupported 类，继承自 TorchDynamoException
# 在初始化函数中，接收 msg 参数，并将其添加到统计数据中
class Unsupported(TorchDynamoException):
    def __init__(self, msg):
        super().__init__(msg)
        self.real_stack = torch._guards.TracingContext.extract_stack()
        self.msg = msg
        self.category: Optional[str] = None
        self.add_to_stats()

    # 从统计数据中移除消息
    def remove_from_stats(self):
        assert self.category is not None
        counters[self.category][self.msg] -= 1
        if counters[self.category][self.msg] <= 0:
            del counters[self.category][self.msg]

    # 添加消息到统计数据中，默认为 "unimplemented" 类别
    def add_to_stats(self, category="unimplemented"):
        self.category = category
        counters[category][self.msg] += 1

# 定义 RecompileError 类，继承自 TorchDynamoException
class RecompileError(TorchDynamoException):
    pass

# 定义 ArgsMismatchError 类，继承自 Unsupported
# 在初始化函数中，接收 msg 参数，并调用父类初始化函数
class ArgsMismatchError(Unsupported):
    def __init__(self, msg):
        super().__init__(msg)

# 定义 AttributeMutationError 类，继承自 Unsupported
class AttributeMutationError(Unsupported):
    pass
    # 定义一个构造函数，初始化父类，并传入消息参数
    def __init__(self, msg):
        # 调用父类的构造函数，传入消息参数
        super().__init__(msg)
class CondOpArgsMismatchError(ArgsMismatchError):
    """
    Internal error from cond() due to arguments mismatch.
    """

    def __init__(self, msg):
        # 调用父类的初始化方法，传入错误消息
        super().__init__(msg)


class UserErrorType(Enum):
    DYNAMIC_CONTROL_FLOW = auto()
    ANTI_PATTERN = auto()
    STANDARD_LIBRARY = auto()
    CONSTRAINT_VIOLATION = auto()
    DYNAMIC_DIM = auto()
    INVALID_INPUT = auto()
    INVALID_OUTPUT = auto()


class UserError(Unsupported):
    def __init__(self, error_type: UserErrorType, msg, case_name=None):
        """
        Type of errors that would be valid in Eager, but not supported in TorchDynamo.
        The error message should tell user about next actions.

        error_type: Type of user error
        msg: Actionable error message
        case_name: (Optional) Unique name (snake case) for the usage example in exportdb.
        """
        if case_name is not None:
            assert isinstance(case_name, str)
            if msg.endswith("."):
                msg += " "
            else:
                msg += "\n"
            msg += exportdb_error_message(case_name)
        # 调用父类Unsupported的初始化方法，传入错误消息
        super().__init__(msg)
        self.error_type = error_type
        self.message = msg


class UserStopIteration(TorchDynamoException):
    value: Optional[Any]

    # 引用 CPython 中 StopIteration_init 的实现
    # https://github.com/python/cpython/blob/3.11/Objects/exceptions.c#L568-L584
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法，传入固定的错误消息
        super().__init__("unhandled `raise StopIteration`")
        if len(args) > 0:
            self.value = args[0]
        else:
            self.value = None


class UnsafeScriptObjectError(TorchDynamoException):
    pass


class UncapturedHigherOrderOpError(TorchDynamoException):
    pass


class IncorrectUsage(Exception):
    pass


class ObservedException(TorchDynamoException):
    pass


# These exceptions are ok to fallback to eager/graph_break.
# 可以回退到 eager/graph_break 的异常类型
exceptions_allowed_to_be_fallback = (
    torch._subclasses.fake_tensor.DataDependentOutputException,
    torch._subclasses.fake_tensor.DynamicOutputShapeException,
    torch._subclasses.fake_tensor.UnsupportedOperatorException,
    torch._subclasses.fake_tensor.UnsupportedFakeTensorException,
)


def unimplemented_with_warning(e: Exception, code, msg: str) -> NoReturn:
    # 这个函数内部调用 unimplemented，最终会导致图形中断或者回退到 eager 模式。
    # unimplemented 本身不会打印任何用户警告，即非常静默。此辅助函数用于在 torch.compile 堆栈中
    # 遇到值得向用户显示警告的错误时使用。例如，如果 AOT Autograd 后端由于伪张量异常而失败，
    # 可以回退到 eager，但不是无声。在这种情况下，我们可以使用此函数记录消息和堆栈跟踪。
    graph_break_msg = format_error_msg_verbose(e, code)
    graph_breaks_log.debug("%s", graph_break_msg)
    log.warning(msg)
    unimplemented(msg, from_exc=e)


_NOTHING = object()
# 当函数调用未实现时，抛出异常并终止程序运行
def unimplemented(msg: str, *, from_exc: Any = _NOTHING) -> NoReturn:
    # 检查环境变量中是否设置了BREAK，并且其值不为假
    assert msg != os.environ.get("BREAK", False)
    # 如果有指定的异常来源，则抛出带有原始异常的Unsupported异常
    if from_exc is not _NOTHING:
        raise Unsupported(msg) from from_exc
    # 否则直接抛出Unsupported异常
    raise Unsupported(msg)


# 记录警告信息，并更新计数器
def warning(msg: str) -> None:
    # 将警告消息计数器加一
    counters["warnings"][msg] += 1
    # 检查环境变量中是否设置了BREAK，并且其值不为假
    assert msg != os.environ.get("BREAK", False)


# 自定义的KeyError异常类，用于定制异常消息
# 详见 https://github.com/python/cpython/blob/3.11/Objects/exceptions.c#L2534 获取更多细节
class KeyErrorMsg:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self) -> str:
        return self.__str__()


# 增强异常消息，可以附加堆栈信息和其他调试相关信息
def augment_exc_message(exc: Exception, msg: str = "\n", export: bool = False) -> None:
    import traceback

    # 清空内部用户帧摘要信息
    exc.innermost_user_frame_summary = None  # type: ignore[attr-defined]

    # 获取真实的堆栈信息
    real_stack = get_real_stack(exc)
    # 如果存在真实的堆栈信息且不为空，则将最内层的用户帧摘要赋值给异常对象
    if real_stack is not None and len(real_stack) > 0:
        exc.innermost_user_frame_summary = real_stack[-1]  # type: ignore[attr-defined]
        # 将用户代码的堆栈信息添加到消息中
        msg += f"\nfrom user code:\n {''.join(traceback.format_list(real_stack))}"

    # 如果配置允许重放记录并且异常对象有记录文件名属性，则添加相关消息
    if config.replay_record_enabled and hasattr(exc, "record_filename"):
        msg += f"\nLast frame execution written to {exc.record_filename}. To run only this frame while debugging, run\
 torch._dynamo.replay('{exc.record_filename}').\n"

    # 如果不是调试模式并且异常对象有真实堆栈属性，则添加更多信息提示
    if not config.verbose and hasattr(exc, "real_stack"):
        msg += '\nSet TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information\n'

    # 如果异常对象有内部异常属性并且该属性有缩小路径信息属性，则添加相关消息
    if hasattr(exc, "inner_exception") and hasattr(
        exc.inner_exception, "minifier_path"
    ):
        if hasattr(exc.inner_exception, "buck_command"):
            msg += (
                f"\nMinifier script written to {exc.inner_exception.minifier_path}. Run "
                f"this buck command to find the smallest traced graph "
                f"which reproduces this error: {exc.inner_exception.buck_command}\n"
            )
        else:
            msg += (
                f"\nMinifier script written to {exc.inner_exception.minifier_path}. Run "
                "this script to find the smallest traced graph which reproduces this error.\n"
            )

    # 如果不是调试模式且不是导出模式，则添加异常抑制和回退到急切模式的提示信息
    if not config.suppress_errors and not export:
        msg += (
            "\n\n"
            "You can suppress this exception and fall back to eager by setting:\n"
            "    import torch._dynamo\n"
            "    torch._dynamo.config.suppress_errors = True\n"
        )

    # 获取旧的异常消息，如果没有则为空字符串
    old_msg = "" if len(exc.args) == 0 else str(exc.args[0])

    # 如果异常类型是KeyError，则使用定制的KeyErrorMsg替换异常消息
    if isinstance(exc, KeyError):
        exc.args = (KeyErrorMsg(old_msg + msg),) + exc.args[1:]
    else:
        # 否则将增强后的消息附加到原始消息后面
        new_msg = old_msg + msg
        exc.args = (new_msg,) + exc.args[1:]


# 获取真实的堆栈信息，如果不存在则返回None
def get_real_stack(exc: Exception, frame=None) -> Optional[StackSummary]:
    real_stack = getattr(exc, "real_stack", None)
    if real_stack is None:
        return None

    # 注意：即使real_stack可能为空列表，我们仍然尝试获取最真实的堆栈信息
    # 始终报告一个堆栈，因为 stack_above_dynamo 可能对调试仍然有用

    # 初始化一个空列表 stack_above_dynamo
    stack_above_dynamo = []
    # 如果 frame 不是 None
    if frame is not None:
        # 注意：在 Python 3.11 及更高版本中，frame 是 PyInterpreterFrame，不是真正的 frame 对象。
        # 不能直接传递给 traceback，因为它没有足够的信息。
        # 解决此问题的技术方法是，实际上应该将 frame 实例化，就像 _PyFrame_GetFrameObject 将会做的那样
        # （但实际上我们不能这样做，因为这会填充 frame_obj 字段，这是默认 eval frame 不喜欢的）。

        # 幸运的是，在这种情况下，我们可以通过 hack 解决：实际上不需要使用真正的顶层 frame，
        # 我们可以从当前位置提取，并依赖于 filter_stack 来消除所有 dynamo frames。
        # 为了方便测试，我们将这种行为应用于所有 Python 版本。
        
        # 调用 extract_stack() 获取当前堆栈信息，并传递给 filter_stack 函数处理 dynamo frames
        stack_above_dynamo = filter_stack(extract_stack())

    # 返回合并后的堆栈信息，类型为 StackSummary
    return cast(StackSummary, stack_above_dynamo + real_stack)
# 过滤掉进入 Dynamo 后的所有堆栈帧
def filter_stack(stack):
    user_stack = []
    # 遍历堆栈中的每个帧
    for frame in stack:
        # 如果文件名中包含 "convert_frame"，则停止遍历
        if "convert_frame" in frame.filename:
            break
        # 如果文件名中包含 "eval_frame" 或者行中包含 "torch._dynamo.optimize("，则跳过该帧
        if "eval_frame" in frame.filename or "torch._dynamo.optimize(" in frame.line:
            continue
        # 将符合条件的帧添加到用户堆栈中
        user_stack.append(frame)

    return user_stack


def format_error_msg_verbose(
    exc: Exception, code, record_filename=None, frame=None
) -> str:
    # 构造详细的错误消息，指明不会转换的函数、文件和行数
    msg = (
        f"WON'T CONVERT {code.co_name} {code.co_filename} line {code.co_firstlineno}\n"
    )
    msg += "=" * 10 + " TorchDynamo Stack Trace " + "=" * 10 + "\n"
    # 添加完整的异常堆栈信息
    msg += format_exc()
    # 获取真实堆栈信息
    real_stack = get_real_stack(exc, frame)
    if real_stack is not None:
        # 如果存在真实堆栈信息，添加相关提示和堆栈内容
        msg += (
            "\n"
            + "=" * 10
            + " The above exception occurred while processing the following code "
            + "=" * 10
            + "\n\n"
        )
        msg += "".join(format_list(real_stack))
        msg += "\n"
        msg += "=" * 10

    return msg


def format_error_msg(exc: Exception, code, record_filename=None, frame=None) -> str:
    # 初始化错误消息字符串
    msg = os.linesep * 2

    # 根据配置决定是否输出详细的错误消息
    if config.verbose:
        # 如果配置为详细模式，则调用详细错误消息生成函数
        msg = format_error_msg_verbose(exc, code, record_filename, frame)
    else:
        # 否则，简要说明不会转换的函数、文件和行数，以及具体原因
        msg = f"WON'T CONVERT {code.co_name} {code.co_filename}\
 line {code.co_firstlineno} \ndue to: \n{format_exc()}"

    return msg
```
# `.\pytorch\torch\distributed\elastic\multiprocessing\errors\__init__.py`

```py
"""
Each host in a distributed PyTorch job runs with a single TorchElastic agent,
and multiple workers (as children processes of the TorchElastic agent).
Since the workers are user-provided (your PyTorch script/job), TorchElastic
has a way to propagate errors on the trainers through the agent and up to the
scheduler, which ultimately informs the end-user about the state of the job
and applies any retry policies.

TorchElastic categorizes errors into 3 categories:

+----------------+----------------+--------------------------------------------------------------+
| Category       | Sub-Category   |  Description                                                 |
+================+================+==============================================================+
| User Error     | Input Error    | invalid inputs to TorchElastic APIs (e.g. min > max nodes)   |
|                +----------------+--------------------------------------------------------------+
|                | Worker Failure | any failures on the worker child process                     |
+----------------+----------------+--------------------------------------------------------------+
| Platform Error |      n/a       | failures caused by the agent                                 |
+----------------+----------------+--------------------------------------------------------------+
| Infra Error    |      n/a       | failures outside the domain of the agent and workers         |
|                |                | (e.g. host failures)                                         |
+----------------+----------------+--------------------------------------------------------------+

All errors other than "Worker Failure" are either raised canonically from the
agent process or implicitly or explicitly crash the agent process. So the
standard language (python) provided exception handling strategies apply.

Worker Failures are special because the exception/failure originates on a different
process from the agent so the error needs to be propagated inter-process
(e.g. the agent cannot simply ``try-catch`` an exception raised on the worker process).

TorchElastic agents use :func:`torch.distributed.elastic.multiprocessing.start_processes`
to launch the workers which has a simple file based inter-process error propagation
built-in.

Any function or binary entrypoint decorated with :func:`record`
will write uncaught exceptions (with the trace information) to a file specified by the
environment variable ``TORCHELASTIC_ERROR_FILE``. The parent process (e.g. agent)
sets this env var on each child it launches, then aggregates the error files for all
children, and propagates the one with the **smallest** timestamp (e.g. the **first** error).
"""
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关功能的模块
import signal  # 导入信号处理相关的模块
import socket  # 导入网络通信相关的模块
import time  # 导入时间相关的模块
import warnings  # 导入警告相关的模块
from dataclasses import dataclass, field  # 导入用于创建数据类的模块
from datetime import datetime  # 导入处理日期和时间的模块
from functools import wraps  # 导入用于创建装饰器的模块
from string import Template  # 导入字符串模板相关的模块
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar  # 导入用于类型提示的模块

from torch.distributed.elastic.utils.logging import get_logger  # 导入获取日志记录器的函数

from .error_handler import ErrorHandler  # 导入自定义的错误处理器类
from .handlers import get_error_handler  # 导入获取错误处理器的函数

__all__ = [
    "ProcessFailure",  # 将类 ProcessFailure 加入到导出的模块列表中
    "ChildFailedError",  # 将异常类 ChildFailedError 加入到导出的模块列表中（未在此代码段定义）
    "record",  # 将函数 record 加入到导出的模块列表中（未在此代码段定义）
    "ErrorHandler",  # 将错误处理器类 ErrorHandler 加入到导出的模块列表中
    "get_error_handler",  # 将获取错误处理器的函数 get_error_handler 加入到导出的模块列表中
]

logger = get_logger(__name__)  # 获取当前模块的日志记录器对象

JSON = Dict  # 定义 JSON 类型别名为字典类型

_EMPTY_ERROR_DATA = {"message": "<NONE>"}  # 定义空错误数据的默认值
_NOT_AVAILABLE = "<N/A>"  # 定义不可用数据的默认显示字符串

T = TypeVar("T")  # 定义一个类型变量 T

@dataclass
class ProcessFailure:
    """
    Represent the failed process result. When the worker process fails, it may record failure root cause into the file.

    Tries to read the failure timestamp from the provided ``error_file``,
    if the ``error_file`` does not exist, the timestamp is the current
    timestamp (seconds since epoch).

    The ``message`` field is a concise explanation of the failure. If
    the error file exists then the message is obtained from the error file.
    Otherwise one is generated based on the failure signature.

    .. note:: It is assumed that the ``error_file`` is written by
              ``torch.distributed.elastic.multiprocessing.errors.error_handler.ErrorHandler``.
              Otherwise the behavior is undefined.

    """

    local_rank: int  # 本地进程的排名
    pid: int  # 进程的 ID
    exitcode: int  # 进程的退出码
    error_file: str  # 记录错误信息的文件路径
    error_file_data: JSON = field(init=False)  # 错误文件中的数据，默认为空字典
    message: str = field(init=False)  # 错误消息，默认为空字符串
    timestamp: int = field(init=False)  # 时间戳，默认为当前时间的秒数

    def __post_init__(self):
        self.error_file_data = _EMPTY_ERROR_DATA  # 初始化错误文件数据为默认空数据
        if os.path.isfile(self.error_file):  # 如果错误文件存在
            try:
                with open(self.error_file) as fp:  # 打开错误文件
                    self.error_file_data = json.load(fp)  # 读取错误文件中的 JSON 数据
                    logger.debug(  # 记录调试信息，显示错误数据
                        "User process failed with error data: %s",
                        json.dumps(self.error_file_data, indent=2),
                    )
                    self.message, self.timestamp = self._get_error_data(  # 获取错误消息和时间戳
                        self.error_file_data
                    )
            except Exception:
                logger.exception("Failed to parse reply file: %s", self.error_file)  # 记录异常信息
                raise
        else:  # 如果错误文件不存在
            self._set_no_reply_file()  # 调用方法设置无回复文件状态

        # 如果没有设置错误消息，则根据退出码生成信息
        if not self.message:
            # 通常信号不会生成错误文件消息
            if self.exitcode < 0:
                self.message = (
                    f"Signal {-self.exitcode} ({self.signal_name()})"
                    f" received by PID {self.pid}"
                )
            else:
                self.message = "To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html"

    def _set_no_reply_file(self):
        """
        Set the state when no reply file is available.
        """
        self.message = _NOT_AVAILABLE  # 设置消息为不可用信息
        self.timestamp = int(time.time())  # 设置时间戳为当前时间的秒数

    def _get_error_data(self, error_data: JSON) -> Tuple[str, int]:
        """
        Extract error message and timestamp from error data.
        """
        message = error_data.get("message", _NOT_AVAILABLE)  # 从错误数据中获取消息，若无则为不可用信息
        timestamp = error_data.get("timestamp", int(time.time()))  # 从错误数据中获取时间戳，若无则为当前时间的秒数
        return message, timestamp

    def signal_name(self) -> str:
        """
        Get the name of the signal corresponding to the exit code.
        """
        return signal.Signals(-self.exitcode).name  # 返回与退出码对应的信号名称
    # 从错误文件数据中获取消息和时间戳，并以元组形式返回
    def _get_error_data(self, error_file_data: Dict[str, Any]) -> Tuple[str, int]:
        # 获取错误消息
        message = error_file_data["message"]
        if isinstance(message, str):
            # 如果消息是字符串类型，则尝试从 error_file_data 中获取时间戳，转换为整数
            timestamp = int(error_file_data.get("timestamp", 0))
        else:
            # 如果消息不是字符串，假设其为字典，获取额外信息中的时间戳并转换为整数
            timestamp = int(message["extraInfo"]["timestamp"])
        # 返回消息和时间戳的元组
        return (message, timestamp)

    # 设置没有回复文件的状态
    def _set_no_reply_file(self):
        # 将错误文件设置为不可用状态
        self.error_file = _NOT_AVAILABLE
        # 设置错误文件数据为空数据
        self.error_file_data = _EMPTY_ERROR_DATA
        # 设置消息为空字符串
        self.message = ""
        # 设置时间戳为当前时间的整数值
        self.timestamp = int(time.time())

    # 返回信号名称的字符串表示
    def signal_name(self) -> str:
        if self.exitcode < 0:
            # 如果退出码小于零，表示为信号退出码
            # 不希望因为找不到信号名称而杀死父进程
            # 如果信号码无法映射到已知名称，则使用不可用状态
            try:
                return signal.Signals(-self.exitcode).name
            except Exception:
                return _NOT_AVAILABLE
        else:
            # 如果退出码不是负数，表示不是信号退出码，返回不可用状态
            return _NOT_AVAILABLE

    # 返回时间戳的 ISO 格式（YYYY-MM-DD_HH:MM:SS）
    def timestamp_isoformat(self):
        """Return timestamp in ISO format (YYYY-MM-DD_HH:MM:SS)."""
        return datetime.fromtimestamp(self.timestamp).isoformat(sep="_")
# 全局变量 GlobalRank，表示一个整数类型
GlobalRank = int

# 失败信息格式模板，用于格式化显示失败的详细信息
_FAILURE_FORMAT_TEMPLATE = """[${idx}]:
  time      : ${time}
  host      : ${hostname}
  rank      : ${rank} (local_rank: ${local_rank})
  exitcode  : ${exitcode} (pid: ${pid})
  error_file: ${error_file}
  traceback : ${message}"""

# 消息格式模板，用于格式化显示复杂的消息和失败信息
_MSG_FORMAT_TEMPLATE = """
${boarder}
${title}
${section}
Failures:
${other_failures}
${section}
Root Cause (first observed failure):
${root_failure}
${boarder}"""

class ChildFailedError(Exception):
    """
    特殊的异常类型，可以从带有“@record”装饰器的函数中引发，
    以便子进程的根异常可以原样传播到调用栈上（例如，不包装在父进程的回溯中）。

    在父进程只是简单的看护进程，而子进程（工作进程）实际上在进行有意义的计算时非常有用。
    在这种情况下，由于父进程没有做任何非平凡的事情，子进程的错误应该传播到调度程序，
    以便进行准确的根本原因诊断。

    .. 注意:: 传播依赖于错误文件而不是异常处理，以支持函数和二进制启动。

    示例：
    ::

     # 主机（容器）上的进程树
     0: scheduler-init-process:
                |- 1: torchelastic_agent:
                         |- 2: trainer_0 (ok)
                         |- 3: trainer_1 (fail) -> error.json
                         |- ...
                         |- n+2: trainer_n (ok)
                |- n+3: other processes
                |- ...

    在上面的示例中，trainer 1 的失败（写入到 error.json 中）是根本原因，
    应该报告给调度程序的初始化进程。
    torchelastic 代理会在检测到 trainer 1 的失败时引发
    ``ChildFailedError("trainer", {1: "trainer_1/error.json"})``，
    这将把 trainer 1 错误文件的内容传播到调度程序的初始化进程。
    """

    def __init__(self, name: str, failures: Dict[GlobalRank, ProcessFailure]):
        self.name = name
        self.failures = failures
        assert (
            self.failures
        )  # 创建 ChildFailedError 时必须有失败信息，否则不合理
        super().__init__(self.format_msg())

    def get_first_failure(self) -> Tuple[GlobalRank, ProcessFailure]:
        # 找到时间戳最早的失败信息的全局排名和对应的进程失败对象
        rank = min(self.failures.keys(), key=lambda r: self.failures[r].timestamp)
        return rank, self.failures[rank]
    def format_msg(self, boarder_delim="=", section_delim="-"):
        # 构造标题，指明失败的名称
        title = f"{self.name} FAILED"
        # 获取第一个失败的等级和失败对象
        root_rank, root_failure = self.get_first_failure()

        # 初始化根失败和其他失败的格式化字符串
        root_failure_fmt: str = ""
        other_failures_fmt: List[str] = []
        # 初始化宽度为标题的长度
        width = len(title)
        
        # 遍历所有失败项，格式化并计算最大宽度
        for idx, (rank, failure) in enumerate(self.failures.items()):
            # 格式化单个失败项，并获取其宽度
            fmt, w = self._format_failure(idx, rank, failure)
            width = max(width, w)
            # 将根失败和其他失败分开存储
            if rank == root_rank:
                root_failure_fmt = fmt
            else:
                other_failures_fmt.append(fmt)

        # 最大宽度不超过60个字符
        width = min(width, 60)

        # 使用模板替换生成最终的格式化消息
        return Template(_MSG_FORMAT_TEMPLATE).substitute(
            boarder=boarder_delim * width,
            title=title,
            section=section_delim * width,
            root_failure=root_failure_fmt,
            other_failures="\n".join(other_failures_fmt or ["  <NO_OTHER_FAILURES>"]),
        )

    def _format_failure(
        self, idx: int, rank: int, failure: ProcessFailure
    ) -> Tuple[str, int]:
        # failure.message 可能是字符串（当失败没有生成回溯时，如信号）或字典形式的 JSON 数据
        # 字典形式包含 "message": $ERROR_MSG, "extraInfo": {"py_callstack": $TRACEBACK, timestamp: $TS}
        # 显示逻辑如下：
        # 1. 如果 failure.message 不是字典，则直接显示为字符串
        # 2. 否则尝试获取回溯（py_callstack）
        # 3.      如果回溯不存在，使用消息内容
        # 4.      如果消息不存在，显示 "<N/A>"
        msg = failure.message
        if isinstance(failure.message, dict):
            msg = (
                failure.message.get("extraInfo", {})
                .get("py_callstack", failure.message.get("message", "<N/A>"))
                .replace("\n", "\n  ")  # 用于正确缩进回溯信息
            )

        # 使用模板替换生成最终的失败消息
        fmt = Template(_FAILURE_FORMAT_TEMPLATE).substitute(
            idx=idx,
            time=failure.timestamp_isoformat(),
            hostname=socket.getfqdn(),
            rank=rank,
            local_rank=failure.local_rank,
            exitcode=failure.exitcode,
            pid=failure.pid,
            error_file=failure.error_file,
            message=msg,
        )
        # 计算格式化后消息的最大宽度
        width = 0
        for line in fmt.split("\n"):
            width = max(width, len(line))
        return fmt, width
# 定义一个装饰器函数record，接受一个函数fn和一个可选的错误处理器error_handler作为参数，并返回一个装饰后的函数
def record(
    fn: Callable[..., T], error_handler: Optional[ErrorHandler] = None
) -> Callable[..., T]:
    """
    Syntactic sugar to record errors/exceptions that happened in the decorated
    function using the provided ``error_handler``.

    Using this decorator is equivalent to:

    ::

     error_handler = get_error_handler()
     error_handler.initialize()
     try:
        foobar()
     except ChildFailedError as e:
        _, failure = e.get_first_failure()
        error_handler.dump_error_file(failure.error_file, failure.exitcode)
        raise
     except Exception as e:
        error_handler.record(e)
        raise

    .. important:: use this decorator once per process at the top level method,
                   typically this is the main method.

    Example

    ::

     @record
     def main():
         pass

     if __name__=="__main__":
        main()

    """
    # 如果没有传入error_handler参数，则获取默认的错误处理器
    if not error_handler:
        error_handler = get_error_handler()

    # 定义装饰函数wrap，接收被装饰的函数f作为参数
    def wrap(f):
        # 使用functools.wraps装饰器来保留原始函数f的元数据
        @wraps(f)
        # 定义内部包装函数wrapper，接收任意位置和关键字参数
        def wrapper(*args, **kwargs):
            # 断言确保error_handler不为None，用于mypy类型检查器
            assert error_handler is not None
            # 初始化错误处理器
            error_handler.initialize()
            try:
                # 调用被装饰的函数f，并返回其结果
                return f(*args, **kwargs)
            except SystemExit as se:
                # 捕获SystemExit异常，通常用于处理由sys.exit()引发的退出
                # 对于基于run_path的入口点，code=0的SystemExit不会退出，这里通过返回None处理
                if se.code == 0:
                    return None
                else:
                    raise
            except ChildFailedError as e:
                # 捕获ChildFailedError异常，获取第一个失败信息
                rank, failure = e.get_first_failure()
                # 如果错误文件不是_NOT_AVAILABLE，则调用错误处理器的dump_error_file方法
                if failure.error_file != _NOT_AVAILABLE:
                    error_handler.dump_error_file(failure.error_file, failure.exitcode)
                else:
                    # 否则记录日志提示错误文件不可用
                    logger.info(
                        (
                            "local_rank %s FAILED with no error file."
                            " Decorate your entrypoint fn with @record for traceback info."
                            " See: https://pytorch.org/docs/stable/elastic/errors.html",
                            rank,
                        )
                    )
                raise  # 继续抛出异常
            except Exception as e:
                # 捕获所有其他异常，调用错误处理器的record_exception方法记录异常
                error_handler.record_exception(e)
                raise  # 继续抛出异常

        return wrapper  # 返回内部包装函数wrapper

    return wrap(fn)  # 返回装饰函数wrap
```
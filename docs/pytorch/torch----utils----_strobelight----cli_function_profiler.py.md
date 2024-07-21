# `.\pytorch\torch\utils\_strobelight\cli_function_profiler.py`

```py
# mypy: disallow-untyped-defs

# 导入必要的模块
import functools  # 提供了创建部分函数的功能
import logging  # 日志记录模块
import os  # 操作系统功能接口
import re  # 正则表达式模块
import subprocess  # 用于执行子进程的模块
import time  # 时间相关操作模块
from threading import Lock  # 多线程同步工具：锁
from typing import Any, List, Optional, Sequence  # 类型注解相关

# 创建一个名为"strobelight_function_profiler"的日志记录器对象
logger = logging.getLogger("strobelight_function_profiler")

# 创建一个控制台日志处理器
console_handler = logging.StreamHandler()

# 设置日志格式
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)

# 将格式应用到控制台处理器
console_handler.setFormatter(formatter)

# 将控制台处理器添加到日志记录器
logger.addHandler(console_handler)

# 设置日志记录器的日志级别为INFO
logger.setLevel(logging.INFO)

# 禁止日志记录器向上传播到父记录器
logger.propagate = False


class StrobelightCLIProfilerError(Exception):
    """
    Raised when an error happens during strobelight profiling
    """


def _pid_namespace_link(pid: Optional[int] = None) -> str:
    """Returns the link to the process's namespace, example: pid:[4026531836]"""
    PID_NAMESPACE_PATH = "/proc/{}/ns/pid"
    # 如果未指定pid，则使用当前进程的PID
    pid = pid or os.getpid()
    # 返回指定PID进程的命名空间链接路径
    return os.readlink(PID_NAMESPACE_PATH.format(pid))


def _pid_namespace(pid: Optional[int] = None) -> int:
    """Returns the process's namespace id"""
    # 如果未指定pid，则使用当前进程的PID
    pid = pid or os.getpid()
    # 获取指定PID进程的命名空间ID
    link = _pid_namespace_link(pid)
    return int(link[link.find("[") + 1: -1])


def _command_to_string(command: Sequence[str]) -> str:
    """Converts a command sequence to a single string"""
    # 将命令序列转换为单个字符串并返回
    return " ".join(command)


class StrobelightCLIFunctionProfiler:
    """
    Note: this is a meta only tool.

    StrobelightCLIFunctionProfiler can be used to profile a python function and
    generate a strobelight link with the results. It works on meta servers but
    does not requries an fbcode target.
    When stop_at_error is false(default), error during profiling does not prevent
    the work function from running.

    Check function_profiler_example.py for an example.
    """

    # This lock is used to make sure only one thread is running the profiler at any point.
    # 这把锁用于确保在任何时刻只有一个线程运行分析器
    _lock = Lock()

    def __init__(
        self,
        *,
        stop_at_error: bool = False,
        max_profile_duration_sec: int = 60 * 10,
        sample_each: float = 1e7,  # sample each sample_each cycles.
        run_user_name: str = "pytorch-strobelight-ondemand",
        timeout_wait_for_running_sec: int = 60,
        timeout_wait_for_finished_sec: int = 60,
        recorded_env_variables: Optional[List[str]] = None,
        sample_tags: Optional[List[str]] = None,
        stack_max_len: int = 127,
        async_stack_max_len: int = 127,
    ):
        # 是否在出错时停止，默认为False
        self.stop_at_error = stop_at_error
        # 最大分析持续时间（秒），默认为600秒（10分钟）
        self.max_profile_duration_sec = max_profile_duration_sec
        # 每次采样周期数，默认为1e7
        self.sample_each = sample_each
        # 运行用户名称，默认为"pytorch-strobelight-ondemand"
        self.run_user_name = run_user_name
        # 等待运行状态的超时时间（秒），默认为60秒
        self.timeout_wait_for_running_sec = timeout_wait_for_running_sec
        # 等待完成状态的超时时间（秒），默认为60秒
        self.timeout_wait_for_finished_sec = timeout_wait_for_finished_sec
        # 最近运行的结果
        # 记录最近一次运行的strobelight运行ID
        self.current_run_id: Optional[int] = None
        # 采样标签，可选参数
        self.sample_tags = sample_tags

        # 堆栈最大长度，默认为127
        self.stack_max_len = stack_max_len
        # 异步堆栈最大长度，默认为127
        self.async_stack_max_len = async_stack_max_len
    # 异步运行方法，执行与性能分析相关的命令
    def _run_async(self) -> None:
        # 获取当前进程的进程号
        processId = os.getpid()
        # 使用进程号获取命名空间
        namespace = _pid_namespace(processId)
        # 构建运行命令列表，包括 strobeclient 命令和各种选项
        command = [
            "strobeclient",
            "run",
            "--profiler",
            "pyperf",
            "--event",
            "cycles",
            "--async",
            "--sample-interval",
            f"{int(self.sample_each)}",
            "--duration-ms",
            f"{int(self.max_profile_duration_sec * 1000)}",
            "--pid",
            f"{namespace}:{processId}",
        ]

        # 如果设置了 sample_tags，则添加到命令中
        if self.sample_tags:
            command.append("--sample-tags")
            command.append(",".join(self.sample_tags))

        # 输出调试信息，显示将要执行的命令
        logger.debug("running command: %s", _command_to_string(command))
        # 执行命令并捕获输出
        result = subprocess.run(command, capture_output=True)
        # 解码标准错误输出作为字符串
        output = result.stderr.decode("utf-8")
        # 输出调试信息，显示命令执行后的输出内容
        logger.debug("output:\n{%s}", output)

        # 如果命令返回码不为 0，则抛出异常，显示运行失败信息
        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in run_async:{output}"
            )

        # 如果在输出中找到 "INFO Run Id: (-?\d+)" 格式的信息，则解析出当前运行的 ID
        if match := re.search(r"INFO Run Id: (-?\d+)", output):
            self.current_run_id = int(match.group(1))
            return

        # 如果未找到匹配信息，则抛出异常，显示无法启动性能分析的错误信息
        raise StrobelightCLIProfilerError(
            f"failed to start strobelight profiling, unexpected result {output}"
        )

    # 等待运行状态为 RUNNING，或者在 PREPARING 状态时进行等待，最多尝试 20 次
    def _wait_for_running(self, counter: int = 0) -> None:
        if counter > 20:
            raise StrobelightCLIProfilerError(
                "wait_for_running called more than 20 times"
            )

        # 构建获取运行状态命令的列表，包括当前运行的 ID
        command = ["strobeclient", "getRunStatus", "--run-id", f"{self.current_run_id}"]
        # 输出调试信息，显示将要执行的命令
        logger.debug("running command: %s", _command_to_string(command))
        # 执行命令并捕获输出
        result = subprocess.run(command, capture_output=True)
        # 解码标准错误输出作为字符串
        output = result.stderr.decode("utf-8")
        # 输出调试信息，显示命令执行后的输出内容
        logger.debug("output:\n{%s}", output)

        # 如果命令返回码不为 0，则抛出异常，显示获取运行状态失败的错误信息
        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in wait_for_running:{output}"
            )

        # 如果在输出中找到 "Profile run status: (.*)" 格式的信息，则获取当前运行状态
        if match := re.search("Profile run status: (.*)", output):
            current_status = match.group(1)
            # 如果当前状态为 "RUNNING"，则返回
            if current_status == "RUNNING":
                return
            # 如果当前状态为 "PREPARING"，则等待 10 秒后再次调用 _wait_for_running 方法
            elif current_status == "PREPARING":
                time.sleep(10)
                self._wait_for_running(counter + 1)
                return
            # 如果状态既不是 "RUNNING" 也不是 "PREPARING"，则抛出异常，显示意外的运行状态信息
            else:
                raise StrobelightCLIProfilerError(f"unexpected {current_status} phase")

        # 如果未找到匹配信息，则抛出异常，显示意外的输出信息
        raise StrobelightCLIProfilerError(f"unexpected output\n: {output} ")
    # 停止当前运行的 strobelight 任务
    def _stop_run(self) -> None:
        # 构建停止运行的命令
        command = ["strobeclient", "stopRun", "--run-id", str(self.current_run_id)]
        # 记录调试信息，显示即将执行的命令
        logger.debug("running command: %s", _command_to_string(command))
        # 执行命令并捕获输出
        result = subprocess.run(command, capture_output=True)
        # 解码标准错误输出
        output = result.stderr.decode("utf-8")
        # 记录调试信息，显示命令的输出内容
        logger.debug("output:\n{%s}", output)

        # 检查命令返回码，若不为 0 则抛出异常
        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to stop strobelight profiling, return code is not 0 :{output}"
            )

        # 使用正则表达式搜索输出中的特定信息
        if match := re.search("INFO ::1:(.*)", output):
            current_status = match.group(1)
            # 检查当前状态中是否包含 "Success!"，表示成功停止
            if current_status.__contains__("Success!"):
                return
            else:
                # 若未成功停止，则抛出异常
                raise StrobelightCLIProfilerError(
                    f"failed to stop strobelight profiling, got {current_status} result"
                )

        # 若未能从输出中匹配到预期的信息，则抛出异常
        raise StrobelightCLIProfilerError(f"unexpected output\n: {output} ")

    # 获取当前 strobelight 任务的运行结果
    def _get_results(self) -> None:
        # 构建获取运行状态的命令
        command = ["strobeclient", "getRunStatus", "--run-id", str(self.current_run_id)]
        # 记录调试信息，显示即将执行的命令
        logger.debug("running command: %s", _command_to_string(command))
        # 执行命令并捕获输出
        result = subprocess.run(command, capture_output=True)
        # 解码标准错误输出
        output = result.stderr.decode("utf-8")
        # 记录调试信息，显示命令的输出内容
        logger.debug("output:\n{%s}", output)

        # 检查命令返回码，若不为 0 则抛出异常
        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to extract profiling results, return code is not 0 : {output}"
            )

        # 使用正则表达式搜索输出中的特定信息
        if match := re.search("INFO ::1:(.*)", output):
            current_status = match.group(1)
            # 检查当前状态中是否包含 "Profile run status: PROCESSING"，表示处理中
            if current_status.__contains__("Profile run status: PROCESSING"):
                # 若状态为处理中，等待 10 秒后递归调用获取结果方法
                time.sleep(10)
                self._get_results()
                return
            # 若未完成且状态不包含 "Profile run finished with SUCCESS"，则抛出异常
            elif not current_status.__contains__("Profile run finished with SUCCESS"):
                raise StrobelightCLIProfilerError(
                    f"failed to extract profiling results, unexpected response {output}"
                )

        # 使用正则表达式遍历输出，记录特定信息到日志
        for item in re.findall(
            r"(Total samples(.*)|GraphProfiler(.*)|Icicle view \(python stack\)(.*))",
            output,
        ):
            logger.info(item[0])

    # 安全地停止 strobelight 任务，若出现异常则记录警告信息而非抛出异常
    def _stop_strobelight_no_throw(
        self,
        collect_results: bool,
    ) -> None:
        try:
            # 调用停止运行方法
            self._stop_run()
            # 记录信息，指示 strobelight 任务已停止
            logger.info("strobelight profiling stopped")

            # 记录调试信息，指示收集已停止
            logger.debug("collection stopped")

            # 若不需要收集结果，则直接返回
            if not collect_results:
                return

            # 否则，调用获取结果方法
            self._get_results()
        except Exception as error:
            # 捕获所有异常，并记录警告信息及异常详情
            logger.warning("error during stop_strobelight", exc_info=True)

    # 如果 strobelight 已启动且正在运行，则返回 True。不抛出异常。
    # 启动 strobelight 分析器的内部方法，返回布尔值表示成功或失败
    def _start_strobelight(self) -> bool:
        strobelight_started = False
        try:
            # 异步运行内部方法
            self._run_async()
            strobelight_started = True
            # 记录当前运行的 ID 到日志
            logger.info("strobelight run id is: %s", self.current_run_id)
            # 等待运行状态
            self._wait_for_running()
            # 记录 strobelight 分析正在运行的消息到日志
            logger.info("strobelight profiling running")
            return True

        except Exception as error:
            # 记录启动 strobelight 过程中的异常信息到警告日志，并捕获异常
            logger.warning("error during start_strobelight:", exc_info=True)
            # 如果 strobelight 已经启动，尝试停止它并不抛出异常
            if strobelight_started:
                self._stop_strobelight_no_throw(collect_results=False)
            return False

    # 进行函数性能分析的方法，返回与工作函数执行相关的结果
    def profile(self, work_function: Any, *args: Any, **kwargs: Any) -> Any:
        self.current_run_id = None

        # 尝试获取锁，避免并发运行问题
        if locked := StrobelightCLIFunctionProfiler._lock.acquire(False):
            # 如果没有获取到锁
            if not locked:
                # 如果设置了在错误时停止，则抛出异常
                if self.stop_at_error:
                    raise StrobelightCLIProfilerError("concurrent runs not supported")
                
                # 记录警告信息：不支持并发运行
                logger.warning("concurrent runs not supported")
                # 直接执行工作函数
                return work_function(*args, **kwargs)

            # 启动 strobelight 分析器
            started = self._start_strobelight()
            # 如果启动失败
            if not started:
                # 如果设置了在错误时停止，则释放锁并抛出异常
                if self.stop_at_error:
                    StrobelightCLIFunctionProfiler._lock.release()
                    raise StrobelightCLIProfilerError(
                        "failed to start strobelight profiling"
                    )
                # 否则，直接执行工作函数
                result = work_function(*args, **kwargs)
                StrobelightCLIFunctionProfiler._lock.release()
                return result

            try:
                # 记录调试信息：开始收集数据
                logger.debug("collection started")
                # 执行工作函数并获取结果
                result = work_function(*args, **kwargs)
                # 尝试停止 strobelight 分析器，收集结果
                self._stop_strobelight_no_throw(collect_results=True)
                StrobelightCLIFunctionProfiler._lock.release()
                return result
            except Exception as error:
                # 记录警告信息：工作函数抛出异常
                logger.warning("work function throw exception", exc_info=True)
                # 停止 strobelight 分析器，但不收集结果
                self._stop_strobelight_no_throw(collect_results=False)
                StrobelightCLIFunctionProfiler._lock.release()
                # 重新抛出捕获到的异常
                raise error
# A function decorator that wraps profile, if no profiler is provided one with
# default args is created. A function can be annotated as:
# @strobelight()
# @strobelight(profiler = StrobelightFunctionProfiler(stop_at_error=True,..))
# @strobelight(stop_at_error=True,...)
def strobelight(
    profiler: Optional[StrobelightCLIFunctionProfiler] = None, **kwargs: Any
) -> Any:
    # 如果没有传入 profiler 参数，则创建一个带有默认参数的 StrobelightCLIFunctionProfiler 对象
    if not profiler:
        profiler = StrobelightCLIFunctionProfiler(**kwargs)

    # 内部函数定义，用于实际包装工作函数
    def strobelight_inner(work_function: Any) -> Any:
        # 包装函数，使用 functools.wraps 保留原始工作函数的元数据
        @functools.wraps(work_function)
        def wrapper_function(*args: Any, **kwargs: Any) -> Any:
            # 调用 profiler 的 profile 方法来执行工作函数，并返回其结果
            return profiler.profile(work_function, *args, **kwargs)

        return wrapper_function

    return strobelight_inner
```
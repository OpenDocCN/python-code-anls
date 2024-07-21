# `.\pytorch\torch\_strobelight\cli_function_profiler.py`

```py
# mypy: disallow-untyped-defs

# 导入所需的模块和库
import functools  # 导入 functools 模块，用于高阶函数和函数修饰器
import logging  # 导入 logging 模块，用于记录日志
import os  # 导入 os 模块，用于操作系统相关功能
import re  # 导入 re 模块，用于正则表达式操作
import subprocess  # 导入 subprocess 模块，用于执行外部命令
import time  # 导入 time 模块，用于时间相关操作
from threading import Lock  # 从 threading 模块导入 Lock 类，用于线程同步
from timeit import default_timer as timer  # 从 timeit 模块导入 default_timer 作为 timer

# 创建名为 'strobelight_function_profiler' 的 logger 对象
logger = logging.getLogger("strobelight_function_profiler")

# 创建一个控制台日志处理器
console_handler = logging.StreamHandler()

# 设置日志格式
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)

# 将格式应用到控制台日志处理器
console_handler.setFormatter(formatter)

# 将控制台日志处理器添加到 logger
logger.addHandler(console_handler)

# 设置 logger 的日志级别为 INFO
logger.setLevel(logging.INFO)

# 禁止 logger 传播日志到其它 logger
logger.propagate = False


class StrobelightCLIProfilerError(Exception):
    """
    Raised when an error happens during strobelight profiling
    """


def _pid_namespace_link(pid: Optional[int] = None) -> str:
    """Returns the link to the process's namespace, example: pid:[4026531836]"""
    PID_NAMESPACE_PATH = "/proc/{}/ns/pid"
    pid = pid or os.getpid()
    return os.readlink(PID_NAMESPACE_PATH.format(pid))


def _pid_namespace(pid: Optional[int] = None) -> int:
    """Returns the process's namespace id"""
    pid = pid or os.getpid()
    link = _pid_namespace_link(pid)
    return int(link[link.find("[") + 1 : -1])


def _command_to_string(command: Sequence[str]) -> str:
    """Converts a list of strings into a single command string"""
    return " ".join(command)


class StrobelightCLIFunctionProfiler:
    """
    Note: this is a Meta only tool.

    StrobelightCLIFunctionProfiler can be used to profile a python function and
    generate a strobelight link with the results. It works on meta servers but
    does not requries an fbcode target.
    When stop_at_error is false(default), error during profiling does not prevent
    the work function from running.

    Check function_profiler_example.py for an example.
    """

    # This lock is used to make sure only one thread is running the profiler at any point.
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
        """
        Initialize the StrobelightCLIFunctionProfiler object.

        Parameters:
        - stop_at_error: If True, stops the profiler on error; otherwise, continues.
        - max_profile_duration_sec: Maximum duration for profiling in seconds.
        - sample_each: Number of cycles to sample each time.
        - run_user_name: Username under which the profiler runs.
        - timeout_wait_for_running_sec: Timeout duration to wait for profiler to start running.
        - timeout_wait_for_finished_sec: Timeout duration to wait for profiler to finish.
        - recorded_env_variables: List of environment variables to record during profiling.
        - sample_tags: List of tags associated with the samples.
        - stack_max_len: Maximum length of stack trace to record.
        - async_stack_max_len: Maximum length of asynchronous stack trace to record.
        """
        self.stop_at_error = stop_at_error
        self.max_profile_duration_sec = max_profile_duration_sec
        self.sample_each = sample_each
        self.run_user_name = run_user_name
        self.timeout_wait_for_running_sec = timeout_wait_for_running_sec
        self.timeout_wait_for_finished_sec = timeout_wait_for_finished_sec
        # Results of the most recent run.
        # Tracks the strobelight run id of the most recent run
        self.current_run_id: Optional[int] = None
        self.profile_result: Optional[List[str]] = None
        self.sample_tags = sample_tags
    # 异步运行方法，用于启动 strobeclient 进行性能分析
    def _run_async(self) -> None:
        # 获取当前进程 ID
        processId = os.getpid()
        # 根据进程 ID 获取命名空间
        namespace = _pid_namespace(processId)
        # 构建 strobeclient 命令行参数列表
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

        # 如果有指定采样标签，则添加到命令行参数中
        if self.sample_tags:
            command.append("--sample-tags")
            command.append(",".join(self.sample_tags))

        # 记录调试日志，显示执行的命令
        logger.debug("running command: %s", _command_to_string(command))
        # 运行命令，并捕获输出
        result = subprocess.run(command, capture_output=True)
        # 解码标准错误输出作为字符串
        output = result.stderr.decode("utf-8")
        # 记录调试日志，显示命令执行的输出内容
        logger.debug("output:\n{%s}", output)

        # 如果命令执行返回非零状态码，则抛出异常
        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in run_async:{output}"
            )

        # 从命令执行输出中匹配并提取运行 ID
        if match := re.search(r"INFO Run Id: (-?\d+)", output):
            self.current_run_id = int(match.group(1))
            return

        # 如果未能从输出中匹配到运行 ID，则抛出异常
        raise StrobelightCLIProfilerError(
            f"failed to start strobelight profiling, unexpected result {output}"
        )

    # 等待性能分析任务运行状态的方法
    def _wait_for_running(self, counter: int = 0) -> None:
        # 如果计数器超过 20 次，则抛出异常
        if counter > 20:
            raise StrobelightCLIProfilerError(
                "wait_for_running called more than 20 times"
            )

        # 构建获取运行状态的 strobeclient 命令行参数列表
        command = ["strobeclient", "getRunStatus", "--run-id", f"{self.current_run_id}"]
        # 记录调试日志，显示执行的命令
        logger.debug("running command: %s", _command_to_string(command))
        # 运行命令，并捕获输出
        result = subprocess.run(command, capture_output=True)
        # 解码标准错误输出作为字符串
        output = result.stderr.decode("utf-8")
        # 记录调试日志，显示命令执行的输出内容
        logger.debug("output:\n{%s}", output)

        # 如果命令执行返回非零状态码，则抛出异常
        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in wait_for_running:{output}"
            )

        # 从输出中匹配运行状态信息
        if match := re.search("Profile run status: (.*)", output):
            current_status = match.group(1)
            # 如果当前状态为 RUNNING，则任务正在运行，直接返回
            if current_status == "RUNNING":
                return
            # 如果当前状态为 PREPARING，则等待 10 秒后递归调用本方法
            elif current_status == "PREPARING":
                time.sleep(10)
                self._wait_for_running(counter + 1)
                return
            # 如果出现其他未预期的状态，则抛出异常
            else:
                raise StrobelightCLIProfilerError(f"unexpected {current_status} phase")

        # 如果未能从输出中匹配到有效的运行状态信息，则抛出异常
        raise StrobelightCLIProfilerError(f"unexpected output\n: {output} ")
    # 停止当前运行的 strobelight 采集任务
    def _stop_run(self) -> None:
        # 构建停止运行命令
        command = ["strobeclient", "stopRun", "--run-id", str(self.current_run_id)]
        # 记录调试信息，输出运行的命令字符串
        logger.debug("running command: %s", _command_to_string(command))
        # 执行命令并捕获输出结果
        result = subprocess.run(command, capture_output=True)
        # 解码标准错误输出
        output = result.stderr.decode("utf-8")
        # 记录调试信息，输出标准错误输出内容
        logger.debug("output:\n{%s}", output)

        # 如果返回码不为 0，则抛出异常
        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to stop strobelight profiling, return code is not 0 :{output}"
            )

        # 从输出中查找匹配的信息
        if match := re.search("INFO ::1:(.*)", output):
            # 获取匹配的状态信息
            current_status = match.group(1)
            # 如果状态信息包含 "Success!"，则正常返回
            if current_status.__contains__("Success!"):
                return
            else:
                # 否则抛出异常，指示停止 strobelight 采集失败
                raise StrobelightCLIProfilerError(
                    f"failed to stop strobelight profiling, got {current_status} result"
                )

        # 如果未能匹配到预期的状态信息，抛出异常，指示输出不符合预期
        raise StrobelightCLIProfilerError(f"unexpected output\n: {output} ")

    # 获取 strobelight 采集的结果
    def _get_results(self) -> None:
        # 构建获取运行状态命令
        command = ["strobeclient", "getRunStatus", "--run-id", str(self.current_run_id)]
        # 记录调试信息，输出运行的命令字符串
        logger.debug("running command: %s", _command_to_string(command))
        # 执行命令并捕获输出结果
        result = subprocess.run(command, capture_output=True)
        # 解码标准错误输出
        output = result.stderr.decode("utf-8")
        # 记录调试信息，输出标准错误输出内容
        logger.debug("output:\n{%s}", output)

        # 如果返回码不为 0，则抛出异常
        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to extract profiling results, return code is not 0 : {output}"
            )

        # 从输出中查找匹配的信息
        if match := re.search("INFO ::1:(.*)", output):
            # 获取匹配的状态信息
            current_status = match.group(1)
            # 如果状态信息包含 "Profile run status: PROCESSING"，则等待一段时间后再次获取结果
            if current_status.__contains__("Profile run status: PROCESSING"):
                time.sleep(10)
                self._get_results()
                return
            # 如果状态信息不包含 "Profile run finished with SUCCESS"，则抛出异常，指示获取结果失败
            elif not current_status.__contains__("Profile run finished with SUCCESS"):
                raise StrobelightCLIProfilerError(
                    f"failed to extract profiling results, unexpected response {output}"
                )

        # 初始化存储结果的列表
        self.profile_result = []
        # 使用正则表达式匹配输出中的各项结果，并添加到结果列表中
        for item in re.findall(
            r"(Total samples(.*)|GraphProfiler(.*)|Icicle view \(python stack\)(.*))",
            output,
        ):
            self.profile_result += item[0]
            # 记录信息日志，输出匹配到的结果项
            logger.info(item[0])

    # 尝试停止 strobelight 采集任务，并根据参数决定是否收集结果，不抛出异常
    def _stop_strobelight_no_throw(
        self,
        collect_results: bool,
    ) -> None:
        try:
            # 调用停止运行方法
            self._stop_run()
            # 记录信息日志，指示 strobelight 采集已停止
            logger.info("strobelight profiling stopped")

            # 记录调试信息，输出收集已停止的消息
            logger.debug("collection stopped")

            # 如果不需要收集结果，则直接返回
            if not collect_results:
                return

            # 否则调用获取结果方法
            self._get_results()
        except Exception as error:
            # 捕获任何异常，记录警告信息，包括异常栈信息
            logger.warning("error during stop_strobelight", exc_info=True)

    # 返回 true，表示 strobelight 已启动并正在运行，不抛出异常
    # 启动 strobelight，进行函数性能分析
    def _start_strobelight(self) -> bool:
        strobelight_started = False  # 初始化 strobelight 启动状态为 False
        try:
            self._run_async()  # 异步运行 strobelight
            strobelight_started = True  # 设置 strobelight 启动状态为 True
            logger.info("strobelight run id is: %s", self.current_run_id)  # 记录 strobelight 的运行 ID
            self._wait_for_running()  # 等待 strobelight 运行就绪
            logger.info("strobelight profiling running")  # 记录 strobelight 正在进行性能分析
            return True  # 返回启动成功

        except Exception as error:
            logger.warning("error during start_strobelight:", exc_info=True)  # 记录启动 strobelight 过程中的错误
            if strobelight_started:
                self._stop_strobelight_no_throw(collect_results=False)  # 如果 strobelight 已启动，则尝试停止它
            return False  # 返回启动失败

    # 进行函数性能分析
    def profile(self, work_function: Any, *args: Any, **kwargs: Any) -> Any:
        self.current_run_id = None  # 初始化当前运行 ID 为 None
        self.profile_result = None  # 初始化性能分析结果为 None

        # 尝试获取锁，以防止并发运行
        if locked := StrobelightCLIFunctionProfiler._lock.acquire(False):
            if not locked:
                # 如果未能获取锁且设置了在错误时停止，则抛出异常
                if self.stop_at_error:
                    raise StrobelightCLIProfilerError("concurrent runs not supported")

                logger.warning("concurrent runs not supported")  # 记录并发运行不支持的警告
                return work_function(*args, **kwargs)  # 直接执行工作函数

            started = self._start_strobelight()  # 启动 strobelight 进行性能分析
            if not started:
                if self.stop_at_error:
                    StrobelightCLIFunctionProfiler._lock.release()
                    raise StrobelightCLIProfilerError(
                        "failed to start strobelight profiling"
                    )  # 如果启动 strobelight 失败且设置了在错误时停止，则抛出异常
                result = work_function(*args, **kwargs)  # 启动失败时执行工作函数
                StrobelightCLIFunctionProfiler._lock.release()
                return result

            try:
                logger.debug("collection started")  # 记录数据收集已开始
                start = timer()
                result = work_function(*args, **kwargs)  # 执行工作函数
                end = timer()
                total_time = end - start  # 计算工作函数执行时间
                logger.info("work function took %s seconds", total_time)  # 记录工作函数执行时间
                self._stop_strobelight_no_throw(collect_results=True)  # 停止 strobelight 并收集结果
                StrobelightCLIFunctionProfiler._lock.release()
                return result

            except Exception as error:
                logger.warning("work function throw exception", exc_info=True)  # 记录工作函数抛出异常
                self._stop_strobelight_no_throw(collect_results=False)  # 停止 strobelight 不收集结果
                StrobelightCLIFunctionProfiler._lock.release()
                raise error  # 重新抛出异常
# 一个函数装饰器，用于包装 profile 函数，如果没有提供分析器，则创建一个带有默认参数的分析器。
# 函数可以被注释为:
# @strobelight()
# @strobelight(profiler = StrobelightFunctionProfiler(stop_at_error=True,..))
# @strobelight(stop_at_error=True,...)
def strobelight(
    profiler: Optional[StrobelightCLIFunctionProfiler] = None, **kwargs: Any
) -> Any:
    # 如果没有提供分析器，则创建一个带有指定参数的分析器
    if not profiler:
        profiler = StrobelightCLIFunctionProfiler(**kwargs)

    # 定义内部函数 strobelight_inner，用于包装工作函数
    def strobelight_inner(work_function: Any) -> Any:
        # 使用 functools.wraps 装饰器保留原始函数的元数据
        @functools.wraps(work_function)
        # 定义包装函数 wrapper_function，用于调用分析器的 profile 方法
        def wrapper_function(*args: Any, **kwargs: Any) -> Any:
            return profiler.profile(work_function, *args, **kwargs)

        return wrapper_function

    return strobelight_inner
```
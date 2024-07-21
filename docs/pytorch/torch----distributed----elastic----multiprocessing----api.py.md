# `.\pytorch\torch\distributed\elastic\multiprocessing\api.py`

```
# 标识当前操作系统是否为 Windows
IS_WINDOWS = sys.platform == "win32"
# 标识当前操作系统是否为 macOS
IS_MACOS = sys.platform == "darwin"

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

class SignalException(Exception):
    """
    定义一个自定义异常类，用于在 torchelastic 代理进程中由终止处理程序引发异常，
    表示进程接收到终止信号。
    """

    def __init__(self, msg: str, sigval: signal.Signals) -> None:
        super().__init__(msg)
        self.sigval = sigval


def _terminate_process_handler(signum: int, frame: Optional[FrameType]) -> None:
    """
    终止处理程序，在主进程接收到终止信号时引发异常。

    当进程接收到死亡信号(SIGTERM, SIGINT)时，将调用此终止处理程序。它会引发 ``SignalException`` 
    异常，用户代码应该处理此异常。Python 在终止处理程序执行完毕后不会立即终止进程，因此不应该忽略异常，
    否则进程将永远不会终止。
    """
    sigval = signal.Signals(signum)
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)


def _get_kill_signal() -> signal.Signals:
    """
    获取杀死进程的信号。对于 Unix 系统返回 SIGKILL，对于 Windows 返回 CTRL_C_EVENT。
    """
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined] # noqa: F821
    else:
        return signal.SIGKILL


def _get_default_signal() -> signal.Signals:
    """
    获取默认的终止信号。对于 Unix 系统返回 SIGTERM，对于 Windows 返回 CTRL_C_EVENT。
    """
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined] # noqa: F821
    else:
        return signal.SIGTERM
# 验证给定字典的键是否与预期的进程数匹配
def _validate_full_rank(d: Dict[int, Any], nprocs: int, what: str):
    # 获取实际字典中的键集合
    actual_keys = set(d.keys())
    # 创建一个预期键的集合，范围从0到nprocs-1
    expected_keys = set(range(nprocs))

    # 如果实际键集合与预期键集合不相等，则引发运行时错误
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"{what}, local rank mapping mismatch,"
            f" expected: {expected_keys}, actual: {actual_keys}"
        )


# 匹配用于映射验证的正则表达式，用于检查形如 "0:1,1:2" 的字符串
_MAPPING_REGEX = r"^(\d:[0123],)*(\d:[0123])$"
# 匹配用于值验证的正则表达式，用于检查形如 "0" 到 "3" 的单个数字字符串
_VALUE_REGEX = r"^[0123]$"


# 定义标准输出和标准错误的枚举类
class Std(IntFlag):
    NONE = 0
    OUT = 1
    ERR = 2
    ALL = OUT | ERR

    @classmethod
    def from_str(cls, vm: str) -> Union["Std", Dict[int, "Std"]]:
        """
        根据输入的字符串返回相应的 Std 枚举值或者映射字典。

        示例:
        ::
        
         from_str("0") -> Std.NONE
         from_str("1") -> Std.OUT
         from_str("0:3,1:0,2:1,3:2") -> {0: Std.ALL, 1: Std.NONE, 2: Std.OUT, 3: Std.ERR}

        其它任何输入都会引发异常
        """

        def to_std(v: str) -> Std:  # type: ignore[return]
            # 将字符串转换为 Std 枚举值
            s = Std(int(v))
            if s in Std:
                return s
            # 不应该到达此处，因为输入已经通过正则表达式检查

        # 如果输入是一个数字字符串（例如 "0"）
        if re.match(_VALUE_REGEX, vm):
            return to_std(vm)
        # 如果输入是一个映射字符串（例如 "0:1,1:2"）
        elif re.match(_MAPPING_REGEX, vm):
            d: Dict[int, Std] = {}
            # 解析映射字符串，创建字典映射
            for m in vm.split(","):
                i, v = m.split(":")
                d[int(i)] = to_std(v)
            return d
        else:
            # 如果输入既不符合 <_VALUE_REGEX> 也不符合 <_MAPPING_REGEX>，则引发值错误异常
            raise ValueError(
                f"{vm} does not match: <{_VALUE_REGEX}> or <{_MAPPING_REGEX}>"
            )


def to_map(
    val_or_map: Union[Std, Dict[int, Std]], local_world_size: int
) -> Dict[int, Std]:
    """
    将单个值或者映射转换为映射字典的便捷方法。

    示例:
    ::

     to_map(Std.OUT, local_world_size=2) # 返回: {0: Std.OUT, 1: Std.OUT}
     to_map({1: Std.OUT}, local_world_size=2) # 返回: {0: Std.NONE, 1: Std.OUT}
     to_map({0: Std.OUT, 1: Std.OUT}, local_world_size=2) # 返回: {0: Std.OUT, 1: Std.OUT}
    """
    # 如果输入是一个 Std 枚举值，则创建一个映射字典，将该值应用到所有本地进程
    if isinstance(val_or_map, Std):
        return dict.fromkeys(range(local_world_size), val_or_map)
    else:
        map = {}
        # 如果输入是一个映射字典，则根据本地进程数创建新的映射字典
        for i in range(local_world_size):
            map[i] = val_or_map.get(i, Std.NONE)
        return map


@dataclass
class LogsDest:
    """
    每种日志类型，保存本地进程 ID 到文件路径的映射。
    """

    stdouts: Dict[int, str] = field(default_factory=dict)
    stderrs: Dict[int, str] = field(default_factory=dict)
    tee_stdouts: Dict[int, str] = field(default_factory=dict)
    tee_stderrs: Dict[int, str] = field(default_factory=dict)
    error_files: Dict[int, str] = field(default_factory=dict)


class LogsSpecs(ABC):
    """
    定义每个工作进程的日志处理和重定向。
    """
    """
    Args:
        log_dir:
            Base directory where logs will be written.
        redirects:
            Streams to redirect to files. Pass a single ``Std``
            enum to redirect for all workers, or a mapping keyed
            by local_rank to selectively redirect.
        tee:
            Streams to duplicate to stdout/stderr.
            Pass a single ``Std`` enum to duplicate streams for all workers,
            or a mapping keyed by local_rank to selectively duplicate.
    """
    
    class MyClass:
        """
        Initializes an instance of MyClass with optional parameters for logging and stream redirection.
    
        Attributes:
            _root_log_dir (Optional[str]): Base directory for log files.
            _redirects (Union[Std, Dict[int, Std]]): Specification for stream redirection.
            _tee (Union[Std, Dict[int, Std]]): Specification for stream duplication.
            _local_ranks_filter (Optional[Set[int]]): Set of local ranks to filter operations on.
    
        Args:
            log_dir (Optional[str]): Base directory for log files.
            redirects (Union[Std, Dict[int, Std]]): Specification for stream redirection.
            tee (Union[Std, Dict[int, Std]]): Specification for stream duplication.
            local_ranks_filter (Optional[Set[int]]): Set of local ranks to filter operations on.
        """
    
        def __init__(
            self,
            log_dir: Optional[str] = None,
            redirects: Union[Std, Dict[int, Std]] = Std.NONE,
            tee: Union[Std, Dict[int, Std]] = Std.NONE,
            local_ranks_filter: Optional[Set[int]] = None,
        ) -> None:
            """
            Initializes MyClass with specified parameters.
    
            Args:
                log_dir (Optional[str]): Base directory for log files.
                redirects (Union[Std, Dict[int, Std]]): Specification for stream redirection.
                tee (Union[Std, Dict[int, Std]]): Specification for stream duplication.
                local_ranks_filter (Optional[Set[int]]): Set of local ranks to filter operations on.
            """
            self._root_log_dir = log_dir
            self._redirects = redirects
            self._tee = tee
            self._local_ranks_filter = local_ranks_filter
    
        @abstractmethod
        def reify(
            self,
            envs: Dict[int, Dict[str, str]],
        ) -> LogsDest:
            """
            Abstract method to build the destination of log files based on environment variables.
    
            Args:
                envs (Dict[int, Dict[str, str]]): Environment variables for each local rank.
    
            Returns:
                LogsDest: Destination of log files.
            """
            pass
    
        @property
        @abstractmethod
        def root_log_dir(self) -> str:
            """
            Abstract property for retrieving the root log directory.
    
            Returns:
                str: Root directory for log files.
            """
            pass
    """
class DefaultLogsSpecs(LogsSpecs):
    """
    Default LogsSpecs implementation:

    - `log_dir` will be created if it doesn't exist
    - Generates nested folders for each attempt and rank.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        redirects: Union[Std, Dict[int, Std]] = Std.NONE,
        tee: Union[Std, Dict[int, Std]] = Std.NONE,
        local_ranks_filter: Optional[Set[int]] = None,
    ) -> None:
        # 检查给定的 log_dir 是否不是 os.devnull
        if log_dir != os.devnull:
            # 如果 log_dir 为空，则创建一个以 "torchelastic_" 开头的临时目录
            if not log_dir:
                log_dir = tempfile.mkdtemp(prefix="torchelastic_")
            # 如果 log_dir 存在但不是文件，则确保目录存在，否则抛出异常
            elif not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            else:
                if os.path.isfile(log_dir):
                    raise NotADirectoryError(f"log_dir: {log_dir} is a file")
        # 调用父类 LogsSpecs 的构造函数
        super().__init__(log_dir, redirects, tee, local_ranks_filter)
        # 初始化一个属性，只在第一次初始化时赋值为 None
        self._run_log_dir = None

    @property
    def root_log_dir(self) -> str:
        # 返回根日志目录的字符串表示形式
        return str(self._root_log_dir)

    def _make_log_dir(self, log_dir: Optional[str], rdzv_run_id: str):
        # 创建日志目录，如果 log_dir 为空，则在 base_log_dir 中创建一个以 rdzv_run_id 开头的临时目录
        base_log_dir = log_dir or tempfile.mkdtemp(prefix="torchelastic_")
        os.makedirs(base_log_dir, exist_ok=True)
        dir = tempfile.mkdtemp(prefix=f"{rdzv_run_id}_", dir=base_log_dir)
        # 记录日志目录设置信息
        logger.info("log directory set to: %s", dir)
        return dir

    def reify(
        self,
        envs: Dict[int, Dict[str, str]],
    ) -> None:
        # 此方法尚未实现，通常用于将对象转换为实体或实例的操作
        pass

    def __repr__(self) -> str:
        # 返回对象的字符串表示形式，包括 _root_log_dir, _redirects, _tee, _local_ranks_filter 属性的信息
        return (
            f"DefaultLogsSpecs(root_log_dir={self._root_log_dir}, redirects={self._redirects}, "
            f"tee={self._tee}, local_ranks_filter={self._local_ranks_filter})"
        )

    def __eq__(self, other: object) -> bool:
        # 检查两个 DefaultLogsSpecs 对象是否相等
        if not isinstance(other, DefaultLogsSpecs):
            return False

        return (
            self._root_log_dir == other._root_log_dir
            and self._redirects == other._redirects
            and self._tee == other._tee
            and self._local_ranks_filter == other._local_ranks_filter
        )


@dataclass
class RunProcsResult:
    """
    Results of a completed run of processes started with ``start_processes()``. Returned by ``PContext``.

    Note the following:

    1. All fields are mapped by local rank
    2. ``return_values`` - only populated for functions (not the binaries).
    3. ``stdouts`` - path to stdout.log (empty string if no redirect)
    4. ``stderrs`` - path to stderr.log (empty string if no redirect)

    """

    return_values: Dict[int, Any] = field(default_factory=dict)
    failures: Dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts: Dict[int, str] = field(default_factory=dict)
    stderrs: Dict[int, str] = field(default_factory=dict)

    def is_failed(self) -> bool:
        # 检查是否有失败的进程
        return len(self.failures) > 0


class PContext(abc.ABC):
    """
    The base class that standardizes operations over a set of processes that are launched via different mechanisms.
    """
    The name ``PContext`` is intentional to disambiguate with ``torch.multiprocessing.ProcessContext``.

    .. warning:: stdouts and stderrs should ALWAYS be a superset of
                 tee_stdouts and tee_stderrs (respectively) this is b/c
                 tee is implemented as a redirect + tail -f <stdout/stderr.log>
    """

    # 定义 PContext 类，用于管理并发进程的上下文环境
    def __init__(
        self,
        name: str,  # 进程上下文的名称
        entrypoint: Union[Callable, str],  # 进程入口点，可以是函数或字符串
        args: Dict[int, Tuple],  # 进程参数的字典，键为进程编号，值为参数元组
        envs: Dict[int, Dict[str, str]],  # 进程环境变量的字典，键为进程编号，值为环境变量字典
        logs_specs: LogsSpecs,  # 日志规格对象，用于指定日志的输出方式和位置
        log_line_prefixes: Optional[Dict[int, str]] = None,  # 日志行前缀的可选字典
    ):
        # 设置进程上下文的名称
        self.name = name

        # 根据进程参数的数量验证所有映射是否具有相同数量的键，并且所有本地进程都已经注册
        nprocs = len(args)

        # TODO 可以扩展 log_line_prefixes
        # 根据环境变量实例化日志输出位置对象
        logs_dest = logs_specs.reify(envs)

        # 验证 stdouts 和 stderrs 是否包含了所有进程的输出，确保日志完整性
        _validate_full_rank(logs_dest.stdouts, nprocs, "stdouts")
        _validate_full_rank(logs_dest.stderrs, nprocs, "stderrs")

        # 设置 PContext 的属性
        self.entrypoint = entrypoint  # 进程的入口点
        self.args = args  # 进程参数的字典
        self.envs = envs  # 进程环境变量的字典
        self.stdouts = logs_dest.stdouts  # 标准输出日志位置
        self.stderrs = logs_dest.stderrs  # 标准错误输出日志位置
        self.error_files = logs_dest.error_files  # 错误日志文件位置
        self.nprocs = nprocs  # 进程数量

        # 初始化标准输出和标准错误日志的尾部跟踪对象
        self._stdout_tail = TailLog(
            name, logs_dest.tee_stdouts, sys.stdout, log_line_prefixes
        )
        self._stderr_tail = TailLog(
            name, logs_dest.tee_stderrs, sys.stderr, log_line_prefixes
        )

    # 启动进程
    def start(self) -> None:
        """Start processes using parameters defined in the constructor."""
        # 设置信号处理函数，处理进程终止的信号
        signal.signal(signal.SIGTERM, _terminate_process_handler)
        signal.signal(signal.SIGINT, _terminate_process_handler)
        if not IS_WINDOWS:
            signal.signal(signal.SIGHUP, _terminate_process_handler)
            signal.signal(signal.SIGQUIT, _terminate_process_handler)
        self._start()  # 调用实际启动进程的方法
        self._stdout_tail.start()  # 启动标准输出日志尾部跟踪
        self._stderr_tail.start()  # 启动标准错误输出日志尾部跟踪

    @abc.abstractmethod
    def _start(self) -> None:
        """Start processes using strategy defined in a particular context."""
        raise NotImplementedError

    @abc.abstractmethod
    def _poll(self) -> Optional[RunProcsResult]:
        """
        Poll the run status of the processes running under this context.
        This method follows an "all-or-nothing" policy and returns
        a ``RunProcessResults`` object if either all processes complete
        successfully or any process fails. Returns ``None`` if
        all processes are still running.
        """
        raise NotImplementedError
    def wait(self, timeout: float = -1, period: float = 1) -> Optional[RunProcsResult]:
        """
        Wait for the specified ``timeout`` seconds, polling every ``period`` seconds
        for the processes to be done. Returns ``None`` if the processes are still running
        on timeout expiry. Negative timeout values are interpreted as "wait-forever".
        A timeout value of zero simply queries the status of the processes (e.g. equivalent
        to a poll).

        ..note: Multiprocessing library registers SIGTERM and SIGINT signal handlers that raise
                ``SignalException`` when the signals received. It is up to the consumer of the code
                to properly handle the exception. It is important not to swallow the exception otherwise
                the process would not terminate. Example of the typical workflow can be:

        .. code-block:: python
            pc = start_processes(...)
            try:
                pc.wait(1)
                .. do some other work
            except SignalException as e:
                pc.shutdown(e.sigval, timeout=30)

        If SIGTERM or SIGINT occurs, the code above will try to shutdown child processes by propagating
        received signal. If child processes will not terminate in the timeout time, the process will send
        the SIGKILL.
        """
        # 如果 timeout 为 0，则立即返回当前进程状态（相当于 poll）
        if timeout == 0:
            return self._poll()

        # 如果 timeout 小于 0，则将 timeout 设置为系统最大值，表示无限等待
        if timeout < 0:
            timeout = sys.maxsize

        # 计算超时的时间点
        expiry = time.time() + timeout
        # 循环直到超时时间点
        while time.time() < expiry:
            # 调用 _poll 方法查询进程状态
            pr = self._poll()
            # 如果进程状态不为 None（即进程已结束），则返回进程状态
            if pr:
                return pr
            # 休眠指定的周期时间
            time.sleep(period)

        # 如果超时未返回进程状态，则返回 None
        return None

    @abc.abstractmethod
    def pids(self) -> Dict[int, int]:
        """Return pids of processes mapped by their respective local_ranks."""
        # 抽象方法，子类需要实现，返回一个字典，包含进程的 pid 和 local_rank 的映射关系
        raise NotImplementedError

    @abc.abstractmethod
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).
        """
        # 抽象方法，子类需要实现，用指定的 death_sig 信号终止所有管理的进程，并清理元资源
        raise NotImplementedError

    def close(
        self, death_sig: Optional[signal.Signals] = None, timeout: int = 30
    ) -> None:
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).

        Args:
            death_sig: Death signal to terminate processes.
            timeout: Time to wait for processes to finish, if process is
                still alive after this time, it will be terminated via SIGKILL.
        """
        # 如果 death_sig 为 None，则获取默认的终止信号
        if not death_sig:
            death_sig = _get_default_signal()
        # 调用 _close 方法终止所有进程
        self._close(death_sig=death_sig, timeout=timeout)
        # 如果存在 stdout_tail，则停止它
        if self._stdout_tail:
            self._stdout_tail.stop()
        # 如果存在 stderr_tail，则停止它
        if self._stderr_tail:
            self._stderr_tail.stop()
def get_std_cm(std_rd: str, redirect_fn):
    # 如果在Windows或者macOS系统上运行，或者std_rd为空，则返回一个nullcontext对象
    if IS_WINDOWS or IS_MACOS or not std_rd:
        return nullcontext()
    else:
        # 否则，使用给定的redirect_fn函数处理std_rd，返回相应的上下文管理器对象
        return redirect_fn(std_rd)


def _wrap(
    local_rank: int,
    fn: Callable,
    args: Dict[int, Tuple],
    envs: Dict[int, Dict[str, str]],
    stdout_redirects: Dict[int, str],  # 标准输出重定向文件路径（如果为None则输出到控制台）
    stderr_redirects: Dict[int, str],  # 标准错误输出重定向文件路径（如果为None则输出到控制台）
    ret_vals: Dict[int, mp.SimpleQueue],
    queue_finished_reading_event: synchronize.Event,
) -> None:
    # 预先获取每个进程参数，如果找不到映射关系则会快速失败
    args_ = args[local_rank]
    env_ = envs[local_rank]
    ret_val_ = ret_vals[local_rank]

    stdout_rd = stdout_redirects[local_rank]
    stderr_rd = stderr_redirects[local_rank]

    # 获取标准输出和标准错误的上下文管理器对象
    stdout_cm = get_std_cm(stdout_rd, redirect_stdout)
    stderr_cm = get_std_cm(stderr_rd, redirect_stderr)

    # 设置当前进程的环境变量
    for k, v in env_.items():
        os.environ[k] = v

    # 使用上下文管理器对象，记录函数的执行结果
    with stdout_cm, stderr_cm:
        ret = record(fn)(*args_)
    # 将执行结果放入返回值队列中
    ret_val_.put(ret)
    # 等待队列读取事件完成
    queue_finished_reading_event.wait()


class MultiprocessContext(PContext):
    """``PContext``用于管理作为函数调用的工作进程。"""

    def __init__(
        self,
        name: str,
        entrypoint: Callable,
        args: Dict[int, Tuple],
        envs: Dict[int, Dict[str, str]],
        start_method: str,
        logs_specs: LogsSpecs,
        log_line_prefixes: Optional[Dict[int, str]] = None,
    ):
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_line_prefixes,
        )

        self.start_method = start_method
        # 每个返回值队列始终只包含一个元素。
        self._ret_vals = {
            local_rank: mp.get_context(self.start_method).SimpleQueue()
            for local_rank in range(self.nprocs)
        }

        # 详细见``join()``函数的注释说明
        self._return_values: Dict[int, Any] = {}
        self._pc: Optional[mp.ProcessContext] = None
        # 注意：仅当所有进程成功完成时才应调用set方法。
        # 如果任何进程在event.wait()时失败，调用set()方法将导致死锁。
        self._worker_finished_event = mp.get_context(self.start_method).Event()
    # 启动方法，初始化多进程
    def _start(self):
        # 如果已经初始化了进程上下文，则抛出数值错误异常
        if self._pc:
            raise ValueError(
                "The process context already initialized."
                " Most likely the start method got called twice."
            )
        # 使用 multiprocessing 模块的 start_processes 方法启动多个进程
        self._pc = mp.start_processes(
            fn=_wrap,
            args=(
                self.entrypoint,
                self.args,
                self.envs,
                self.stdouts,
                self.stderrs,
                self._ret_vals,
                self._worker_finished_event,
            ),
            nprocs=self.nprocs,
            join=False,
            daemon=False,
            start_method=self.start_method,
        )

    # 检查是否所有进程都已完成
    def _is_done(self) -> bool:
        return len(self._return_values) == self.nprocs

    # 返回一个字典，包含进程 ID 到其对应 PID 的映射
    def pids(self) -> Dict[int, int]:
        # 确保进程上下文已经初始化，用于类型检查的断言
        assert self._pc is not None  
        return dict(enumerate(self._pc.pids()))

    # 关闭进程池中的所有进程
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        # 如果进程上下文未初始化，则直接返回
        if not self._pc:
            return
        # 遍历所有进程，发送信号以关闭它们
        for proc in self._pc.processes:
            if proc.is_alive():
                logger.warning(
                    "Closing process %s via signal %s", proc.pid, death_sig.name
                )
                try:
                    os.kill(proc.pid, death_sig)
                except ProcessLookupError:
                    # 如果进程已退出，则忽略 ProcessLookupError 异常
                    pass
        # 设置超时时间，等待进程结束
        end = time.monotonic() + timeout
        for proc in self._pc.processes:
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            proc.join(time_to_wait)
        # 如果进程仍然存活，则强制退出
        for proc in self._pc.processes:
            if proc.is_alive():
                logger.warning(
                    "Unable to shutdown process %s via %s, forcefully exiting via %s",
                    proc.pid,
                    death_sig,
                    _get_kill_signal(),
                )
                try:
                    os.kill(proc.pid, _get_kill_signal())
                except ProcessLookupError:
                    # 如果进程已退出，则忽略 ProcessLookupError 异常
                    pass
            proc.join()
    # 定义了一个名为 SubprocessContext 的子类，继承自 PContext 类
    """``PContext`` holding worker processes invoked as a binary."""
    
    # 初始化方法，接受多个参数来配置上下文
    def __init__(
        self,
        name: str,  # 上下文的名称
        entrypoint: str,  # 启动程序的入口点，必须是一个字符串
        args: Dict[int, Tuple],  # 参数字典，键为进程编号，值为参数元组
        envs: Dict[int, Dict[str, str]],  # 环境变量字典，键为进程编号，值为环境变量字典
        logs_specs: LogsSpecs,  # 日志规范对象，定义了日志的相关规则
        log_line_prefixes: Optional[Dict[int, str]] = None,  # 可选的日志行前缀字典
    ):
        
        # 调用父类 PContext 的初始化方法，传入所有参数
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_line_prefixes,
        )
        
        # 状态向量；_vdone[local_rank] -> local_rank 是否已完成
        self._running_local_ranks: Set[int] = set(range(self.nprocs))  # 当前正在运行的进程编号集合
        self._failures: Dict[int, ProcessFailure] = {}  # 记录进程失败情况的字典，键为进程编号，值为 ProcessFailure 对象
        self.subprocess_handlers: Dict[int, SubprocessHandler] = {}  # 子进程处理器字典，键为进程编号，值为 SubprocessHandler 对象

    # 启动方法，初始化子进程处理器
    def _start(self):
        # 如果 subprocess_handlers 字典已经初始化，抛出 ValueError 异常
        if self.subprocess_handlers:
            raise ValueError(
                "The subprocess handlers already initialized. Most likely the start method got called twice."
            )
        
        # 使用推导式为每个本地进程编号创建子进程处理器，并存储在 subprocess_handlers 字典中
        self.subprocess_handlers = {
            local_rank: get_subprocess_handler(
                entrypoint=self.entrypoint,  # 子进程入口点，始终为字符串类型
                args=self.args[local_rank],  # 当前本地进程的参数元组
                env=self.envs[local_rank],  # 当前本地进程的环境变量字典
                stdout=self.stdouts[local_rank],  # 当前本地进程的标准输出流
                stderr=self.stderrs[local_rank],  # 当前本地进程的标准错误流
                local_rank_id=local_rank,  # 当前本地进程的编号
            )
            for local_rank in range(self.nprocs)  # 遍历所有本地进程编号的范围
        }
    # 定义一个方法 _poll，返回类型为 Optional[RunProcsResult]
    def _poll(self) -> Optional[RunProcsResult]:
        # 初始化一个空集合，用于存储已完成的本地进程排名
        done_local_ranks = set()
        # 遍历正在运行的本地进程排名
        for local_rank in self._running_local_ranks:
            # 获取处理程序
            handler = self.subprocess_handlers[local_rank]
            # 查询进程的退出码
            exitcode = handler.proc.poll()
            # 如果退出码不为 None，表示进程已经结束
            if exitcode is not None:
                # 将本地排名添加到已完成集合中
                done_local_ranks.add(local_rank)
                # 如果退出码不为 0，表示进程执行失败或被信号中断
                if exitcode != 0:
                    # 将失败信息存储到失败字典中
                    self._failures[local_rank] = ProcessFailure(
                        local_rank=local_rank,
                        pid=handler.proc.pid,
                        exitcode=exitcode,
                        error_file=self.error_files[local_rank],
                    )
                # else: --> 进程成功执行，不需要额外处理

        # 从正在运行的本地进程排名集合中移除已完成的排名
        self._running_local_ranks.difference_update(done_local_ranks)

        # 如果所有进程都已完成或有任何进程失败
        if not self._running_local_ranks or self._failures:
            # 关闭所有运行中的进程
            self.close()
            # 创建 RunProcsResult 对象，包括失败信息、标准输出和标准错误
            result = RunProcsResult(
                failures=self._failures,
                stdouts=self.stdouts,
                stderrs=self.stderrs,
            )
            # 如果有任何失败，则记录第一个失败的详细信息到日志中
            if result.is_failed():
                first_failure = min(result.failures.values(), key=lambda f: f.timestamp)
                logger.error(
                    "failed (exitcode: %s)"
                    " local_rank: %s (pid: %s)"
                    " of binary: %s",
                    first_failure.exitcode,
                    first_failure.local_rank,
                    first_failure.pid,
                    self.entrypoint,
                )
            else:
                # 否则，返回值中填充虚拟值，保持与 MultiprocessingHandler 的一致性
                result.return_values = dict.fromkeys(range(self.nprocs))

            return result
        else:  # 没有失败且仍有进程在运行
            return None

    # 定义一个方法 pids，返回类型为 Dict[int, int]
    def pids(self) -> Dict[int, int]:
        # 返回一个字典，包含本地排名到其对应处理程序进程 PID 的映射
        return {
            local_rank: sh.proc.pid
            for local_rank, sh in self.subprocess_handlers.items()
        }
    # 如果没有子进程处理器，则直接返回，不执行关闭操作
    if not self.subprocess_handlers:
        return

    # 遍历所有子进程处理器
    for handler in self.subprocess_handlers.values():
        # 如果子进程仍在运行（未结束）
        if handler.proc.poll() is None:
            # 记录警告日志，发送关闭信号给对应的子进程
            logger.warning(
                "Sending process %s closing signal %s",
                handler.proc.pid,
                death_sig.name,
            )
            # 调用处理器的关闭方法，发送指定的关闭信号
            handler.close(death_sig=death_sig)

    # 设置结束时间，计算超时时刻
    end = time.monotonic() + timeout

    # 再次遍历所有子进程处理器
    for handler in self.subprocess_handlers.values():
        # 计算还需等待的时间
        time_to_wait = end - time.monotonic()
        # 如果等待时间小于等于0，即超时
        if time_to_wait <= 0:
            break
        try:
            # 等待子进程关闭，限定等待时间为剩余时间
            handler.proc.wait(time_to_wait)
        except subprocess.TimeoutExpired:
            # 忽略超时异常，因为子进程将会被强制通过 SIGKILL 终止
            pass

    # 最后一次遍历子进程处理器
    for handler in self.subprocess_handlers.values():
        # 如果子进程仍在运行（未结束）
        if handler.proc.poll() is None:
            # 记录警告日志，无法通过给定信号关闭进程，强制通过获取的杀死信号退出
            logger.warning(
                "Unable to shutdown process %s via %s, forcefully exiting via %s",
                handler.proc.pid,
                death_sig,
                _get_kill_signal(),
            )
            # 调用处理器的关闭方法，强制使用获取的杀死信号关闭子进程
            handler.close(death_sig=_get_kill_signal())
            # 等待子进程关闭
            handler.proc.wait()
```
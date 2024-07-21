# `.\pytorch\torch\_strobelight\compile_time_profiler.py`

```py
# Import logging and os modules for logging configuration and environment variable handling
import logging
import os

# Import datetime module for timestamp creation and socket module for hostname retrieval
from datetime import datetime
from socket import gethostname
# Import Any and Optional types for type annotations
from typing import Any, Optional

# Import StrobelightCLIFunctionProfiler from torch._strobelight.cli_function_profiler module
from torch._strobelight.cli_function_profiler import StrobelightCLIFunctionProfiler

# Initialize logger with name "strobelight_compile_time_profiler"
logger = logging.getLogger("strobelight_compile_time_profiler")

# Create a console handler for logging and configure its format
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)
console_handler.setFormatter(formatter)

# Add console handler to logger and set logging level to INFO
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
# Disable propagation of log messages to parent loggers
logger.propagate = False

# Define a class for StrobelightCompileTimeProfiler
class StrobelightCompileTimeProfiler:
    # Initialize class variables for profiling counts, flags, and identifiers
    success_profile_count: int = 0
    failed_profile_count: int = 0
    ignored_profile_runs: int = 0
    inside_profile_compile_time: bool = False
    enabled: bool = False
    # Optional identifier for profiling runs
    identifier: Optional[str] = None

    # Optional current profiling phase
    current_phase: Optional[str] = None

    # Optional profiler object
    profiler: Optional[Any] = None

    # Initialize class variables from environment variables or defaults
    max_stack_length: int = int(
        os.environ.get("COMPILE_STROBELIGHT_MAX_STACK_LENGTH", 127)
    )
    max_profile_time: int = int(
        os.environ.get("COMPILE_STROBELIGHT_MAX_PROFILE_TIME", 60 * 30)
    )
    # Sampling rate for profiling
    sample_each: int = int(
        float(os.environ.get("COMPILE_STROBELIGHT_SAMPLE_RATE", 1e7))
    )

    # Class method to enable profiling with a specified profiler class
    @classmethod
    def enable(cls, profiler_class: Any = StrobelightCLIFunctionProfiler) -> None:
        # If profiling is already enabled, log a message and return
        if cls.enabled:
            logger.info("compile time strobelight profiling already enabled")
            return

        # Log message indicating profiling is enabled
        logger.info("compile time strobelight profiling enabled")

        # Check if the provided profiler class is StrobelightCLIFunctionProfiler
        if profiler_class is StrobelightCLIFunctionProfiler:
            import shutil

            # Check if strobeclient executable is available; if not, log a message and return
            if not shutil.which("strobeclient"):
                logger.info(
                    "strobeclient not found, cant enable compile time strobelight profiling, seems"
                    "like you are not on a FB machine."
                )
                return

        # Set profiling flag to True and initialize the profiler object
        cls.enabled = True
        cls._cls_init()
        # Initialize profiler with specified parameters
        cls.profiler = profiler_class(
            sample_each=cls.sample_each,
            max_profile_duration_sec=cls.max_profile_time,
            stack_max_len=cls.max_stack_length,
            async_stack_max_len=cls.max_stack_length,
            # Construct run_user_name using environment variables USER or USERNAME
            run_user_name="pt2-profiler/" + os.environ.get("USER", os.environ.get("USERNAME", "")),
            # Set sample_tags to include the identifier for profiling runs
            sample_tags={cls.identifier},
        )

    # Additional class methods and code would follow here
    # 定义一个类方法 _cls_init，用于初始化类属性
    def _cls_init(cls) -> None:
        # 使用当前时间、进程 ID 和主机名生成一个唯一标识符
        cls.identifier = "{date}{pid}{hostname}".format(
            date=datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            pid=os.getpid(),
            hostname=gethostname(),
        )

        # 记录日志，显示本次运行的唯一样本标签
        logger.info("Unique sample tag for this run is: %s", cls.identifier)
        
        # 记录日志，提供一个链接用于在运行结束时访问 strobelight profile
        logger.info(
            "You can use the following link to access the strobelight profile at the end of the run: %s",
            (
                "https://www.internalfb.com/intern/scuba/query/?dataset=pyperf_experime"
                "ntal%2Fon_demand&drillstate=%7B%22purposes%22%3A[]%2C%22end%22%3A%22no"
                "w%22%2C%22start%22%3A%22-30%20days%22%2C%22filterMode%22%3A%22DEFAULT%"
                "22%2C%22modifiers%22%3A[]%2C%22sampleCols%22%3A[]%2C%22cols%22%3A[%22n"
                "amespace_id%22%2C%22namespace_process_id%22]%2C%22derivedCols%22%3A[]%"
                "2C%22mappedCols%22%3A[]%2C%22enumCols%22%3A[]%2C%22return_remainder%22"
                "%3Afalse%2C%22should_pivot%22%3Afalse%2C%22is_timeseries%22%3Afalse%2C"
                "%22hideEmptyColumns%22%3Afalse%2C%22timezone%22%3A%22America%2FLos_Ang"
                "eles%22%2C%22compare%22%3A%22none%22%2C%22samplingRatio%22%3A%221%22%2"
                "C%22metric%22%3A%22count%22%2C%22aggregation_field%22%3A%22async_stack"
                "_complete%22%2C%22top%22%3A10000%2C%22aggregateList%22%3A[]%2C%22param"
                "_dimensions%22%3A[%7B%22dim%22%3A%22py_async_stack%22%2C%22op%22%3A%22"
                "edge%22%2C%22param%22%3A%220%22%2C%22anchor%22%3A%220%22%7D]%2C%22orde"
                "r%22%3A%22weight%22%2C%22order_desc%22%3Atrue%2C%22constraints%22%3A[["
                "%7B%22column%22%3A%22sample_tags%22%2C%22op%22%3A%22all%22%2C%22value%"
                f"22%3A[%22[%5C%22{cls.identifier}%5C%22]%22]%7D]]%2C%22c_constraints%22%3A[[]]%2C%22b"
                "_constraints%22%3A[[]]%2C%22ignoreGroupByInComparison%22%3Afalse%7D&vi"
                "ew=GraphProfilerView&&normalized=1712358002&pool=uber"
            ),
        )

    # 定义一个类方法 _log_stats，用于记录统计信息
    @classmethod
    def _log_stats(cls) -> None:
        # 记录日志，显示 strobelight 成功运行的数量和总的非递归编译事件数量
        logger.info(
            "%s strobelight success runs out of %s non-recursive compilation events.",
            cls.success_profile_count,
            cls.success_profile_count + cls.failed_profile_count,
        )

    # TODO use threadlevel meta data to tags to record phases.
    # 定义一个类方法 profile_compile_time，用于记录编译时间的 profile
    @classmethod
    def profile_compile_time(
        cls, func: Any, phase_name: str, *args: Any, **kwargs: Any
    # 定义一个类方法，用于条件性地启用性能分析功能
    # 如果功能未启用，则直接调用原始函数并返回结果
    def profile_if_enabled(cls, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        # 检查是否启用了性能分析
        if not cls.enabled:
            return func(*args, **kwargs)

        # 检查性能分析器是否已设置
        if cls.profiler is None:
            # 如果性能分析器未设置，则记录错误并返回
            logger.error("profiler is not set")
            return

        # 如果正在进行编译时间的性能分析，则记录忽略的运行次数并返回
        if cls.inside_profile_compile_time:
            cls.ignored_profile_runs += 1
            logger.info(
                "profile_compile_time is requested for phase: %s while already in running phase: %s, recursive call ignored",
                phase_name,
                cls.current_phase,
            )
            return func(*args, **kwargs)

        # 设置正在进行编译时间的性能分析，并记录当前阶段
        cls.inside_profile_compile_time = True
        cls.current_phase = phase_name

        # 执行性能分析器的分析功能
        work_result = cls.profiler.profile(func, *args, **kwargs)

        # 根据性能分析结果更新成功或失败的计数
        if cls.profiler.profile_result is not None:
            cls.success_profile_count += 1
        else:
            cls.failed_profile_count += 1

        # 记录性能统计信息
        cls._log_stats()

        # 完成编译时间的性能分析，重置相关标志
        cls.inside_profile_compile_time = False

        # 返回执行结果
        return work_result
```
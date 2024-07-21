# `.\pytorch\torch\utils\benchmark\utils\timer.py`

```
"""Timer class based on the timeit.Timer class, but torch aware."""
# 导入所需的模块和库
import enum
import timeit
import textwrap
from typing import overload, Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union

import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import TimerClass, TimeitModuleType
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface

# 定义可以公开的类和函数
__all__ = ["Timer", "timer", "Language"]

# 根据 CUDA 是否可用定义计时器函数
if torch.backends.cuda.is_built() and torch.cuda.is_available():  # type: ignore[no-untyped-call]
    def timer() -> float:
        # 同步 CUDA 设备
        torch.cuda.synchronize()
        return timeit.default_timer()
elif torch._C._get_privateuse1_backend_name() != "privateuseone":
    # 根据 PrivateUseOne 后端名称定义计时器函数
    privateuse1_device_handler = getattr(torch, torch._C._get_privateuse1_backend_name(), None) \
        if torch._C._get_privateuse1_backend_name() != "cpu" else None

    def timer() -> float:
        if privateuse1_device_handler:
            # 同步 PrivateUseOne 设备
            privateuse1_device_handler.synchronize()
        return timeit.default_timer()
else:
    # 默认使用 Python 自带的计时器
    timer = timeit.default_timer

# 枚举类，用于定义编程语言
class Language(enum.Enum):
    PYTHON = 0
    CPP = 1

# CPPTimer 类，用于执行 C++ 语句的计时
class CPPTimer:
    def __init__(
        self,
        stmt: str,
        setup: str,
        global_setup: str,
        timer: Callable[[], float],
        globals: Dict[str, Any],
    ) -> None:
        if timer is not timeit.default_timer:
            # 如果使用的不是 Python 默认的计时器，抛出异常
            raise NotImplementedError(
                "PyTorch was built with CUDA and a GPU is present; however "
                "Timer does not yet support GPU measurements. If your "
                "code is CPU only, pass `timer=timeit.default_timer` to the "
                "Timer's constructor to indicate this. (Note that this will "
                "produce incorrect results if the GPU is in fact used, as "
                "Timer will not synchronize CUDA.)"
            )

        if globals:
            # 如果有全局变量，则不支持 C++ 计时
            raise ValueError("C++ timing does not support globals.")

        self._stmt: str = textwrap.dedent(stmt)
        self._setup: str = textwrap.dedent(setup)
        self._global_setup: str = textwrap.dedent(global_setup)
        self._timeit_module: Optional[TimeitModuleType] = None

    def timeit(self, number: int) -> float:
        if self._timeit_module is None:
            # 编译 C++ 模板，准备进行计时
            self._timeit_module = cpp_jit.compile_timeit_template(
                stmt=self._stmt,
                setup=self._setup,
                global_setup=self._global_setup,
            )

        # 执行计时并返回结果
        return self._timeit_module.timeit(number)

# Timer 类，用于测量 PyTorch 语句的执行时间
class Timer:
    """Helper class for measuring execution time of PyTorch statements.

    For a full tutorial on how to use this class, see:
    https://pytorch.org/tutorials/recipes/recipes/benchmark.html

    The PyTorch Timer is based on `timeit.Timer` (and in fact uses
    `timeit.Timer` internally), but with several key differences:
    """
    # 运行时环境设置说明:
    # Timer 类会执行预热操作（因为 PyTorch 的某些元素是惰性初始化的），设置线程池大小以确保比较的公平性，并在必要时同步异步 CUDA 函数。
    
    # 焦点在于复制实验:
    # 在测量代码性能时，特别是复杂的核心算法或模型时，运行到运行的变异是一个重要的混杂因素。预期所有测量都应包括复制实验以量化噪音，并允许计算中位数，这比均值更为稳健。
    # 因此，这个类在概念上结合了 `timeit.Timer.repeat` 和 `timeit.Timer.autorange`，与 `timeit` API 有所不同。
    # （具体算法在方法的文档字符串中有讨论。）`timeit` 方法用于那些不希望使用自适应策略的情况。
    
    # 可选的元数据:
    # 在定义 Timer 时，可以选择性地指定 `label`、`sub_label`、`description` 和 `env`（稍后定义）这些字段。这些字段包含在结果对象的表示中，并由 `Compare` 类用于分组和显示比较结果。
    
    # 指令计数:
    # 除了测量墙上时间外，Timer 可以在 Callgrind 下运行语句并报告执行的指令数量。
    
    # 与 `timeit.Timer` 构造函数参数直接对应的说明:
    # `stmt`, `setup`, `timer`, `globals`
    
    # PyTorch Timer 特定的构造函数参数:
    # `label`, `sub_label`, `description`, `env`, `num_threads`
    # 定义一个类型别名 _timer_cls，指定为 TimerClass 类型，用于计时器
    _timer_cls: Type[TimerClass] = timeit.Timer
    def __init__(
        self,
        stmt: str = "pass",
        setup: str = "pass",
        global_setup: str = "",
        timer: Callable[[], float] = timer,
        globals: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        sub_label: Optional[str] = None,
        description: Optional[str] = None,
        env: Optional[str] = None,
        num_threads: int = 1,
        language: Union[Language, str] = Language.PYTHON,
    ):
        # 检查 stmt 参数是否为字符串，当前仅支持字符串作为语句
        if not isinstance(stmt, str):
            raise ValueError("Currently only a `str` stmt is supported.")

        # 复制 globals 参数，以防止外部对其进行修改泄漏
        # （例如，`eval` 函数会添加 `__builtins__` 键）
        self._globals = dict(globals or {})

        timer_kwargs = {}
        if language in (Language.PYTHON, "py", "python"):
            # 如果语言为 Python，确保默认包含 `torch`，以方便使用
            self._globals.setdefault("torch", torch)
            self._language: Language = Language.PYTHON
            # 如果指定了 global_setup，则抛出异常，因为 global_setup 仅适用于 C++ 语言
            if global_setup:
                raise ValueError(
                    f"global_setup is C++ only, got `{global_setup}`. Most "
                    "likely this code can simply be moved to `setup`."
                )

        elif language in (Language.CPP, "cpp", "c++"):
            # 如果语言为 C++，设置计时器类为 CPPTimer
            assert self._timer_cls is timeit.Timer, "_timer_cls has already been swapped."
            self._timer_cls = CPPTimer
            # 如果 setup 参数为 "pass"，则将其置空
            setup = ("" if setup == "pass" else setup)
            self._language = Language.CPP
            timer_kwargs["global_setup"] = global_setup

        else:
            # 对于不支持的语言类型，抛出异常
            raise ValueError(f"Invalid language `{language}`.")

        # 为了方便，去除多行代码片段的缩进，避免 Python 的 IndentationError 或 C++ 中看起来奇怪
        # 去除前导的换行符，用于处理定义块字符串时出现的初始换行符
        stmt = textwrap.dedent(stmt)
        stmt = (stmt[1:] if stmt and stmt[0] == "\n" else stmt).rstrip()
        setup = textwrap.dedent(setup)
        setup = (setup[1:] if setup and setup[0] == "\n" else setup).rstrip()

        # 初始化计时器对象，传入参数 stmt, setup, timer, globals，以及额外的 timer_kwargs
        self._timer = self._timer_cls(
            stmt=stmt,
            setup=setup,
            timer=timer,
            globals=valgrind_timer_interface.CopyIfCallgrind.unwrap_all(self._globals),
            **timer_kwargs,
        )

        # 初始化任务规范对象，传入参数 stmt, setup, global_setup, label, sub_label, description, env, num_threads
        self._task_spec = common.TaskSpec(
            stmt=stmt,
            setup=setup,
            global_setup=global_setup,
            label=label,
            sub_label=sub_label,
            description=description,
            env=env,
            num_threads=num_threads,
        )
    def _timeit(self, number: int) -> float:
        # 即使在 C++ 中调用计时器也需要大约 50 纳秒，因此实际操作不应该少于 1 纳秒。
        # （这也避免了除以零的错误。）
        return max(self._timer.timeit(number), 1e-9)

    def timeit(self, number: int = 1000000) -> common.Measurement:
        """模仿 timeit.Timer.timeit() 的语义。

        执行主语句 (`stmt`) `number` 次。
        https://docs.python.org/3/library/timeit.html#timeit.Timer.timeit
        """
        with common.set_torch_threads(self._task_spec.num_threads):
            # 预热阶段
            self._timeit(number=max(int(number // 100), 2))

            return common.Measurement(
                number_per_run=number,
                raw_times=[self._timeit(number=number)],
                task_spec=self._task_spec
            )

    def repeat(self, repeat: int = -1, number: int = -1) -> None:
        # 抛出未实现错误，参见 `Timer.blocked_autorange.`
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def autorange(self, callback: Optional[Callable[[int, float], NoReturn]] = None) -> None:
        # 抛出未实现错误，参见 `Timer.blocked_autorange.`
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def _threaded_measurement_loop(
        self,
        number: int,
        time_hook: Callable[[], float],
        stop_hook: Callable[[List[float]], bool],
        min_run_time: float,
        max_run_time: Optional[float] = None,
        callback: Optional[Callable[[int, float], NoReturn]] = None
    ) -> List[float]:
        total_time = 0.0
        can_stop = False
        times: List[float] = []
        with common.set_torch_threads(self._task_spec.num_threads):
            while (total_time < min_run_time) or (not can_stop):
                time_spent = time_hook()
                times.append(time_spent)
                total_time += time_spent
                if callback:
                    callback(number, time_spent)
                can_stop = stop_hook(times)
                if max_run_time and total_time > max_run_time:
                    break
        return times

    def _estimate_block_size(self, min_run_time: float) -> int:
        with common.set_torch_threads(self._task_spec.num_threads):
            # 估算所需的块大小，使得测量变得可以忽略不计，与内部循环相比。
            # 这也作为预热阶段。
            overhead = torch.tensor([self._timeit(0) for _ in range(5)]).median().item()
            number = 1
            while True:
                time_taken = self._timeit(number)
                relative_overhead = overhead / time_taken
                if relative_overhead <= 1e-4 and time_taken >= min_run_time / 1000:
                    break
                if time_taken > min_run_time:
                    break
                # 避免在 C++ pybind11 接口中溢出
                if number * 10 > 2147483647:
                    break
                number *= 10
        return number
    def blocked_autorange(
        self,
        callback: Optional[Callable[[int, float], NoReturn]] = None,
        min_run_time: float = 0.2,
    ) -> common.Measurement:
        """Measure many replicates while keeping timer overhead to a minimum.

        At a high level, blocked_autorange executes the following pseudo-code::

            `setup`

            total_time = 0
            while total_time < min_run_time
                start = timer()
                for _ in range(block_size):
                    `stmt`
                total_time += (timer() - start)

        Note the variable `block_size` in the inner loop. The choice of block
        size is important to measurement quality, and must balance two
        competing objectives:

            1) A small block size results in more replicates and generally
               better statistics.

            2) A large block size better amortizes the cost of `timer`
               invocation, and results in a less biased measurement. This is
               important because CUDA synchronization time is non-trivial
               (order single to low double digit microseconds) and would
               otherwise bias the measurement.

        blocked_autorange sets block_size by running a warmup period,
        increasing block size until timer overhead is less than 0.1% of
        the overall computation. This value is then used for the main
        measurement loop.

        Returns:
            A `Measurement` object that contains measured runtimes and
            repetition counts, and can be used to compute statistics.
            (mean, median, etc.)
        """
        # 估算合适的 block_size，以确保计时器的开销最小化
        number = self._estimate_block_size(min_run_time)

        def time_hook() -> float:
            # 调用 _timeit 方法，返回执行时间
            return self._timeit(number)

        def stop_hook(times: List[float]) -> bool:
            # 始终返回 True，表示始终继续执行测量循环
            return True

        # 调用多线程测量循环方法，获取测量的时间列表
        times = self._threaded_measurement_loop(
            number, time_hook, stop_hook,
            min_run_time=min_run_time,
            callback=callback)

        # 返回一个 Measurement 对象，包含测量的运行时间和重复次数
        return common.Measurement(
            number_per_run=number,
            raw_times=times,
            task_spec=self._task_spec
        )

    def adaptive_autorange(
            self,
            threshold: float = 0.1,
            *,
            min_run_time: float = 0.01,
            max_run_time: float = 10.0,
            callback: Optional[Callable[[int, float], NoReturn]] = None,
    ) -> common.Measurement:
        """Similar to `blocked_autorange` but also checks for variablility in measurements
        and repeats until iqr/median is smaller than `threshold` or `max_run_time` is reached.
        
        
        At a high level, adaptive_autorange executes the following pseudo-code::
        
            `setup`
        
            times = []
            while times.sum < max_run_time
                start = timer()
                for _ in range(block_size):
                    `stmt`
                times.append(timer() - start)
        
                enough_data = len(times)>3 and times.sum > min_run_time
                small_iqr=times.iqr/times.mean<threshold
        
                if enough_data and small_iqr:
                    break
        
        Args:
            threshold: value of iqr/median threshold for stopping
                控制变异性的 IQR/Median 阈值，用于停止测量的条件之一
        
            min_run_time: total runtime needed before checking `threshold`
                在检查 `threshold` 之前需要的总运行时间
        
            max_run_time: total runtime  for all measurements regardless of `threshold`
                所有测量的最大总运行时间，无论 `threshold` 是多少
        
        Returns:
            A `Measurement` object that contains measured runtimes and
            repetition counts, and can be used to compute statistics.
            (mean, median, etc.)
                返回一个 `Measurement` 对象，包含测量的运行时间和重复次数，可用于计算统计数据
        """
        number = self._estimate_block_size(min_run_time=0.05)
            计算合适的 block 大小，确保每次运行的时间超过 0.05 秒
        
        def time_hook() -> float:
            return self._timeit(number)
                返回执行时间的钩子函数
        
        def stop_hook(times: List[float]) -> bool:
            if len(times) > 3:
                return common.Measurement(
                    number_per_run=number,
                    raw_times=times,
                    task_spec=self._task_spec
                ).meets_confidence(threshold=threshold)
                    如果测量次数大于 3 次，则返回一个包含测量时间的 `Measurement` 对象，并检查是否满足置信度条件
            return False
                否则返回 False，表示未满足停止条件
        
        times = self._threaded_measurement_loop(
            number, time_hook, stop_hook, min_run_time, max_run_time, callback=callback)
                使用多线程测量循环来执行测量任务，传入合适的参数和回调函数
        
        return common.Measurement(
            number_per_run=number,
            raw_times=times,
            task_spec=self._task_spec
        )
            返回一个包含测量结果的 `Measurement` 对象，包括每次运行的数量、原始测量时间和任务规范信息
    ) -> Any:
        """
        使用 Callgrind 收集指令计数。

        与墙上时间不同，指令计数是确定性的（除了程序本身的非确定性和来自Python解释器的少量抖动）。这使它们非常适合详细的性能分析。该方法在单独的进程中运行 `stmt`，以便Valgrind可以对程序进行仪器化。然而，由于仪器化，性能严重下降，不过一般少量迭代就足以获得良好的测量结果。

        要使用这种方法，必须安装 `valgrind`、`callgrind_control` 和 `callgrind_annotate`。

        由于调用者（本进程）与 `stmt` 执行之间存在进程边界，因此 `globals` 不能包含任意的内存数据结构（与计时方法不同）。相反，globals 限制为内置对象、`nn.Modules` 和 TorchScripted 函数/模块，以减少由序列化和后续反序列化引起的意外因素。`GlobalsBridge` 类提供了更多关于此主题的详细信息。特别注意 `nn.Modules`：它们依赖于pickle，可能需要在 `setup` 中添加导入才能正确传输。

        默认情况下，将收集空语句的配置文件并缓存，以指示有多少指令来自驱动 `stmt` 的Python循环。

        返回：
            一个 `CallgrindStats` 对象，提供指令计数和一些基本的分析和操作结果的工具。
        """
        if not isinstance(self._task_spec.stmt, str):
            raise ValueError("`collect_callgrind` 目前仅支持字符串 `stmt`")

        if repeats is not None and repeats < 1:
            raise ValueError("如果指定了 `repeats`，必须 >= 1")

        # 检查语句是否有效。这不保证成功，但是在父进程中为错误的 `stmt` 或 `setup` 抛出异常比在Valgrind子进程中更简单且更快。
        self._timeit(1)
        is_python = (self._language == Language.PYTHON)
        assert is_python or not self._globals
        result = valgrind_timer_interface.wrapper_singleton().collect_callgrind(
            task_spec=self._task_spec,
            globals=self._globals,
            number=number,
            repeats=repeats or 1,
            collect_baseline=collect_baseline and is_python,
            is_python=is_python,
            retain_out_file=retain_out_file,
        )

        return (result[0] if repeats is None else result)
```
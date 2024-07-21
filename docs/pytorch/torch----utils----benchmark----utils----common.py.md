# `.\pytorch\torch\utils\benchmark\utils\common.py`

```
# 导入必要的模块和库
import collections  # 提供了额外的数据容器
import contextlib  # 上下文管理工具
import dataclasses  # 支持创建不可变对象的装饰器
import os  # 提供与操作系统交互的功能
import shutil  # 提供高级文件操作功能
import tempfile  # 提供创建临时文件和目录的功能
import textwrap  # 提供文本包装和填充功能
import time  # 提供时间相关的功能
from typing import cast, Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple  # 强类型标注相关
import uuid  # 提供生成 UUID 的功能

import torch  # PyTorch 深度学习库


__all__ = ["TaskSpec", "Measurement", "select_unit", "unit_to_english", "trim_sigfig", "ordered_unique", "set_torch_threads"]  # 模块的公开接口列表


_MAX_SIGNIFICANT_FIGURES = 4  # 最大有效数字位数
_MIN_CONFIDENCE_INTERVAL = 25e-9  # 最小置信区间，单位为秒，25纳秒

# Measurement 如果分布可疑会包含警告。预期所有运行都会有一定的变化；这些参数设定了阈值。
_IQR_WARN_THRESHOLD = 0.1  # 第一四分位数和第三四分位数的差值警告阈值
_IQR_GROSS_WARN_THRESHOLD = 0.25  # 第一四分位数和第三四分位数的差值严重警告阈值


@dataclasses.dataclass(init=True, repr=False, eq=True, frozen=True)
class TaskSpec:
    """定义一个用于描述定时任务的容器类。(除了全局变量)"""
    stmt: str  # 待测量的语句
    setup: str  # 初始化代码
    global_setup: str = ""  # 全局初始化代码
    label: Optional[str] = None  # 标签
    sub_label: Optional[str] = None  # 子标签
    description: Optional[str] = None  # 描述
    env: Optional[str] = None  # 环境描述
    num_threads: int = 1  # 使用的线程数

    @property
    def title(self) -> str:
        """尝试生成一个适合测量的字符串标签。"""
        if self.label is not None:
            return self.label + (f": {self.sub_label}" if self.sub_label else "")
        elif "\n" not in self.stmt:
            return self.stmt + (f": {self.sub_label}" if self.sub_label else "")
        return (
            f"stmt:{f' ({self.sub_label})' if self.sub_label else ''}\n"
            f"{textwrap.indent(self.stmt, '  ')}"
        )

    def setup_str(self) -> str:
        """返回设置代码的字符串表示。"""
        return (
            "" if (self.setup == "pass" or not self.setup)
            else f"setup:\n{textwrap.indent(self.setup, '  ')}" if "\n" in self.setup
            else f"setup: {self.setup}"
        )

    def summarize(self) -> str:
        """构建 TaskSpec 的部分 repr 字符串，用于其他容器。"""
        sections = [
            self.title,
            self.description or "",
            self.setup_str(),
        ]
        return "\n".join([f"{i}\n" if "\n" in i else i for i in sections if i])


_TASKSPEC_FIELDS = tuple(i.name for i in dataclasses.fields(TaskSpec))  # 获取 TaskSpec 中所有字段名的元组


@dataclasses.dataclass(init=True, repr=False)
class Measurement:
    """定时器测量结果。

    这个类存储给定语句的一个或多个测量结果。它是可序列化的，并为下游消费者提供多种便利方法
    （包括详细的 __repr__）。
    """
    number_per_run: int  # 每次运行的测量次数
    raw_times: List[float]  # 原始时间列表
    task_spec: TaskSpec  # 相关的 TaskSpec 对象
    metadata: Optional[Dict[Any, Any]] = None  # 保留给用户的元数据字段

    def __post_init__(self) -> None:
        self._sorted_times: Tuple[float, ...] = ()  # 排序后的时间元组
        self._warnings: Tuple[str, ...] = ()  # 警告信息元组
        self._median: float = -1.0  # 中位数，默认为 -1.0
        self._mean: float = -1.0  # 平均数，默认为 -1.0
        self._p25: float = -1.0  # 25% 分位数，默认为 -1.0
        self._p75: float = -1.0  # 75% 分位数，默认为 -1.0
    # 定义一个特殊方法 __getattr__，用于动态获取对象的属性
    def __getattr__(self, name: str) -> Any:
        # 如果属性名在 _TASKSPEC_FIELDS 中，则从 self.task_spec 中获取对应属性的值并返回
        if name in _TASKSPEC_FIELDS:
            return getattr(self.task_spec, name)
        # 否则调用父类的 __getattribute__ 方法获取属性值并返回
        return super().__getattribute__(name)

    # =========================================================================
    # == Convenience methods for statistics ===================================
    # =========================================================================
    #
    # 这些方法使用原始时间除以 number_per_run 得到结果；这是一种推断，隐藏了不同 number_per_run
    # 导致的开销分摊不同的事实，但如果 Timer 已经选择了适当的 number_per_run，那这个问题就不存在了，
    # 强制用户处理这种除法会导致用户体验不佳。
    
    @property
    def times(self) -> List[float]:
        # 返回每个 raw_times 中的时间除以 number_per_run 的结果组成的列表
        return [t / self.number_per_run for t in self.raw_times]

    @property
    def median(self) -> float:
        # 懒惰初始化（lazy initialization）并返回 _median 属性的值
        self._lazy_init()
        return self._median

    @property
    def mean(self) -> float:
        # 懒惰初始化并返回 _mean 属性的值
        self._lazy_init()
        return self._mean

    @property
    def iqr(self) -> float:
        # 懒惰初始化并返回 _p75 和 _p25 之差，即四分位距（interquartile range, IQR）
        self._lazy_init()
        return self._p75 - self._p25

    @property
    def significant_figures(self) -> int:
        """Approximate significant figure estimate.

        这个属性用于估算显著数字的近似值。

        This property is intended to give a convenient way to estimate the
        precision of a measurement. It only uses the interquartile region to
        estimate statistics to try to mitigate skew from the tails, and
        uses a static z value of 1.645 since it is not expected to be used
        for small values of `n`, so z can approximate `t`.

        The significant figure estimation used in conjunction with the
        `trim_sigfig` method to provide a more human interpretable data
        summary. __repr__ does not use this method; it simply displays raw
        values. Significant figure estimation is intended for `Compare`.
        """
        # 懒惰初始化并计算显著数字的估算值
        self._lazy_init()
        n_total = len(self._sorted_times)
        lower_bound = int(n_total // 4)
        upper_bound = int(torch.tensor(3 * n_total / 4).ceil())
        interquartile_points: Tuple[float, ...] = self._sorted_times[lower_bound:upper_bound]
        std = torch.tensor(interquartile_points).std(unbiased=False).item()
        sqrt_n = torch.tensor(len(interquartile_points)).sqrt().item()

        # 粗略的估算，这些并不是统计学上严格的。
        confidence_interval = max(1.645 * std / sqrt_n, _MIN_CONFIDENCE_INTERVAL)
        relative_ci = torch.tensor(self._median / confidence_interval).log10().item()
        num_significant_figures = int(torch.tensor(relative_ci).floor())
        return min(max(num_significant_figures, 1), _MAX_SIGNIFICANT_FIGURES)

    @property
    def has_warnings(self) -> bool:
        # 懒惰初始化并返回 _warnings 是否为真（True）
        self._lazy_init()
        return bool(self._warnings)
    def _lazy_init(self) -> None:
        if self.raw_times and not self._sorted_times:
            # 如果原始时间存在且已经没有排序过的时间数据
            self._sorted_times = tuple(sorted(self.times))
            # 将时间数据排序后转换为 torch 的 double 类型 tensor
            _sorted_times = torch.tensor(self._sorted_times, dtype=torch.float64)
            # 计算中位数并存储在实例变量中
            self._median = _sorted_times.quantile(.5).item()
            # 计算平均值并存储在实例变量中
            self._mean = _sorted_times.mean().item()
            # 计算第一四分位数并存储在实例变量中
            self._p25 = _sorted_times.quantile(.25).item()
            # 计算第三四分位数并存储在实例变量中
            self._p75 = _sorted_times.quantile(.75).item()

            def add_warning(msg: str) -> None:
                # 计算 IQR 相对于中位数的百分比
                rel_iqr = self.iqr / self.median * 100
                # 将警告信息添加到警告列表中
                self._warnings += (
                    f"  WARNING: Interquartile range is {rel_iqr:.1f}% "
                    f"of the median measurement.\n           {msg}",
                )

            # 如果不满足置信度阈值_IQR_GROSS_WARN_THRESHOLD，则添加警告
            if not self.meets_confidence(_IQR_GROSS_WARN_THRESHOLD):
                add_warning("This suggests significant environmental influence.")
            # 否则，如果不满足置信度阈值_IQR_WARN_THRESHOLD，则添加警告
            elif not self.meets_confidence(_IQR_WARN_THRESHOLD):
                add_warning("This could indicate system fluctuation.")


    def meets_confidence(self, threshold: float = _IQR_WARN_THRESHOLD) -> bool:
        # 检查当前 IQR 是否满足给定的置信度阈值
        return self.iqr / self.median < threshold

    @property
    def title(self) -> str:
        # 返回测量对象的任务标题
        return self.task_spec.title

    @property
    def env(self) -> str:
        # 返回测量对象的环境描述，如果未指定则返回默认值
        return (
            "Unspecified env" if self.taskspec.env is None
            else cast(str, self.taskspec.env)
        )

    @property
    def as_row_name(self) -> str:
        # 返回用于行名的字符串表示，如果子标签或语句未定义则返回"[Unknown]"
        return self.sub_label or self.stmt or "[Unknown]"

    def __repr__(self) -> str:
        """
        Example repr:
            <utils.common.Measurement object at 0x7f395b6ac110>
              Broadcasting add (4x8)
              Median: 5.73 us
              IQR:    2.25 us (4.01 to 6.26)
              372 measurements, 100 runs per measurement, 1 thread
              WARNING: Interquartile range is 39.4% of the median measurement.
                       This suggests significant environmental influence.
        """
        # 惰性初始化对象属性
        self._lazy_init()
        # 跳过行和换行符的定义
        skip_line, newline = "MEASUREMENT_REPR_SKIP_LINE", "\n"
        n = len(self._sorted_times)
        time_unit, time_scale = select_unit(self._median)
        # 如果测量数量小于4，则过滤掉 IQR 信息
        iqr_filter = '' if n >= 4 else skip_line

        repr_str = f"""
{
super().__repr__()
{self.task_spec.summarize()}
  {'Median: ' if n > 1 else ''}{self._median / time_scale:.2f} {time_unit}
  {iqr_filter}IQR:    {self.iqr / time_scale:.2f} {time_unit} ({self._p25 / time_scale:.2f} to {self._p75 / time_scale:.2f})
  {n} measurement{'s' if n > 1 else ''}, {self.number_per_run} runs {'per measurement,' if n > 1 else ','} {self.num_threads} thread{'s' if self.num_threads > 1 else ''}
{newline.join(self._warnings)}""".strip()  # noqa: B950


# 调用父类的 __repr__() 方法返回对象的字符串表示形式
super().__repr__()
# 调用实例的 task_spec 对象的 summarize() 方法，返回任务规格的摘要信息
{self.task_spec.summarize()}
# 根据 n 的值决定输出字符串，如果 n 大于 1，则输出'Median: '，否则为空字符串
{'Median: ' if n > 1 else ''}
# 输出格式化后的中位数，除以时间刻度并保留两位小数，加上时间单位
{self._median / time_scale:.2f} {time_unit}
# 根据 iqr_filter 的条件输出对应的字符串'IQR:    '或空字符串
{iqr_filter}IQR:    {self.iqr / time_scale:.2f} {time_unit} ({self._p25 / time_scale:.2f} to {self._p75 / time_scale:.2f})
# 根据 n 的值输出测量的数量及相关描述
{n} measurement{'s' if n > 1 else ''}, {self.number_per_run} runs {'per measurement,' if n > 1 else ','} {self.num_threads} thread{'s' if self.num_threads > 1 else ''}
# 输出由 self._warnings 中元素组成的字符串，每行一个，并移除开头和结尾的空白字符
{newline.join(self._warnings)}""".strip()  # noqa: B950
    """Create a temporary directory. The caller is responsible for cleanup.
    
    This function is conceptually similar to `tempfile.mkdtemp`, but with
    the key additional feature that it will use shared memory if the
    `BENCHMARK_USE_DEV_SHM` environment variable is set. This is an
    implementation detail, but an important one for cases where many Callgrind
    measurements are collected at once. (Such as when collecting
    microbenchmarks.)
    
    This is an internal utility, and is exported solely so that microbenchmarks
    can reuse the util.
    """
    
    # 从环境变量 `BENCHMARK_USE_DEV_SHM` 中获取是否使用共享内存的设置
    use_dev_shm: bool = (os.getenv("BENCHMARK_USE_DEV_SHM") or "").lower() in ("1", "true")
    
    # 如果设置了使用共享内存
    if use_dev_shm:
        # 设置临时目录的根路径为 `/dev/shm/pytorch_benchmark_utils`
        root = "/dev/shm/pytorch_benchmark_utils"
        
        # 断言当前操作系统为 POSIX，因为 tmpfs (/dev/shm) 只支持 POSIX 系统
        assert os.name == "posix", f"tmpfs (/dev/shm) is POSIX only, current platform is {os.name}"
        
        # 断言 `/dev/shm` 存在，确保系统支持 tmpfs (/dev/shm)
        assert os.path.exists("/dev/shm"), "This system does not appear to support tmpfs (/dev/shm)."
        
        # 创建目录，如果目录已存在则忽略
        os.makedirs(root, exist_ok=True)
        
        # 如果开启了垃圾收集 (gc_dev_shm)，清理所有未使用的临时目录
        if gc_dev_shm:
            # 遍历根目录下的所有文件和目录
            for i in os.listdir(root):
                # 获取每个目录的 owner.pid 文件路径
                owner_file = os.path.join(root, i, "owner.pid")
                
                # 如果 owner.pid 文件不存在，则跳过当前目录
                if not os.path.exists(owner_file):
                    continue
                
                # 读取 owner.pid 文件中的进程 PID
                with open(owner_file) as f:
                    owner_pid = int(f.read())
                
                # 如果 owner_pid 是当前进程的 PID，则跳过当前目录
                if owner_pid == os.getpid():
                    continue
                
                try:
                    # 检查 owner_pid 对应的进程是否存在
                    os.kill(owner_pid, 0)
                except OSError:
                    # 如果进程不存在，则输出信息并删除该目录
                    print(f"Detected that {os.path.join(root, i)} was orphaned in shared memory. Cleaning up.")
                    shutil.rmtree(os.path.join(root, i))
    
    else:
        # 如果未设置使用共享内存，则临时目录的根路径为系统默认临时目录
        root = tempfile.gettempdir()
    
    # 生成临时目录的名称，包含前缀、时间戳和 UUID，确保唯一性和排序
    name = f"{prefix or tempfile.gettempprefix()}__{int(time.time())}__{uuid.uuid4()}"
    # 拼接得到完整的临时目录路径
    path = os.path.join(root, name)
    # 创建临时目录，如果目录已存在则抛出异常
    os.makedirs(path, exist_ok=False)
    
    # 如果使用共享内存，则在临时目录下创建 owner.pid 文件，并写入当前进程的 PID
    if use_dev_shm:
        with open(os.path.join(path, "owner.pid"), "w") as f:
            f.write(str(os.getpid()))
    
    # 返回创建的临时目录路径
    return path
```
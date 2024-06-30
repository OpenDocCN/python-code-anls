# `D:\src\scipysrc\scipy\benchmarks\benchmarks\common.py`

```
"""
Airspeed Velocity benchmark utilities
"""
# 导入系统和操作系统相关模块
import sys
import os
# 正则表达式模块
import re
# 时间模块
import time
# 文本包装模块
import textwrap
# 子进程管理模块
import subprocess
# 迭代工具模块
import itertools
# 随机数模块
import random


class Benchmark:
    """
    Base class with sensible options
    """
    pass


def is_xslow():
    """
    Check if the 'SCIPY_XSLOW' environment variable is set to 1.

    Returns
    -------
    bool
        True if 'SCIPY_XSLOW' is set to 1, False otherwise.
    """
    try:
        return int(os.environ.get('SCIPY_XSLOW', '0'))
    except ValueError:
        return False


class LimitedParamBenchmark(Benchmark):
    """
    Limits parameter combinations to `max_number` choices, chosen
    pseudo-randomly with fixed seed.
    Raises NotImplementedError (skip) if not in active set.
    """
    num_param_combinations = 0

    def setup(self, *args, **kwargs):
        """
        Setup method to configure parameter combinations.

        Parameters
        ----------
        *args
            Variable positional arguments.
        **kwargs
            Additional keyword arguments:
            - param_seed: Seed for random number generation.
            - params: List of parameter choices.
            - num_param_combinations: Maximum number of parameter combinations.

        Raises
        ------
        NotImplementedError
            If the provided combination of arguments is not in the active choices.
        """
        slow = is_xslow()

        if slow:
            # no need to skip
            return

        param_seed = kwargs.pop('param_seed', None)
        if param_seed is None:
            param_seed = 1

        params = kwargs.pop('params', None)
        if params is None:
            params = self.params

        num_param_combinations = kwargs.pop('num_param_combinations', None)
        if num_param_combinations is None:
            num_param_combinations = self.num_param_combinations

        all_choices = list(itertools.product(*params))

        rng = random.Random(param_seed)
        rng.shuffle(all_choices)
        active_choices = all_choices[:num_param_combinations]

        if args not in active_choices:
            raise NotImplementedError("skipped")


def get_max_rss_bytes(rusage):
    """
    Extract the max RSS value in bytes.

    Parameters
    ----------
    rusage : resource.struct_rusage
        Resource usage object containing memory statistics.

    Returns
    -------
    int or None
        Maximum RSS (Resident Set Size) value in bytes.
        Returns None if rusage is empty.
    """
    if not rusage:
        return None

    if sys.platform.startswith('linux'):
        # On Linux getrusage() returns ru_maxrss in kilobytes
        # https://man7.org/linux/man-pages/man2/getrusage.2.html
        return rusage.ru_maxrss * 1024
    elif sys.platform == "darwin":
        # on macOS ru_maxrss is in bytes
        return rusage.ru_maxrss
    else:
        # Unknown platform, return whatever is available.
        return rusage.ru_maxrss


def run_monitored_wait4(code):
    """
    Run code in a new Python process, and monitor peak memory usage.

    Parameters
    ----------
    code : str
        Python code to execute in the subprocess.

    Returns
    -------
    duration : float
        Duration in seconds (including Python startup time).
    peak_memusage : int
        Peak memory usage in bytes of the child Python process.

    Raises
    ------
    AssertionError
        If the execution of the code returns a non-zero exit code.

    Notes
    -----
    Works on Unix platforms (Linux, macOS) that have `os.wait4()`.
    """
    code = textwrap.dedent(code)

    start = time.time()
    process = subprocess.Popen([sys.executable, '-c', code])
    pid, returncode, rusage = os.wait4(process.pid, 0)
    duration = time.time() - start
    max_rss_bytes = get_max_rss_bytes(rusage)

    if returncode != 0:
        raise AssertionError("Running failed:\n%s" % code)

    return duration, max_rss_bytes


def run_monitored_proc(code):
    """
    Run code in a new Python process, and monitor peak memory usage.

    Parameters
    ----------
    code : str
        Python code to execute in the subprocess.

    Returns
    -------
    duration : float
        Duration in seconds (including Python startup time).
    """
    pass
    # peak_memusage : float
    # Peak memory usage (rough estimate only) in bytes
    """
    检查当前系统是否为 Linux，否则抛出运行时错误
    """
    if not sys.platform.startswith('linux'):
        raise RuntimeError("Peak memory monitoring only works on Linux")

    # 格式化代码文本，去除缩进
    code = textwrap.dedent(code)
    # 在子进程中执行 Python 代码
    process = subprocess.Popen([sys.executable, '-c', code])

    # 初始化最高内存使用量为负一
    peak_memusage = -1

    # 记录开始时间
    start = time.time()
    # 循环监视子进程状态
    while True:
        ret = process.poll()
        # 如果子进程结束则退出循环
        if ret is not None:
            break

        # 读取子进程的状态信息
        with open('/proc/%d/status' % process.pid) as f:
            procdata = f.read()

        # 使用正则表达式匹配并获取当前进程的物理内存使用量
        m = re.search(r'VmRSS:\s*(\d+)\s*kB', procdata, re.S | re.I)
        if m is not None:
            # 将获取的内存使用量转换为字节
            memusage = float(m.group(1)) * 1e3
            # 更新最高内存使用量
            peak_memusage = max(memusage, peak_memusage)

        # 等待一段时间再继续检查
        time.sleep(0.01)

    # 等待子进程结束
    process.wait()

    # 计算代码执行的时长
    duration = time.time() - start

    # 如果子进程返回值不为 0，抛出断言错误
    if process.returncode != 0:
        raise AssertionError("Running failed:\n%s" % code)

    # 返回代码执行的时长和最高内存使用量
    return duration, peak_memusage
# 定义一个函数 `run_monitored`，用于在新的 Python 进程中运行代码，并监控内存峰值使用情况。
def run_monitored(code):
    """
    Run code in a new Python process, and monitor peak memory usage.

    Returns
    -------
    duration : float
        Duration in seconds (including Python startup time)
    peak_memusage : float or int
        Peak memory usage (rough estimate only) in bytes

    """

    # 检查操作系统模块是否具有 `wait4` 属性，选择相应的监控方式
    if hasattr(os, 'wait4'):
        return run_monitored_wait4(code)
    else:
        return run_monitored_proc(code)


# 定义函数 `get_mem_info`，用于获取可用内存信息
def get_mem_info():
    """Get information about available memory"""
    # 导入 psutil 模块，获取虚拟内存信息
    import psutil
    vm = psutil.virtual_memory()
    # 返回包含总内存和可用内存信息的字典
    return {
        "memtotal": vm.total,
        "memavailable": vm.available,
    }


# 定义函数 `set_mem_rlimit`，用于设置地址空间的资源限制
def set_mem_rlimit(max_mem=None):
    """
    Set address space rlimit
    """
    # 导入 resource 模块
    import resource
    # 如果未指定最大内存限制，则根据系统内存总量的 70% 计算
    if max_mem is None:
        mem_info = get_mem_info()
        max_mem = int(mem_info['memtotal'] * 0.7)
    # 获取当前地址空间限制
    cur_limit = resource.getrlimit(resource.RLIMIT_AS)
    # 如果当前限制大于 0，则将最大内存限制设置为当前限制与计算值的较小者
    if cur_limit[0] > 0:
        max_mem = min(max_mem, cur_limit[0])

    try:
        # 尝试设置地址空间的新限制
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, cur_limit[1]))
    except ValueError:
        # 在 macOS 上可能会引发异常：当前限制超过最大限制
        pass


# 定义函数 `with_attributes`，返回一个装饰器函数，用于为函数添加属性
def with_attributes(**attrs):
    def decorator(func):
        # 遍历属性字典，为函数设置相应的属性
        for key, value in attrs.items():
            setattr(func, key, value)
        return func
    return decorator


# 定义 `safe_import` 类，实现上下文管理器用于安全导入模块
class safe_import:

    def __enter__(self):
        # 进入上下文时，初始化错误标志为 False，并返回自身实例
        self.error = False
        return self

    def __exit__(self, type_, value, traceback):
        # 退出上下文时，检查是否有异常发生
        if type_ is not None:
            self.error = True
            # 根据环境变量 `SCIPY_ALLOW_BENCH_IMPORT_ERRORS` 控制是否抑制 ImportError 异常
            suppress = not (
                os.getenv('SCIPY_ALLOW_BENCH_IMPORT_ERRORS', '1').lower() in
                ('0', 'false') or not issubclass(type_, ImportError))
            return suppress
```
# `.\pytorch\torch\_inductor\runtime\runtime_utils.py`

```py
# mypy: allow-untyped-defs
# 导入未被类型化的函数定义支持
from __future__ import annotations

# 导入标准库模块
import functools
import getpass
import inspect
import operator
import os
import re
import tempfile
import time

# 导入第三方库
import torch


def conditional_product(*args):
    # 对传入参数进行条件乘积运算
    return functools.reduce(operator.mul, [x for x in args if x])


def ceildiv(numer: int, denom: int) -> int:
    # 向上取整除法运算
    return -(numer // -denom)


def is_power_of_2(n: int) -> bool:
    """Returns whether n = 2 ** m for some integer m."""
    # 检查 n 是否是 2 的整数次幂
    return n > 0 and n & n - 1 == 0


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    # 返回大于等于 n 的最小的 2 的幂次方
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def get_num_bytes(*args: torch.Tensor, num_in_out_args: int = 0) -> int:
    """
    Return the total number of bytes the arguments of tensor type takes.

    For in/out args, tensor sizes are counted twice: once for reading and
    once for writing.

    The first num_in_out_args arguments are in out tensors.
    """
    # 计算所有张量类型参数所占用的总字节数
    return sum(
        arg.numel() * arg.element_size() * (1 + int(i < num_in_out_args))
        for i, arg in enumerate(args)
        if isinstance(arg, torch.Tensor)
    )


def triton_config_to_hashable(cfg):
    """
    Convert triton config to a tuple that can uniquely identify it. We can use
    the return value as a dictionary key.
    """
    # 将 Triton 配置转换为可唯一标识的元组，用作字典键
    items = sorted(cfg.kwargs.items())
    items.append(("num_warps", cfg.num_warps))
    items.append(("num_stages", cfg.num_stages))
    return tuple(items)


def create_bandwidth_info_str(ms, num_gb, gb_per_s, prefix="", suffix="", color=True):
    # 创建描述带宽信息的字符串，包括延迟、数据量和速率
    info_str = f"{prefix}{ms:.3f}ms    \t{num_gb:.3f} GB \t {gb_per_s:7.2f}GB/s{suffix}"
    slow = ms > 0.012 and gb_per_s < 650
    return red_text(info_str) if color and slow else info_str


def get_max_y_grid():
    # 返回最大的 Y 轴网格数
    return 65535


def do_bench(fn, fn_args, fn_kwargs, **kwargs):
    from torch._inductor.utils import is_cpu_device

    # 获取所有函数参数，并根据设备类型选择性执行基准测试
    args = list(fn_args)
    args.extend(fn_kwargs.values())
    if is_cpu_device(args):
        return do_bench_cpu(lambda: fn(*fn_args, **fn_kwargs), **kwargs)
    else:
        return do_bench_gpu(lambda: fn(*fn_args, **fn_kwargs), **kwargs)


def do_bench_gpu(*args, **kwargs):
    @functools.lru_cache(None)
    # 使用 functools.lru_cache 对 GPU 上的基准测试进行缓存
    def load_triton():
        try:
            # 尝试延迟加载 Triton，因为导入 Triton 模块速度较慢
            # 参见 https://github.com/openai/triton/issues/1599
            from triton.testing import do_bench as triton_do_bench
        except ImportError as exc:
            # 如果导入失败，抛出 NotImplementedError 异常
            raise NotImplementedError("requires Triton") from exc

        # triton 的 PR https://github.com/openai/triton/pull/1513 更改了
        # quantile 字段名从 'percentiles' 变为 'quantiles'
        # 并将默认值从 (0.5, 0.2, 0.8) 更改为 None。
        # 这可能会在调用者期望获得一个元组时获取一个项目时导致问题。
        #
        # 添加一个包装器来保持对 Inductor 的相同行为。
        # 或许我们应该有自己的实现来处理这个函数？
        return triton_do_bench, (
            "quantiles"
            if inspect.signature(triton_do_bench).parameters.get("quantiles")
            is not None
            else "percentiles"
        )

    # 加载 Triton 模块并获取相关函数及量化字段名
    triton_do_bench, quantile_field_name = load_triton()

    # 如果参数中没有 quantile_field_name 指定的字段名，则设置默认值 (0.5, 0.2, 0.8)
    if quantile_field_name not in kwargs:
        kwargs[quantile_field_name] = (0.5, 0.2, 0.8)
    
    # 调用 triton_do_bench 函数，并返回其第一个返回值
    return triton_do_bench(*args, **kwargs)[0]
# 定义一个函数，用于执行 CPU 性能测试
def do_bench_cpu(fn, warmup=5, times=20):
    # 断言测试次数大于0
    assert times > 0
    # 预热阶段，执行指定次数的函数调用
    for _ in range(warmup):
        fn()
    durations = []
    # 进行多次性能测试
    for _ in range(times):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        # 计算每次执行的时间并添加到持续时间列表中
        durations.append((t1 - t0) * 1000)
    # 返回中位数时间作为性能测试结果
    sorted_durations = sorted(durations)
    if times % 2 == 0:
        return (sorted_durations[times // 2 - 1] + sorted_durations[times // 2]) / 2
    else:
        return sorted_durations[times // 2]


# 获取或创建缓存目录路径
def cache_dir() -> str:
    # 获取环境变量中的缓存目录路径
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        # 如果环境变量中未设置缓存目录，则使用默认的缓存目录路径
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir = default_cache_dir()
    # 确保缓存目录存在，如果不存在则创建
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


# 生成默认的缓存目录路径
def default_cache_dir():
    # 获取当前用户的用户名，并进行字符替换，生成安全的用户名字符串
    sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
    # 返回由临时目录和用户名组成的缓存目录路径
    return os.path.join(
        tempfile.gettempdir(),
        "torchinductor_" + sanitized_username,
    )


try:
    # 尝试导入 colorama 模块
    import colorama
    # 设置标志，表示 colorama 已成功导入
    HAS_COLORAMA = True
except ModuleNotFoundError:
    # 若导入失败，则设置标志为 False，并将 colorama 设为 None
    HAS_COLORAMA = False
    colorama = None  # type: ignore[assignment]


# 根据消息和颜色生成带有颜色的文本
def _color_text(msg, color):
    # 若 colorama 未导入，则直接返回原始消息
    if not HAS_COLORAMA:
        return msg
    # 使用 colorama 库设置指定颜色的文本
    return getattr(colorama.Fore, color.upper()) + msg + colorama.Fore.RESET


# 返回绿色文本
def green_text(msg):
    return _color_text(msg, "green")


# 返回黄色文本
def yellow_text(msg):
    return _color_text(msg, "yellow")


# 返回红色文本
def red_text(msg):
    return _color_text(msg, "red")


# 返回蓝色文本
def blue_text(msg):
    return _color_text(msg, "blue")


# 获取对象的第一个可用属性
def get_first_attr(obj, *attrs):
    """
    Return the first available attribute or throw an exception if none is present.
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    # 若无任何可用属性，则抛出异常
    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


try:
    # 尝试获取 torch._dynamo.utils.dynamo_timed 的别名
    dynamo_timed = torch._dynamo.utils.dynamo_timed
except AttributeError:
    # 若属性错误，说明当前环境下没有实际的 dynamo_timed 函数，因此定义一个简单的占位函数
    # 在编译工作者（如编译后的环境）中，只提供一个 mock 版本的 torch 模块
    def dynamo_timed(original_function=None, phase_name=None, fwd_only=True):
        if original_function:
            return original_function
        return dynamo_timed
```